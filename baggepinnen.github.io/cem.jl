# # Reinforcement learning with Cross-entropy method optimization (CEM)

# CEM performs a policy search with exploration directly in parameter space. A collection of sample policies are sampled from a distribution and tried through experiments. A new, weighted distribution is fit to the sampled policy parameters, where the reward achieved by the policy determines its weight. This way, policies that achieved high rewards are more likely to be sampled next time. This procedure is iterated until performance is adequate.

# CEM works well when simulation is computationally cheap and the parameter dimension is small. We therefore use a linear combination of the states as policy in this example, since the linear combination of basis function leads to a high-dimensional parameter space.

# We focus on Gaussian distributions to represent policy distributions in parameter space, and illustrate two methods of fitting new weighted distributions: rank-based and Boltzmann-based. The rank-based fit fits and unweighted Gaussian to the top fraction of policies, whereas the Boltzmann-based strategy assigns a weight of $\exp(r_i)$ to policy $i$, where $r_i$ is the total reward achieved by that policy.

# For further introduction, see [Levine 2017 (slides)](http://rll.berkeley.edu/deeprlcourse/f17docs/lecture_7_advanced_q_learning.pdf)

# This tutorial is available as an [iJulia notebook](https://github.com/baggepinnen/baggepinnen.github.io/blob/master/cem.ipynb)

# We start out by importing some packages. If it's your first time, you might have to install some packages using the commands in the initial comment. OpenAI gym is installed with instructions available at https://github.com/JuliaML/OpenAIGym.jl

# using Pkg
# Pkg.add("Plots")
# Pkg.add("BasisFunctionExpansions")
# Pkg.add("ValueHistories")
# Pkg.add("StatsBase"); Pkg.add("Distributions")
# Pkg.add("https://github.com/JuliaML/OpenAIGym.jl")

using OpenAIGym, ValueHistories, Plots, LinearAlgebra, Random, Statistics, StatsBase, Distributions


# If you want to perform plotting in the loop, the following function helps keeping the plot clean.
default(size=(1200,800)) # Set the default plot size
function update_plot!(p; max_history = 10)
    num_series = length(p.series_list)
    if num_series > 1
        if num_series > max_history
            deleteat!(p.series_list,1:num_series-max_history)
        end
    end
end


# Next, we define an environment from the OpenAIGym framework, we'll use the [cartpole environment](https://gym.openai.com/envs/CartPole-v0/) and a function that runs an entire episode.

env = GymEnv("CartPole-v0")

function collect_reward(ep)
    r = Vector{Float64}()
    for (ss,aa,rr,ss1) in ep
        push!(r,rr)
    end
    r
end


# We also define a linear policy object that is a linear combination of the states.

struct Policy <: AbstractPolicy
    θ::Vector{Float64}
end

(π::Policy)(s) = π.θ's + 0.5 > 0.5 ? 1 :  0 # policy function

Reinforce.action(π::Policy, r, s, A) = π(s)

num_params  = length(env.state)
num_epochs  = 15
num_samples = 25
const π     = Policy(zeros(num_params)) # π is now our policy object


# We have now arrived at the main algorithm. We wrap it in a function for the Julia JIT complier to have it run faster.

function fit_new_distribution_weighted(θs, w)
    weights = ProbabilityWeights(exp.(w))
    μ       = mean(θs, weights, dims=2)[:]
    Σ       = cov(θs, weights, 2, corrected=true) + 1e-5I
    MvNormal(μ, Σ)
end

function fit_new_distribution_rank(θs, w)
    fraction = 2
    p        = sortperm(w)
    indicies = p[end-end÷fraction:end]
    μ        = mean(θs[:,indicies], dims=2)[:]
    Σ        = cov(θs[:,indicies], dims=2) + 1e-5I
    MvNormal(μ, Σ)
end

gr() # Set GR as plot backend, try pyplot() if GR is not working
function CME(π, num_epochs, num_samples; plotting=true)
    dist = MvNormal(zeros(num_params), 0.2Matrix{Float64}(I,num_params,num_params)) # Initial noise
    plotting && (fig = plot(layout=2, show=true))
    reward_history = ValueHistories.History(Float64)
    for i = 1:num_epochs
        θs = rand(dist, num_samples) # Sample new policies
        weights = zeros(num_samples)
        for n = 1:num_samples # Collect N trajectories for Expectation estimation
            π.θ       .= θs[:,n] # Set policy parameters
            ep         = Episode(env, π)
            s,a,r,s1   = collect_reward(ep)
            weights[n] = ep.total_reward
            push!(reward_history, (i-1)*num_samples+n, ep.total_reward)
        end
        dist = fit_new_distribution(θs, weights)

        # Printing and plotting
        println("Epoch: $i, reward: $(reward_history.values[end])")
        if plotting
            plot!(reward_history, subplot=1, show=false)
            scatter!(θs, subplot=2, c=:red, title="Policy parameters", show=false, legend=false)
            update_plot!(fig[1], max_history=1)
            update_plot!(fig[2], max_history=num_samples)
            gui(fig)
        end
    end
    plot(reward_history, title="Rewards", xlabel="Episodes", show=true)
end


# We now call our function, with the two different distribution-refit strategies:

fit_new_distribution = fit_new_distribution_weighted
Random.seed!(0);
CME(π, num_epochs, num_samples; plotting=false)

#

fit_new_distribution = fit_new_distribution_rank
Random.seed!(0);
CME(π, num_epochs, num_samples; plotting=false)


# If we run many experiments, we can plot the average reward history. The following plot depicts the average over 50 parallel runs.

\fig{cem.png}
# As we can see, the weight-based method finds a good solution faster, but fails to zoom in on a perfect policy like the rank-based method does after a while. Note, very little time was spent on tuning the hyper parameters of the two methods. The Boltzmann method can be tuned with a temperature parameter that controls the entropy of the distribution, this will allow for also this method to zoom in on a good policy.
