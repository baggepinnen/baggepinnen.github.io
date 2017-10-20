
# Pkg.clone("https://github.com/tbreloff/MetaPkg.jl")
# using MetaPkg
# meta_add("MetaRL")
# Pkg.add("Plots")
# Pkg.add("BasisFunctionExpansions")
# Pkg.add("ValueHistories")
@everywhere begin
    using OpenAIGym, ValueHistories, Plots


    default(size=(1200,800)) # Set the default plot size
    function update_plot!(p; max_history = 10)
        num_series = length(p.series_list)
        if num_series > 1
            if num_series > max_history
                deleteat!(p.series_list,1:num_series-max_history)
            end
        end
    end


    env = GymEnv("CartPole-v0")

    function collect_reward(ep)
        r = Vector{Float64}(0)
        for (ss,aa,rr::Float64,ss1) in ep
            push!(r,rr)
        end
        r
    end


    struct Policy <: AbstractPolicy
        θ::Vector{Float64}
    end

    (π::Policy)(s) = π.θ's + 0.5 > 0.5 ? 1 :  0 # policy function

    Reinforce.action(π::Policy, r, s, A) = π(s)

    num_params  = length(env.state)
    num_epochs  = 15
    num_samples = 25
    const π     = Policy(zeros(num_params)) # π is now our policy object


    function fit_new_distribution_weighted(θs, w)
        weights = ProbabilityWeights(exp.(w))
        μ       = mean(θs, weights, 2)
        Σ       = cov(θs, weights, 2, corrected=true)
        Σ       = chol(Symmetric(Σ + 1e-5I))'
        μ, Σ
    end

    function fit_new_distribution_rank(θs, w)
        fraction = 2
        p        = sortperm(w)
        indicies = p[end-end÷fraction:end]
        μ        = mean(θs[:,indicies], 2)
        Σ        = cov(θs[:,indicies], 2)
        Σ        = chol(Symmetric(Σ))'
        μ, Σ
    end


    gr() # Set GR as plot backend, try pyplot() if GR is not working
    function CME(π, num_epochs, num_samples; plotting=true)
        Σ = 0.2eye(num_params) # Initial noise chol(covariance)
        μ = zeros(num_params)  # Initial policy mean
        plotting && (fig = plot(layout=2, show=true))
        reward_history = ValueHistories.History(Float64)
        for i = 1:num_epochs
            θs = Σ*randn(num_params, num_samples) .+ μ # Sample new policies
            weights = zeros(num_samples)
            for n = 1:num_samples # Collect N trajectories for Expectation estimation
                π.θ       .= θs[:,n] # Set policy parameters
                ep         = Episode(env, π)
                s,a,r,s1   = collect_reward(ep)
                weights[n] = ep.total_reward
                push!(reward_history, (i-1)*num_samples+n, ep.total_reward)
            end
            μ, Σ = fit_new_distribution(θs, weights)

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
        plot(reward_history, title="Rewards", xlabel="Episodes", show=true),reward_history
    end
end


N = 50
gr()
run_many(N) = pmap(i->CME(π, num_epochs, num_samples; plotting=false)[2], 1:N)
function plot_progress(rewards, pf=plot; kwargs...)
    mat = hcat([r.values for r in rewards]...)
    m = mean(mat,2)[:]
    mat = sort(mat, 2)
    sn = mat[:,end÷10]
    sp = mat[:,end-end÷10]
    pf(m; fillrange=sn, title="Average reward history", xlabel="Iteration", kwargs...)
    plot!(m; fillrange=sp, kwargs...)
end


@everywhere fit_new_distribution = fit_new_distribution_weighted
@everywhere srand(1);
results_weighted = run_many(N)

@everywhere fit_new_distribution = fit_new_distribution_rank
@everywhere srand(1);
results_rank = run_many(N)

# using JLD
@save "cem.jld"
pyplot(size=(640,480))
plot_progress(results_weighted, plot, c=:blue, lab="Weight based", opacity = 0.5, linewidth=3)
plot_progress(results_rank, plot!, c=:red, lab="Rank based", opacity = 0.4, linewidth=3)
