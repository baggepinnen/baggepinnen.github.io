include("/local/home/fredrikb/.julia/dev/KalmanTree/src/tree_tools.jl")
include("/local/home/fredrikb/.julia/dev/KalmanTree/src/domain_tools.jl")
include("/local/home/fredrikb/.julia/dev/KalmanTree/src/models.jl")
# using Pkg
# Pkg.add("Plots")
# Pkg.add("BasisFunctionExpansions")
# Pkg.add("ValueHistories")
# Pkg.add("https://github.com/JuliaML/OpenAIGym.jl")

using OpenAIGym, ValueHistories, Plots, Random, LinearAlgebra, Hyperopt
default(size=(1800,1000))


nx,nu = 4,1
domain = [(0,1), (-1.1,1.1) ,(-3.5,3.5) ,(-0.3,0.3) ,(-3.5,3.5)]

splitter = NormalizedTraceSplitter(2:5)

struct Qfun{T1,T2}
    grid::T1
    splitter::T2
end

(Q::Qfun)(s,a) = predict(Q.grid, s, a)

"""This function makes for a nice syntax of updating the Q-function"""
function Base.setindex!(Q::Qfun, q, s, a)
    update!(Q.grid, s, a, q)
end


mutable struct ϵGreedyPolicy{T} <: AbstractPolicy
    ϵ::Float64
    decay_rate::Float64
    Q::T
end

"""Calling this function decays the ϵ"""
function decay!(p::ϵGreedyPolicy)
    p.ϵ *= p.decay_rate
end

"""This is our ϵ-greedy action function"""
function Reinforce.action(p::ϵGreedyPolicy, r, s, A)
    rand() < p.ϵ ? rand(0:1) : p.Q(s,1) > p.Q(s,0) ? 1 : 0
end

"""max(Q(s,a)) over a"""
function max_a(Q, s)
    max(Q(s,1), Q(s,0))
end

argmax_a(Q,s) = Q(s,1) > Q(s,0) ? 1 : 0

env = GymEnv("CartPole-v0");

function Qlearning(Q, policy, num_episodes; plotting=true, target_update_interval=100)
    γ            = 0.99; # Discounting factor
    Qt = deepcopy(Q)
    plotting && (fig = plot(layout=2, show=true))
    reward_history = ValueHistories.History(Float64)
    for i = 1:num_episodes
        ep = Episode(env, policy)
        decay!(policy) # Decay greedyness
        if i % target_update_interval == 0
            Qt = deepcopy(Q)
            # splitter(Q.grid)
            # @show countnodes(Q.grid)
        end
        for (s,a,r,s1) in ep # An episode object is iterable
            # Q[s,a] = r + γ*max_a(Q, s1) # Update Q using Q-learning
            # TODO: if using newton updater, use inverse variance as preconditioner in calculating newton direction
            Q[s,a] = r + γ*Qt(s1,argmax_a(Q, s1)) # Update Q using double Q-learning
        end
        push!(reward_history, i, ep.total_reward)
        i % 20 == 0 && println("Episode: $i, reward: $(ep.total_reward)")
        if plotting && i % 50 == 0
            p1 = plot(reward_history, show=false)
            p2 = gridmat(Q.grid, show=false, axis=false)
            plot(p1,p2); gui()
        end
    end
    plot(reward_history, title="Rewards", xlabel="Episode", show=true)
    reward_history
end
λ, tui, P0 = 0.01, 5, 10
tuir = round.(Int, exp10.(range(0, stop=2, length=200)))
ho = @hyperopt for i=20, sampler = BlueNoiseSampler(),
    α   = exp10.(range(-5, stop=-0.5, length=200)),
    # λ   = exp10.(LinRange(-5,3,200)),
    # λ = 1 .- exp10.(LinRange(-4,-2,50)),
    tui = tuir
    # P0  = exp10.(range(-1, stop=3, length=200))
    #     @benchmark begin
    # updater = NewtonUpdater(α, 0.999)
    updater = GradientUpdater(α, 0.999)
    m     = QuadraticModel(nx+nu; actiondims = 1:1, updater=updater)
    # m     = QuadraticModel(nx+nu; actiondims = 1:1, λ = λ, P0 = P0)
    gridm = Grid(domain, m, splitter, initial_split =0)
    Q     = Qfun(gridm, splitter); # Q is now our Q-function approximator
    ϵ     = 0.5 # Initial chance of choosing random action
    num_episodes = 400
    decay_rate   = 0.995 # decay rate for learning rate and ϵ
    policy       = ϵGreedyPolicy(ϵ, decay_rate, Q);
    rh = Qlearning(Q, policy, num_episodes, plotting = false, target_update_interval=tui)
    mean(rh.values[end-100:end])
end

plot(ho);gui()
##
error()
# julia> maximum(ho)
# (Real[0.67, 0.999854, 28, 6.25055], 200.0)

##
struct BoltzmannPolicy <: AbstractPolicy end

decay!(policy::BoltzmannPolicy) = nothing

"""This is our Boltzmann exploration action function"""
function Reinforce.action(policy::BoltzmannPolicy, r, s, A)
    Q1,Q0 = Q(s,1), Q(s,0)
    prob1 = exp(Q1)/(exp(Q1)+exp(Q0))
    rand() < prob1 ? 1 : 0
end

policy = BoltzmannPolicy()
updater = NewtonUpdater(0.1, 0.999)
m     = QuadraticModel(nx+nu; actiondims = 1:1, updater=updater)
# m     = QuadraticModel(nx+nu; actiondims = 1:1, λ = λ, P0 = P0)
gridm = Grid(domain, m, splitter, initial_split =0)
Q     = Qfun(gridm, splitter); # Q is now our Q-function approximator

@time Qlearning(Q,policy, num_episodes, α, plotting = true, target_update_interval=tui)
