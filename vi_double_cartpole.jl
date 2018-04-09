using BasisFunctionExpansions, ValueHistories, Plots, OrdinaryDiffEq, StaticArrays, BayesianTools.ProductDistributions, Optim
default(size=(1200,800)) # Set the default plot size

const g  = 9.82
const l1 = 0.1
const l2 = 0.5
const h  = 0.05
function fsys(xd,x,u)
    xd[1] = x[2]
    xd[2] = -g/l1 * sin(x[1]) + u/l1 * cos(x[1])
    xd[3] = x[4]
    xd[4] = -g/l2 * sin(x[3]) + u/l2 * cos(x[3])
    xd[5] = x[6]
    xd[6] = u
    xd
end

fsys(xd,x) = fsys(xd,x,argmax_a(Q,x))

function simulate()
    N = 100
    x0    = Float64[0,0,0,0,0,0] + 0.01randn(6)
    x = Vector{Vector{Float64}}(N+1)
    x[1] = x0
    a = Vector{Float64}(N)
    xd = similar(x0)
    @progress for i = 1:N
        ai = argmax_a(Q,x[i])
        fsys(xd,x[i],ai)
        a[i] = ai
        x[i+1] = x[i] + h*xd
        x[i+1][1] = mod(x[i+1][1],2π)
        x[i+1][3] = mod(x[i+1][3],2π)
        x[i+1][5] = clamp.(x[i+1][5], -1, 1)
        if abs(x[i+1][5]) == 1
            x[i+1][6] *= -1
        end
    end
    x,a
end

eulerstep(x,u) = x .+ h.*fsys(similar(x),x,u)
const dists = ProductDistribution(Uniform(0,2π),Uniform(-5,5),Uniform(0,2π),Uniform(-5,5),Uniform(-1,1),Uniform(-3,3))

function sample(s)
    rand!(dists,s)
end

function reward(s)
    r = - 0.01s[2]^2 - 0.01s[4]^2 - 0.01s[5]^2 - 0.01s[6]^2 - 10*(abs(s[5]) > 0.9) + 10
    r -= 10sin((s[1]-pi)/2)^2
    r -= 10sin((s[3]-pi)/2)^2
end
bfe = MultiUniformRBFE([linspace(0,2π) linspace(-5,5) linspace(0,2π) linspace(-5,5) linspace(-1,1) linspace(-3,3)], [6,6,6,6,6,6])

struct Qfun
    θ::Vector{Float64}
    bfe::MultiUniformRBFE
end

(Q::Qfun)(s) = Q.bfe(s)⋅Q.θ # This row makes our type Qfun callable
(Q::Qfun)(s,a) = Q.bfe(eulerstep(s,a))⋅Q.θ # This row makes our type Qfun callable

"""This function makes for a nice syntax of updating the Q-function"""
function Base.setindex!(Q::Qfun, q, s)
    Q.θ .+= Q.bfe(s)* q
end

const Q = Qfun(zeros(size(bfe.μ,2)), bfe) # Q is now our Q-function approximator

num_episodes     = 10000
α                = 0.5   # Initial learning rate
const decay_rate = 0.995 # decay rate for learning rate and ϵ
const γ          = 0.99  # Discounting factor

"""max(Q(s,a)) over a"""
function max_a(Q, s)
    optimize(a->-Q(s,a[]), -5,5).minimum
end

function argmax_a(Q, s)
    optimize(a->-Q(s,a[]), -5,5).minimizer[]
end

gr() # Enable the pyplot backend, try gr insted if pyplot is slow
# gr()

function VIlearning(num_episodes, α; plotting=true)
    # plotting && (fig = plot(layout=2, show=true))
    reward_history = ValueHistories.History(Float64)
    s = zeros(6)
    @progress for ep = 1:num_episodes
        sample(s)
        r = reward(s)
        δ = r + γ*max_a(Q, s) - Q(s)
        Q[s] = α*δ # Update the Q-function approximator using Q-learning
        push!(reward_history, ep, δ)
        if plotting && ep % 10 == 0
            plot(reward_history, layout=2, subplot=1, reuse = true, show=false)
            scatter!(Q.θ, subplot=2, reuse=true)
            gui()
        end
        if plotting && (ep-5) % 30 == 0
            x,a = simulate()
            @show R = sum(reward,x)
            plot(hcat(x...)', layout=7)
            plot!(a, subplot=7)
            gui()
        end
    end
    reward_history
end

@time VIlearning(num_episodes, α, plotting = true)
