using BasisFunctionExpansions, ValueHistories, Plots, OrdinaryDiffEq, StaticArrays, BayesianTools.ProductDistributions, Optim, ControlSystems, OnlineStats
default(size=(1400,1000)) # Set the default plot size

cd(@__DIR__)
const g  = 9.82
const l1 = 0.15
const l2 = 0.5
const h  = 0.05
function fsys(xd, x, u = argmax_a(V,x))
    xd[1] = x[2]
    xd[2] = -g/l1 * sin(x[1]) + u/l1 * cos(x[1])
    xd[3] = x[4]
    xd[4] = -g/l2 * sin(x[3]) + u/l2 * cos(x[3])
    xd[5] = x[6]
    xd[6] = u
    xd
end

function simulate(N = 100)
    x0    = Float64[0,0,0,0,0,0] + 0.01randn(6)
    x = Vector{Vector{Float64}}(N+1)
    x[1] = x0
    a = Vector{Float64}(N)
    xd = similar(x0)
    @progress for i = 1:N
        ai = argmax_a(V,x[i])
        fsys(xd,x[i],ai)
        a[i] = ai
        x[i+1] = x[i] + h*xd
        x[i+1][1] = mod(x[i+1][1],2π)
        x[i+1][3] = mod(x[i+1][3],2π)
        x[i+1][5] = clamp.(x[i+1][5], -1, 1)
        if abs(x[i+1][5]) == 1
            x[i+1][6] *= -0.2
        end
    end
    x,a
end

eulerstep(x,u) = x .+ h.*fsys(similar(x),x,u)
const dists = ProductDistribution(Uniform(0,2π),Uniform(-10,10),Uniform(0,2π),Uniform(-10,10),Uniform(-1,1),Uniform(-3,3))

function sample!(s)
    rand!(dists,s)
end

function reward(s)
    r = - 0.1s[2]^2 - 0.1s[4]^2 - 0.01s[5]^2 - 0.01s[6]^2 - 10*(abs(s[5]) > 0.9)
    r -= 10sin((s[1]-pi)/2)^2
    r -= 10sin((s[3]-pi)/2)^2
end
bfe = MultiUniformRBFE([linspace(0,2π) linspace(-5,5) linspace(0,2π) linspace(-5,5) linspace(-1,1) linspace(-3,3)], [7,5,7,5,7,5])

struct Vfun{N}
    θ::Vector{Float64}
    bfe::MultiUniformRBFE{N}
end

(V::Vfun)(s) = V.bfe(s)⋅V.θ # This row makes our type Vfun callable
(V::Vfun)(s,a) = V.bfe(eulerstep(s,a))⋅V.θ # Calculate maxₐ V(s⁺,a)

"""This function makes for a nice syntax of updating the V-function"""
function Base.setindex!(V::Vfun, q, s)
    V.θ .+= V.bfe(s)* q
end

const V = Vfun(zeros(size(bfe.μ,2)), bfe) # V is now our V-function approximator
const omean = Mean(weight = x -> .01)

iters            = 10_000
α                = 0.1   # Initial learning rate
const decay_rate = 0.995 # decay rate for learning rate and ϵ
const γ          = 0.99  # Discounting factor

"""max(V(s,a)) over a"""
function max_a(V, s)
    optimize(a->-V(s,a[]), -5,5).minimum
end

function argmax_a(V, s)
    optimize(a->-V(s,a[]), -5,5).minimizer[]
end

gr() # Enable the pyplot backend, try gr insted if pyplot is slow
# gr()

function save(filename, θ)
    open(filename,"w") do f
        serialize(f,θ)
    end
end

function load(filename)
    local θ
    open(filename) do f
        θ = deserialize(f)
    end
    return θ
end

V.θ .= load("valueparams")

"""
VIlearning(iters, α; plotting=true)
Run `iters` iterations of value iteration with stepsize `α`. Instead of integrating over the dynamics, this function samples states and uses the policy a = argmaxₐ(V(s⁺))
"""
function VIlearning(iters, α; plotting=true)
    # plotting && (fig = plot(layout=2, show=true))
    temporal_diffs = ValueHistories.History(Float64)
    temporal_diffsm = ValueHistories.History(Float64)
    s = zeros(6) # Preallocated container to store samples
    @progress for iter = 1:iters
        sample!(s)
        r = reward(s)
        δ = r + γ*max_a(V, s) - V(s) # Temporal difference (TD) error
        V[s] = α*δ # Update the V-function approximator using V-learning
        fit!(omean, δ)
        push!(temporal_diffs, iter, δ)
        push!(temporal_diffsm, iter, omean.μ)
        if plotting && iter % 100 == 0
            save("valueparams",V.θ)
            plot(temporal_diffs, layout=2, subplot=1, reuse = true, show=false)
            plot!(temporal_diffsm, subplot=1)
            scatter!(V.θ, subplot=2, reuse=true)
            gui()
        end
    end
    temporal_diffs
end

trace = VIlearning(50_000, α, plotting = true)


x,a = simulate(200)
@show R = sum(reward,x)
X = hcat(x...)'
ControlSystems.unwrap!(@view(X[:,[1,3]]))
scatter(X, layout=8)
plot!(a, subplot=7)
scatter!(V.θ, subplot=8)
gui()


function fit_policy(V, α, iters = 1000)
    μ = Vfun(zeros(size(bfe.μ,2)), bfe)
    errors = ValueHistories.History(Float64)
    s = zeros(6) # Preallocated container to store samples
    @progress for iter = 1:iters
        α *= 1-1/iters
        sample!(s)
        δ = argmax_a(V, s) - μ(s) # Temporal difference (TD) error
        μ[s] = α*δ # Update the V-function approximator using V-learning
        push!(errors, iter, δ)
        if iter % 10 == 0
            save("policyparams",μ.θ)
            plot(errors, layout=2, subplot=1, reuse = true, show=false)
            scatter!(μ.θ, subplot=2, reuse=true)
            gui()
        end
    end
end

fit_policy(V, 1., 1000)
