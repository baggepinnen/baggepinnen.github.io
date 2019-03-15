using Flux, ValueHistories, Plots, OrdinaryDiffEq, StaticArrays, BayesianTools.ProductDistributions, Optim, ControlSystems, OnlineStats
using Flux: ADAM, params, data
default(size=(1400,1000)) # Set the default plot size

cd(@__DIR__)
const g  = 9.82
const l1 = 0.15
const l2 = 0.5
const h  = 0.05
const nstates = 4
function fsys(xd, x, u = argmax_a(V,x))
    xd[1] = x[2]
    xd[2] = u
    xd[3] = x[4]
    xd[4] = -g/l2 * sin(x[3])
    if x[1] != 1 && x[1] != -1
        xd[4] += u/l2 * cos(x[3])
    end
    # xd[5] = x[6]
    # xd[6] = -g/l1 * sin(x[5]) + u/l1 * cos(x[5])
    xd
end

function simulate(N = 100)
    x0   = 0.01randn(nstates) # + [0,0,pi,0]
    x    = Vector{Vector{Float64}}(N+1)
    x[1] = x0
    a    = Vector{Float64}(N)
    xd   = similar(x0)
    @progress for i = 1:N
        ai = argmax_a(V,x[i])
        fsys(xd,x[i],ai)
        a[i] = ai
        x[i+1] = x[i] + h*xd
        x[i+1][1] = clamp.(x[i+1][1], -1, 1)
        x[i+1][3] = mod(x[i+1][3],2π)
        # x[i+1][5] = mod(x[i+1][5],2π)
        if abs(x[i+1][1]) == 1
            x[i+1][2] *= -0.2
        end
    end
    x,a
end

eulerstep(x,u) = x .+ h.*fsys(similar(x),x,u)
const dists = ProductDistribution(Uniform(-1,1),Uniform(-3,3),Uniform(0,2π),Uniform(-10,10))#,Uniform(0,2π),Uniform(-10,10))

function sample!(s)
    rand!(dists,s)
end

function reward(s,a)
    r = - s[1]^2 - s[2]^2 - 10*(abs(s[1]) > 0.9)
    r -= 10sin((s[3]-pi)/2)^2 +  0.1s[4]^2
    # r -= 10sin((s[5]-pi)/2)^2 +  0.1s[6]^2
    r -= 0.5a^2
end


iters            = 10_000
α                = 0.1   # Initial learning rate
const decay_rate = 0.995 # decay rate for learning rate and ϵ
const γ          = 0.99  # Discounting factor

function opt_a(V, s)
    optimize(a->-data(V(s,a))[], -5,5)
end

"""max(V(s,a)) over a"""
max_a(V, s) = opt_a(V, s).minimum
argmax_a(V, s) = opt_a(V, s).minimizer[]


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

const V = Chain(Dense(nstates,20,Flux.σ), Dense(20,10,Flux.σ), Dense(10,1))
opt = Flux.ADAM(params(V))
(V::typeof(V))(s,a) = V(eulerstep(s,a)) # Calculate maxₐ V(s⁺,a)

"""
VIlearning(iters, α; plotting=true)
Run `iters` iterations of value iteration with stepsize `α`. Instead of integrating over the dynamics, this function samples states and uses the policy a = argmaxₐ(V(s⁺))
"""
function VIlearning(V,opt, iters; plotting=true)
    Vc = deepcopy(V)
    # plotting && (fig = plot(layout=2, show=true))
    losses = ValueHistories.History(Float64)
    loss(s,y) = sum(abs.(V(s).-y))
    s = zeros(nstates) # Preallocated container to store samples
    @progress for iter = 1:iters
        sample!(s)
        res = opt_a(Vc,s)
        a = res.minimizer[]
        r = reward(s,a)
        y = r + γ*res.minimum
        tl = Flux.train!(loss, [(s,y)], opt)
        Flux.train!(loss, [([0,0,pi,0],0)], opt)
        if iter % 100 == 0
            push!(losses, iter, tl)
        end
        if iter % 10000 == 0
            Vc = deepcopy(V)
            save("valueparams",V)
        end
        if iter % 1000 == 0
            plot(losses, reuse = true, layout=5)
            plot!(linspace(-1,1), s->data(V([s,0,pi,0]))[], subplot=2, xlabel="Pos (at top)")
            plot!(linspace(-3,3), s->data(V([0,s,pi,0]))[], subplot=3, xlabel="Vel (at top)")
            plot!(linspace(0,2pi), s->data(V([0,0,s,0]))[], subplot=4, xlabel="Angle")
            plot!(linspace(-5,5), a->data(V([0,0,pi,0],a))[], subplot=5, xlabel="Action")
            gui()
        end
    end
    losses
end

trace = VIlearning(V,opt, 1_000_000, plotting = true)


x,a = simulate(200)
@show R = sum(x->reward(x...),zip(x,a))
X = hcat(x...)'
ControlSystems.unwrap!(@view(X[:,[3]]))
scatter(X, layout=(4,2))
plot!(a, subplot=7)
gui()

#
# function fit_policy(V, α, iters = 1000)
#     μ = Vfun(zeros(size(bfe.μ,2)), bfe)
#     errors = ValueHistories.History(Float64)
#     s = zeros(6) # Preallocated container to store samples
#     @progress for iter = 1:iters
#         α *= 1-1/iters
#         sample!(s)
#         δ = argmax_a(V, s) - μ(s) # Temporal difference (TD) error
#         μ[s] = α*δ # Update the V-function approximator using V-learning
#         push!(errors, iter, δ)
#         if iter % 10 == 0
#             save("policyparams",μ.θ)
#             plot(errors, layout=2, subplot=1, reuse = true, show=false)
#             scatter!(μ.θ, subplot=2, reuse=true)
#             gui()
#         end
#     end
# end
#
# fit_policy(V, 1., 1000)
