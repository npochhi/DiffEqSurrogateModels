using DifferentialEquations
using LatinHypercubeSampling
using Flux, Flux.Data.MNIST, Statistics
using Flux: onehotbatch, onecold, mse, throttle
using Base.Iterators: repeated

f = @ode_def LotkaVolterra begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a b c d

N = 1000
gens = 5

p = LHCoptim(N, 4, gens)[1] ./ (N / 10)
p += ones(size(p))

println("Generated param vals!")

u0 = [1.0; 1.0]
range_t = range(0.0, length=N, stop=10.0)
tspan = (0.0, 10.5)
sols = []
t = collect(range_t)

println("Started solving equations!")

for p_i in 1:size(p, 1)
  prob = ODEProblem(f, u0, tspan, p[p_i, :])
  sol = solve(prob, maxiters=1e7, force_dtmin=true)
  append!(sols, [sol(10.0)])
end

println("Collected diff eq data!")
sols = hcat(sols...)
p = p'

m = Chain(
  Dense(4, 5, relu),
  Dense(5, 2))

loss(x, y) = mse(m(x), y)
opt = ADAM()
dataset = repeated((p, sols), 10)

Flux.train!(loss, params(m), dataset, opt)

println(mean(loss(p, sols)))
