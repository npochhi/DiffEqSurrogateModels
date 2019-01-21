using DifferentialEquations
using LatinHypercubeSampling
using Optim

f = @ode_def LotkaVolterra begin
  dx = a*x - b*x*y
  dy = -c*y + d*x*y
end a b c d

N = 1000
gens = 10

p = LHCoptim(N, 4, gens)[1] ./ (N / 10)
p += ones(size(p))

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

loss_x(coeffs_x) = sum([(sum(coeffs_x[1:end-1].*p[idx, :]) + coeffs_x[end] - sols[idx][1])^2 for idx in 1:size(p, 1)]) / size(p, 1)
loss_y(coeffs_y) = sum([(sum(coeffs_y[1:end-1].*p[idx, :]) + coeffs_y[end] - sols[idx][2])^2 for idx in 1:size(p, 1)]) / size(p, 1)

res_y = Optim.minimizer(optimize(loss_y, randn(size(p)[2] + 1), GradientDescent()))
res_x = Optim.minimizer(optimize(loss_x, randn(size(p)[2] + 1), GradientDescent()))

println(res_x)
println(res_y)

println("L2 Loss corresponding to x: $(loss_x(res_x))")
println("L2 Loss corresponding to y: $(loss_y(res_y))")
