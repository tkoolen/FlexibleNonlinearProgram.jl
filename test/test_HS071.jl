# Test code adapted from MathProgBase (test/nlp.jl)

type HS071 <: MathProgBase.AbstractNLPEvaluator
end

# hs071
# min x1 * x4 * (x1 + x2 + x3) + x3
# st  x1 * x2 * x3 * x4 >= 25
#     x1^2 + x2^2 + x3^2 + x4^2 = 40
#     1 <= x1, x2, x3, x4 <= 5
# Start at (1,5,5,1)
# End at (1.000..., 4.743..., 3.821..., 1.379...)

function MathProgBase.initialize(d::HS071, requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in [:Grad, :Jac, :Hess])
            error("Unsupported feature $feat")
            # TODO: implement Jac-vec and Hess-vec products
            # for solvers that need them
        end
    end
end

MathProgBase.features_available(d::HS071) = [:Grad, :Jac, :Hess]

MathProgBase.eval_f(d::HS071, x) = x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]

function MathProgBase.eval_g(d::HS071, g, x)
    g[1] = x[1]   * x[2]   * x[3]   * x[4]
    g[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
end

function MathProgBase.eval_grad_f(d::HS071, grad_f, x)
    grad_f[1] = x[1] * x[4] + x[4] * (x[1] + x[2] + x[3])
    grad_f[2] = x[1] * x[4]
    grad_f[3] = x[1] * x[4] + 1
    grad_f[4] = x[1] * (x[1] + x[2] + x[3])
end

MathProgBase.jac_structure(d::HS071) = [1,1,1,1,2,2,2,2],[1,2,3,4,1,2,3,4]
# lower triangle only
MathProgBase.hesslag_structure(d::HS071) = [1,2,2,3,3,3,4,4,4,4],[1,1,2,1,2,3,1,2,3,4]


function MathProgBase.eval_jac_g(d::HS071, J, x)
    # Constraint (row) 1
    J[1] = x[2]*x[3]*x[4]  # 1,1
    J[2] = x[1]*x[3]*x[4]  # 1,2
    J[3] = x[1]*x[2]*x[4]  # 1,3
    J[4] = x[1]*x[2]*x[3]  # 1,4
    # Constraint (row) 2
    J[5] = 2*x[1]  # 2,1
    J[6] = 2*x[2]  # 2,2
    J[7] = 2*x[3]  # 2,3
    J[8] = 2*x[4]  # 2,4
end

function MathProgBase.eval_hesslag(d::HS071, H, x, σ, μ)
    # Again, only lower left triangle
    # Objective
    H[1] = σ * (2*x[4])               # 1,1
    H[2] = σ * (  x[4])               # 2,1
    H[3] = 0                          # 2,2
    H[4] = σ * (  x[4])               # 3,1
    H[5] = 0                          # 3,2
    H[6] = 0                          # 3,3
    H[7] = σ* (2*x[1] + x[2] + x[3])  # 4,1
    H[8] = σ * (  x[1])               # 4,2
    H[9] = σ * (  x[1])               # 4,3
    H[10] = 0                         # 4,4

    # First constraint
    H[2] += μ[1] * (x[3] * x[4])  # 2,1
    H[4] += μ[1] * (x[2] * x[4])  # 3,1
    H[5] += μ[1] * (x[1] * x[4])  # 3,2
    H[7] += μ[1] * (x[2] * x[3])  # 4,1
    H[8] += μ[1] * (x[1] * x[3])  # 4,2
    H[9] += μ[1] * (x[1] * x[2])  # 4,3

    # Second constraint
    H[1]  += μ[2] * 2  # 1,1
    H[3]  += μ[2] * 2  # 2,2
    H[6]  += μ[2] * 2  # 3,3
    H[10] += μ[2] * 2  # 4,4

end

function full_hesslag{T}(evaluator::MathProgBase.AbstractNLPEvaluator, x::AbstractVector{T}, σ::T, μ::AbstractVector{T})
    hesslag_rows, hesslag_cols = MathProgBase.hesslag_structure(evaluator)
    Hvec = similar(x, length(hesslag_rows))
    MathProgBase.eval_hesslag(evaluator, Hvec, x, σ, μ)
    H = zeros(length(x), length(x))
    for (hesslag_row, hesslag_col, hesslag_val) in zip(hesslag_rows, hesslag_cols, Hvec)
        if hesslag_row >= hesslag_col
            H[hesslag_row, hesslag_col] += hesslag_val
            H[hesslag_col, hesslag_row] += hesslag_val
        end
    end
    H
end

function test_hs071(prog::NonlinearProgram)
    solver = MathProgBase.defaultNLPsolver
    m = MathProgBase.NonlinearModel(solver)
    loadproblem!(m, prog)
    MathProgBase.setwarmstart!(m,[1,5,5,1])

    MathProgBase.optimize!(m)
    stat = MathProgBase.status(m)

    @test stat == :Optimal
    x = MathProgBase.getsolution(m)
    @test_approx_eq_eps x[1] 1.0000000000000000 1e-5
    @test_approx_eq_eps x[2] 4.7429996418092970 1e-5
    @test_approx_eq_eps x[3] 3.8211499817883077 1e-5
    @test_approx_eq_eps x[4] 1.3794082897556983 1e-5
    @test_approx_eq_eps MathProgBase.getobjval(m) 17.014017145179164 1e-5

    # Test that a second call to optimize! works
    MathProgBase.setwarmstart!(m,[1,5,5,1])
    MathProgBase.optimize!(m)
    stat = MathProgBase.status(m)
    @test stat == :Optimal

    # Test that hesslag matches manually coded version.
    x = rand(num_variables(prog))
    σ = rand()
    μ = zeros(constraint_length(prog))
    hesslag_manual = full_hesslag(HS071(), x, σ, μ)
    hesslag_prog = full_hesslag(prog, x, σ, μ)
    @test isapprox(hesslag_manual, hesslag_prog)
end

function unconstrained_hs071()
    prog = NonlinearProgram()
    x = add_variables!(prog, "x", Float64[1,1,1,1], Float64[5,5,5,5])
    set_objective!(prog, x -> x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3], x)
    prog, x
end

@testset "HS071 separate constraints" begin
    prog, x = unconstrained_hs071()

    function constraint1!(g, x)
        g[1] = x[1]   * x[2]   * x[3]   * x[4]
    end
    add_constraint!(prog, Constraint(constraint1!, [25.], [Inf]), x)

    function constraint2!(g, x)
        g[1] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
    end
    add_constraint!(prog, Constraint(constraint2!, [40.], [40.]), x)

    test_hs071(prog)

    # test without hesslag
    prog.featuresAvailable = [:Grad, :Jac] # TODO: improve API
    test_hs071(prog)
end

@testset "HS071 single constraint" begin
    prog, x = unconstrained_hs071()

    function eval_my_constraint!(g, x)
        g[1] = x[1]   * x[2]   * x[3]   * x[4]
        g[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
    end
    add_constraint!(prog, Constraint(eval_my_constraint!, Float64[25, 40], Float64[Inf, 40]), x)

    test_hs071(prog)

    # test without hesslag
    prog.featuresAvailable = [:Grad, :Jac] # TODO: improve API
    test_hs071(prog)
end
