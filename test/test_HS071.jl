# Test code adapted from MathProgBase (test/nlp.jl)
# Hock-Schittkowski nonlinear programming test problem 71.
# http://www.ai7.uni-bayreuth.de/test_problem_coll.pdf

function full_hesslag{T}(evaluator::MathProgBase.AbstractNLPEvaluator, x::AbstractVector{T}, σ::T, μ::AbstractVector{T})
    hesslag_rows, hesslag_cols = MathProgBase.hesslag_structure(evaluator)
    Hvec = similar(x, length(hesslag_rows))
    MathProgBase.eval_hesslag(evaluator, Hvec, x, σ, μ)
    H = zeros(length(x), length(x))
    for (hesslag_row, hesslag_col, hesslag_val) in zip(hesslag_rows, hesslag_cols, Hvec)
        H[hesslag_row, hesslag_col] += hesslag_val
        H[hesslag_col, hesslag_row] += hesslag_val
    end

    H
end

function test_hs071(prog::NonlinearProgram)
    solver = IpoptSolver(print_level=0)
    m = MathProgBase.NonlinearModel(solver)
    loadproblem!(m, prog)
    MathProgBase.setwarmstart!(m,[1,5,5,1])

    MathProgBase.optimize!(m)
    stat = MathProgBase.status(m)

    @test stat == :Optimal
    x = MathProgBase.getsolution(m)
    @test isapprox(x[1], 1.0000000000000000, atol = 1e-5)
    @test isapprox(x[2], 4.7429996418092970, atol = 1e-5)
    @test isapprox(x[3], 3.8211499817883077, atol = 1e-5)
    @test isapprox(x[4], 1.3794082897556983, atol = 1e-5)
    @test isapprox(MathProgBase.getobjval(m), 17.014017145179164, atol = 1e-5)

    # Test that a second call to optimize! works
    MathProgBase.setwarmstart!(m,[1,5,5,1])
    MathProgBase.optimize!(m)
    stat = MathProgBase.status(m)
    @test stat == :Optimal

    # Test that hesslag matches manually coded version.
    x = rand(num_variables(prog))
    σ = rand()
    μ = rand(constraint_length(prog))
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
    prog.available_features = [:Grad, :Jac] # TODO: improve API
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
    prog.available_features = [:Grad, :Jac] # TODO: improve API
    test_hs071(prog)
end
