# Test adapted from MathProgBase (test/nlp.jl)

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
end

@testset "HS071 single constraint" begin
    prog, x = unconstrained_hs071()

    function eval_my_constraint!(g, x)
        g[1] = x[1]   * x[2]   * x[3]   * x[4]
        g[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
    end
    add_constraint!(prog, Constraint(eval_my_constraint!, Float64[25, 40], Float64[Inf, 40]), x)

    test_hs071(prog)
end
