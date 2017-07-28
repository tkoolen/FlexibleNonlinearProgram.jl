# Hock-Schittkowski nonlinear programming test problem 41.
# http://www.ai7.uni-bayreuth.de/test_problem_coll.pdf
# Problem with sparsity in objective function.

@testset "HS041" begin
    prog = NonlinearProgram()
    x = add_variables!(prog, "x", Float64[0, 0, 0, 0], Float64[1, 1, 1, 2])
    set_objective!(prog, x -> 2 - x[1] * x[2] * x[3], x[1 : 3])
    function constr!(g, x)
        g[1] = x[1] + 2 * x[2] + 2 * x[3] - x[4]
    end
    add_constraint!(prog, Constraint(constr!, [0.], [0.]), x)

    solver = IpoptSolver(print_level=0)
    m = MathProgBase.NonlinearModel(solver)
    loadproblem!(m, prog)
    MathProgBase.setwarmstart!(m,[2, 2, 2, 2])

    MathProgBase.optimize!(m)
    stat = MathProgBase.status(m)

    @test stat == :Optimal
    x = MathProgBase.getsolution(m)
    @test isapprox(x[1], 2/3, atol = 1e-5)
    @test isapprox(x[2], 1/3, atol = 1e-5)
    @test isapprox(x[3], 1/3, atol = 1e-5)
    @test isapprox(x[4], 2., atol = 1e-5)

end
