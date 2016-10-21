# Hock-Schittkowski nonlinear programming test problem 39.
# http://www.ai7.uni-bayreuth.de/test_problem_coll.pdf
# Problem with sparsity in constraints and objective.

@testset "HS039" begin
    prog = NonlinearProgram()
    n = 4
    x = add_variables!(prog, "x", fill(-Inf, n), fill(Inf, n))
    set_objective!(prog, x -> -x[1], [x[1]])
    add_constraint!(prog, Constraint((g, x) -> g[1] = x[2] - x[1]^3 - x[3]^2, [0.], [0.]), x[1 : 3])
    add_constraint!(prog, Constraint((g, x) -> g[1] = x[1]^2 - x[2] - x[3]^2, [0.], [0.]), [x[1]; x[2]; x[4]])

    solver = MathProgBase.defaultNLPsolver
    m = MathProgBase.NonlinearModel(solver)
    loadproblem!(m, prog)
    MathProgBase.setwarmstart!(m,[2, 2, 2, 2])

    MathProgBase.optimize!(m)
    stat = MathProgBase.status(m)

    @test stat == :Optimal
    x = MathProgBase.getsolution(m)
    @test isapprox(x, [1, 1, 0, 0], atol = 1e-5)
end
