# FlexibleNonlinearProgram

FlexibleNonlinearProgram provides a simple way to set up a nonlinear program in the format accepted by [MathProgBase](https://github.com/JuliaOpt/MathProgBase.jl), which allows the program to be solved using a variety of solvers. FlexibleNonlinearProgram.jl uses [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl) for gradient computation. It's not nearly as fancy as [JuMP](https://github.com/JuliaOpt/JuMP.jl), but does support vector-valued constraints, a big limitation in JuMP's current nonlinear programming interface.

Check out the tests to see some basic examples.
