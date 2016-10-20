module FlexibleNonlinearProgram

using ForwardDiff
using MathProgBase
import Base: length
import MathProgBase: loadproblem!

include("decision_variable.jl")
include("constraint.jl")
include("program.jl")

export
    # types
    Constraint,
    DecisionVariable,
    NonlinearProgram,
    # functions
    lower_bound,
    upper_bound,
    name,
    num_variables,
    constraints,
    constraint_length,
    variable_lower_bound,
    variable_upper_bound,
    constraint_lower_bound,
    constraint_upper_bound,
    add_variable!,
    add_variables!,
    set_objective!,
    set_sense!,
    add_constraint!,
    loadproblem!

end # module
