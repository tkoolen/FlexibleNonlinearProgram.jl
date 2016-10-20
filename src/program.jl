type NonlinearProgram <: MathProgBase.AbstractNLPEvaluator
    variables::Vector{DecisionVariable}
    objective::Function
    objectiveVarIndices::Vector{Int64}
    sense::Symbol
    constraintVarIndices::Dict{Constraint, Vector{Int64}}
    featuresAvailable::Vector{Symbol}

    function NonlinearProgram()
        new([], () -> 0., [], :Min, Dict(), [:Grad, :Jac])
    end
end

num_variables(p::NonlinearProgram) = length(p.variables)
constraints(p::NonlinearProgram) = keys(p.constraintVarIndices)
constraint_length(p::NonlinearProgram) = sum(length(c) for c in constraints(p))
variable_lower_bound(p::NonlinearProgram) = [lower_bound(var) for var in p.variables]
variable_upper_bound(p::NonlinearProgram) = [upper_bound(var) for var in p.variables]
constraint_lower_bound(p::NonlinearProgram) = vcat([lower_bound(c) for c in constraints(p)]...)
constraint_upper_bound(p::NonlinearProgram) = vcat([upper_bound(c) for c in constraints(p)]...)

add_variable!(p::NonlinearProgram, var::DecisionVariable) = begin push!(p.variables, var); num_variables(p) end

function add_variables!(p::NonlinearProgram, baseName::String, lowerBound::Vector{Float64}, upperBound::Vector{Float64})
    length(lowerBound) == length(upperBound) || error("lower bound and upper bound must have the same length")
    variables = DecisionVariable[]
    for i = 1 : length(lowerBound)
        var = DecisionVariable("$(baseName)_$i", lowerBound[i], upperBound[i])
        add_variable!(p, var)
        push!(variables, var)
    end
    variables
end

function find_variable!(p::NonlinearProgram, var::DecisionVariable)
    index = findfirst(p.variables, var)
    if index == 0
        index = add_variable!(p, var)
    end
    index
end

function set_objective!(p::NonlinearProgram, objective::Function, vars::Vector{DecisionVariable})
    p.objective = objective
    empty!(p.objectiveVarIndices)
    for var in vars
        index = find_variable!(p, var)
        push!(p.objectiveVarIndices, index)
    end
end

function set_sense!(p::NonlinearProgram, sense::Symbol)
    @assert sense ∈ (:Min, :Max)
    p.sense = sense
end

function add_constraint!(p::NonlinearProgram, c::Constraint, vars::Vector{DecisionVariable})
    varIndices = [find_variable!(p, var) for var in vars]
    p.constraintVarIndices[c] = varIndices
    nothing
end

# MathProgBase interface

MathProgBase.features_available(p::NonlinearProgram) = p.featuresAvailable

function MathProgBase.initialize(p::NonlinearProgram, requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in MathProgBase.features_available(p))
            error("Unsupported feature $feat")
        end
    end
end

function MathProgBase.eval_f(p::NonlinearProgram, x)
    p.objective(view(x, p.objectiveVarIndices))
end

function MathProgBase.eval_grad_f(p::NonlinearProgram, g, x)
    g[:] = zero(eltype(x))
    ForwardDiff.gradient!(view(g, p.objectiveVarIndices), p.objective, view(x, p.objectiveVarIndices))
end

function MathProgBase.eval_g(p::NonlinearProgram, g, x)
    gStartIndex = 1
    for (constraint, varIndices) in p.constraintVarIndices
        constraintLength = length(constraint)
        gView = view(g, gStartIndex : gStartIndex + constraintLength - 1)
        xView = view(x, varIndices)
        constraint.f!(gView, xView)
        gStartIndex += constraintLength
    end
    g
end

function MathProgBase.jac_structure(p::NonlinearProgram)
    rowStart = 1
    rows = Int64[]
    cols = Int64[]
    for (constraint, varIndices) in p.constraintVarIndices
        numRows = length(constraint)
        for col in varIndices
            for i = 0 : numRows - 1
                row = rowStart + i
                push!(rows, row)
                push!(cols, col)
            end
        end
        rowStart += numRows
    end
    rows, cols
end

function MathProgBase.eval_jac_g(p::NonlinearProgram, J, x)
    startIndex = 1
    for (constraint, varIndices) in p.constraintVarIndices
        xConstraint = view(x, varIndices)
        out = JacobianResult(similar(x, length(constraint)), similar(x, length(constraint), length(xConstraint))) # TODO: preallocate.
        ForwardDiff.jacobian!(out, constraint.f!, xConstraint)
        sparseJac = ForwardDiff.jacobian(out)
        sparseJacLength = length(sparseJac)
        Jview = view(J, startIndex : startIndex + sparseJacLength - 1)
        copy!(Jview, reshape(sparseJac, sparseJacLength))
        startIndex += sparseJacLength
    end
end

function loadproblem!(model::MathProgBase.AbstractNonlinearModel, p::NonlinearProgram)
    n = num_variables(p)
    m = constraint_length(p)
    l = variable_lower_bound(p)
    u = variable_upper_bound(p)
    lb = constraint_lower_bound(p)
    ub = constraint_upper_bound(p)
    sense = p.sense
    MathProgBase.loadproblem!(model, n, m, l, u, lb, ub, sense, p)
end
