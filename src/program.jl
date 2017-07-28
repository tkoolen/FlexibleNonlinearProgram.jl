type NonlinearProgram <: MathProgBase.AbstractNLPEvaluator
    variables::Vector{DecisionVariable}
    objective::Function # TODO: type unstable
    objective_var_indices::Vector{Int64}
    sense::Symbol
    constraint_var_indices::OrderedDict{Constraint, Vector{Int64}} # TODO: type unstable
    available_features::Vector{Symbol}

    function NonlinearProgram()
        new([], () -> 0., [], :Min, OrderedDict{Constraint, Vector{Int64}}(), [:Grad, :Jac, :Hess])
    end
end

num_variables(p::NonlinearProgram) = length(p.variables)
constraints(p::NonlinearProgram) = keys(p.constraint_var_indices)
constraint_length(p::NonlinearProgram) = sum(length(c) for c in constraints(p))
variable_lower_bound(p::NonlinearProgram) = [lower_bound(var) for var in p.variables]
variable_upper_bound(p::NonlinearProgram) = [upper_bound(var) for var in p.variables]
constraint_lower_bound(p::NonlinearProgram) = vcat([lower_bound(c) for c in constraints(p)]...)
constraint_upper_bound(p::NonlinearProgram) = vcat([upper_bound(c) for c in constraints(p)]...)

add_variable!(p::NonlinearProgram, var::DecisionVariable) = begin push!(p.variables, var); num_variables(p) end

function add_variables!(p::NonlinearProgram, base_name::String, lower_bound::Vector{Float64}, upper_bound::Vector{Float64})
    length(lower_bound) == length(upper_bound) || error("lower bound and upper bound must have the same length")
    variables = DecisionVariable[]
    for i = 1 : length(lower_bound)
        var = DecisionVariable("$(base_name)_$i", lower_bound[i], upper_bound[i])
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
    empty!(p.objective_var_indices)
    for var in vars
        index = find_variable!(p, var)
        push!(p.objective_var_indices, index)
    end
end

function set_sense!(p::NonlinearProgram, sense::Symbol)
    @assert sense ∈ (:Min, :Max)
    p.sense = sense
end

function add_constraint!(p::NonlinearProgram, c::Constraint, vars::Vector{DecisionVariable})
    var_indices = [find_variable!(p, var) for var in vars]
    p.constraint_var_indices[c] = var_indices
    nothing
end

# MathProgBase interface

MathProgBase.features_available(p::NonlinearProgram) = p.available_features

function MathProgBase.initialize(p::NonlinearProgram, requested_features::Vector{Symbol})
    for feat in requested_features
        if !(feat in MathProgBase.features_available(p))
            error("Unsupported feature $feat")
        end
    end
end

function MathProgBase.eval_f(p::NonlinearProgram, x)
    p.objective(view(x, p.objective_var_indices))
end

function MathProgBase.eval_grad_f(p::NonlinearProgram, g, x)
    g[:] = zero(eltype(x))
    ForwardDiff.gradient!(view(g, p.objective_var_indices), p.objective, view(x, p.objective_var_indices))
end

function MathProgBase.eval_g(p::NonlinearProgram, g, x)
    start_index = 1
    for (constraint, var_indices) in p.constraint_var_indices
        constraintLength = length(constraint)
        gview = view(g, start_index : start_index + constraintLength - 1)
        xview = view(x, var_indices)
        constraint.f!(gview, xview)
        start_index += constraintLength
    end
    g
end

function MathProgBase.jac_structure(p::NonlinearProgram)
    rowStart = 1
    rows = Int64[]
    cols = Int64[]
    for (constraint, var_indices) in p.constraint_var_indices
        numRows = length(constraint)
        for col in var_indices
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

function MathProgBase.hesslag_structure(p::NonlinearProgram)
    rows = Int64[]
    cols = Int64[]
    allvar_indices = Vector{Vector{Int64}}()
    push!(allvar_indices, p.objective_var_indices)
    for var_indices in values(p.constraint_var_indices)
        push!(allvar_indices, var_indices)
    end
    for var_indices in allvar_indices
        for col in var_indices
            for row in var_indices
                if row >= col # lower triangle only
                    push!(rows, row)
                    push!(cols, col)
                end
            end
        end
    end
    rows, cols
end

function MathProgBase.eval_jac_g(p::NonlinearProgram, J, x)
    start = 1
    for (constraint, var_indices) in p.constraint_var_indices
        xconstraint = view(x, var_indices)
        yconstraint = similar(x, length(constraint))
        out = DiffBase.DiffResult(yconstraint, similar(x, length(constraint), length(xconstraint))) # TODO: preallocate.
        ForwardDiff.jacobian!(out, constraint.f!, yconstraint, xconstraint)
        sparse_jac = DiffBase.jacobian(out)
        Jview = view(J, start : start + length(sparse_jac) - 1)
        copy!(Jview, vec(sparse_jac))
        start += length(sparse_jac)
    end
end

function MathProgBase.eval_hesslag(p::NonlinearProgram, H, x, σ, μ)
    start = 1

    # objective
    xobjective = view(x, p.objective_var_indices)
    out = DiffBase.HessianResult(xobjective) # TODO: preallocate
    ForwardDiff.hessian!(out, p.objective, xobjective)
    Hobj = DiffBase.hessian(out)
    scale!(Hobj, σ)
    Hobjlength = num_triangular_elements(size(Hobj, 1))
    copy_lower_triangle_column_major!(view(H, start : start + Hobjlength - 1), Hobj)
    start += Hobjlength

    # constraints
    constraintstart = 1
    for (constraint, var_indices) in p.constraint_var_indices
        xconstraint = view(x, var_indices)
        μConstraint = view(μ, constraintstart : constraintstart + length(constraint) - 1)
        out = DiffBase.HessianResult(xconstraint) # TODO: preallocate
        ForwardDiff.hessian!(out, LagrangianContribution(constraint, μConstraint), xconstraint)
        Hconstraint = DiffBase.hessian(out)
        Hconstraintlength = num_triangular_elements(size(Hconstraint, 1))
        copy_lower_triangle_column_major!(view(H, start : start + Hconstraintlength - 1), Hconstraint)
        start += Hconstraintlength
        constraintstart += length(constraint)
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
