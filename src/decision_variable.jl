mutable struct DecisionVariable
    name::String
    lower_bound::Float64
    upper_bound::Float64
end
DecisionVariable(name::String) = DecisionVariable(name, -Inf, Inf)
name(var::DecisionVariable) = var.name
lower_bound(var::DecisionVariable) = var.lower_bound
upper_bound(var::DecisionVariable) = var.upper_bound
