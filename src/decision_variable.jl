type DecisionVariable
    name::String
    lowerBound::Float64
    upperBound::Float64
end
DecisionVariable(name::String) = DecisionVariable(name, -Inf, Inf)
name(var::DecisionVariable) = var.name
lower_bound(var::DecisionVariable) = var.lowerBound
upper_bound(var::DecisionVariable) = var.upperBound
