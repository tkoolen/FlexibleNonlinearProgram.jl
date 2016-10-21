immutable Constraint{F}
    f!::F
    lowerBound::Vector{Float64}
    upperBound::Vector{Float64}
    name::String

    function Constraint(f!::F, lower::Vector{Float64}, upper::Vector{Float64}, name::String)
        length(lower) == length(upper) || error("lower bound and upper bound must have the same length")
        new(f!, lower, upper, name)
    end
end
Constraint{F}(f!::F, lower::Vector{Float64}, upper::Vector{Float64}, name::String) = Constraint{F}(f!, lower, upper, name)
Constraint(f!, lower::Vector{Float64}, upper::Vector{Float64}) = Constraint(f!, lower, upper, "")

lower_bound(c::Constraint) = c.lowerBound
upper_bound(c::Constraint) = c.upperBound
name(c::Constraint) = c.name
length(c::Constraint) = length(c.lowerBound)

evaluate!(out::AbstractVector, c::Constraint, arg::AbstractVector) = c.f!(out, arg)

type LagrangianContribution{F, V<:AbstractVector}
    constraint::Constraint{F}
    μ::V
end

function (functor::LagrangianContribution)(arg::AbstractVector)
    out = similar(arg, length(functor.constraint))
    functor.constraint.f!(out, arg)
    functor.μ ⋅ out
end
