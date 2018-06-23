struct Constraint{F}
    f!::F
    lower_bound::Vector{Float64}
    upper_bound::Vector{Float64}
    name::String

    function Constraint(f!::F, lower::Vector{Float64}, upper::Vector{Float64}, name::String) where {F}
        length(lower) == length(upper) || error("lower bound and upper bound must have the same length")
        new{F}(f!, lower, upper, name)
    end
end
Constraint(f!, lower::Vector{Float64}, upper::Vector{Float64}) = Constraint(f!, lower, upper, "")

lower_bound(c::Constraint) = c.lower_bound
upper_bound(c::Constraint) = c.upper_bound
name(c::Constraint) = c.name
Base.length(c::Constraint) = length(c.lower_bound)

evaluate!(out::AbstractVector, c::Constraint, arg::AbstractVector) = c.f!(out, arg)

mutable struct LagrangianContribution{F, V<:AbstractVector}
    constraint::Constraint{F}
    μ::V
end

function (functor::LagrangianContribution)(arg::AbstractVector)
    out = similar(arg, length(functor.constraint))
    functor.constraint.f!(out, arg)
    functor.μ ⋅ out
end
