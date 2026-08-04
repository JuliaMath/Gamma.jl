####################################
## Public entry points
####################################

"""
    gamma(a, z)

Compute the unnormalised upper incomplete gamma function `Γ(a,z)`.
"""
gamma(a::T, z::T) where {T<:AbstractFloat} = _gamma(a, z)
gamma(a::Complex{T}, z::Complex{T}) where {T<:AbstractFloat} = _gamma(a, z)
function gamma(a::Number, z::Number)
    a, z = promote(float(a), float(z))
    return _gamma(a, z)
end

"""
    gamma_lower(a, z)

Compute the unnormalised lower incomplete gamma function `γ(a,z)`.
"""
gamma_lower(a::T, z::T) where {T<:AbstractFloat} = _gamma_lower(a, z)
gamma_lower(a::Complex{T}, z::Complex{T}) where {T<:AbstractFloat} =
    _gamma_lower(a, z)
function gamma_lower(a::Number, z::Number)
    a, z = promote(float(a), float(z))
    return _gamma_lower(a, z)
end

"""
    gamma_inc(a, z)

Return the regularised lower and upper incomplete gamma functions `(P, Q)`,
with `P + Q = 1`.
"""
gamma_inc(a::T, z::T) where {T<:AbstractFloat} = _gamma_inc(a, z)
gamma_inc(a::Complex{T}, z::Complex{T}) where {T<:AbstractFloat} =
    _gamma_inc(a, z)
function gamma_inc(a::Number, z::Number)
    a, z = promote(float(a), float(z))
    return _gamma_inc(a, z)
end

####################################
## Shared numerical methods
####################################

function _gamma_lower_series(a::T, z::T;
                             maxiter::Union{Nothing,Int}=nothing) where {T}
    R = typeof(real(z))
    tol = 8 * eps(one(R))
    cap = isnothing(maxiter) ? 50_000 : maxiter
    term = one(z) / a
    total = term
    stable = 0
    for n = 1:cap
        term *= z / (a + n)
        total += term
        if _En_converged(term, total, zero(total), tol)
            stable += 1
            stable >= 2 &&
                return _En_safeexpmult(a * log(z) - z, total)
        else
            stable = 0
        end
    end
    throw(IncompleteGammaConvergenceError(:lower_incomplete_gamma_series, cap))
end

function _gamma_upper_cf(a::T, z::T) where {T<:AbstractFloat}
    # Γ(a,z) = z^a E_{1-a}(z).
    ν = 1 - a
    scaled, _, _ = _En_cf_nogamma(ν, z)
    return _En_safeexpmult(a * log(z) - z, scaled)
end

function _gamma_upper_cf(
    a::Complex{T}, z::Complex{T}
) where {T<:AbstractFloat}
    ν = 1 - a
    scaled = if real(z) > 0
        _En_cf_nogamma(ν, z)[1]
    else
        _expint_left_halfplane(ν, z, true)
    end
    return _En_safeexpmult(a * log(z) - z, scaled)
end

function _gamma_lower_direct(a::T, z::T) where {T<:AbstractFloat}
    if a > 0 && z >= 0
        return z <= a + oftype(a, 0.1)
    end
    absa = abs(a)
    return abs(z) < absa + one(absa)
end

function _gamma_lower_direct(
    a::Complex{T}, z::Complex{T}
) where {T<:AbstractFloat}
    if isreal(a) && isreal(z) && real(a) > 0 && real(z) >= 0
        return real(z) <= real(a) + oftype(real(a), 0.1)
    end
    absa = abs(a)
    return abs(z) < absa + one(real(absa))
end

function _gamma_upper_unsafe(a::T, z::T) where {T}
    if _gamma_lower_direct(a, z)
        lower = _gamma_lower_series(a, z)
        return gamma(a) - lower
    end
    return _gamma_upper_cf(a, z)
end

function _gamma_lower_unsafe(a::T, z::T) where {T}
    if _gamma_lower_direct(a, z)
        return _gamma_lower_series(a, z)
    end
    complete = gamma(a)
    upper = _gamma_upper_cf(a, z)
    return complete - upper
end

####################################
## Upper incomplete gamma
####################################

_gamma(a::Float16, z::Float16) =
    Float16(_gamma(Float32(a), Float32(z)))
_gamma(a::ComplexF16, z::ComplexF16) =
    ComplexF16(_gamma(ComplexF32(a), ComplexF32(z)))

function _gamma(a::T, z::T) where {T<:AbstractFloat}
    z < 0 && !isinteger(a) &&
        throw(DomainError(
            z,
            "a negative real z has a complex principal value; pass complex(z)",
        ))
    if !(a > 0)
        iszero(z) && throw(DomainError(
            z, "the upper incomplete gamma is singular at z = 0"
        ))
        isinteger(a) && return _gamma_upper_cf(a, z)
    end
    return _gamma_upper_unsafe(a, z)
end

function _gamma(a::Complex{T}, z::Complex{T}) where {T<:AbstractFloat}
    if !(real(a) > 0)
        iszero(z) && throw(DomainError(
            z, "the upper incomplete gamma is singular at z = 0"
        ))
        isreal(a) && isinteger(real(a)) && return _gamma_upper_cf(a, z)
    end
    return _gamma_upper_unsafe(a, z)
end

function _gamma(a::BigFloat, z::BigFloat)
    z < 0 && !isinteger(a) &&
        throw(DomainError(
            z,
            "a negative real z has a complex principal value; pass complex(z)",
    ))
    if !(a > 0)
        iszero(z) && throw(DomainError(
            z, "the upper incomplete gamma is singular at z = 0"
        ))
        isinteger(a) && return _gamma_upper_cf(a, z)
    end
    p = precision(BigFloat)
    guard = max(24, ndigits(p; base=2) + 12)
    return setprecision(p + guard) do
        value = _gamma_upper_unsafe(BigFloat(a), BigFloat(z))
        setprecision(p) do
            BigFloat(value)
        end
    end
end

function _gamma(a::Complex{BigFloat}, z::Complex{BigFloat})
    if !(real(a) > 0)
        iszero(z) && throw(DomainError(
            z, "the upper incomplete gamma is singular at z = 0"
        ))
        isreal(a) && isinteger(real(a)) && return _gamma_upper_cf(a, z)
    end
    p = precision(BigFloat)
    guard = max(24, ndigits(p; base=2) + 12)
    return setprecision(p + guard) do
        ahi = Complex{BigFloat}(BigFloat(real(a)), BigFloat(imag(a)))
        zhi = Complex{BigFloat}(BigFloat(real(z)), BigFloat(imag(z)))
        value = _gamma_upper_unsafe(ahi, zhi)
        setprecision(p) do
            Complex{BigFloat}(BigFloat(real(value)), BigFloat(imag(value)))
        end
    end
end

####################################
## Lower incomplete gamma
####################################

_gamma_lower(a::Float16, z::Float16) =
    Float16(_gamma_lower(Float32(a), Float32(z)))
_gamma_lower(a::ComplexF16, z::ComplexF16) =
    ComplexF16(_gamma_lower(ComplexF32(a), ComplexF32(z)))

function _gamma_lower(a::T, z::T) where {T<:AbstractFloat}
    z < 0 && !isinteger(a) &&
        throw(DomainError(
            z,
            "a negative real z has a complex principal value; pass complex(z)",
        ))
    if !(a > 0)
        isinteger(a) &&
            throw(DomainError(a, "the lower incomplete gamma has a pole at a"))
        iszero(z) && throw(DomainError(
            z, "the lower incomplete gamma is singular at z = 0"
        ))
    end
    return _gamma_lower_unsafe(a, z)
end

function _gamma_lower(a::Complex{T}, z::Complex{T}) where {T<:AbstractFloat}
    if !(real(a) > 0)
        if isreal(a) && isinteger(real(a))
            throw(DomainError(a, "the lower incomplete gamma has a pole at a"))
        end
        iszero(z) && throw(DomainError(
            z, "the lower incomplete gamma is singular at z = 0"
        ))
    end
    return _gamma_lower_unsafe(a, z)
end

function _gamma_lower(a::BigFloat, z::BigFloat)
    z < 0 && !isinteger(a) &&
        throw(DomainError(
            z,
            "a negative real z has a complex principal value; pass complex(z)",
        ))
    if !(a > 0)
        isinteger(a) &&
            throw(DomainError(a, "the lower incomplete gamma has a pole at a"))
        iszero(z) && throw(DomainError(
            z, "the lower incomplete gamma is singular at z = 0"
        ))
    end
    p = precision(BigFloat)
    guard = max(24, ndigits(p; base=2) + 12)
    return setprecision(p + guard) do
        value = _gamma_lower_unsafe(BigFloat(a), BigFloat(z))
        setprecision(p) do
            BigFloat(value)
        end
    end
end

function _gamma_lower(a::Complex{BigFloat}, z::Complex{BigFloat})
    if !(real(a) > 0)
        if isreal(a) && isinteger(real(a))
            throw(DomainError(a, "the lower incomplete gamma has a pole at a"))
        end
        iszero(z) && throw(DomainError(
            z, "the lower incomplete gamma is singular at z = 0"
        ))
    end
    p = precision(BigFloat)
    guard = max(24, ndigits(p; base=2) + 12)
    return setprecision(p + guard) do
        ahi = Complex{BigFloat}(BigFloat(real(a)), BigFloat(imag(a)))
        zhi = Complex{BigFloat}(BigFloat(real(z)), BigFloat(imag(z)))
        value = _gamma_lower_unsafe(ahi, zhi)
        setprecision(p) do
            Complex{BigFloat}(BigFloat(real(value)), BigFloat(imag(value)))
        end
    end
end

####################################
## Regularised incomplete gamma
####################################

Base.@noinline function _ig_transition_exponent(a::T, z::T) where {T}
    ratio = z / a
    return a * log(a) - a + a * (Base.@inline logmxp1(ratio))
end

function _ig_exponent(a::T, z::T) where {T<:AbstractFloat}
    if a >= oftype(a, 50) && z > 0
        if oftype(a, 0.75) * a <= z <= oftype(a, 1.25) * a
            z == a && return a * log(a) - a
            return _ig_transition_exponent(a, z)
        end
    end
    return a * log(z) - z
end

# Adapted from SpecialFunctions.jl 2.8.0 (MIT license).
function _gamma_inc_taylor_x(a::Float64, x::Float64)
    l = 3.0
    c = x
    total = x / (a + 3.0)
    tol = 15e-15 / (a + 1.0)
    while true
        l += 1.0
        c *= -x / l
        term = c / (a + l)
        total += term
        abs(term) <= tol && break
    end

    correction =
        a * x * ((total / 6.0 - 0.5 / (a + 2.0)) * x + 1.0 / (a + 1.0))
    exponent = a * log(x)

    h = if a < 0.1
        top = evalpoly(a, (
            0.577215664901533, -0.409078193005776, -0.230975380857675,
            0.0597275330452234, 0.007669681649490,
            -0.00514889771323592, 0.000589597428611429,
        ))
        bottom = evalpoly(a, (
            1.0, 0.427569613095214, 0.158451672430138,
            0.0261132021441447, 0.00423244297896961,
        ))
        a * top / bottom
    else
        inv(gamma(a + 1.0)) - 1.0
    end
    inverse_gamma = 1.0 + h

    if (x < 0.25 && exponent > -0.13394) || a < x / 2.59
        power_minus_one = expm1(exponent)
        power = 1.0 + power_minus_one
        q = max(
            (power * correction - power_minus_one) * inverse_gamma - h,
            0.0,
        )
        return 1.0 - q, q
    end

    p = exp(exponent) * inverse_gamma * (1.0 - correction)
    return p, 1.0 - p
end

function _gamma_lower_series_normalized(
    a::T, z::T; maxiter::Union{Nothing,Int}=nothing
) where {T}
    iszero(z) && return zero(z)
    R = typeof(real(z))
    tol = 8 * eps(one(R))
    cap = isnothing(maxiter) ? 50_000 : maxiter
    term = one(z)
    total = term
    stable = 0
    for n = 1:cap
        term *= z / (a + n)
        total += term
        if _En_converged(term, total, zero(total), tol)
            stable += 1
            if stable >= 2
                exponent = a * log(z) - z
                gamma_argument = a + one(a)
                return _gamma_lower_series_normalized_result(
                    exponent, gamma_argument, total
                )
            end
        else
            stable = 0
        end
    end
    throw(IncompleteGammaConvergenceError(
        :normalized_lower_incomplete_gamma_series, cap
    ))
end

function _gamma_lower_series_normalized_result(
    exponent::T, gamma_argument::T, total::T
) where {T<:AbstractFloat}
    logabs, sign = logabsgamma(gamma_argument)
    return sign * _En_safeexpmult(exponent - logabs, total)
end

function _gamma_lower_series_normalized_result(
    exponent::Complex{T}, gamma_argument::Complex{T}, total::Complex{T}
) where {T<:AbstractFloat}
    if isreal(gamma_argument)
        logabs, sign = logabsgamma(real(gamma_argument))
        return sign * _En_safeexpmult(exponent - logabs, total)
    end
    return _En_safeexpmult(exponent - loggamma(gamma_argument), total)
end

function _gamma_inc_unsafe(a::Float64, z::Float64)
    if 0 < a < 1 && 0 < z < 1.1
        return _gamma_inc_taylor_x(a, z)
    end
    if _gamma_lower_direct(a, z)
        p = _gamma_lower_series_normalized(a, z)
        return p, one(p) - p
    end
    ν = 1 - a
    scaled, _, _ = _En_cf_nogamma(ν, z)
    exponent = _ig_exponent(a, z)
    return _gamma_inc_from_upper(a, scaled, exponent)
end

function _gamma_inc_unsafe(a::T, z::T) where {T<:AbstractFloat}
    if _gamma_lower_direct(a, z)
        p = _gamma_lower_series_normalized(a, z)
        return p, one(p) - p
    end
    ν = 1 - a
    scaled, _, _ = _En_cf_nogamma(ν, z)
    exponent = _ig_exponent(a, z)
    return _gamma_inc_from_upper(a, scaled, exponent)
end

function _gamma_inc_unsafe(
    a::Complex{T}, z::Complex{T}
) where {T<:AbstractFloat}
    if _gamma_lower_direct(a, z)
        p = _gamma_lower_series_normalized(a, z)
        return p, one(p) - p
    end
    ν = 1 - a
    scaled = if real(z) > 0
        _En_cf_nogamma(ν, z)[1]
    else
        _expint_left_halfplane(ν, z, true)
    end
    exponent = a * log(z) - z
    return _gamma_inc_from_upper(a, scaled, exponent)
end

function _gamma_inc_from_upper(a::Float64, scaled::Float64,
                               exponent::Float64)
    if 0 < a <= 11.5
        denominator = gamma(a)
        numerator = exp(exponent)
        q = numerator * scaled / denominator
        return one(q) - q, q
    end
    logabs, sign = logabsgamma(a)
    q = sign * _En_safeexpmult(exponent - logabs, scaled)
    return one(q) - q, q
end

function _gamma_inc_from_upper(a::T, scaled::T,
                               exponent::T) where {T<:AbstractFloat}
    logabs, sign = logabsgamma(a)
    q = sign * _En_safeexpmult(exponent - logabs, scaled)
    return one(q) - q, q
end

function _gamma_inc_from_upper(a::ComplexF64, scaled::ComplexF64,
                               exponent::ComplexF64)
    if isreal(a) && 0 < real(a) <= 11.5
        denominator = gamma(real(a))
        numerator = exp(exponent)
        if isfinite(numerator) && !iszero(numerator)
            q = numerator * scaled / denominator
            isfinite(q) && !iszero(q) && return one(q) - q, q
        end
    end
    q = if isreal(a)
        logabs, sign = logabsgamma(real(a))
        sign * _En_safeexpmult(exponent - logabs, scaled)
    else
        _En_safeexpmult(exponent - loggamma(a), scaled)
    end
    return one(q) - q, q
end

function _gamma_inc_from_upper(
    a::Complex{T}, scaled::Complex{T}, exponent::Complex{T}
) where {T<:AbstractFloat}
    q = if isreal(a)
        logabs, sign = logabsgamma(real(a))
        sign * _En_safeexpmult(exponent - logabs, scaled)
    else
        _En_safeexpmult(exponent - loggamma(a), scaled)
    end
    return one(q) - q, q
end

function _gamma_inc(a::Float16, z::Float16)
    p, q = _gamma_inc(Float32(a), Float32(z))
    return Float16(p), Float16(q)
end

function _gamma_inc(a::ComplexF16, z::ComplexF16)
    p, q = _gamma_inc(ComplexF32(a), ComplexF32(z))
    return ComplexF16(p), ComplexF16(q)
end

function _gamma_inc(a::T, z::T) where {T<:AbstractFloat}
    z < 0 && !isinteger(a) &&
        throw(DomainError(
            z,
            "a negative real z has a complex principal value; pass complex(z)",
        ))
    if !(a > 0)
        isinteger(a) &&
            throw(DomainError(a, "the lower incomplete gamma has a pole at a"))
        iszero(z) && throw(DomainError(
            z, "the upper incomplete gamma is singular at z = 0"
        ))
    end
    return _gamma_inc_unsafe(a, z)
end

function _gamma_inc(a::Complex{T}, z::Complex{T}) where {T<:AbstractFloat}
    if !(real(a) > 0)
        if isreal(a) && isinteger(real(a))
            throw(DomainError(a, "the lower incomplete gamma has a pole at a"))
        end
        iszero(z) && throw(DomainError(
            z, "the upper incomplete gamma is singular at z = 0"
        ))
    end
    return _gamma_inc_unsafe(a, z)
end

function _gamma_inc(a::BigFloat, z::BigFloat)
    z < 0 && !isinteger(a) &&
        throw(DomainError(
            z,
            "a negative real z has a complex principal value; pass complex(z)",
        ))
    if !(a > 0)
        isinteger(a) &&
            throw(DomainError(a, "the lower incomplete gamma has a pole at a"))
        iszero(z) && throw(DomainError(
            z, "the upper incomplete gamma is singular at z = 0"
        ))
    end
    p = precision(BigFloat)
    guard = max(24, ndigits(p; base=2) + 12)
    return setprecision(p + guard) do
        lower, upper = _gamma_inc_unsafe(BigFloat(a), BigFloat(z))
        setprecision(p) do
            BigFloat(lower), BigFloat(upper)
        end
    end
end

function _gamma_inc(a::Complex{BigFloat}, z::Complex{BigFloat})
    if !(real(a) > 0)
        if isreal(a) && isinteger(real(a))
            throw(DomainError(a, "the lower incomplete gamma has a pole at a"))
        end
        iszero(z) && throw(DomainError(
            z, "the upper incomplete gamma is singular at z = 0"
        ))
    end
    p = precision(BigFloat)
    guard = max(24, ndigits(p; base=2) + 12)
    return setprecision(p + guard) do
        ahi = Complex{BigFloat}(BigFloat(real(a)), BigFloat(imag(a)))
        zhi = Complex{BigFloat}(BigFloat(real(z)), BigFloat(imag(z)))
        lower, upper = _gamma_inc_unsafe(ahi, zhi)
        setprecision(p) do
            (
                Complex{BigFloat}(
                    BigFloat(real(lower)), BigFloat(imag(lower))
                ),
                Complex{BigFloat}(
                    BigFloat(real(upper)), BigFloat(imag(upper))
                ),
            )
        end
    end
end
