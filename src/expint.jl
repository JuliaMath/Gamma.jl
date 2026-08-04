# Adapted from SpecialFunctions.jl, src/expint.jl, version 2.7.2, MIT License.

"""
    IncompleteGammaConvergenceError(algorithm, iterations)

The requested precision was not reached by an incomplete-gamma algorithm.
"""
struct IncompleteGammaConvergenceError <: Exception
    algorithm::Symbol
    iterations::Int
end

function Base.showerror(io::IO, err::IncompleteGammaConvergenceError)
    print(io, err.algorithm, " did not converge after ", err.iterations,
          " iterations at the requested precision")
end

####################################
## Public entry points
####################################

"""
    expint(z)
    expint(ν, z)

Compute the generalized exponential integral
`E_ν(z) = ∫₁^∞ exp(-z*t)/t^ν dt` on its principal branch.
"""
expint(ν::T, z::T) where {T<:AbstractFloat} = _expint(ν, z, false)
expint(ν::Complex{T}, z::Complex{T}) where {T<:AbstractFloat} =
    _expint(ν, z, false)
function expint(ν::Number, z::Number)
    ν, z = promote(float(ν), float(z))
    return _expint(ν, z, false)
end
expint(z::Number) = expint(one(z), z)

"""
    expintx(z)
    expintx(ν, z)

Compute the exponentially scaled generalized exponential integral
`exp(z) * E_ν(z)`.
"""
expintx(ν::T, z::T) where {T<:AbstractFloat} = _expint(ν, z, true)
expintx(ν::Complex{T}, z::Complex{T}) where {T<:AbstractFloat} =
    _expint(ν, z, true)
function expintx(ν::Number, z::Number)
    ν, z = promote(float(ν), float(z))
    return _expint(ν, z, true)
end
expintx(z::Number) = expintx(one(z), z)

####################################
## Evaluation dispatch
####################################

_expint(ν::Float16, z::Float16, expscaled::Bool) =
    Float16(_expint(Float32(ν), Float32(z), expscaled))
_expint(ν::ComplexF16, z::ComplexF16, expscaled::Bool) =
    ComplexF16(_expint(ComplexF32(ν), ComplexF32(z), expscaled))

function _expint(ν::T, z::T, expscaled::Bool) where {T<:AbstractFloat}
    if isnan(ν) || isnan(z)
        return oftype(z, NaN) * z
    elseif z < 0
        throw(DomainError(z,
            "expint has a complex value for negative z; pass complex(z)"))
    elseif iszero(z)
        return oftype(z, ν > 1 ? inv(ν - 1) : T(Inf))
    elseif iszero(ν)
        return expscaled ? inv(z) : exp(-z) / z
    end
    return _expint_unsafe(ν, z, expscaled)
end

function _expint(ν::Complex{T}, z::Complex{T},
                 expscaled::Bool) where {T<:AbstractFloat}
    if isnan(ν) || isnan(z)
        return oftype(z, NaN) * z
    elseif iszero(z)
        return oftype(z, real(ν) > 1 ? inv(ν - 1) : T(Inf))
    elseif iszero(ν)
        return expscaled ? inv(z) : exp(-z) / z
    end
    return _expint_unsafe(ν, z, expscaled)
end

function _expint_unsafe(ν::T, z::T,
                        expscaled::Bool) where {T<:AbstractFloat}
    if abs2(z) < 9
        positive_integer = isinteger(ν) && ν > 0
        if !positive_integer || ν <= typemax(Int)
            result = positive_integer ? _En_expand_origin_posint(ν, z) :
                                        _En_expand_origin_general(ν, z)
            return expscaled ? _En_safeexpmult(z, result) : result
        end
    end
    cf, _, _ = _En_cf_nogamma(ν, z)
    return expscaled ? cf : _En_safeexpmult(-z, cf)
end

function _expint_unsafe(ν::Complex{T}, z::Complex{T},
                        expscaled::Bool) where {T<:AbstractFloat}
    if abs2(z) < 9
        realν = real(ν)
        positive_integer = isreal(ν) && isinteger(realν) && realν > 0
        if !positive_integer || realν <= typemax(Int)
            result = positive_integer ? _En_expand_origin_posint(realν, z) :
                                        _En_expand_origin_general(ν, z)
            return expscaled ? _En_safeexpmult(z, result) : result
        end
    end
    if real(z) > 0
        cf, _, _ = _En_cf_nogamma(ν, z)
        return expscaled ? cf : _En_safeexpmult(-z, cf)
    end
    return _expint_left_halfplane(ν, z, expscaled)
end

####################################
## Shared numerical helpers
####################################

@inline function _En_converged(
    delta::T, first::T, second::T, tol
) where {T<:Real}
    return abs(delta) <= tol * max(abs(first), abs(second))
end

@inline function _En_converged(
    delta::Complex{T}, first::Complex{T}, second::Complex{T}, tol
) where {T<:AbstractFloat}
    delta2 = abs2(delta)
    scale2 = max(abs2(first), abs2(second))
    if isfinite(delta2) && !iszero(delta2) &&
       isfinite(scale2) && !iszero(scale2)
        return delta2 <= abs2(tol) * scale2
    end
    return abs(delta) <= tol * max(abs(first), abs(second))
end

function _En_safeexpmult(z::T, value::T) where {T<:Real}
    ez = exp(z)
    if isinf(ez) || iszero(ez)
        return sign(value) * exp(z + log(abs(value)))
    end
    return ez * value
end

function _En_safeexpmult(
    z::Complex{T}, value::Complex{T}
) where {T<:AbstractFloat}
    ez = exp(z)
    if isinf(ez) || iszero(ez)
        return exp(z + log(value))
    end
    return ez * value
end

####################################
## Continued fraction
####################################

# Gamma-free continued fraction for E_ν(z):
# https://functions.wolfram.com/GammaBetaErf/ExpIntegralE/10/0001/
function _En_cf_nogamma(
    ν::T, z::T; maxiter::Union{Nothing,Int}=nothing,
    throw_on_failure::Bool=true
) where {T}
    R = typeof(real(z))
    tol = 8 * eps(one(R))
    cap = isnothing(maxiter) ? 50_000 : maxiter
    B = z + ν
    Bprev::typeof(B) = z
    A::typeof(B) = one(B)
    Aprev::typeof(B) = one(B)
    previous = Aprev / Bprev
    stable = 0

    for i = 2:cap
        previous_coefficient = i - 1
        A, Aprev = z * A + previous_coefficient * Aprev, A
        B, Bprev = z * B + previous_coefficient * Bprev, B
        coefficient = ν + previous_coefficient
        A, Aprev = A + coefficient * Aprev, A
        B, Bprev = B + coefficient * Bprev, B

        scale = max(abs(A), abs(Aprev), abs(B), abs(Bprev))
        A /= scale
        Aprev /= scale
        B /= scale
        Bprev /= scale

        current = A / B
        if i > 4 && _En_converged(
            current - previous, current, previous, tol
        )
            stable += 1
            stable >= 2 && return current, i, true
        else
            stable = 0
        end
        previous = current
    end

    throw_on_failure &&
        throw(IncompleteGammaConvergenceError(:expint_continued_fraction, cap))
    return A / B, cap, false
end

####################################
## Origin expansions
####################################

function _En_expand_origin_posint(
    n, z::T; maxiter::Union{Nothing,Int}=nothing
) where {T<:AbstractFloat}
    m = n - 1
    gammaterm = if m < 100
        result = one(z)
        for k = 1:Int(m)
            result *= -z / k
        end
        result
    else
        sign = isodd(m) ? -one(m) : one(m)
        sign * exp(m * log(abs(z)) - loggamma(m + 1))
    end
    return _En_expand_origin_posint(n, z, gammaterm, maxiter)
end

function _En_expand_origin_posint(
    n, z::Complex{T}; maxiter::Union{Nothing,Int}=nothing
) where {T<:AbstractFloat}
    m = n - 1
    gammaterm = if m < 100
        result = one(z)
        for k = 1:Int(m)
            result *= -z / k
        end
        result
    else
        exp(m * log(-z) - loggamma(m + 1))
    end
    return _En_expand_origin_posint(n, z, gammaterm, maxiter)
end

function _En_expand_origin_posint(n, z::T, gammaterm::T,
                                  maxiter::Union{Nothing,Int}) where {T}
    frac = one(z)
    gammaterm *= digamma(oftype(real(z), n)) - log(z)
    sumterm = n == 1 ? zero(z) : frac / (1 - n)
    R = typeof(real(z))
    tol = 8 * eps(one(R))
    cap = isnothing(maxiter) ? 50_000 : maxiter
    stable = 0

    for k = 1:cap
        frac *= -z / k
        if k != n - 1
            term = frac / (k + 1 - n)
            sumterm += term
            if _En_converged(term, sumterm, one(sumterm), tol)
                stable += 1
                stable >= 2 && return gammaterm - sumterm
            else
                stable = 0
            end
        end
    end
    throw(IncompleteGammaConvergenceError(:expint_origin_series, cap))
end

function _En_expand_origin_general(
    ν::T, z::T; maxiter::Union{Nothing,Int}=nothing
) where {T}
    gammaterm = gamma(1 - ν) * z^(ν - 1)
    frac = one(z)
    sumterm = frac / (1 - ν)
    R = typeof(real(z))
    tol = 8 * eps(one(R))
    cap = isnothing(maxiter) ? 50_000 : maxiter
    stable = 0

    for k = 1:cap
        frac *= -z / k
        term = frac / (k + 1 - ν)
        sumterm += term
        if _En_converged(term, sumterm, one(sumterm), tol)
            stable += 1
            stable >= 2 && return gammaterm - sumterm
        else
            stable = 0
        end
    end
    throw(IncompleteGammaConvergenceError(:expint_origin_series, cap))
end

####################################
## Left-half-plane continuation
####################################

function _En_taylor(ν::T, start::T, z0::T, delta::T;
                    maxiter::Union{Nothing,Int}=nothing) where {T}
    a = exp(z0) * start
    total = a
    delta_prod_fact = -delta
    R = typeof(real(total))
    tol = 8 * eps(one(R))
    cap = isnothing(maxiter) ? max(32, 2 * precision(R)) : maxiter
    stable = 0

    for k = 0:cap
        a = (delta_prod_fact + a * delta * (ν - k - 1) / (k + 1)) / z0
        total += a
        if _En_converged(a, total, one(total), tol)
            stable += 1
            stable >= 2 && return exp(-z0) * total
        else
            stable = 0
        end
        delta_prod_fact *= -delta / (k + 2)
    end
    throw(IncompleteGammaConvergenceError(:expint_continuation_series, cap))
end

function _expint_left_halfplane(
    ν::Complex{T}, z::Complex{T}, expscaled
) where {T<:AbstractFloat}
    reflect = imag(z) < 0
    rez, imz = real(z), abs(imag(z))
    original_z = z
    z = reflect ? conj(z) : z
    ν = reflect ? conj(ν) : ν
    R = typeof(real(z))
    quick = max(64, 2 * precision(R))
    target = -log(8 * eps(one(R)))
    s = R(5) * target / (R(16) * sqrt(R(quick)))
    estimate = 2 * s * sqrt(s^2 - real(z))
    imstart = max(imz, estimate)
    z0 = complex(rez, imstart)
    cf, _, converged = _En_cf_nogamma(ν, z0; maxiter=quick,
                                      throw_on_failure=false)

    doublings = 0
    while !converged
        imstart *= 2
        z0 = complex(rez, imstart)
        cf, _, converged = _En_cf_nogamma(ν, z0; maxiter=quick,
                                          throw_on_failure=false)
        doublings += 1
        doublings > 4 * precision(R) + 64 &&
            throw(IncompleteGammaConvergenceError(:expint_left_halfplane, quick))
    end

    if imz > 0 && z0 == z
        result = expscaled ? cf : _En_safeexpmult(-z0, cf)
        return reflect ? conj(result) : result
    end

    start = _En_safeexpmult(-z0, cf)
    distance = imstart - imz
    nsteps = ceil(Int, 2 * distance)
    nsteps > 1_000_000 &&
        throw(IncompleteGammaConvergenceError(:expint_left_halfplane, nsteps))
    delta = (imz - imstart) * im / nsteps
    for _ = 1:nsteps
        start = _En_taylor(ν, start, z0, delta)
        z0 += delta
    end

    result = reflect ? conj(start) : start
    if iszero(imz)
        x = real(z)
        upper_lip = !signbit(imag(original_z))
        branchsign = upper_lip ? 1 : -1
        if isreal(ν)
            logabs, sign = logabsgamma(real(ν))
            jump_half = sign * oftype(x, π) *
                        exp((real(ν) - 1) * log(-x) - logabs)
            result = real(result) - branchsign * jump_half * im
        elseif !upper_lip
            branchjump = 2 * oftype(x, π) *
                         exp((ν - 1) * log(-x) - loggamma(ν)) * im
            result += branchjump
        end
    end
    return expscaled ? _En_safeexpmult(original_z, result) : result
end
