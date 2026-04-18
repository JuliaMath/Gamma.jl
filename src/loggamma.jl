# Pure Julia loggamma and logabsgamma implementations
# Partly adapted from SpecialFunctions.jl (MIT license)
# using Stirling asymptotic series, Taylor series at z=1 and z=2,
# reflection formula, and shift recurrence.
# See: D. E. G. Hare, "Computing the principal branch of log-Gamma,"
# J. Algorithms 25, pp. 221-236 (1997)

####################################
## Typed getters returning literals
####################################

# Stirling coefficients
function _stirling_coeffs(::Type{Float64})
    return (
        8.333333333333333333333368e-02, -2.777777777777777777777778e-03,
        7.936507936507936507936508e-04, -5.952380952380952380952381e-04,
        8.417508417508417508417510e-04, -1.917526917526917526917527e-03,
        6.410256410256410256410257e-03, -2.955065359477124183006535e-02
    )
end
function _stirling_coeffs(::Type{Float32})
    return (
        8.333333333333333333333368f-02, -2.777777777777777777777778f-03,
        7.936507936507936507936508f-04, -5.952380952380952380952381f-04,
        8.417508417508417508417510f-04
    )
end

# Taylor series around 1 and 2
function _taylor1(::Type{Float64})
    return (
        -5.7721566490153286060651188e-01, 8.2246703342411321823620794e-01,
        -4.0068563438653142846657956e-01, 2.705808084277845478790009e-01,
        -2.0738555102867398526627303e-01, 1.6955717699740818995241986e-01,
        -1.4404989676884611811997107e-01, 1.2550966952474304242233559e-01,
        -1.1133426586956469049087244e-01, 1.000994575127818085337147e-01,
        -9.0954017145829042232609344e-02, 8.3353840546109004024886499e-02,
        -7.6932516411352191472827157e-02, 7.1432946295361336059232779e-02,
        -6.6668705882420468032903454e-02
    )
end
function _taylor1(::Type{Float32})
    return (
        -5.7721566490153286060651188f-01, 8.2246703342411321823620794f-01,
        -4.0068563438653142846657956f-01, 2.705808084277845478790009f-01,
        -2.0738555102867398526627303f-01, 1.6955717699740818995241986f-01,
        -1.4404989676884611811997107f-01, 1.2550966952474304242233559f-01,
        -1.1133426586956469049087244f-01, 1.000994575127818085337147f-01
    )
end

function _taylor2(::Type{Float64})
    return (
        4.2278433509846713939348812e-01, 3.2246703342411321823620794e-01,
        -6.7352301053198095133246196e-02, 2.0580808427784547879000897e-02,
        -7.3855510286739852662729527e-03, 2.8905103307415232857531201e-03,
        -1.1927539117032609771139825e-03, 5.0966952474304242233558822e-04,
        -2.2315475845357937976132853e-04, 9.9457512781808533714662972e-05,
        -4.4926236738133141700224489e-05, 2.0507212775670691553131246e-05
    )
end
function _taylor2(::Type{Float32})
    return (
        4.2278433509846713939348812f-01, 3.2246703342411321823620794f-01,
        -6.7352301053198095133246196f-02, 2.0580808427784547879000897f-02,
        -7.3855510286739852662729527f-03, 2.8905103307415232857531201f-03,
        -1.1927539117032609771139825f-03, 5.0966952474304242233558822f-04
    )
end

# Typed constant getters
_half_log2pi(::Type{Float64}) = 9.1893853320467274178032927e-01
_half_log2pi(::Type{Float32}) = 9.1893853320467274178032927f-01

_logpi(::Type{Float64}) = 1.1447298858494002
_logpi(::Type{Float32}) = 1.1447298858494002f0

_two_pi(::Type{Float64}) = 6.2831853071795864769252842
_two_pi(::Type{Float32}) = 6.2831853071795864769252842f0

# Generic loggamma entry points
loggamma(x::Union{Float32, Float64}) = _loggamma(x)
loggamma(x::Float16) = Float16(_loggamma(Float32(x)))
loggamma(x::Rational) = loggamma(float(x))
loggamma(x::Integer) = loggamma(float(x))
loggamma(z::Complex{Float64}) = _loggamma(z)
loggamma(z::Complex{Float32}) = _loggamma(z)
loggamma(z::Complex{Float16}) = Complex{Float16}(_loggamma(Complex{Float32}(z)))
loggamma(z::Complex{<:Integer}) = _loggamma(Complex{Float64}(z))
loggamma(z::Complex{<:Rational}) = loggamma(float(z))
function loggamma(x::BigFloat)
    # For now we use the same implementation for BigFloat as Complex{BigFloat}. This is not ideal since it does more work than necessary.
    if isnan(x)
        return x
    elseif isinf(x)
        return x > 0 ? x : BigFloat(NaN)
    elseif x <= 0
        iszero(x) && return BigFloat(Inf)
        isinteger(x) && return BigFloat(Inf)  # negative integer pole
        y, sgn = _logabsgamma(x)
        sgn < 0 && throw(DomainError(x, "`gamma(x)` must be non-negative"))
        return y
    end
    return real(_loggamma_complex_bigfloat(Complex{BigFloat}(x, zero(BigFloat))))
end
loggamma(z::Complex{BigFloat}) = _loggamma(z)

"""
    logfactorial(x)

Compute the logarithmic factorial of a nonnegative integer `x` via loggamma.
"""
function logfactorial(x::Integer)
    if x < 0
        throw(DomainError(x, "`x` must be non-negative."))
    end
    return loggamma(float(x + oneunit(x)))
end

####################################
## Float64 / Float32 loggamma and logabsgamma implementations
####################################

"""
    logabsgamma(x::Real)

Returns a tuple `(log(abs(Γ(x))), sign(Γ(x)))` for real `x`.
"""
logabsgamma(x::Float32) = _logabsgamma(x)
logabsgamma(x::Real) = _logabsgamma(float(x))
function logabsgamma(x::Float16)
    y, s = _logabsgamma(Float32(x))
    return Float16(y), s
end

function _logabsgamma(x::T) where T<:Union{Float32,Float64}
    if isnan(x)
        return x, 1
    elseif x > zero(x)
        return _loggamma_unsafe_pos(x), 1
    elseif iszero(x)
        return T(Inf), Int(sign(1 / x))
    else
        s = sinpi(x)
        iszero(s) && return T(Inf), 1
        sgn = signbit(s) ? -1 : 1
        return _logpi(T) - log(abs(s)) - _loggamma(T(1) - x), sgn
    end
end

# Generic unsafe-positive loggamma
function _loggamma_unsafe_pos(x::T) where T<:Union{Float32,Float64}
    if x < 7
        n = 7 - floor(Int, x)
        z = x
        prod = one(x)
        for i in 0:n-1
            prod *= z + i
        end
        return _loggamma_stirling(z + n) - log(prod)
    else
        return _loggamma_stirling(x)
    end
end

# logabsgamma without safety checks (used to avoid double checks)
function _logabsgamma_unsafe_sub0(x::T) where T<:Union{Float32,Float64}
    s = sinpi(x)
    sgn = signbit(s) ? -1 : 1
    return _logpi(T) - log(abs(s)) - _loggamma(T(1) - x), sgn
end

function _loggamma_stirling(x)
    T = typeof(real(x))
    tinv = inv(x)
    w = tinv * tinv
    tail = tinv * @evalpoly(w, _stirling_coeffs(T)...)
    return muladd(x - one(T)/2, log(x), -x + _half_log2pi(T) + tail)
end

"""
    loggamma(x::Real)

Returns the log of the absolute value of ``\\Gamma(x)`` for real `x`.
Throws a `DomainError` if ``\\Gamma(x)`` is negative.

For complex arguments, `exp(loggamma(x))` matches `gamma(x)` up to floating-point error
but may differ from `log(gamma(x))` by an integer multiple of ``2\\pi i``.

External links: [DLMF](https://dlmf.nist.gov/5.4), [Wikipedia](https://en.wikipedia.org/wiki/Gamma_function#The_log-gamma_function)
"""
function _loggamma(x::T) where T<:Union{Float32,Float64}
    if isnan(x)
        return x
    elseif isinf(x)
        return (x > 0 ? T(Inf) : T(NaN))
    elseif x ≤ 0
        if iszero(x)
            return T(Inf)
        elseif isinteger(x)
            return T(Inf)
        else
            y, sgn = _logabsgamma_unsafe_sub0(x)
            sgn < 0 && throw(DomainError(x, "`gamma(x)` must be non-negative"))
            return y
        end
    end
    # Use Taylor expansions near x=1 and x=2
    if abs(x - 1) < 0.1
        w = x - one(T)
        return w * @evalpoly(w, _taylor1(T)...)
    elseif abs(x - 2) < 0.1
        w = x - 2
        return w * @evalpoly(w, _taylor2(T)...)
    elseif x < 7
        n = 7 - floor(Int, x)
        z = x
        prod = one(x)
        for i in 0:n-1
            prod *= z + i
        end
        return _loggamma_stirling(z + n) - log(prod)
    else
        return _loggamma_stirling(x)
    end
end

####################################
## Complex{Float64} / Complex{Float32} loggamma implementation
####################################

function _loggamma(z::Complex{T}) where T<:Union{Float32,Float64}
    x, y = reim(z)
    yabs = abs(y)

    if !isfinite(x) || !isfinite(y)
        if isinf(x) && isfinite(y)
            return Complex{T}(x, x > 0 ? (iszero(y) ? y : copysign(T(Inf), y)) : copysign(T(Inf), -y))
        elseif isfinite(x) && isinf(y)
            return Complex{T}(-T(Inf), y)
        else
            return Complex{T}(T(NaN), T(NaN))
        end
    elseif x > 7 || yabs > 7
        return _loggamma_stirling(z)
    elseif x < 0.1
        if iszero(x) && iszero(y)
            imagpart = signbit(x) ? copysign(T(π), -y) : -y
            return Complex{T}(T(Inf), imagpart)
        end
        LOGPI_T = _logpi(T)
        TWO_PI_T = _two_pi(T)
        return Complex(LOGPI_T, copysign(TWO_PI_T, y) * floor((one(T)/2) * x + one(T)/4)) -
            log(sinpi(z)) - _loggamma(Complex{T}(1 - x, -y))
    elseif abs(x - 1) + yabs < 0.1
        w = Complex{T}(x - one(T), y)
        return w * @evalpoly(w, _taylor1(T)...)
    elseif abs(x - 2) + yabs < 0.1
        w = Complex{T}(x - 2, y)
        return w * @evalpoly(w, _taylor2(T)...)
    else
        shiftprod = Complex{T}(x, yabs)
        xshift = x + one(T)
        sb = false
        signflips = 0
        while xshift ≤ 7
            shiftprod *= Complex{T}(xshift, yabs)
            sbp = signbit(imag(shiftprod))
            signflips += sbp & (sbp != sb)
            sb = sbp
            xshift += one(T)
        end
        shift = log(shiftprod)
        TWO_PI_T = _two_pi(T)
        if signbit(y)
            shift = Complex(real(shift), signflips * -TWO_PI_T - imag(shift))
        else
            shift = Complex(real(shift), imag(shift) + signflips * TWO_PI_T)
        end
        return _loggamma_stirling(Complex{T}(xshift, y)) - shift
    end
end

####################################
## Complex{BigFloat} loggamma implementation
####################################
# Adapted from SpecialFunctions.jl (MIT license)
# Uses Stirling series with Bernoulli numbers computed via Akiyama-Tanigawa,
# reflection formula, upward recurrence, and branch correction via Float64 oracle.

# Scaled Stirling coefficients B_{2k}/(2k*(2k-1)) * zr^(1-2k) for k=0,...,n
# Bernoulli numbers computed inline via the Akiyama-Tanigawa algorithm
function _scaled_stirling_coeffs(n::Integer, zr::Complex{BigFloat})
    mmax = 2n
    A = Vector{Rational{BigInt}}(undef, mmax + 1)
    E = Vector{Complex{BigFloat}}(undef, n + 1)
    @inbounds for m = 0:mmax
        A[m+1] = 1 // (m + 1)
        for j = m:-1:1
            A[j] = j * (A[j] - A[j+1])
        end
        if iseven(m)
            k = m ÷ 2
            E[k+1] = A[1] / (2k * (2k - 1)) * zr^(1 - 2k)
        end
    end
    return E
end

function _loggamma_complex_bigfloat(z::Complex{BigFloat})
    bigpi = big(π)
    x = real(z)
    y = imag(z)

    if !isfinite(x) || !isfinite(y)
        inf = BigFloat(Inf)
        nan = BigFloat(NaN)
        if isinf(x) && isfinite(y)
            yim = x > 0 ? (iszero(y) ? y : copysign(inf, y)) : copysign(inf, -y)
            return Complex{BigFloat}(x, yim)
        elseif isfinite(x) && isinf(y)
            return Complex{BigFloat}(-inf, y)
        else
            return Complex{BigFloat}(nan, nan)
        end
    end

    # reflection formula
    if x < 0.5
        val = log(bigpi) - log(sinpi(z)) - _loggamma_complex_bigfloat(1 - z)
        return _loggamma_branchcorrect(val, z)
    end

    # upward recurrence: shift z into the Stirling region
    p = precision(BigFloat)
    r = max(0, Int(ceil(p - abs(z))))
    zr = z + r

    # Stirling series
    N = max(10, p ÷ 15)
    B = _scaled_stirling_coeffs(N, zr)
    lg = sum(B[2:end]) + (zr - big"0.5") * log(zr) - zr + log(sqrt(2 * bigpi))

    # undo upward shift via log of product
    if r > 0
        prodarg = prod(z + (i - 1) for i in 1:r)
        lg -= log(prodarg)
    end

    return _loggamma_branchcorrect(lg, z)
end

# Branch correction: offset by multiples of 2πi to match the Float64 branch
function _loggamma_branchcorrect(val::Complex{BigFloat}, z::Complex{BigFloat})
    zf = _loggamma_oracle64_point(z)
    val_f = _loggamma(zf)
    imv = imag(val)
    k = round(Int, (Float64(imv) - imag(val_f)) / (2π))
    return Complex{BigFloat}(real(val), imv - 2 * big(π) * k)
end

# Map a BigFloat complex point to Float64 for branch-cut determination
function _loggamma_oracle64_point(z::Complex{BigFloat})
    xr = Float64(real(z))
    xi = Float64(imag(z))
    n = round(Int, xr)
    if n ≤ 0 && isapprox(xr, Float64(n); atol=eps(Float64)) && abs(xi) ≤ 2eps(Float64)
        xr = real(z) > n ? nextfloat(xr) : prevfloat(xr)
    end
    return Complex{Float64}(xr, xi)
end

# Complex{BigFloat} entry point with guard precision
function _loggamma(z::Complex{BigFloat})
    imz = imag(z)
    rez = real(z)
    if iszero(imz)
        return Complex(loggamma(rez))
    end
    p0 = precision(BigFloat)
    guard = 16
    setprecision(p0 + guard) do
        zhi = Complex{BigFloat}(rez, imz)
        rhi = _loggamma_complex_bigfloat(zhi)
        setprecision(p0) do
            return Complex{BigFloat}(real(rhi), imag(rhi))
        end
    end
end

function _logabsgamma(x::BigFloat)
    if isnan(x)
        return x, 1
    elseif isinf(x)
        return x > 0 ? (x, 1) : (BigFloat(NaN), 1)
    elseif x > 0
        return real(_loggamma_complex_bigfloat(Complex{BigFloat}(x, zero(BigFloat)))), 1
    elseif iszero(x)
        return BigFloat(Inf), Int(sign(1 / x))
    end

    s = sinpi(x)
    iszero(s) && return BigFloat(Inf), 1
    return real(_loggamma_complex_bigfloat(Complex{BigFloat}(x, zero(BigFloat)))), (signbit(s) ? -1 : 1)
end