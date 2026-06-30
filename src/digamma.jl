# Partly adapted from SpecialFunctions.jl (MIT license).

"""
    digamma(x)

Compute the digamma function of `x`, the logarithmic derivative of `gamma(x)`.

External links: [DLMF](https://dlmf.nist.gov/5.2#E2), [Wikipedia](https://en.wikipedia.org/wiki/Digamma_function)
"""
digamma(x::Union{Integer, Rational}) = digamma(float(x))
digamma(x::Float64) = _digamma(x)
digamma(x::Float32) = _digamma(x)
digamma(x::Float16) = Float16(_digamma(Float32(x)))
digamma(x::BigFloat) = _digamma(x)
digamma(z::Complex{Float64}) = _digamma(z)
digamma(z::Complex{Float32}) = _digamma(z)
digamma(z::Complex{Float16}) = Complex{Float16}(_digamma(Complex{Float32}(z)))
digamma(z::Complex{BigFloat}) = _digamma(z)
function digamma(z::Complex{<:Union{Integer, Rational}})
    zf = Complex(float.(reim(z))...)
    return digamma(zf)
end

_cotpi(x::Union{Float32, Float64, BigFloat}) = inv(tanpi(x))

function _cotpi(z::Complex{T}) where T<:Union{Float32, Float64, BigFloat}
    x, y = reim(z)
    iszero(y) && return Complex{T}(_cotpi(x), -zero(T))
    e = exp(-2T(π) * abs(y))
    e2 = e * e
    sech = 2e / (one(T) + e2)
    tanh = (one(T) - e2) / (one(T) + e2)
    denom = one(T) - cospi(2x) * sech
    return Complex{T}(sinpi(2x) * sech / denom, -copysign(tanh, y) / denom)
end

# Float64 implementation
function _digamma(z::Union{Float64, Complex{Float64}})
    x = real(z)
    if x <= 0
        ψ = -π * _cotpi(z)
        z = 1 - z
        x = real(z)
    else
        ψ = zero(z)
    end
    X = 8
    if x < X
        n = X - floor(Int, x)
        for ν = 1:n-1
            ψ -= inv(z + ν)
        end
        ψ -= inv(z)
        z += n
    end
    t = inv(z)
    ψ += log(z) - 0.5 * t
    t *= t
    return ψ - t * evalpoly(t, (
        0.08333333333333333, -0.008333333333333333,
        0.003968253968253968, -0.004166666666666667,
        0.007575757575757576, -0.021092796092796094,
        0.08333333333333333, -0.4432598039215686
    ))
end

# Float32 implementation
function _digamma(z::Union{Float32, Complex{Float32}})
    x = real(z)
    if x <= 0f0
        ψ = -Float32(π) * _cotpi(z)
        z = one(z) - z
        x = real(z)
    else
        ψ = zero(z)
    end
    X = 8f0
    if x < X
        n = 8 - floor(Int, x)
        for ν = 1:n-1
            ψ -= inv(z + ν)
        end
        ψ -= inv(z)
        z += n
    end
    t = inv(z)
    ψ += log(z) - 0.5f0 * t
    t *= t
    return ψ - t * evalpoly(t, (
        0.083333336f0, -0.008333334f0,
        0.003968254f0, -0.004166667f0,
        0.007575758f0, -0.021092797f0
    ))
end

# BigFloat implementation
# Following MPFR, cache the integers B[2n] * (2n+1)! and absorb the
# factorial into the asymptotic recurrence.
const _BERNOULLI_SCALED = BigInt[
    1,
    -4,
    120,
    -12096,
    3024000,
    -1576143360,
    1525620096000,
    -2522591034163200,
    6686974460694528000,
    -27033456071346536448000,
    160078872315904478576640000,
    -1342964491649083924630732800000,
    15522270327163593186886877184000000,
    -241364461951740682229320388129587200000,
    4946702463524336786354770193822515200000000,
    -131259771031915242949184995422830881406976000000,
]
const _BERNOULLI_SCALED_LOCK = ReentrantLock()

function _isprime_odd(n::Int)
    d = 3
    while d <= n ÷ d
        n % d == 0 && return false
        d += 2
    end
    return true
end

function _bernoulli_denominator(n::Int)
    den = BigInt(6)
    for p in 5:2:n+1
        n % (p - 1) == 0 && _isprime_odd(p) && (den *= p)
    end
    return den
end

function _scaled_bernoulli_candidate(n::Int, den::BigInt, p::Int)
    scale = BigInt(1) << p
    zeta = copy(scale)
    zeta += cld(scale, BigInt(1) << n)

    k = 3
    while true
        kn = BigInt(k)^n
        kn > scale && break
        zeta += scale ÷ kn
        k += 1
    end
    zeta += cld(scale, (n - 1) * BigInt(k)^(n - 1))

    setrounding(BigFloat, RoundNearest) do
        setprecision(p + 32) do
            x = BigFloat(zeta)
            x *= 2den * factorial(BigInt(n))
            x = ldexp(x, -p) / (2big(π))^n
            return round(BigInt, x)
        end
    end
end

function _scaled_bernoulli(n::Int, guard::Int=16)
    m = 2n
    den = _bernoulli_denominator(m)
    bits = 1 + ndigits(den; base=2) +
           loggamma(Float64(m + 1)) / log(2) - m * log2(2π)
    p = max(32, ceil(Int, bits) + guard)
    a = _scaled_bernoulli_candidate(m, den, p)
    b = _scaled_bernoulli_candidate(m, den, p + 32)
    while a != b
        p += 32
        a = b
        b = _scaled_bernoulli_candidate(m, den, p + 32)
    end
    b *= factorial(BigInt(m + 1)) ÷ den
    return isodd(n) ? b : -b
end

function _bernoulli_scaled(n::Int)
    lock(_BERNOULLI_SCALED_LOCK) do
        while length(_BERNOULLI_SCALED) < n
            push!(_BERNOULLI_SCALED, _scaled_bernoulli(length(_BERNOULLI_SCALED) + 1))
        end
        return _BERNOULLI_SCALED[n]
    end
end

function _digamma_asymptotic_big(z::BigFloat)
    psi = log(z) - inv(2z)
    w = inv(z * z)
    threshold = ldexp(one(BigFloat), -precision(BigFloat) - 8) *
                max(abs(psi), one(BigFloat))
    t = one(z)
    n = 1
    while true
        t *= w
        t /= 2n
        t /= 2n + 1
        term = t * _bernoulli_scaled(n) / 2n
        abs(term) <= threshold && break
        psi -= term
        n += 1
    end
    return psi
end

function _digamma_asymptotic_big(z::Complex{BigFloat})
    psi = log(z) - inv(2z)
    w = inv(z * z)
    threshold = ldexp(one(BigFloat), -precision(BigFloat) - 8) *
                max(abs(psi), one(BigFloat))
    t = one(z)
    n = 1
    while true
        t *= w
        t /= 2n
        t /= 2n + 1
        term = t * _bernoulli_scaled(n) / 2n
        abs(term) <= threshold && break
        psi -= term
        n += 1
    end
    return psi
end

_digamma_cutoff_big() = max(2, (precision(BigFloat) + 3) ÷ 4)

function _digamma_positive_big(z::BigFloat)
    cutoff = _digamma_cutoff_big()
    s = zero(z)
    while z < cutoff
        s += inv(z)
        z += 1
    end
    return _digamma_asymptotic_big(z) - s
end

function _digamma_positive_big(z::Complex{BigFloat})
    cutoff = _digamma_cutoff_big()
    cutoff2 = BigFloat(cutoff)^2
    s = zero(z)
    while real(z) < cutoff && abs2(z) < cutoff2
        s += inv(z)
        z += 1
    end
    return _digamma_asymptotic_big(z) - s
end

function _digamma_big(z::Union{BigFloat, Complex{BigFloat}})
    if real(z) < big"0.5"
        return _digamma_positive_big(1 - z) - big(π) * _cotpi(z)
    else
        return _digamma_positive_big(z)
    end
end

function _digamma(x::BigFloat)
    isnan(x) && return x
    isinf(x) && return x > 0 ? x : BigFloat(NaN)
    iszero(x) && return copysign(BigFloat(Inf), -x)
    x < 0 && isinteger(x) && return BigFloat(NaN)

    p0 = precision(BigFloat)
    e = exponent(x)
    e <= -2 * max(precision(x), p0) && return -inv(x)
    x > 0 && p0 + 30 < e && return log(x)

    setprecision(p0 + max(16, ndigits(p0; base=2) + 8)) do
        xhi = BigFloat(x)
        yhi = _digamma_big(xhi)
        setprecision(p0) do
            return BigFloat(yhi)
        end
    end
end

function _digamma(z::Complex{BigFloat})
    x, y = reim(z)
    if isnan(x) || isnan(y)
        return Complex{BigFloat}(BigFloat(NaN), BigFloat(NaN))
    elseif isinf(x) || isinf(y)
        return log(z)
    elseif iszero(x) && iszero(y)
        return Complex{BigFloat}(_digamma(x), zero(BigFloat))
    end

    p0 = precision(BigFloat)
    setprecision(p0 + max(16, ndigits(p0; base=2) + 8)) do
        zhi = Complex{BigFloat}(x, y)
        yhi = _digamma_big(zhi)
        setprecision(p0) do
            return Complex{BigFloat}(real(yhi), imag(yhi))
        end
    end
end
