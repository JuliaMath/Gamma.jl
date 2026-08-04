using Gamma: IncompleteGammaConvergenceError

include("inc_gamma_mpmath_cases.jl")

_refmetric(got, ref) = iszero(ref) ? abs(got - ref) : abs((got - ref) / ref)

const _FLOAT64_LOWER_RTOL = Dict(
    "small-z" => 5e-14,
    "crossover" => 4e-14,
    "large-z" => 3e-14,
    "negative-a" => 4e-15,
    "complex-quadrants" => 5e-15,
    "negative-axis" => 8e-15,
    "complex-crossover" => 1.1e-14,
)

const _FLOAT64_UPPER_RTOL = Dict(
    "small-z" => 5e-16,
    "crossover" => 2e-12,
    "large-z" => 3e-14,
    "negative-a" => 4e-10,
    "complex-quadrants" => 5e-15,
    "negative-axis" => 8e-15,
    "complex-crossover" => 1.1e-14,
)

function _parse_float64_case(case)
    _, ar, ai, zr, zi, lr, li, ur, ui, regime = case
    a = complex(parse(Float64, ar), parse(Float64, ai))
    z = complex(parse(Float64, zr), parse(Float64, zi))
    lower = complex(Float64(parse(BigFloat, lr)), Float64(parse(BigFloat, li)))
    upper = complex(Float64(parse(BigFloat, ur)), Float64(parse(BigFloat, ui)))
    return a, z, lower, upper, regime
end

function _parse_bigfloat_case(case)
    _, ar, ai, zr, zi, lr, li, ur, ui, regime = case
    a = complex(parse(BigFloat, ar), parse(BigFloat, ai))
    z = complex(parse(BigFloat, zr), parse(BigFloat, zi))
    lower = complex(parse(BigFloat, lr), parse(BigFloat, li))
    upper = complex(parse(BigFloat, ur), parse(BigFloat, ui))
    return a, z, lower, upper, regime
end

@testset "incomplete gamma API" begin
    for T in (
        Float16, Float32, Float64, BigFloat,
        Complex{Float16}, Complex{Float32}, ComplexF64, Complex{BigFloat},
    )
        a, z = T(2), T(3)
        @test @inferred(gamma(a, z)) isa T
        @test @inferred(gamma_lower(a, z)) isa T
        p, q = @inferred Gamma.gamma_inc(a, z)
        @test p isa T
        @test q isa T
        @test p + q == one(T)
    end

    @test @inferred(gamma(2, 3)) isa Float64
    @test @inferred(gamma_lower(2, 3)) isa Float64
    @test @inferred(Gamma.gamma_inc(2, 3)) isa Tuple{Float64,Float64}
    @test @inferred(gamma(2, 3.0)) isa Float64
    @test @inferred(gamma_lower(2, 3.0)) isa Float64
    @test @inferred(Gamma.gamma_inc(2, 3.0)) isa Tuple{Float64,Float64}
    @test @inferred(gamma(Float32(2), 3.0)) isa Float64
    @test @inferred(gamma_lower(Float32(2), 3.0)) isa Float64
    @test @inferred(Gamma.gamma_inc(Float32(2), 3.0)) isa
          Tuple{Float64,Float64}
    @test @inferred(gamma(2, π)) isa Float64
    @test @inferred(gamma_lower(2, π)) isa Float64
    @test @inferred(Gamma.gamma_inc(2, π)) isa Tuple{Float64,Float64}
    @test @inferred(gamma(2 + 0im, 3.0)) isa ComplexF64
    @test @inferred(gamma_lower(2 + 0im, 3.0)) isa ComplexF64
    @test @inferred(Gamma.gamma_inc(2 + 0im, 3.0)) isa
          Tuple{ComplexF64,ComplexF64}
    @test @inferred(gamma(big"2", 3.0 + 0im)) isa Complex{BigFloat}
    @test @inferred(gamma_lower(big"2", 3.0 + 0im)) isa Complex{BigFloat}
    @test @inferred(Gamma.gamma_inc(big"2", 3.0 + 0im)) isa
          Tuple{Complex{BigFloat},Complex{BigFloat}}

    @test_throws MethodError Gamma.gamma_inc(2.0, 3.0, 0)
    @test_throws DomainError gamma_lower(0.0, 1.0)
    @test_throws DomainError gamma_lower(-1.0, 1.0)
    @test_throws DomainError gamma_lower(0.5, -1.0)
    @test_throws DomainError gamma(0.5, -1.0)
    @test_throws DomainError Gamma.gamma_inc(0.5, -1.0)
    @test_throws DomainError Gamma.gamma_inc(0.0, 1.0)
    for T in (ComplexF64, Complex{BigFloat})
        a, z = T(0), T(1)
        @test gamma(a, z) ≈ Gamma.expint(T(1), z)
        @test_throws DomainError gamma_lower(a, z)
        @test_throws DomainError Gamma.gamma_inc(a, z)
    end
    @test_throws IncompleteGammaConvergenceError Gamma._En_cf_nogamma(
        0.5, 2.0; maxiter=5
    )

    Gamma.gamma_inc(2.5, 2.0)
    Gamma.gamma_inc(2.5 + 0.2im, 2.0 + 0.1im)
    @test (@allocated Gamma.gamma_inc(2.5, 2.0)) == 0
    @test (@allocated Gamma.gamma_inc(2.5 + 0.2im, 2.0 + 0.1im)) == 0
end

@testset "Float16 evaluation through Float32" begin
    a, z = Float16(2.5), Float16(2)
    a32, z32 = Float32(a), Float32(z)
    @test gamma(a, z) === Float16(gamma(a32, z32))
    @test gamma_lower(a, z) === Float16(gamma_lower(a32, z32))
    p32, q32 = Gamma.gamma_inc(a32, z32)
    @test Gamma.gamma_inc(a, z) === (Float16(p32), Float16(q32))

    a, z = ComplexF16(0.5, 0.25), ComplexF16(2, 1)
    a32, z32 = ComplexF32(a), ComplexF32(z)
    @test gamma(a, z) === ComplexF16(gamma(a32, z32))
    @test gamma_lower(a, z) === ComplexF16(gamma_lower(a32, z32))
    p32, q32 = Gamma.gamma_inc(a32, z32)
    @test Gamma.gamma_inc(a, z) === (ComplexF16(p32), ComplexF16(q32))
    @test Gamma.gamma_inc(real(a), z) ===
          Gamma.gamma_inc(ComplexF16(real(a)), z)

    @test Gamma.gamma_inc(Float16(25), Float16(1000)) ===
          (Float16(1), Float16(0))
    @test Gamma.gamma_inc(Float16(300), Float16(300)) ===
          (Float16(0.508), Float16(0.4924))
end

@testset "closed forms and identities" begin
    for T in (Float64, ComplexF64)
        z = T(3)
        for n = 1:8
            upper_big, lower_big = setprecision(128) do
                zb = BigFloat(3)
                complete = BigFloat(factorial(n - 1))
                ub = complete * exp(-zb) *
                     sum(zb^k / factorial(k) for k = 0:n-1)
                ub, complete - ub
            end
            upper, lower = T(upper_big), T(lower_big)
            @test gamma(T(n), z) ≈ upper rtol=16eps(Float64)
            @test gamma_lower(T(n), z) ≈ lower rtol=16eps(Float64)
        end
    end

    for T in (Float64, BigFloat)
        a, z = T(1.75), T(0.8)
        lower = gamma_lower(a, z)
        upper = gamma(a, z)
        @test lower + upper ≈ gamma(a) rtol=128eps(T)
        @test gamma_lower(a + one(T), z) ≈
              a * lower - z^a * exp(-z) rtol=256eps(T)
    end

    a, z = 0.5, 1.25
    @test gamma_lower(a, z) ≈ sqrt(π) * SpecialFunctions.erf(sqrt(z))
    @test gamma(a, z) ≈ sqrt(π) * SpecialFunctions.erfc(sqrt(z))
    @test gamma(a, z) ≈ z^a * Gamma.expint(1 - a, z) rtol=32eps(Float64)
end

@testset "generalized exponential integral" begin
    for T in (
        Float16, Float32, Float64, BigFloat,
        Complex{Float16}, Complex{Float32}, ComplexF64, Complex{BigFloat},
    )
        ν, z = T(2), T(2)
        @test @inferred(Gamma.expint(ν, z)) isa T
        @test @inferred(Gamma.expintx(ν, z)) isa T
    end
    for f in (Gamma.expint, Gamma.expintx)
        ν, z = Float16(0.5), Float16(2)
        @test f(ν, z) === Float16(f(Float32(ν), Float32(z)))
        ν, z = ComplexF16(0.5, 0.25), ComplexF16(2, 1)
        @test f(ν, z) === ComplexF16(f(ComplexF32(ν), ComplexF32(z)))
        @test f(real(ν), z) === f(ComplexF16(real(ν)), z)
    end
    @test @inferred(Gamma.expint(1, 2.0)) isa Float64
    @test @inferred(Gamma.expintx(1, 2.0)) isa Float64
    @test @inferred(Gamma.expint(Float32(1), 2.0)) isa Float64
    @test @inferred(Gamma.expintx(Float32(1), 2.0)) isa Float64
    @test @inferred(Gamma.expint(1, π)) isa Float64
    @test @inferred(Gamma.expintx(1, π)) isa Float64
    @test @inferred(Gamma.expint(1 + 0im, 2.0)) isa ComplexF64
    @test @inferred(Gamma.expintx(1 + 0im, 2.0)) isa ComplexF64
    @test @inferred(Gamma.expint(big"1", 2.0 + 0im)) isa Complex{BigFloat}
    @test @inferred(Gamma.expintx(big"1", 2.0 + 0im)) isa Complex{BigFloat}

    setprecision(256) do
        cases = (
            (complex(big"0.5"), complex(big"2"),
             complex(big"0.05702612399289204827645887193117990741350550876172169370280621084224744842308610367344716230811960771")),
            (complex(big"1"), complex(big"1"),
             complex(big"0.21938393439552027367716377546012164903104729340690820757797861307356869855914154472221025103513725")),
            (complex(big"2.3"), complex(big"1", big"2"),
             complex(big"-0.08900094783292969603230068696548775791852855369872728695579144430812641514561403479949250371475292285",
                     big"-0.04708186526834273439842211340143026024172539176077138569967436983433323606340448808894006124274060858")),
            (complex(big"-0.5"), complex(big"-3", big"0.1"),
             complex(big"-5.279703637332678779535765471007746416904813944781438259335574021808117056613629975047922595892107188",
                     big"0.5748584955040566259508322825190196921611540523181709746994239790504698808465109404900441709276139088")),
        )
        for (ν, z, reference) in cases
            value = Gamma.expint(ν, z)
            @test _refmetric(value, reference) < big"1e-72"
            @test Gamma.expintx(ν, z) ≈ exp(z) * value rtol=big"1e-72"
        end
    end

    @test isnan(Gamma.expint(NaN, 1.0))
    @test isnan(Gamma.expint(1.0, NaN))
    @test isnan(Gamma.expint(complex(NaN), 1.0 + 0im))
    @test isnan(Gamma.expint(1.0 + 0im, complex(NaN)))
    @test_throws DomainError Gamma.expint(1.0, -1.0)
    @test Gamma.expint(2.0, 0.0) == 1.0
    @test isinf(Gamma.expint(1.0, 0.0))
    @test Gamma.expint(2.0 + 0im, 0.0 + 0im) == 1.0 + 0im
    @test isinf(real(Gamma.expint(1.0 + 0im, 0.0 + 0im)))
    @test Gamma.expint(0.0, 2.0) ≈ exp(-2) / 2
    @test Gamma.expintx(0.0, 2.0) == 0.5
    @test Gamma.expint(0.0 + 0im, 2.0 + 1im) ==
          exp(-(2.0 + 1im)) / (2.0 + 1im)
    @test Gamma.expintx(0.0 + 0im, 2.0 + 1im) == inv(2.0 + 1im)
    @test Gamma.expint(2.0) == Gamma.expint(1.0, 2.0)
    @test Gamma.expintx(2.0) == Gamma.expintx(1.0, 2.0)
    @test Gamma.expintx(0.5, 1e20) == 1e-20
    @test Gamma.expintx(1e20, 1e20) == 5e-21
    for T in (Float64, BigFloat)
        ν, z = parse(T, "1e20"), T(0.5)
        value = Gamma.expint(ν, z)
        @test isfinite(value) && value > 0
        @test Gamma.expint(complex(ν), complex(z)) == complex(value)
    end

    reference = sqrt(π / 4) * SpecialFunctions.erfc(2)
    @test Gamma.expint(0.5, 4.0) ≈ reference rtol=8eps()
    @test Gamma.expintx(0.5, 4.0) ≈ exp(4) * reference rtol=8eps()

    z = 0.5
    for n in (2, 101)
        @test n * Gamma.expint(n + 1, z) ≈
              exp(-z) - z * Gamma.expint(n, z) rtol=8eps()
    end

    branch_cases = (
        (1.0, complex(-3.0, 0.0),
         -9.933832570625416558 - 3.1415926535897932385im),
        (1.0, complex(-3.0, -0.0),
         -9.933832570625416558 + 3.1415926535897932385im),
        (0.5 + 0.25im, complex(-3.0, 0.0),
         -7.6931876545950181363 - 1.7637064918779949669im),
        (0.5 + 0.25im, complex(-3.0, -0.0),
         -9.258828300557080910 - 0.0037977027808670030im),
    )
    for (ν, zcut, expected) in branch_cases
        @test Gamma.expint(ν, zcut) ≈ expected rtol=8eps()
    end
end

@testset "incomplete gamma special values" begin
    @test gamma_lower(2.5, 0.0) == 0.0
    @test_throws DomainError gamma_lower(-0.5, 0.0)
    @test_throws DomainError Gamma.gamma_inc(BigFloat(0), BigFloat(1))
    @test_throws DomainError Gamma.gamma_inc(BigFloat(-0.5), BigFloat(0))
    @test_throws DomainError Gamma.gamma_inc(
        complex(big"-0.5", big"0.25"), complex(big"0")
    )
    @test gamma(2.5, 0.0) == gamma(2.5)
    @test gamma(2.5 + 0im, 0.0 + 0im) == gamma(2.5 + 0im)
    @test_throws DomainError gamma(0.0, 0.0)
    @test_throws DomainError gamma(0.0 + 0im, 0.0 + 0im)

    @test gamma(0.0, 1.0) ≈ Gamma.expint(1.0)
    @test gamma(-1.0, 1.0) ≈ exp(-1) - Gamma.expint(1.0)

    @test Gamma.gamma_inc(2.5, 0.0) == (0.0, 1.0)
    @test Gamma.gamma_inc(Inf, 0.0) == (0.0, 1.0)
    @test_throws DomainError Gamma.gamma_inc(-0.5, 0.0)
    @test Gamma.gamma_inc(2.5 + 0im, 0.0 + 0im) ==
          (0.0 + 0im, 1.0 + 0im)
    @test Gamma.gamma_inc(complex(Inf), complex(0.0)) ==
          (0.0 + 0im, 1.0 + 0im)
    @test_throws DomainError Gamma.gamma_inc(-0.5 + 0im, 0.0 + 0im)

    complex_p, complex_q = Gamma.gamma_inc(2.0 + 0im, 1.0 + 0im)
    real_p, real_q = Gamma.gamma_inc(2.0, 1.0)
    @test complex_p ≈ real_p
    @test complex_q ≈ real_q
    fallback_p, fallback_q = Gamma.gamma_inc(12.0 + 0im, 15.0 + 0im)
    @test fallback_p + fallback_q == 1.0 + 0im

    a, z = 0.5 + 0.2im, -4.0 + 1.0im
    p, q = Gamma.gamma_inc(a, z)
    complete = gamma(a)
    @test p + q ≈ 1
    @test p ≈ gamma_lower(a, z) / complete rtol=2e-14
    @test q ≈ gamma(a, z) / complete rtol=2e-14
end

@testset "incomplete gamma convergence reporting" begin
    @test Gamma.gamma_inc(1e19, 1.0) == (0.0, 1.0)

    _, iterations, converged = Gamma._En_cf_nogamma(
        0.5, 2.0; maxiter=2, throw_on_failure=false
    )
    @test iterations == 2
    @test !converged

    forced_failures = (
        (:expint_origin_series, 1,
         () -> Gamma._En_expand_origin_posint(2, 0.5; maxiter=1)),
        (:expint_origin_series, 1,
         () -> Gamma._En_expand_origin_general(0.5, 0.5; maxiter=1)),
        (:expint_continuation_series, 1,
         () -> Gamma._En_taylor(
             0.5 + 0im, Gamma.expint(0.5, 3 + 3im), 3.0 + 3.0im, 0.1im;
             maxiter=1
         )),
        (:lower_incomplete_gamma_series, 1,
         () -> Gamma._gamma_lower_series(0.5, 0.5; maxiter=1)),
        (:normalized_lower_incomplete_gamma_series, 1,
         () -> Gamma._gamma_lower_series_normalized(0.5, 0.5; maxiter=1)),
    )
    for (algorithm, iterations, f) in forced_failures
        err = try
            f()
            nothing
        catch caught
            caught
        end
        @test err isa IncompleteGammaConvergenceError
        @test err.algorithm == algorithm
        @test err.iterations == iterations
        @test sprint(showerror, err) ==
              "$algorithm did not converge after $iterations iterations at the requested precision"
    end

    err = try
        Gamma.expint(Float16(100), Complex{Float16}(-100, 1))
        nothing
    catch caught
        caught
    end
    @test err isa IncompleteGammaConvergenceError
    @test err.algorithm == :expint_continuation_series

    err = try
        Gamma._expint_left_halfplane(
            ComplexF16(1000), ComplexF16(-10, 1), false
        )
        nothing
    catch caught
        caught
    end
    @test err isa IncompleteGammaConvergenceError
    @test err.algorithm == :expint_left_halfplane
end

@testset "mpmath Float64 and ComplexF64 references" begin
    for case in INCOMPLETE_GAMMA_MPMATH_FLOAT64_CASES
        case[10] == "temme-gap" && continue
        a, z, lower_ref, upper_ref, regime = _parse_float64_case(case)
        lower, upper = gamma_lower(a, z), gamma(a, z)
        @test _refmetric(lower, lower_ref) <= _FLOAT64_LOWER_RTOL[regime]
        @test _refmetric(upper, upper_ref) <= _FLOAT64_UPPER_RTOL[regime]
    end
end

@testset "mpmath BigFloat and Complex{BigFloat} references" begin
    for bits in (256, 512)
        setprecision(bits) do
            tolerance = BigFloat(10)^6 * eps(BigFloat)
            for case in INCOMPLETE_GAMMA_MPMATH_CASES
                a, z, lower_ref, upper_ref, _ = _parse_bigfloat_case(case)
                @test _refmetric(gamma_lower(a, z), lower_ref) <= tolerance
                @test _refmetric(gamma(a, z), upper_ref) <= tolerance
            end
        end
    end
end

@testset "real mpmath references" begin
    float_cases = filter(c -> c[3] == "0" && c[5] == "0" &&
                             !startswith(c[4], "-") &&
                             c[10] != "temme-gap",
                         INCOMPLETE_GAMMA_MPMATH_FLOAT64_CASES)
    for case in float_cases
        _, ar, _, zr, _, lr, _, ur, _, regime = case
        a, z = parse(Float64, ar), parse(Float64, zr)
        lower_ref = Float64(parse(BigFloat, lr))
        upper_ref = Float64(parse(BigFloat, ur))
        @test _refmetric(gamma_lower(a, z), lower_ref) <=
              _FLOAT64_LOWER_RTOL[regime]
        @test _refmetric(gamma(a, z), upper_ref) <=
              _FLOAT64_UPPER_RTOL[regime]
    end

    big_cases = filter(c -> c[3] == "0" && c[5] == "0" &&
                           !startswith(c[4], "-") &&
                           c[10] != "temme-gap",
                       INCOMPLETE_GAMMA_MPMATH_CASES)
    for bits in (256, 512)
        setprecision(bits) do
            tolerance = BigFloat(10)^6 * eps(BigFloat)
            for case in big_cases
                _, ar, _, zr, _, lr, _, ur, _, _ = case
                a, z = parse(BigFloat, ar), parse(BigFloat, zr)
                lower_ref, upper_ref = parse(BigFloat, lr), parse(BigFloat, ur)
                @test _refmetric(gamma_lower(a, z), lower_ref) <= tolerance
                @test _refmetric(gamma(a, z), upper_ref) <= tolerance
            end
        end
    end
end

@testset "cross precision" begin
    selected = (
        ("0.5", "2.0"),
        ("5.0", "5.9"),
        ("100.0", "101.1"),
        ("-0.5001", "3.0"),
    )
    values256 = setprecision(256) do
        [(gamma_lower(parse(BigFloat, a), parse(BigFloat, z)),
          gamma(parse(BigFloat, a), parse(BigFloat, z))) for (a, z) in selected]
    end
    setprecision(512) do
        for ((a, z), (lower256, upper256)) in zip(selected, values256)
            lower512 = gamma_lower(parse(BigFloat, a), parse(BigFloat, z))
            upper512 = gamma(parse(BigFloat, a), parse(BigFloat, z))
            @test _refmetric(BigFloat(lower256), lower512) < big"1e-74"
            @test _refmetric(BigFloat(upper256), upper512) < big"1e-74"
        end
    end
end

@testset "SpecialFunctions comparison and small z" begin
    positive_real_cases = filter(
        c -> c[3] == "0" && c[5] == "0" &&
             parse(BigFloat, c[2]) > 0 && parse(BigFloat, c[4]) >= 0,
        INCOMPLETE_GAMMA_MPMATH_FLOAT64_CASES,
    )
    for case in positive_real_cases
        _, ar, _, zr, _, lr, _, ur, _, regime = case
        a, z = parse(Float64, ar), parse(Float64, zr)
        lower_big, upper_big = parse(BigFloat, lr), parse(BigFloat, ur)
        p_ref = lower_big / (lower_big + upper_big)
        q_ref = upper_big / (lower_big + upper_big)
        p, q = Gamma.gamma_inc(a, z)
        @test isfinite(p)
        @test isfinite(q)
        @test p + q == 1
        use_p = p_ref <= q_ref
        got, reference = use_p ? p : q, use_p ? p_ref : q_ref
        rounded = Float64(reference)
        if iszero(rounded) || abs(rounded) < floatmin(Float64)
            @test iszero(got)
        else
            @test abs(BigFloat(got) - reference) / abs(reference) <= 3e-12
        end
    end

    for (a, z) in (
        (1e-16, prevfloat(1.1)),
        (0.25, prevfloat(1.1)),
        (prevfloat(0.5), prevfloat(1.1)),
        (0.5, 1e-300),
        (prevfloat(1.0), 1e-300),
    )
        p, q = Gamma.gamma_inc(a, z)
        p_ref, q_ref = setprecision(512) do
            Gamma.gamma_inc(BigFloat(a), BigFloat(z))
        end
        use_p = abs(p_ref) <= abs(q_ref)
        got, reference = use_p ? p : q, use_p ? p_ref : q_ref
        @test _refmetric(BigFloat(got), reference) <= 384eps(Float64)
    end

    setprecision(256) do
        a, z = big"2.5", big"1e-100"
        p, q = Gamma.gamma_inc(a, z)
        sp, sq = SpecialFunctions.gamma_inc(a, z)
        @test p > 0
        @test sp == 0
        @test q == 1 - p
        @test sq == 1
    end
end
