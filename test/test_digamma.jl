using Gamma: digamma
const γ = Base.MathConstants.eulergamma
include("digamma_mpmath_cases.jl")

function _digamma_real_reference(x::T) where T<:AbstractFloat
    return T(SpecialFunctions.digamma(Float64(x)))
end

function _digamma_region_values(::Type{T}, maxval, seed) where T<:AbstractFloat
    rng = MersenneTwister(seed)
    values = T[]
    append!(values, T[0.0, -0.0, 0.5, 1, 2, 8, -0.5, -1.1, -2.5])

    tiny_lo = T === Float16 ? -3 : -12
    for _ in 1:40
        push!(values, T(10.0 ^ (tiny_lo * rand(rng))))
        push!(values, T(rand(rng)) * T(8) + eps(T))
        push!(values, T(rand(rng)) * T(maxval) + one(T))
    end

    for _ in 1:80
        x = T(rand(rng)) * T(maxval)
        isinteger(x) || push!(values, -x)
    end

    for n in 0:30
        δ = T((rand(rng) < 0.5 ? -1 : 1) * (1e-3 + 0.2rand(rng)))
        iszero(δ) || push!(values, -T(n) + δ)
    end

    return values
end

function _digamma_complex_region_values(::Type{T}, maxval, seed) where T<:AbstractFloat
    rng = MersenneTwister(seed)
    values = Complex{T}[]
    append!(values, Complex{T}[
        complex(T(0), T(1//8)),
        complex(T(1//2), T(1//8)),
        complex(T(1), T(1)),
        complex(T(7), T(3)),
        complex(T(-1//2), T(1//8)),
        complex(T(-2), T(1//4)),
        complex(T(maxval), T(2)),
    ])

    for _ in 1:50
        push!(values, complex((T(2) * T(rand(rng)) - one(T)) * T(3),
                              (T(2) * T(rand(rng)) - one(T)) * T(3)))
        push!(values, complex((T(2) * T(rand(rng)) - one(T)) * T(maxval),
                              (T(2) * T(rand(rng)) - one(T)) * T(maxval / 4)))
    end

    for n in 0:20
        δx = T((rand(rng) < 0.5 ? -1 : 1) * (1e-3 + 0.1rand(rng)))
        δy = T((rand(rng) < 0.5 ? -1 : 1) * (1e-3 + 0.1rand(rng)))
        push!(values, complex(-T(n) + δx, δy))
    end

    filter!(z -> !iszero(imag(z)), values)
    return values
end

function _finite_sf_digamma(z::Complex)
    ref = SpecialFunctions.digamma(ComplexF64(z))
    return all(isfinite, reim(ref)) ? ref : nothing
end

@testset "digamma type inference and return type" begin
    @testset "T: $T" for T in (Float16, Float32, Float64,
              Complex{Float16}, Complex{Float32}, Complex{Float64},
              BigFloat, Complex{BigFloat}, Int32, Int64)
        @inferred digamma(one(T))
        @test digamma(one(T)) isa float(T)
    end
end

@testset "digamma SpecialFunctions cases" begin
    for T in (Float32, Float64)
        @test digamma(T(9)) ≈ T(2.140641477955609996536345)
        @test digamma(T(2.5)) ≈ T(0.7031566406452431872257)
        @test digamma(T(0.1)) ≈ T(-10.42375494041107679516822)
        @test digamma(T(7e-4)) ≈ T(-1429.147493371120205005198)
        @test digamma(T(7e-5)) ≈ T(-14286.29138623969227538398)
        @test digamma(T(7e-6)) ≈ T(-142857.7200612932791081972)
        @test digamma(T(2e-6)) ≈ T(-500000.5772123750382073831)
        @test digamma(T(1e-6)) ≈ T(-1000000.577214019968668068)
        @test digamma(T(7e-7)) ≈ T(-1428572.005785942019703646)
        @test digamma(T(-0.5)) ≈ T(0.03648997397857652055902367)
        @test digamma(T(-1.1)) ≈ T(10.15416395914385769902271)
        @test digamma(T(0)) == T(-Inf)
        @test digamma(T(-1)) == T(-Inf)

        @test digamma(T(1/2)) ≈ T(-γ - log(4))
        @test digamma(T(1)) ≈ T(-γ)
        @test digamma(T(2)) ≈ T(1 - γ)
        @test digamma(T(3)) ≈ T(3/2 - γ)
        @test digamma(T(4)) ≈ T(11/6 - γ)
        @test digamma(T(5)) ≈ T(25/12 - γ)
        @test digamma(T(10)) ≈ T(7129/2520 - γ)
    end
end

@testset "digamma complex cases" begin
    for x in -10.2:0.3456:50
        @test digamma(x + 0im) ≈ digamma(x)
    end
    @test digamma(7 + 0im) ≈ 1.872784335098467139393487909917597568957840664060076401194232
    @test digamma(7im) ≈ 1.94761433458434866917623737015561385331974500663251349960124 +
                         1.642224898223468048051567761191050945700191089100087841536im
    @test digamma(-3.2 + 0.1im) ≈ 4.65022505497781398615943030397508454861261537905047116427511 +
                                  2.32676364843128349629415011622322040021960602904363963042380im
    @test digamma(ComplexF32(7, 0)) ≈ ComplexF32(digamma(7 + 0im))
    @test digamma(ComplexF32(0, 7)) ≈ ComplexF32(digamma(7im))
end

@testset "digamma seeded accuracy" begin
    for (T, maxval, scale) in ((Float16, 13, 2.0), (Float32, 43, 4.0), (Float64, 170, 4.0))
        @testset "real $T" begin
            for x in _digamma_region_values(T, maxval, 5000 + sizeof(T))
                isinteger(x) && x < 0 && continue
                ref = _digamma_real_reference(x)
                @test isapprox(digamma(x), ref; atol=scale*eps(T), rtol=scale*eps(T))
            end
        end
    end

    for (T, maxval, scale) in ((Float16, 13, 4.0), (Float32, 43, 64.0), (Float64, 170, 64.0))
        @testset "complex $T" begin
            for z in _digamma_complex_region_values(T, maxval, 6000 + sizeof(T))
                ref64 = _finite_sf_digamma(z)
                ref64 === nothing && continue
                ref = Complex{T}(ref64)
                @test isapprox(digamma(z), ref; atol=scale*eps(T), rtol=scale*eps(T))
            end
        end
    end

    setprecision(256) do
        @testset "real BigFloat" begin
            for x in BigFloat.(_digamma_region_values(Float64, 170, 7000))
                isinteger(x) && x < 0 && continue
                @test isapprox(digamma(x), SpecialFunctions.digamma(x); rtol=8eps(BigFloat))
            end
        end

        @testset "Complex{BigFloat} mpmath" begin
            tol = big"1e-58"
            for (xr, xi, yr, yi) in DIGAMMA_COMPLEX_BIGFLOAT_MPMATH_CASES
                z = Complex{BigFloat}(BigFloat(xr), BigFloat(xi))
                ref = Complex{BigFloat}(BigFloat(yr), BigFloat(yi))
                @test isapprox(digamma(z), ref; atol=tol, rtol=tol)
            end
        end
    end
end

@testset "BigFloat digamma" begin
    for p in (128, 256, 512)
        setprecision(p) do
            mpfr_tol = 10eps(BigFloat)
            for x in BigFloat.(["0.1", "0.5", "1.0", "2.5", "9.0", "50.0", "-0.5", "-1.1"])
                @test isapprox(digamma(x), SpecialFunctions.digamma(x); rtol=mpfr_tol)
            end
            @test digamma(big"0.0") == big"-Inf"
            @test digamma(BigFloat(-0.0)) == big"Inf"
            @test isnan(digamma(big"-1.0"))

            for x in (BigFloat(0.0), BigFloat(-0.0)), y in (BigFloat(0.0), BigFloat(-0.0))
                expected = Complex{BigFloat}(digamma(x), BigFloat(0.0))
                @test isequal(digamma(Complex{BigFloat}(x, y)), expected)
            end

            quoted_tol = max(big"1e-57", mpfr_tol)
            @test isapprox(digamma(Complex{BigFloat}(big"7.0", big"0.0")),
                big"1.872784335098467139393487909917597568957840664060076401194232";
                rtol=quoted_tol)
            @test isapprox(digamma(Complex{BigFloat}(big"0.0", big"7.0")),
                big"1.94761433458434866917623737015561385331974500663251349960124" +
                big"1.642224898223468048051567761191050945700191089100087841536" * im;
                rtol=quoted_tol)
            @test isapprox(digamma(Complex{BigFloat}(big"-3.2", big"0.1")),
                big"4.65022505497781398615943030397508454861261537905047116427511" +
                big"2.32676364843128349629415011622322040021960602904363963042380" * im;
                rtol=quoted_tol)

            z = Complex{BigFloat}(big"1.4", big"3.7")
            @test ComplexF64(digamma(z)) ≈ digamma(ComplexF64(z))

            identity_tol = 10eps(BigFloat)
            for z in (
                Complex{BigFloat}(big"1.3", big"0.7"),
                Complex{BigFloat}(big"7.0", big"3.0"),
                Complex{BigFloat}(big"-2.4", big"0.2"),
                Complex{BigFloat}(big"0.2", big"5.0"),
                Complex{BigFloat}(big"-6.7", big"-1.3"),
            )
                @test isapprox(digamma(z + 1), digamma(z) + inv(z); rtol=identity_tol)
                @test isapprox(digamma(conj(z)), conj(digamma(z)); rtol=identity_tol)
                @test isapprox(digamma(1 - z) - digamma(z), big(π) * cospi(z) / sinpi(z);
                               rtol=identity_tol)
                @test ComplexF64(digamma(z)) ≈ digamma(ComplexF64(z))
            end

            derivative_tol = 2cbrt(eps(BigFloat)^2)
            for z in (
                Complex{BigFloat}(big"1.3", big"0.7"),
                Complex{BigFloat}(big"3.5", big"-1.2"),
                Complex{BigFloat}(big"0.8", big"4.0"),
            )
                h = exp2(-precision(BigFloat) ÷ 3)
                dloggamma = (loggamma(z + h) - loggamma(z - h)) / (2h)
                @test isapprox(digamma(z), dloggamma; rtol=derivative_tol)
            end
        end
    end

    @testset "nonfinite Complex{BigFloat}" begin
        nan = BigFloat(NaN)
        inf = BigFloat(Inf)
        for z in (
            Complex{BigFloat}(nan, 0),
            Complex{BigFloat}(0, nan),
            Complex{BigFloat}(nan, inf),
        )
            @test all(isnan, reim(digamma(z)))
        end

        for z in (
            Complex{BigFloat}(inf, 1),
            Complex{BigFloat}(-inf, 1),
            Complex{BigFloat}(1, inf),
            Complex{BigFloat}(1, -inf),
        )
            @test isequal(digamma(z), log(z))
        end
    end
end

@testset "high precision digamma" begin
    setprecision(2048) do
        x = big"1.25"
        @test isapprox(digamma(x), SpecialFunctions.digamma(x); rtol=10eps(BigFloat))

        z = Complex{BigFloat}(big"1.3", big"0.7")
        @test isapprox(digamma(z + 1), digamma(z) + inv(z); rtol=10eps(BigFloat))
    end
end

@testset "scaled Bernoulli precision retry" begin
    expected = big"4439271675277203871382284388240112245696102400000000"
    @test Gamma._scaled_bernoulli(17, 0) == expected
end

@testset "digamma against SpecialFunctions" begin
    for (T, max, scale) in ((Float16, 13, 2.0), (Float32, 43, 64.0), (Float64, 170, 8.0))
        for _ in 1:NUM_RUNS
            x = rand(T) * max
            @test isapprox(digamma(x), T(SpecialFunctions.digamma(Float64(x)));
                           atol=scale*eps(T), rtol=scale*eps(T))
            xn = -x
            isinteger(xn) && continue
            @test isapprox(digamma(xn), T(SpecialFunctions.digamma(Float64(xn)));
                           atol=scale*eps(T), rtol=scale*eps(T))
        end
    end

    for (T, max, rtol) in ((Float32, 43, 64.0), (Float64, 170, 64.0))
        for _ in 1:NUM_RUNS
            z = Complex{T}(rand(T) * 2T(max) - T(max), rand(T) * 2T(max) - T(max))
            abs(imag(z)) < sqrt(eps(T)) && continue
            ref = Complex{T}(SpecialFunctions.digamma(ComplexF64(z)))
            all(isfinite, reim(ref)) || continue
            @test isapprox(digamma(z), ref; atol=rtol*eps(T), rtol=rtol*eps(T))
        end
    end
end
