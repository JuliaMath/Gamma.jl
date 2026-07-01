using Gamma
using Test
import SpecialFunctions
using Random
Random.seed!(1993)

# how many iters to run randomized tests for
const NUM_RUNS = 10000


@testset "gamma type inference and return type" begin
    @testset "T: $T" for T in (Float16, Float32, Float64, BigFloat,
              Complex{Float16}, Complex{Float32}, Complex{Float64}, Complex{BigFloat},
              Int32 , Int64, BigInt)
        @inferred gamma(one(T))
        @test gamma(one(T)) isa float(T)
    end
end

@testset "gamma(::$T)" for (T, max, rtol) in ((Float16, 13, 1.0), (Float32, 43, 1.0), (Float64, 170, 7))
    @inferred gamma(one(T))
    for _ in 1:NUM_RUNS
        x = rand(T)*max
        @test isapprox(T(SpecialFunctions.gamma(widen(x))), gamma(x), rtol=rtol*eps(T))
        if isinteger(x) && x != 0
            @test_throws DomainError gamma(-x)
        else
            @test isapprox(T(SpecialFunctions.gamma(widen(-x))), gamma(-x), atol=nextfloat(T(0.),2), rtol=rtol*eps(T))
        end
    end
    @test isnan(gamma(T(NaN)))
    @test isinf(gamma(T(Inf)))
    @test_throws DomainError isinf(gamma(-T(Inf)))
end

x = [0, 1, 2, 3, 8, 15, 20, 30]
@test SpecialFunctions.gamma.(x) ≈ gamma.(x)
@inferred gamma(1)

# gamma_near_1 exact at expansion point (constant term = Γ(1) = 1)
@test Gamma.gamma_near_1(1.0) === 1.0
# nearby values: degree-3 Taylor gives O((x-1)^4) error,
# ≲ 1e-4 for |x-1| ≤ 0.1
for x in (0.90, 0.95, 1.0, 1.05, 1.10)
    @test Gamma.gamma_near_1(x) ≈ gamma(x) atol=1e-3
end

include("test_loggamma.jl")
include("test_digamma.jl")
