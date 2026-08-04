# Gamma.jl
[![Build Status](https://github.com/JuliaMath/Gamma.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaMath/Gamma.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaMath/Gamma.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaMath/Gamma.jl)

Simple and fast Gamma function.

This library provides a light-weight Julia native implementation of the [Gamma function](https://en.wikipedia.org/wiki/Gamma_function) and related utilities.

Supports `Integer`, `Float16`, `Float32`, and `Float64`, `Complex` and `BigFloat` arguments.

## Supported functions

```julia
gamma(x)           # gamma function Γ(x)
loggamma(x)        # logarithm of the gamma function
logabsgamma(x)     # (log(abs(Γ(x))), sign(Γ(x))) for real x
logfactorial(n)    # logarithm of n!
digamma(x)         # logarithmic derivative ψ(x) of Γ(x)
gamma(a, z)        # upper incomplete gamma Γ(a,z)
gamma_lower(a, z)  # lower incomplete gamma γ(a,z)
gamma_inc(a, z)    # normalized pair (P, Q)
expint(ν, z)       # generalized exponential integral Eν(z)
expintx(ν, z)      # exp(z)Eν(z)
```

## Contributing

Contributions are very welcome, as are feature requests, suggestions or general discussions.
Please open an issue for discussion on newer implementations, share papers, new features, or if you encounter any problems.
Our goal is to provide high quality Julia native implementations of Gamma functions that match or exceed the accuracy of the implementations provided by SpecialFunctions.jl.
Please let us know if you encounter any accuracy or performance issues.
