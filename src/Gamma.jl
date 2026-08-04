module Gamma

    using LogExpFunctions: logmxp1

    include("gamma_implementation.jl")
    include("loggamma.jl")
    include("digamma.jl")
    include("expint.jl")
    include("incomplete_gamma.jl")
    include("precompile.jl")

    export gamma, gamma_lower, gamma_inc, expint, expintx,
           loggamma, logabsgamma, logfactorial, digamma

end
