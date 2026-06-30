module Gamma

    include("gamma_implementation.jl")
    include("loggamma.jl")
    include("digamma.jl")
    include("precompile.jl")

    export gamma, loggamma, logabsgamma, logfactorial, digamma

end
