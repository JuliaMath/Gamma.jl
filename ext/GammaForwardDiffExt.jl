module GammaForwardDiffExt

using Gamma
using ForwardDiff

function Gamma.gamma(xx::ForwardDiff.Dual{T,V}) where {T,V}
    x = ForwardDiff.value(xx)
    fx = Gamma.gamma(x)
    dx = Gamma.digamma(x) * fx
    return Dual{T}(fx,dx*partials(d)) 
end

function Gamma.loggamma(xx::ForwardDiff.Dual{T,V}) where {T,V}
    x = ForwardDiff.value(xx)
    fx = Gamma.loggamma(x)
    dx = Gamma.digamma(x)
    return Dual{T}(fx,dx*partials(d)) 
end

function Gamma.logfactorial(xx::ForwardDiff.Dual{T,V}) where {T,V}
    x = ForwardDiff.value(xx)
    fx = Gamma.logfactorial(x)
    dx = Gamma.digamma(x + 1)
    return Dual{T}(fx,dx*partials(d)) 
end

end #module
