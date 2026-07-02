module GammaForwardDiffExt
    using Gamma
    using ForwardDiff

    function Gamma.gamma(xx::ForwardDiff.Dual{T,V}) where {T,V}
        x = ForwardDiff.value(xx)
        dx = Gamma.digamma(x) * Gamma.gamma(x)
        return Dual{T}(x,dx*partials(d)) 
    end

    function Gamma.loggamma(xx::ForwardDiff.Dual{T,V}) where {T,V}
        x = ForwardDiff.value(xx)
        dx = Gamma.digamma(x)
        return Dual{T}(x,dx*partials(d)) 
    end
end #module
