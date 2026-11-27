module GammaEnzymeExt

using Gamma
using Enzyme
using Enzyme.EnzymeRules

EnzymeRules.@easy_rule(
    Gamma.gamma(x::AbstractFloat),
    @setup(),
    (Gamma.digamma(x) * Gamma.gamma(x),)
)

EnzymeRules.@easy_rule(
    Gamma.loggamma(x::AbstractFloat),
    @setup(),
    (Gamma.digamma(x),)
)

EnzymeRules.@easy_rule(
    Gamma.logfactorial(x::AbstractFloat),
    @setup(),
    (Gamma.digamma(x + 1),)
)

end #module