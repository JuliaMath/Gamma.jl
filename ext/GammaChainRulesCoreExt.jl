module GammaChainRulesCoreExt
    using Gamma
    using ChainRulesCore

    ChainRulesCore.@scalar_rule Gamma.gamma(x)      Gamma.digamma(x) * Gamma.gamma(x)
    ChainRulesCore.@scalar_rule Gamma.loggamma(x)   Gamma.digamma(x)

end #module
