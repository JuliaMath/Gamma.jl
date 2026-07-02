module GammaMooncakeExt

using Gamma
using Mooncake
using Mooncake.ChainRulesCore
import Base: IEEEFloat
import Mooncake: @from_chainrules, DefaultCtx

#copied from https://github.com/chalk-lab/Mooncake.jl/blob/main/ext/MooncakeSpecialFunctionsExt.jl
@from_chainrules DefaultCtx Tuple{typeof(gamma),IEEEFloat}
@from_chainrules DefaultCtx Tuple{typeof(loggamma),IEEEFloat}

end #module