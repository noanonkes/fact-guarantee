class SMLAOptimizer:
	pass	

from .cma import CMAESOptimizer

OPTIMIZERS = { opt.cli_key():opt for opt in SMLAOptimizer.__subclasses__() }