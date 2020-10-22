"""
Data structures and common utility functions
"""
from collections import namedtuple

'''
Each of these should be a function 
of z == ln(M_N/T). The entries carrying a flavor index
_a also take an additional integer parameter a (0,1,2)
corresponding to (e,mu,tau). 
'''
IntegratedRates = namedtuple('IntegratedRates', [
    "GammaBar_nu_a", # Scalar
    "GammaBarTilde_nu_a", # 2x2
    "HamiltonianBar_N", # 2x2
    "GammaBar_N", # 2x2
    "GammaBarTilde_alpha_N", # 2x2
    "Seq" # 2x2
])

'''
These are the (interpolated) temperature dependent coefficients 
that are multiplied by the model dependent parts to get the 
integrated rates. Each entry is a function of T. 
'''
TDependentRateCoeffs = namedtuple('TDependentRateCoeffs', [
    "nugp",
    "nugm",
    "hnlgp",
    "hnlgm",
    "hnlhp",
    "hnlhm",
    "hnlh0",
    "hnldeq"
])

'''
Point in model parameter space (reals)
'''
ModelParams = namedtuple('ModelParams', [
    "M", "dM", "Imw", "Rew", "delta", "eta"
])