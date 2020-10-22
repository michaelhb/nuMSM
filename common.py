"""
Data structures and common utility functions
"""
from collections import namedtuple

'''
Each of these should be a function 
of z == ln(M_N/T). 
'''
IntegratedRates = namedtuple('IntegratedRates', [
    "GammaBar_nu_a", # (3)
    "GammaBarTilde_nu_a", # (3,2,2)
    "GammaBarTilde_N_a",  # (3,2,2)
    "HamiltonianBar_N", # (2,2)
    "GammaBar_N", # (2,2)
    "Seq" # (2,2)
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