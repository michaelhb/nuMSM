"""
Data structures and common utility functions
"""
from collections import namedtuple

'''
Sphaleron freeze out temperature
'''
Tsph = 131.7

'''
Each of these should be a function 
of z == ln(M_N/T). 
'''
Rates = namedtuple('Rates', [
    "GB_nu_a", # (3)
    "GBt_nu_a", # (3,2,2)
    "GBt_N_a",  # (3,2,2)
    "HB_N", # (2,2)
    "GB_N", # (2,2)
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
Interpolated values of the entropy s and effective degrees of
freedom geff as a function of T.
'''
SMData = namedtuple('SMData', ['s', 'geff'])

'''
Point in model parameter space (reals)
'''
ModelParams = namedtuple('ModelParams', [
    "M", "dM", "Imw", "Rew", "delta", "eta"
])

'''
Averaged equations state vector legend. 
'''
AveragedStateVector = namedtuple("AveragedStateVector",
                                 ["n_delta_e", "n_delta_mu", "n_delta_tau",
                                  "rp_1", "rp_2", "rp_3", "rp_4",
                                  "rm_1", "rm_2", "rm_3", "rm_4"])