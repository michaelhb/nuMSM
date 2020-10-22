import numpy as np
from collections import namedtuple
from integrated_rates import IntegratedRates
from susceptibility import susceptibility_matrix

'''
Averaged equations state vector legend. 
'''
AveragedStateVector = namedtuple("AveragedStateVector",
    ["n_delta_e", "n_delta_mu", "n_delta_tau",
     "n_plus_11", "n_plus_22", "n_plus_12",
     "n_minus_11", "n_minus_22", "n_minus_12"])

