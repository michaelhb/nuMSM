import numpy as np
from common import AveragedStateVector

def get_T0(mp):
    '''
    :param mp: see common.ModelParams
    :return: Appropriate initial temp to begin integration
    '''
    M0 = 7.16e17
    Tosc = (0.304 * mp.M * mp.dM * M0) ** (1 / 3.0)
    return max(1000.0, 10 * Tosc)

def get_initial_state(T0, smdata):
    zeta3 = 1.20206

    #equilibrium number density for a relativistic fermion, T = T0
    neq = (3.*zeta3*(T0**3))/(4*(np.pi**2))

    #entropy at T0
    s = smdata.s(T0)

    return AveragedStateVector(
        n_delta_e=0,
        n_delta_mu=0,
        n_delta_tau=0,
        n_plus_11=-(neq/s),
        n_plus_22=-(neq/s),
        re_n_plus_12=0,
        im_n_plus_12=0,
        n_minus_11=0,
        n_minus_22=0,
        re_n_minus_12=0,
        im_n_minus_12=0
    )


