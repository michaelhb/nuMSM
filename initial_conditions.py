import numpy as np
from averaged_equations import tau
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

    n_plus0 = -np.identity(2)*neq/s
    r_plus0 = np.einsum('kij,ji->k',tau,n_plus0)

    return np.real(np.concatenate([[0,0,0],r_plus0,[0,0,0,0]]))


