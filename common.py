import numpy as np

"""
Data structures and common utility functions
"""
from collections import namedtuple

'''
Sphaleron freeze out temperature
'''
Tsph = 131.7

''' riemann zeta(3) '''
zeta3 = 1.20206


def get_T0(mp):
    '''
    :param mp: see common.ModelParams
    :return: Appropriate initial temp to begin integration
    '''
    M0 = 7.16e17
    Tosc = (0.304 * mp.M * mp.dM * M0) ** (1 / 3.0)
    return max(1000.0, 10 * Tosc)

# Equilibrium distribution for SM neutrinos
def f_nu(kc):
    return 1.0/(np.exp(kc) + 1)

# Equilibrium distribution for HNLs
def f_N(T, M, kc):
    return 1.0/(np.exp(np.sqrt(M**2 + (T**2)*(kc**2))/T) + 1.0)

# def Tz(z, M):
#     return M*np.exp(-z)
#
# def zT(T, M):
#     return np.log(M/T)
#
def Tz(z, M):
    return Tsph*np.exp(-z)

def zT(T, M):
    return np.log(Tsph/T)

def MpStar(z, mp, smdata):
    Mp = 1.22e19  # Planck mass
    T = Tz(z, mp.M)
    geff = smdata.geff(T)
    return Mp*np.sqrt(45.0/(4*np.pi**3*geff))

'''
Factor arising from the change of variables T -> z
'''
def jacobian(z, mp, smdata):
    """
    :param z: integration coordinate z = ln(M/T)
    :param mp: see common.ModelParams
    :param smdata: see common.SMData
    :return: jacobian for transformation T -> Z
    """
    T = Tz(z, mp.M)
    return MpStar(z, mp, smdata) / (T ** 2)

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

"""Basis of Hermitian generators"""
tau = np.array([
    [[1,0],[0,0]],
    [[0,0],[0,1]],
    (1./np.sqrt(2))*np.array([[0,1],[1,0]]),
    (1./np.sqrt(2))*np.array([[0,1j],[-1j,0]])
])

"""Structure constants for commutators and anticommutators"""
def gen_Cijk():
    return 1j*np.imag(np.einsum('iab,jbc,kca->ijk',2*tau, tau, tau))

def gen_Aijk():
    return np.real(np.einsum('iab,jbc,kca->ijk',2*tau, tau, tau))

# Generate on module load
Cijk = gen_Cijk()
Aijk = gen_Aijk()

"""Convert commutators and anticommutators to matrix multiplication."""

def Ch(H):
    return np.einsum('inm,mn,ijk->kj',tau,H,Cijk)

def Ah(H):
    return np.einsum('inm,mn,ijk->kj',tau,H,Aijk)

def tr_h(H_a):
    """
    :param H_a: a (3,2,2) tensor; the first index being flavor, and the last two
        Hermitian or skew-Hermitian.
    :return: (3,4) matrix X_ai = Tr[tau_i.H_a].
    """
    return np.einsum('ijk,akj->ai', tau, H_a)

def trapezoidal_weights(points):
    if (len(points)) == 1:
        return [1.0]
    else:
        weights = [0.5 * (points[1] - points[0])]

        for i in range(1, len(points) - 1):
            weights.append(0.5 * (points[i + 1] - points[i - 1]))

        weights.append(0.5 * (points[-1] - points[-2]))
        return weights

