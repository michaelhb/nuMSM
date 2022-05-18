import numpy as np

"""
Data structures and common utility functions
"""
from collections import namedtuple

"""
Pass no args to solvers by default
"""
ode_par_defaults = {}

'''
Sphaleron freeze out temperature
'''
Tsph = 131.7

''' riemann zeta(3) '''
zeta3 = 1.20206

''' active neutrino masses, per http://www.nu-fit.org/?q=node/228 '''
msol, matm = np.sqrt([7.40e-5*1e-18, 2.494e-3*1e-18])
msolIH, matmIH = np.sqrt([7.40e-5*1e-18, 2.465e-3*1e-18])
m1NH, m2NH, m3NH = 0.0, msol, matm  #NH
m1IH, m2IH, m3IH = np.sqrt(matmIH**2-msolIH**2), matmIH, 0.0  #IH

''' Imw for a given common mass and total mixing. Returns a positive solution. '''
def imw(U2, M, H):
    if H == 1:
        mn = m2NH + m3NH
    elif H == 2:
        mn = m1IH + m2IH

    return 0.5*np.log((M*U2)/mn + np.sqrt(((M*U2)/mn)**2 - 1))


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

def f_Ndot(kc, T, mp, smdata):
    E_k = np.sqrt(mp.M**2 + (T**2)*(kc**2))
    Mpl = MpStar(zT(T, mp.M), mp, smdata)
    # return -(T/Mpl)*(E_k*np.exp(E_k/T))/((np.exp(E_k/T) + 1)**2)
    return -1*((T*E_k)/Mpl)*np.exp(E_k/T)/((1 + np.exp(E_k/T))**2)

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
    return np.einsum('ilm,jmn,knl', tau, tau, tau) - np.einsum('ilm,kmn,jnl', tau, tau, tau)
    # return 1j*np.imag(np.einsum('iab,jbc,kca->ijk',2*tau, tau, tau))

def gen_Aijk():
    return np.einsum('imn,jnl,klm', tau, tau, tau) + np.einsum('jmn,inl,klm', tau, tau, tau)
    # return np.real(np.einsum('iab,jbc,kca->ijk',2*tau, tau, tau))

# Generate on module load
Cijk = gen_Cijk()
Aijk = gen_Aijk()

"""Convert commutators and anticommutators to matrix multiplication."""

def Ch(H):
    # return np.einsum('inm,mn,ijk->kj',tau,H,Cijk)
    return np.einsum('inm,mn,jik->kj', tau, H, Cijk)

def Ah(H):
    # return np.einsum('inm,mn,ijk->kj',tau,H,Aijk)
    return np.einsum('inm,mn,jik->kj', tau, H, Aijk)

def tr_h(H_a):
    """
    :param H_a: a (3,2,2) tensor; the first index being flavor, and the last two
        Hermitian or skew-Hermitian.
    :return: (3,4) matrix X_ai = Tr[tau_i.H_a].
    """
    return np.einsum('ijk,akj->ai', tau, H_a)

