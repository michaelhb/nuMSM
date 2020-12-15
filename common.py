import numpy as np

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
Factor arising from the change of variables T -> z
'''
def jacobian(z, mp, smdata):
    """
    :param z: integration coordinate z = ln(M/T)
    :param mp: see common.ModelParams
    :param smdata: see common.SMData
    :return: jacobian for transformation T -> Z
    """
    Mp = 1.22e19  # Planck mass
    T = mp.M*np.exp(-z)
    geff = smdata.geff(T)
    MpStar = Mp*np.sqrt(45.0/(4*np.pi**3*geff))
    return MpStar/(T**2)

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
    return np.einsum('ijk,akj->ai',tau,H_a)