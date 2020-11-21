import numpy as np
import sys
from collections import namedtuple
from integrated_rates import IntegratedRates

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
    #    return np.einsum('inm,mn,ijk->kj',tau,H,Cijk)
    return np.einsum('inm,mn,ijk->jk',tau,H,Cijk)

def Ah(H):
    return np.einsum('inm,mn,ijk->jk',tau,H,Aijk)
    # return np.einsum('inm,mn,ijk->kj',tau,H,Aijk)

def tr_h(H_a):
    """
    :param H_a: a (3,2,2) tensor; the first index being flavor, and the last two
        Hermitian or skew-Hermitian.
    :return: (3,4) matrix X_ai = Tr[tau_i.H_a].
    """
    return np.einsum('ijk,akj->ai',tau,H_a)

def gamma_omega(z, rt, susc):
    """
    :param z: integration coordinate z = ln(M/T)
    :param rt: see common.ModelParams
    :param smdata: see common.SMData
    :return: (3,3) matrix which mixes the n_delta in the upper left corner of the evolution matrix,
    proportional to T^2/6: -Re(GammaBar_nu_alpha).omega_alpha_beta (no sum over alpha)
    """
    GB_nu_a = rt.GB_nu_a(z)
    return (np.real(GB_nu_a).T*susc).T

def Yr(z, rt, susc):
    """
    :param z: integration coordinate z = ln(M/T)
    :param rt: see common.ModelParams
    :return: (4,3) matrix appearing in the (3,1) block of the evolution matrix
    """
    reGBt_nu_a = np.real(rt.GBt_N_a(z))
    return np.einsum('kij,aji,ab->kb',tau,reGBt_nu_a,susc)

def Yi(z, rt, susc):
    """
    :param z: integration coordinate z = ln(M/T)
    :param rt: see common.ModelParams
    :return: (4,3) matrix appearing in the (2,1) block of the evolution matrix
    """
    imGBt_nu_a = np.imag(rt.GBt_N_a(z))
    return np.einsum('kij,aji,ab->kb',tau,imGBt_nu_a,susc)

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

    # MpStar = Mp * np.sqrt(45.0 / (4*(np.pi**3)*smdata.geff(mp.M * np.exp(-z))))
    # return (MpStar * np.exp(-2 * z)) / (mp.M ** 2)

def inhomogeneous_part(z, rt):
    """
    :param rt: see common.IntegratedRates
    :return: inhomogeneous part of the ODE system
    """
    Seq = rt.Seq(z)
    seq = np.einsum('kij,ji->k',tau,Seq)
    return np.real(np.concatenate([[0,0,0],-seq,[0,0,0,0]]))

def coefficient_matrix(z, rt, mp, suscT):
    '''
    :param z: integration coordinate z = ln(M/T)
    :param rt: see common.IntegratedRates
    :param mp: see common.ModelParams
    :param suscT: Suscepibility matrix (2,2) (function of T)
    :param smdata: see common.SMData
    :return: z-dependent coefficient matrix of the ODE system
    '''

    T = mp.M * np.exp(-z)
    GB_nu_a, GBt_nu_a, GBt_N_a, HB_N, GB_N, Seq = [R(z) for R in rt]
    susc = suscT(T)

    # print("GB_nu_a", GB_nu_a*114421265882493.08)
    # print("GBt_nu_a", GBt_nu_a*114421265882493.08)
    # print("GBt_N_a", GBt_N_a*114421265882493.08)
    # print("HB_N", HB_N*114421265882493.08)
    # print("GB_N", GB_N*114421265882493.08)
    # print("Seq", Seq*114421265882493.08)

    b11 = -gamma_omega(z, rt, susc)*(T**2)/6.
    b12 = 2j*tr_h(np.imag(GBt_nu_a))
    b13 = -tr_h(np.real(GBt_nu_a))
    b21 = -(1j/2.)*Yi(z, rt, susc)*(T**2)/6.
    b22 = -1j*Ch(np.real(HB_N)) - (1./2.)*Ah(np.real(GB_N))
    b23 = (1./2.)*Ch(np.imag(HB_N)) - (1j/4.)*Ah(np.imag(GB_N))
    b31 = -Yr(z, rt, susc)*(T**2)/6.
    b32 = 2*Ch(np.imag(HB_N)) - 1j*Ah(np.imag(GB_N))
    b33 = -1j*Ch(np.real(HB_N)) - (1./2.)*Ah(np.real(GB_N))

    return np.real(np.block([
        [b11,b12,b13],
        [b21,b22,b23],
        [b31,b32,b33]
    ]))
