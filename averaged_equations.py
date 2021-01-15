import numpy as np
import sys

from common import *

def get_initial_state_avg(T0, smdata):

    #equilibrium number density for a relativistic fermion, T = T0
    neq = (3.*zeta3*(T0**3))/(4*(np.pi**2))

    #entropy at T0
    s = smdata.s(T0)

    n_plus0 = -np.identity(2)*neq/s
    r_plus0 = np.einsum('kij,ji->k',tau,n_plus0)

    return np.real(np.concatenate([[0,0,0],r_plus0,[0,0,0,0]]))

'''
Averaged equations state vector legend (just for documentation purposes...)
'''
AveragedStateVector = namedtuple("AveragedStateVector",
                                 ["n_delta_e", "n_delta_mu", "n_delta_tau",
                                  "rp_1", "rp_2", "rp_3", "rp_4",
                                  "rm_1", "rm_2", "rm_3", "rm_4"])

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

def inhomogeneous_part(z, rt):
    """
    :param rt: see common.IntegratedRates
    :return: inhomogeneous part of the ODE system
    """
    Seq = rt.Seq(z)
    seq = np.einsum('kij,ji->k', tau, Seq)
    return np.real(np.concatenate([[0, 0, 0], -seq, [0, 0, 0, 0]]))

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
