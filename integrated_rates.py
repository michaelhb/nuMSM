from collections import namedtuple
import numpy as np
from yukawasCI import FM

'''
Each of these should be a (2*2) matrix valued function 
of z == ln(M_N/T). The entries carrying a flavor index
_a also take an additional integer parameter a (0,1,2)
corresponding to (e,mu,tau). 
'''
IntegratedRates = namedtuple('IntegratedRates', [
    "GammaBar_nu_a",
    "GammaBarTilde_nu_a",
    "HamiltonianBar_N",
    "GammaBar_N",
    "GammaBarTilde_alpha_N",
    "Seq"
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

def get_integrated_rates(mp, tc):
    '''
    :param mp: ModelParams
    :param tc: TDependentRateCoeffs
    :return: IntegratedRates
    '''

    # Get the Yukawas
    Fmb = FM(mp.M, mp.dM, mp.Imw, mp.Rew, mp.delta, mp.eta)

    # Construct the Y-matrices (6.9 in 1808.10833)
    UNt = 1 / np.sqrt(2) * np.array([[-1j, 1j], [1, 1]])
    h = np.dot(Fmb, UNt)
    hc = np.conj(h)

    # a is the flavor index
    YNplus = np.array(
        [[[h[a,1]*hc[a,1],-h[a,1]*hc[a,0]],
        [np.conj(- h[a,1]*hc[a,0]),h[a,0]*hc[a,0]]]
            for a in range(3)])

    YNminus = np.array(
        [[[h[a,0]*hc[a,0],- h[a,1]*hc[a,0]],
        [np.conj(-h[a,1]*hc[a,0]),h[a,1]*hc[a,1]]]
            for a in range(3)])

    # Yukawa part of expression for Gamma_nu_alpha (6.3)
    hhc = np.array(
        [h[a, 0] * hc[a, 0] + h[a, 1] * hc[a, 1]
         for a in range(3)])

    def T(z):
        return mp.M*np.exp(-z)

    # Construct the integrated rate functions
    def Gammabar_nu_a(z, a):
        return hhc(a)*(tc.nugp(T(z)) + tc.nugm(T(z)))

    def GammaBarTilde_nu_a(z, a):
        return -tc.hnlgp(T(z))*YNplus(a) + tc.hnlgm(T(z))*YNminus(a)

    def HamiltonianBar_N(z):
        pass

    def GammaBar_N(z):
        return tc.hnlgp*np.sum(YNplus, axis=1) + tc.hnlgm*np.sum(YNminus, axis=1)

    def GammaBarTilde_alpha_N(z):
        pass

    def Seq(z):
        pass

    return IntegratedRates(
        Gammabar_nu_a,
        GammaBarTilde_nu_a,
        HamiltonianBar_N,
        GammaBar_N,
        GammaBarTilde_alpha_N,
        Seq
    )