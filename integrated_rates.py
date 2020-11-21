from collections import namedtuple
import sys
import numpy as np
from yukawasCI import FM
from common import IntegratedRates

"""
Given the temp dependent rate coefficients and model parameters,
construct the momentum-averaged rate matrices (as functions of z).
"""

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
        [[[h[a,1]*hc[a,1], -h[a,1]*hc[a,0]],
        [np.conj(- h[a,1]*hc[a,0]),h[a,0]*hc[a,0]]]
            for a in range(3)])

    YNminus = np.array(
        [[[h[a,0]*hc[a,0],- h[a,1]*hc[a,0]],
        [np.conj(-h[a,1]*hc[a,0]),h[a,1]*hc[a,1]]]
            for a in range(3)])

    # Yukawa part of expression for Gamma_nu_alpha (6.3)
    hhc = np.array(
        [h[a,0]*hc[a,0] + h[a,1]*hc[a, 1]
         for a in range(3)])

    def Tz(z):
        return mp.M*np.exp(-z)

    # Construct the integrated rate functions
    def GammaBar_nu_a(z):
        return np.array([hhc[a]*(tc.nugp(Tz(z)) + tc.nugm(Tz(z))) for a in range(3)])

    def GammaBarTilde_nu_a(z):
        return np.array([-tc.hnlgp(Tz(z))*YNplus[a].T + tc.hnlgm(Tz(z))*YNminus[a].T for a in range(3)])

    def HamiltonianBar_N(z):
        return mp.dM*np.array([[0,1],[1,0]])*tc.hnlh0(Tz(z)) \
            + tc.hnlhp(Tz(z))*np.sum(YNplus, axis=0) \
            + tc.hnlhm(Tz(z))*np.sum(YNminus, axis=0)

    def GammaBar_N(z):
        return tc.hnlgp(Tz(z))*np.sum(YNplus, axis=0) + tc.hnlgm(Tz(z))*np.sum(YNminus, axis=0)

    def GammaBarTilde_N_a(z):
        return np.array([-tc.nugp(Tz(z))*YNplus[a] + tc.nugm(Tz(z))*YNminus[a] for a in range(3)])

    def Seq(z):
        return tc.hnldeq(Tz(z))*np.identity(2)

    return IntegratedRates(
        GammaBar_nu_a,
        GammaBarTilde_nu_a,
        GammaBarTilde_N_a,
        HamiltonianBar_N,
        GammaBar_N,
        Seq
    )