import common
from abc import ABC, abstractmethod
from collections import namedtuple
from functools import cache
from scipy.interpolate import interp1d
import numpy as np
from yukawasCI import FM
from os import path

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
Each of these should be a function 
of z == ln(M_N/T). 
'''
Rates = namedtuple('Rates', [
    "Gamma_nu_a", # (3)
    "GammaTilde_nu_a", # (3,2,2)
    "GammaTilde_N_a",  # (3,2,2)
    "Hamiltonian_N", # (2,2)
    "Hamiltonian_N_Int",
    "Gamma_N", # (2,2)
    "Seq", # (2,2)
    ], # (2,2)
    defaults=[None, None, None, None, None, None, None]
)

class Rates_Interface(ABC):

    @abstractmethod
    def get_averaged_rates(self):
        pass

    @abstractmethod
    def get_rates(self, kc):
        pass

class Rates_Fortran(Rates_Interface):

    def __init__(self, mp, H):
        self.mp = mp
        self.H = H
        self.rates_dir = path.abspath(path.join(path.dirname(__file__), 'test_data/'))

    @cache
    def get_averaged_rates(self):
        path_rates = path.join(self.rates_dir,
            "rates/Int_OrgH_MN{}E-1_kcAve.dat".format(int(self.mp.M * 10)))
        tc = self.get_rate_coefficients(path_rates)
        return self.rates_from_tc(tc)

    @cache
    def get_rates(self, kc):
        path_rates = path.join(self.rates_dir,
            "rates/Int_OrgH_MN{}E-1_kc{}E-1.dat".format(
                int(self.mp.M*10),int(kc * 10)))
        tc = self.get_rate_coefficients(path_rates)
        return self.rates_from_tc(tc)

    def get_rate_coefficients(path):

        T, nugp, nugm, hnlgp, hnlgm, hnlhp, hnlhm, hnlh0, hnldeq = np.flipud(np.loadtxt(path)).T

        return TDependentRateCoeffs(
            interp1d(T, nugp, fill_value="extrapolate"),
            interp1d(T, nugm, fill_value="extrapolate"),
            interp1d(T, hnlgp, fill_value="extrapolate"),
            interp1d(T, hnlgm, fill_value="extrapolate"),
            interp1d(T, hnlhp, fill_value="extrapolate"),
            interp1d(T, hnlhm, fill_value="extrapolate"),
            interp1d(T, hnlh0, fill_value="extrapolate"),
            interp1d(T, hnldeq, fill_value="extrapolate")
        )

    def rates_from_tc(self, tc):
        '''
        :param mp: ModelParams
        :param tc: TDependentRateCoeffs
        :param H: NH (H = 1) or IH (H = 2)
        :return: IntegratedRates
        '''

        # Get the Yukawas
        Fmb = FM(self.mp.M, self.mp.dM, self.mp.Imw,
                 self.mp.Rew, self.mp.delta, self.mp.eta, self.H)

        # Construct the Y-matrices (6.9 in 1808.10833)
        UNt = 1 / np.sqrt(2) * np.array([[-1j, 1j], [1, 1]])
        h = np.dot(Fmb, UNt)
        hc = np.conj(h)

        # a is the flavor index
        YNplus = np.array(
            [[[h[a, 1] * hc[a, 1], -h[a, 1] * hc[a, 0]],
              [np.conj(- h[a, 1] * hc[a, 0]), h[a, 0] * hc[a, 0]]]
             for a in range(3)])

        YNminus = np.array(
            [[[h[a, 0] * hc[a, 0], - h[a, 1] * hc[a, 0]],
              [np.conj(-h[a, 1] * hc[a, 0]), h[a, 1] * hc[a, 1]]]
             for a in range(3)])

        # Yukawa part of expression for Gamma_nu_alpha (6.3)
        hhc = np.array(
            [h[a, 0] * hc[a, 0] + h[a, 1] * hc[a, 1]
             for a in range(3)])

        def Tz(z):
            # return mp.M*np.exp(-z)
            return common.Tz(z, self.mp.M)

        # Construct the integrated rate functions
        def Gamma_nu_a(z):
            return np.array([hhc[a] * (tc.nugp(Tz(z)) + tc.nugm(Tz(z))) for a in range(3)])

        def GammaTilde_nu_a(z):
            res = np.array([-tc.hnlgp(Tz(z)) * YNplus[a].T + tc.hnlgm(Tz(z)) * YNminus[a].T for a in range(3)])
            return res

        def Hamiltonian_N(z):
            return self.mp.dM * np.array([[0, 1], [1, 0]]) * tc.hnlh0(Tz(z)) \
                   + tc.hnlhp(Tz(z)) * np.sum(YNplus, axis=0) \
                   + tc.hnlhm(Tz(z)) * np.sum(YNminus, axis=0)

        def Gamma_N(z):
            return tc.hnlgp(Tz(z)) * np.sum(YNplus, axis=0) + tc.hnlgm(Tz(z)) * np.sum(YNminus, axis=0)

        def GammaTilde_N_a(z):
            return np.array([-tc.nugp(Tz(z)) * YNplus[a] + tc.nugm(Tz(z)) * YNminus[a] for a in range(3)])

        def Seq(z):
            return tc.hnldeq(Tz(z)) * np.identity(2)

        def Hamiltonian_N_Int(z):
            return tc.hnlhp(Tz(z)) * np.sum(YNplus, axis=0) + tc.hnlhm(Tz(z)) * np.sum(YNminus, axis=0)

        return Rates(
            Gamma_nu_a,
            GammaTilde_nu_a,
            GammaTilde_N_a,
            Hamiltonian_N,
            Hamiltonian_N_Int,
            Gamma_N,
            Seq
        )