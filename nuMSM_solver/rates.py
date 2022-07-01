from nuMSM_solver.common import *
from abc import ABC, abstractmethod
from collections import namedtuple
from functools import lru_cache
from scipy.interpolate import interp1d
import numpy as np
from nuMSM_solver.yukawasCI import FM
from os import path

from numba import vectorize

from leptotools.momentumDep import interpHFast, interpFast
from leptotools.scantools import leptogenesisScanSetup


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


def rates_from_tc(tc, mp, H):
    '''
    :param mp: ModelParams
    :param tc: TDependentRateCoeffs
    :param H: NH (H = 1) or IH (H = 2)
    :return: IntegratedRates
    '''

    # Get the Yukawas
    Fmb = FM(mp.M, mp.dM, mp.Imw,
             mp.Rew, mp.delta, mp.eta, H)

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

    # Construct the integrated rate functions
    #@njit
    def Gamma_nu_a(z):
        T = Tz(z, mp.M)
        return np.array([hhc[a] * (T*tc.nugp(T) + T*tc.nugm(T)) for a in range(3)]).T

    @njit
    def GammaTilde_nu_a(z):
        T = Tz(z, mp.M)
        res = np.array([-T*tc.hnlgp(T) * YNplus[a].T + T*tc.hnlgm(T) * YNminus[a].T for a in range(3)])
        return res

    @njit
    def Hamiltonian_N(z):
        T = Tz(z, mp.M)
        return mp.dM * np.array([[0, 1], [1, 0]]) * T*tc.hnlh0(T) \
               + T*tc.hnlhp(T) * np.sum(YNplus, axis=0) \
               + T*tc.hnlhm(T) * np.sum(YNminus, axis=0)

    @njit
    def Gamma_N(z):
        T = Tz(z, mp.M)
        return T*tc.hnlgp(T) * np.sum(YNplus, axis=0) + T*tc.hnlgm(T) * np.sum(YNminus, axis=0)

    @njit
    def GammaTilde_N_a(z):
        T = Tz(z, mp.M)
        return np.array([-T*tc.nugp(T) * YNplus[a] + T*tc.nugm(T) * YNminus[a] for a in range(3)])

    @njit
    def Seq(z):
        T = Tz(z, mp.M)
        return T*tc.hnldeq(T) * np.identity(2)

    @njit
    def Hamiltonian_N_Int(z):
        T = Tz(z, mp.M)
        return T*tc.hnlhp(T) * np.sum(YNplus, axis=0) + tc.hnlhm(Tz(z, mp.M)) * np.sum(YNminus, axis=0)

    return Rates(
        Gamma_nu_a,
        GammaTilde_nu_a,
        GammaTilde_N_a,
        Hamiltonian_N,
        Hamiltonian_N_Int,
        Gamma_N,
        Seq
    )

class Rates_Interface(ABC):

    @abstractmethod
    def get_averaged_rates(self):
        pass

    @abstractmethod
    def get_rates(self, kc_list):
        pass

class Rates_Fortran(Rates_Interface):

    def __init__(self, mp, H):
        self.mp = mp
        self.H = H
        self.rates_dir = path.abspath(path.join(path.dirname(__file__), '../test_data/'))

    @lru_cache(maxsize=None)
    def get_averaged_rates(self):
        path_rates = path.join(self.rates_dir,
            "rates/Int_OrgH_MN{}E-1_kcAve.dat".format(int(self.mp.M * 10)))
        tc = self.get_rate_coefficients(path_rates)
        return rates_from_tc(tc, self.mp, self.H)

    @lru_cache(maxsize=None)
    def get_rates(self, kc_list):
        rates = []
        for kc in kc_list:
            path_rates = path.join(self.rates_dir,
                "rates/Int_OrgH_MN{}E-1_kc{}E-1.dat".format(
                    int(self.mp.M*10),int(kc * 10)))
            tc = self.get_rate_coefficients(path_rates)
            rates.append(rates_from_tc(tc, self.mp, self.H))

    def get_rate_coefficients(self, path):

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


class Rates_Jurai(Rates_Interface):

    def __init__(self, mp, H, kc_list, tot=False):
        self.mp = mp
        self.H = H
        self.kc_list = kc_list

        self.gP, self.gM = interpFast(mp.M, kc_list, tot=tot)
        self.hP, self.hM = interpHFast(mp.M, kc_list)

        # # We will need caching to retain the efficiency boost from 2D interpolation
        # self.gP_all = lru_cache(maxsize=None)(lambda T: gP_(zT(T, mp.M)))
        # self.gM_all = lru_cache(maxsize=None)(lambda T: gM_(zT(T, mp.M)))
        # self.hP_all = lru_cache(maxsize=None)(lambda T: hP_(zT(T, mp.M)))
        # self.hM_all = lru_cache(maxsize=None)(lambda T: hM_(zT(T, mp.M)))

    @lru_cache(maxsize=None)
    def get_averaged_rates(self):
        ls = leptogenesisScanSetup(self.mp.M)
        if self.H == 1:
            ls.set_ordering("NO")
        else:
            ls.set_ordering("IO")

        ls.set_dM(self.mp.dM)
        ls.set_romega(self.mp.Rew)
        ls.set_iomega(self.mp.Imw)
        ls.set_delta(self.mp.delta)
        ls.set_eta(self.mp.eta)
        ls.set_xi(1)
        ls.set_CI()

        nugp = lambda T: ls.gammanuP(zT(T, self.mp.M)) * T
        nugm = lambda T: ls.gammanuM(zT(T, self.mp.M)) * T
        # nugm = lambda T: 0
        hnlgp = lambda T: ls.gammaNP(zT(T, self.mp.M)) * T
        hnlgm = lambda T: ls.gammaNM(zT(T, self.mp.M)) * T
        # hnlgm = lambda T: 0
        hnlhp = lambda T: ls.hNP(zT(T, self.mp.M)) * T
        hnlhm = lambda T: ls.hNM(zT(T, self.mp.M)) * T
        hnlh0 = lambda T: ls.hNM0(zT(T, self.mp.M)) * T
        hnldeq = lambda T: 0 #TODO!

        tc = TDependentRateCoeffs(
            nugp=nugp, nugm=nugm, hnlgp=hnlgp, hnlgm=hnlgm,
            hnlhp=hnlhp,hnlhm=hnlhm,hnlh0=hnlh0,hnldeq=hnldeq
        )

        return rates_from_tc(tc, self.mp, self.H)

    def get_rates(self, kc_list):
        rates = []

        # gP = lambda T: self.gP(T) * T
        # gM = lambda T: self.gM(T) * T
        # hP = lambda T: self.hP(T) * T
        # hM = lambda T: self.hM(T) * T
        h0 = lambda T: [-self.mp.M/np.sqrt((kc*T)**2 + self.mp.M**2) for kc in kc_list]
        hnldeq = lambda T: 0 # Don't call this!

        tc = TDependentRateCoeffs(
            nugp=self.gP, nugm=self.gM, hnlgp=self.gP, hnlgm=self.gM, hnlhp=self.hP, hnlhm=self.hM,
            hnlh0=h0, hnldeq=hnldeq
        )

        return rates_from_tc(tc, self.mp, self.H)