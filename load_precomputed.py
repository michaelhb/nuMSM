import numpy as np
from scipy.interpolate import interp1d
from common import TDependentRateCoeffs, SMData
from fast_interp import interp1d as fast_interp1d
from numba import njit
import sys

def fast_interpolant(T, data):
    # T should be ascending, log-uniform

    # fast_interp1d requires evenly spaced data, so we work with log(T)
    a = np.log(T[0])
    b = np.log(T[-1])
    h = np.log(T[1]) - np.log(T[0])

    interp = fast_interp1d(a, b, h, np.array(data))

    # Return functor that undoes log
    return lambda T : interp(np.log(T))

def get_rate_coefficients(path):
    '''
    :param path: Path to file containing tabulated temp
        dependent rate coefficients
        :return: TDependentRateCoeffs tuple containing the
            interpolated values as a function of T
        '''

    # T, nugp, nugm, hnlgp, hnlgm, hnlhp, hnlhm, hnlh0, hnldeq = np.loadtxt(path).T
    T, nugp, nugm, hnlgp, hnlgm, hnlhp, hnlhm, hnlh0, hnldeq = np.flipud(np.loadtxt(path)).T

    # TESTING
    zeroes = [0]*len(nugp)

    return TDependentRateCoeffs(
        fast_interpolant(T, nugp),
        fast_interpolant(T, zeroes),
        fast_interpolant(T, hnlgp),
        fast_interpolant(T, zeroes),
        fast_interpolant(T, hnlhp),
        fast_interpolant(T, hnlhm),
        fast_interpolant(T, hnlh0),
        fast_interpolant(T, hnldeq)
    )

    # return TDependentRateCoeffs(
    #     interp1d(T, nugp, fill_value="extrapolate"),
    #     interp1d(T, zeroes, fill_value="extrapolate"),
    #     interp1d(T, hnlgp, fill_value="extrapolate"),
    #     interp1d(T, zeroes, fill_value="extrapolate"),
    #     interp1d(T, hnlhp, fill_value="extrapolate"),
    #     interp1d(T, hnlhm, fill_value="extrapolate"),
    #     interp1d(T, hnlh0, fill_value="extrapolate"),
    #     interp1d(T, hnldeq, fill_value="extrapolate")
    # )

    # return TDependentRateCoeffs(
    #     interp1d(T, nugp, fill_value="extrapolate"),
    #     interp1d(T, nugm, fill_value="extrapolate"),
    #     interp1d(T, hnlgp, fill_value="extrapolate"),
    #     interp1d(T, hnlgm, fill_value="extrapolate"),
    #     interp1d(T, hnlhp, fill_value="extrapolate"),
    #     interp1d(T, hnlhm, fill_value="extrapolate"),
    #     interp1d(T, hnlh0, fill_value="extrapolate"),
    #     interp1d(T, hnldeq, fill_value="extrapolate")
    # )

def get_susceptibility_matrix(path):
    '''
    :param path: path to file containing tabulated susceptibility data
    :return: (3,3) matrix valued function of T
    '''
    # Tsus, asus, bsus, csus, dsus = np.loadtxt(path).T
    # ci = interp1d(Tsus, csus, bounds_error = False, fill_value = (csus[-1], csus[0]))
    # di = interp1d(Tsus, dsus, bounds_error = False, fill_value = (dsus[-1], dsus[0]))

    Tsus, asus, bsus, csus, dsus = np.flipud(np.loadtxt(path)).T
    ci = fast_interpolant(Tsus, csus)
    di = fast_interpolant(Tsus, dsus)

    # closures are super
    def susc(T):
        c, d = ci(T), di(T)
        return 1 / T ** 2 * np.array([[c, d, d], [d, c, d], [d, d, c]])

    return susc

def get_sm_data(path):
    # SM data file is not log-uniform, can't use fast_interpolant :(
    # Tsm, psm, esm, ssm, csm, wsm, csm, gsm, hsm, ism = np.loadtxt(path).T
    # Tsm /= 1000
    #
    # sSM = interp1d(Tsm, ssm * Tsm ** 3, assume_sorted=True)
    # geff = interp1d(Tsm, gsm, fill_value="extrapolate", assume_sorted=True)

    # Use same approximations as Jurai
    geff = lambda T : 106.75

    # entropy density
    @njit
    def sSM(T):
        return 106.75 * 2 * np.pi ** 2 * T ** 3 / 45.0

    return SMData(sSM, geff)

