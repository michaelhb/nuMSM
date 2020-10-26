import numpy as np
from scipy.interpolate import interp1d
from common import TDependentRateCoeffs, SMData

def get_rate_coefficients(path):
    '''
    :param path: Path to file containing tabulated temp
        dependent rate coefficients
    :return: TDependentRateCoeffs tuple containing the
        interpolated values as a function of T
    '''

    T, nugp, nugm, hnlgp, hnlgm, hnlhp, hnlhm, hnlh0, hnldeq = np.loadtxt(path).T

    return TDependentRateCoeffs(
        interp1d(T, nugp),
        interp1d(T, nugm),
        interp1d(T, hnlgp),
        interp1d(T, hnlgm),
        interp1d(T, hnlhp),
        interp1d(T, hnlhm),
        interp1d(T, hnlh0),
        interp1d(T, hnldeq)
    )

def get_susceptibility_matrix(path):
    '''
    :param path: path to file containing tabulated susceptibility data
    :return: (3,3) matrix valued function of T
    '''
    Tsus, asus, bsus, csus, dsus = np.loadtxt(path).T
    ci = interp1d(Tsus, csus, bounds_error = False, fill_value = (csus[-1], csus[0]))
    di = interp1d(Tsus, dsus, bounds_error = False, fill_value = (dsus[-1], dsus[0]))

    # closures are super
    def susc(T):
        c, d = ci(T), di(T)
        return 1 / T ** 2 * np.array([[c, d, d], [d, c, d], [d, d, c]])

    return susc

def get_sm_data(path):
    Tsm, psm, esm, ssm, csm, wsm, csm, gsm, hsm, ism = np.loadtxt(path).T
    Tsm /= 1000

    sSM = interp1d(Tsm, ssm * Tsm ** 3, assume_sorted=True)
    geff = interp1d(Tsm, gsm, fill_value="extrapolate", assume_sorted=True)

    return SMData(sSM, geff)

