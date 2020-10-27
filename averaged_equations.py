import numpy as np
from collections import namedtuple
from integrated_rates import IntegratedRates

def coefficient_matrix(z, rt, mp, suscT, smdata):
    '''
    :param z: integration coordinate z = ln(M/T)
    :param rt: see common.IntegratedRates
    :param mp: see common.ModelParams
    :param suscT: Suscepibility matrix (2,2) (function of T)
    :param smdata: see common.SMData

    Ugliness = generated in Mathematica :)
    '''
    GB_nu_a, GBt_nu_a, GBt_N_a, HB_N, GB_N, Seq = [R(z) for R in rt]

    susc = lambda z: suscT(mp.M*np.exp(-z))

    #prefactor compensating for change of variables T -> z
    Mp = 1.22e19 # Planck mass
    MpStar = Mp * np.sqrt(45.0 / (4 * np.pi ** 3 * smdata.geff(mp.M * np.exp(-z))))
    jac = (MpStar*np.exp(-2*z))/(mp.M**2)

    return jac*np.array([[-(np.power(mp.M * np.exp(-z), 2) * np.real(GB_nu_a[0](z)) * susc[0, 0](z)) / 6,
             -(np.power(mp.M * np.exp(-z), 2) * np.real(GB_nu_a[0](z)) * susc[0, 1](z)) / 6,
             -(np.power(mp.M * np.exp(-z), 2) * np.real(GB_nu_a[0](z)) * susc[0, 2](z)) / 6,
             (2 * 1j) * np.imag(GBt_nu_a[0, 0, 0](z)), (-2 * 1j) * np.imag(GBt_nu_a[0, 0, 1](z)),
             (2 * 1j) * np.imag(GBt_nu_a[0, 1, 1](z)), -np.real(GBt_nu_a[0, 0, 0](z)), -np.real(GBt_nu_a[0, 0, 1](z)),
             -np.real(GBt_nu_a[0, 1, 1](z))],
            [-(np.power(mp.M * np.exp(-z), 2) * np.real(GB_nu_a[1](z)) * susc[1, 0](z)) / 6,
             -(np.power(mp.M * np.exp(-z), 2) * np.real(GB_nu_a[1](z)) * susc[1, 1](z)) / 6,
             -(np.power(mp.M * np.exp(-z), 2) * np.real(GB_nu_a[1](z)) * susc[1, 2](z)) / 6,
             (2 * 1j) * np.imag(GBt_nu_a[1, 0, 0](z)), (-2 * 1j) * np.imag(GBt_nu_a[1, 0, 1](z)),
             (2 * 1j) * np.imag(GBt_nu_a[1, 1, 1](z)), -np.real(GBt_nu_a[1, 0, 0](z)), -np.real(GBt_nu_a[1, 0, 1](z)),
             -np.real(GBt_nu_a[1, 1, 1](z))],
            [-(np.power(mp.M * np.exp(-z), 2) * np.real(GB_nu_a[2](z)) * susc[2, 0](z)) / 6,
             -(np.power(mp.M * np.exp(-z), 2) * np.real(GB_nu_a[2](z)) * susc[2, 1](z)) / 6,
             -(np.power(mp.M * np.exp(-z), 2) * np.real(GB_nu_a[2](z)) * susc[2, 2](z)) / 6,
             (2 * 1j) * np.imag(GBt_nu_a[2, 0, 0](z)), (-2 * 1j) * np.imag(GBt_nu_a[2, 0, 1](z)),
             (2 * 1j) * np.imag(GBt_nu_a[2, 1, 1](z)), -np.real(GBt_nu_a[2, 0, 0](z)), -np.real(GBt_nu_a[2, 0, 1](z)),
             -np.real(GBt_nu_a[2, 1, 1](z))], [
                (-1j / 12) * np.imag(GBt_N_a[0, 0, 0](z)) * np.power(mp.M * np.exp(-z), 2) * susc[0, 0](z) - (
                            1j / 12) * np.imag(GBt_N_a[1, 0, 0](z)) * np.power(mp.M * np.exp(-z), 2) * susc[1, 0](z) - (
                            1j / 12) * np.imag(GBt_N_a[2, 0, 0](z)) * np.power(mp.M * np.exp(-z), 2) * susc[2, 0](z),
                (-1j / 12) * np.imag(GBt_N_a[0, 0, 0](z)) * np.power(mp.M * np.exp(-z), 2) * susc[0, 1](z) - (
                            1j / 12) * np.imag(GBt_N_a[1, 0, 0](z)) * np.power(mp.M * np.exp(-z), 2) * susc[1, 1](z) - (
                            1j / 12) * np.imag(GBt_N_a[2, 0, 0](z)) * np.power(mp.M * np.exp(-z), 2) * susc[2, 1](z),
                (-1j / 12) * np.imag(GBt_N_a[0, 0, 0](z)) * np.power(mp.M * np.exp(-z), 2) * susc[0, 2](z) - (
                            1j / 12) * np.imag(GBt_N_a[1, 0, 0](z)) * np.power(mp.M * np.exp(-z), 2) * susc[1, 2](z) - (
                            1j / 12) * np.imag(GBt_N_a[2, 0, 0](z)) * np.power(mp.M * np.exp(-z), 2) * susc[2, 2](z),
                -np.real(GB_N[0, 0](z)), -np.real(GB_N[0, 1](z)) / 2 + 1j * np.real(HB_N[0, 1](z)), 0,
                (-1j / 2) * np.imag(GB_N[0, 0](z)), (1j / 4) * np.imag(GB_N[0, 1](z)) + np.imag(HB_N[0, 1](z)) / 2, 0],
            [(-1j / 12) * np.imag(GBt_N_a[0, 0, 1](z)) * np.power(mp.M * np.exp(-z), 2) * susc[0, 0](z) - (
                        1j / 12) * np.imag(GBt_N_a[1, 0, 1](z)) * np.power(mp.M * np.exp(-z), 2) * susc[1, 0](z) - (
                         1j / 12) * np.imag(GBt_N_a[2, 0, 1](z)) * np.power(mp.M * np.exp(-z), 2) * susc[2, 0](z),
             (-1j / 12) * np.imag(GBt_N_a[0, 0, 1](z)) * np.power(mp.M * np.exp(-z), 2) * susc[0, 1](z) - (
                         1j / 12) * np.imag(GBt_N_a[1, 0, 1](z)) * np.power(mp.M * np.exp(-z), 2) * susc[1, 1](z) - (
                         1j / 12) * np.imag(GBt_N_a[2, 0, 1](z)) * np.power(mp.M * np.exp(-z), 2) * susc[2, 1](z),
             (-1j / 12) * np.imag(GBt_N_a[0, 0, 1](z)) * np.power(mp.M * np.exp(-z), 2) * susc[0, 2](z) - (
                         1j / 12) * np.imag(GBt_N_a[1, 0, 1](z)) * np.power(mp.M * np.exp(-z), 2) * susc[1, 2](z) - (
                         1j / 12) * np.imag(GBt_N_a[2, 0, 1](z)) * np.power(mp.M * np.exp(-z), 2) * susc[2, 2](z),
             -np.real(GB_N[0, 1](z)) / 2 + 1j * np.real(HB_N[0, 1](z)),
             -np.real(GB_N[0, 0](z)) / 2 - np.real(GB_N[1, 1](z)) / 2 - 1j * np.real(HB_N[0, 0](z)) + 1j * np.real(
                 HB_N[1, 1](z)), -np.real(GB_N[0, 1](z)) / 2 - 1j * np.real(HB_N[0, 1](z)),
             (-1j / 4) * np.imag(GB_N[0, 1](z)) - np.imag(HB_N[0, 1](z)) / 2,
             (-1j / 4) * np.imag(GB_N[0, 0](z)) - (1j / 4) * np.imag(GB_N[1, 1](z)) + np.imag(
                 HB_N[0, 0](z)) / 2 - np.imag(HB_N[1, 1](z)) / 2,
             (-1j / 4) * np.imag(GB_N[0, 1](z)) + np.imag(HB_N[0, 1](z)) / 2], [
                (-1j / 12) * np.imag(GBt_N_a[0, 1, 1](z)) * np.power(mp.M * np.exp(-z), 2) * susc[0, 0](z) - (
                            1j / 12) * np.imag(GBt_N_a[1, 1, 1](z)) * np.power(mp.M * np.exp(-z), 2) * susc[1, 0](z) - (
                            1j / 12) * np.imag(GBt_N_a[2, 1, 1](z)) * np.power(mp.M * np.exp(-z), 2) * susc[2, 0](z),
                (-1j / 12) * np.imag(GBt_N_a[0, 1, 1](z)) * np.power(mp.M * np.exp(-z), 2) * susc[0, 1](z) - (
                            1j / 12) * np.imag(GBt_N_a[1, 1, 1](z)) * np.power(mp.M * np.exp(-z), 2) * susc[1, 1](z) - (
                            1j / 12) * np.imag(GBt_N_a[2, 1, 1](z)) * np.power(mp.M * np.exp(-z), 2) * susc[2, 1](z),
                (-1j / 12) * np.imag(GBt_N_a[0, 1, 1](z)) * np.power(mp.M * np.exp(-z), 2) * susc[0, 2](z) - (
                            1j / 12) * np.imag(GBt_N_a[1, 1, 1](z)) * np.power(mp.M * np.exp(-z), 2) * susc[1, 2](z) - (
                            1j / 12) * np.imag(GBt_N_a[2, 1, 1](z)) * np.power(mp.M * np.exp(-z), 2) * susc[2, 2](z), 0,
                -np.real(GB_N[0, 1](z)) / 2 - 1j * np.real(HB_N[0, 1](z)), -np.real(GB_N[1, 1](z)), 0,
                (1j / 4) * np.imag(GB_N[0, 1](z)) - np.imag(HB_N[0, 1](z)) / 2, (-1j / 2) * np.imag(GB_N[1, 1](z))], [
                -(np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[0, 0, 0](z)) * susc[0, 0](z)) / 6 - (
                            np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[1, 0, 0](z)) * susc[1, 0](z)) / 6 - (
                            np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[2, 0, 0](z)) * susc[2, 0](z)) / 6,
                -(np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[0, 0, 0](z)) * susc[0, 1](z)) / 6 - (
                            np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[1, 0, 0](z)) * susc[1, 1](z)) / 6 - (
                            np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[2, 0, 0](z)) * susc[2, 1](z)) / 6,
                -(np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[0, 0, 0](z)) * susc[0, 2](z)) / 6 - (
                            np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[1, 0, 0](z)) * susc[1, 2](z)) / 6 - (
                            np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[2, 0, 0](z)) * susc[2, 2](z)) / 6,
                (-2 * 1j) * np.imag(HB_N[0, 0](z)), (2 + 1j) * np.imag(HB_N[0, 1](z)), 0, -np.real(HB_N[0, 0](z)),
                (-1 / 2 + 1j) * np.real(HB_N[0, 1](z)), 0], [
                -(np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[0, 0, 1](z)) * susc[0, 0](z)) / 6 - (
                            np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[1, 0, 1](z)) * susc[1, 0](z)) / 6 - (
                            np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[2, 0, 1](z)) * susc[2, 0](z)) / 6,
                -(np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[0, 0, 1](z)) * susc[0, 1](z)) / 6 - (
                            np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[1, 0, 1](z)) * susc[1, 1](z)) / 6 - (
                            np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[2, 0, 1](z)) * susc[2, 1](z)) / 6,
                -(np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[0, 0, 1](z)) * susc[0, 2](z)) / 6 - (
                            np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[1, 0, 1](z)) * susc[1, 2](z)) / 6 - (
                            np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[2, 0, 1](z)) * susc[2, 2](z)) / 6,
                (-2 - 1j) * np.imag(HB_N[0, 1](z)),
                (2 - 1j) * np.imag(HB_N[0, 0](z)) - (2 + 1j) * np.imag(HB_N[1, 1](z)),
                (2 - 1j) * np.imag(HB_N[0, 1](z)), (-1 / 2 + 1j) * np.real(HB_N[0, 1](z)),
                (-1 / 2 - 1j) * np.real(HB_N[0, 0](z)) - (1 / 2 - 1j) * np.real(HB_N[1, 1](z)),
                (-1 / 2 - 1j) * np.real(HB_N[0, 1](z))], [
                -(np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[0, 1, 1](z)) * susc[0, 0](z)) / 6 - (
                            np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[1, 1, 1](z)) * susc[1, 0](z)) / 6 - (
                            np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[2, 1, 1](z)) * susc[2, 0](z)) / 6,
                -(np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[0, 1, 1](z)) * susc[0, 1](z)) / 6 - (
                            np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[1, 1, 1](z)) * susc[1, 1](z)) / 6 - (
                            np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[2, 1, 1](z)) * susc[2, 1](z)) / 6,
                -(np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[0, 1, 1](z)) * susc[0, 2](z)) / 6 - (
                            np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[1, 1, 1](z)) * susc[1, 2](z)) / 6 - (
                            np.power(mp.M * np.exp(-z), 2) * np.real(GBt_N_a[2, 1, 1](z)) * susc[2, 2](z)) / 6, 0,
                (-2 + 1j) * np.imag(HB_N[0, 1](z)), (-2 * 1j) * np.imag(HB_N[1, 1](z)), 0,
                (-1 / 2 - 1j) * np.real(HB_N[0, 1](z)), -np.real(HB_N[1, 1](z))]])
