import numpy as np
from collections import namedtuple
from integrated_rates import IntegratedRates


def jacobian(z, mp, smdata):
    """
    :param z: integration coordinate z = ln(M/T)
    :param mp: see common.ModelParams
    :param smdata: see common.SMData
    :return: jacobian for transformation T -> Z
    """
    Mp = 1.22e19  # Planck mass
    MpStar = Mp * np.sqrt(45.0 / (4 * np.pi ** 3 * smdata.geff(mp.M * np.exp(-z))))
    return (MpStar * np.exp(-2 * z)) / (mp.M ** 2)


def inhomogeneous_part(z, rt):
    """
    :param rt: see common.IntegratedRates
    :return: inhomogeneous part of the ODE system
    """
    Seq = rt.Seq(z)
    return np.array([0, 0, 0, -Seq[0, 0], -Seq[1, 1], 0, 0, 0, 0, 0, 0])


def coefficient_matrix(z, rt, mp, suscT, smdata):
    '''
    :param z: integration coordinate z = ln(M/T)
    :param rt: see common.IntegratedRates
    :param mp: see common.ModelParams
    :param suscT: Suscepibility matrix (2,2) (function of T)
    :param smdata: see common.SMData
    :return: z-dependent coefficient matrix of the ODE system

    Ugliness = generated in Mathematica :)
    '''
    GB_nu_a, GBt_nu_a, GBt_N_a, HB_N, GB_N, Seq = [R(z) for R in rt]
    susc = suscT(mp.M * np.exp(-z))

    return np.array([[-(np.power(mp.M * np.exp(-z), 2) * np.real(GB_nu_a[0]) * susc[0, 0]) / 6,
                      -(np.power(mp.M * np.exp(-z), 2) * np.real(GB_nu_a[0]) * susc[0, 1]) / 6,
                      -(np.power(mp.M * np.exp(-z), 2) * np.real(GB_nu_a[0]) * susc[0, 2]) / 6, 0, 0, 0, 0,
                      -np.real(GBt_nu_a[0, 0, 0]), -np.real(GBt_nu_a[0, 1, 1]), 0, 0],
                     [-(np.power(mp.M * np.exp(-z), 2) * np.real(GB_nu_a[1]) * susc[1, 0]) / 6,
                      -(np.power(mp.M * np.exp(-z), 2) * np.real(GB_nu_a[1]) * susc[1, 1]) / 6,
                      -(np.power(mp.M * np.exp(-z), 2) * np.real(GB_nu_a[1]) * susc[1, 2]) / 6, 0, 0, 0, 0,
                      -np.real(GBt_nu_a[1, 0, 0]), -np.real(GBt_nu_a[1, 1, 1]), 0, 0],
                     [-(np.power(mp.M * np.exp(-z), 2) * np.real(GB_nu_a[2]) * susc[2, 0]) / 6,
                      -(np.power(mp.M * np.exp(-z), 2) * np.real(GB_nu_a[2]) * susc[2, 1]) / 6,
                      -(np.power(mp.M * np.exp(-z), 2) * np.real(GB_nu_a[2]) * susc[2, 2]) / 6, 0, 0, 0, 0,
                      -np.real(GBt_nu_a[2, 0, 0]), -np.real(GBt_nu_a[2, 1, 1]), 0, 0],
                     [0, 0, 0, -np.real(GB_N[0, 0]), 0, -np.real(GB_N[0, 1]), -2 * np.real(HB_N[0, 1]), 0, 0,
                      np.imag(HB_N[0, 1]), -np.imag(GB_N[0, 1]) / 2],
                     [0, 0, 0, 0, -np.real(GB_N[1, 1]), -np.real(GB_N[0, 1]), 2 * np.real(HB_N[0, 1]), 0, 0,
                      -np.imag(HB_N[0, 1]), -np.imag(GB_N[0, 1]) / 2],
                     [0, 0, 0, -np.real(GB_N[0, 1]) / 2, -np.real(GB_N[0, 1]) / 2,
                      (-np.real(GB_N[0, 0]) - np.real(GB_N[1, 1])) / 2, np.real(HB_N[0, 0]) - np.real(HB_N[1, 1]),
                      -np.imag(HB_N[0, 1]) / 2, np.imag(HB_N[0, 1]) / 2, 0, 0], [-(np.power(mp.M * np.exp(-z), 2) * (
                    np.imag(GBt_N_a[0, 0, 1]) * susc[0, 0] + np.imag(GBt_N_a[1, 0, 1]) * susc[1, 0] + np.imag(
                GBt_N_a[2, 0, 1]) * susc[2, 0])) / 12, -(np.power(mp.M * np.exp(-z), 2) * (
                    np.imag(GBt_N_a[0, 0, 1]) * susc[0, 1] + np.imag(GBt_N_a[1, 0, 1]) * susc[1, 1] + np.imag(
                GBt_N_a[2, 0, 1]) * susc[2, 1])) / 12, -(np.power(mp.M * np.exp(-z), 2) * (
                    np.imag(GBt_N_a[0, 0, 1]) * susc[0, 2] + np.imag(GBt_N_a[1, 0, 1]) * susc[1, 2] + np.imag(
                GBt_N_a[2, 0, 1]) * susc[2, 2])) / 12, np.real(HB_N[0, 1]), -np.real(HB_N[0, 1]),
                                                                                 -np.real(HB_N[0, 0]) + np.real(
                                                                                     HB_N[1, 1]), (-np.real(
                GB_N[0, 0]) - np.real(GB_N[1, 1])) / 2, -np.imag(GB_N[0, 1]) / 4, -np.imag(GB_N[0, 1]) / 4, 0, 0], [-(
                    np.power(mp.M * np.exp(-z), 2) * (
                        np.real(GBt_N_a[0, 0, 0]) * susc[0, 0] + np.real(GBt_N_a[1, 0, 0]) * susc[1, 0] + np.real(
                    GBt_N_a[2, 0, 0]) * susc[2, 0])) / 6, -(np.power(mp.M * np.exp(-z), 2) * (
                    np.real(GBt_N_a[0, 0, 0]) * susc[0, 1] + np.real(GBt_N_a[1, 0, 0]) * susc[1, 1] + np.real(
                GBt_N_a[2, 0, 0]) * susc[2, 1])) / 6, -(np.power(mp.M * np.exp(-z), 2) * (
                    np.real(GBt_N_a[0, 0, 0]) * susc[0, 2] + np.real(GBt_N_a[1, 0, 0]) * susc[1, 2] + np.real(
                GBt_N_a[2, 0, 0]) * susc[2, 2])) / 6, 0, 0, 4 * np.imag(HB_N[0, 1]), -2 * np.imag(HB_N[0, 1]), -np.real(
            HB_N[0, 0]), 0, -np.real(HB_N[0, 1]), -2 * np.real(HB_N[0, 1])], [-(np.power(mp.M * np.exp(-z), 2) * (
                    np.real(GBt_N_a[0, 1, 1]) * susc[0, 0] + np.real(GBt_N_a[1, 1, 1]) * susc[1, 0] + np.real(
                GBt_N_a[2, 1, 1]) * susc[2, 0])) / 6, -(np.power(mp.M * np.exp(-z), 2) * (
                    np.real(GBt_N_a[0, 1, 1]) * susc[0, 1] + np.real(GBt_N_a[1, 1, 1]) * susc[1, 1] + np.real(
                GBt_N_a[2, 1, 1]) * susc[2, 1])) / 6, -(np.power(mp.M * np.exp(-z), 2) * (
                    np.real(GBt_N_a[0, 1, 1]) * susc[0, 2] + np.real(GBt_N_a[1, 1, 1]) * susc[1, 2] + np.real(
                GBt_N_a[2, 1, 1]) * susc[2, 2])) / 6, 0, 0, -4 * np.imag(HB_N[0, 1]), -2 * np.imag(HB_N[0, 1]), 0,
                                                                              -np.real(HB_N[1, 1]),
                                                                              -np.real(HB_N[0, 1]),
                                                                              2 * np.real(HB_N[0, 1])], [-(
                    np.power(mp.M * np.exp(-z), 2) * (
                        np.real(GBt_N_a[0, 0, 1]) * susc[0, 0] + np.real(GBt_N_a[1, 0, 1]) * susc[1, 0] + np.real(
                    GBt_N_a[2, 0, 1]) * susc[2, 0])) / 6, -(np.power(mp.M * np.exp(-z), 2) * (
                    np.real(GBt_N_a[0, 0, 1]) * susc[0, 1] + np.real(GBt_N_a[1, 0, 1]) * susc[1, 1] + np.real(
                GBt_N_a[2, 0, 1]) * susc[2, 1])) / 6, -(np.power(mp.M * np.exp(-z), 2) * (
                    np.real(GBt_N_a[0, 0, 1]) * susc[0, 2] + np.real(GBt_N_a[1, 0, 1]) * susc[1, 2] + np.real(
                GBt_N_a[2, 0, 1]) * susc[2, 2])) / 6, -2 * np.imag(HB_N[0, 1]), 2 * np.imag(HB_N[0, 1]), 0, 0, -np.real(
            HB_N[0, 1]) / 2, -np.real(HB_N[0, 1]) / 2, (-np.real(HB_N[0, 0]) - np.real(HB_N[1, 1])) / 2, np.real(
            HB_N[0, 0]) - np.real(HB_N[1, 1])],
                     [0, 0, 0, -np.imag(HB_N[0, 1]), -np.imag(HB_N[0, 1]), 0, 0, np.real(HB_N[0, 1]),
                      -np.real(HB_N[0, 1]), -np.real(HB_N[0, 0]) + np.real(HB_N[1, 1]),
                      (-np.real(HB_N[0, 0]) - np.real(HB_N[1, 1])) / 2]])
