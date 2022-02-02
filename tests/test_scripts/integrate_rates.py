"""
Quick and dirty sanity check
"""

from load_precomputed import get_rate_coefficients
from nuMSM_solver.common import zeta3
import numpy as np
import matplotlib.pyplot as plt
from os import path

# Equilibrium distribution for SM neutrinos
def f_nu(kc):
    return 1.0/(np.exp(kc) + 1)

# Equilibrium distribution for HNLs
def f_N(T, M, kc):
    return 1.0/(np.exp(np.sqrt(M**2 + (T**2)*(kc**2))/T) + 1.0)

def quad_weights(points):
    weights = [0.5 * (points[1] - points[0])]

    for i in range(1, len(points) - 1):
        weights.append(0.5 * (points[i + 1] - points[i - 1]))

    weights.append(0.5 * (points[-1] - points[-2]))
    return weights

def integrand_hnl(T, kc, tc, weight, neq):
    return (T**3/(2*(np.pi**2)))*(1.0/neq)*weight*(kc**2)*f_N(T, 1.0, kc)*tc

def integrate_hnl(T, kc_list, tc_vals, weights, neq):
    res = 0

    for kc, tc, weight in zip(kc_list, tc_vals, weights):
        res += integrand_hnl(T, kc, tc, weight, neq)

    return res

def integrand_nu(kc, tc, weight):
    return (6.0/(np.pi**2))*weight*(kc**2)*np.exp(kc)*(f_nu(kc)**2)*tc

def integrate_nu(kc_list, tc_vals, weights):
    res = 0

    for kc, tc, weight in zip(kc_list, tc_vals, weights):
        res += integrand_nu(kc, tc, weight)

    return res

def plot_rate(kc_list, tc_vals, name):
    plt.xlabel("kc")
    plt.ylabel(name)
    plt.plot(kc_list, tc_vals)
    plt.show()

def plot_integrand_hnl(T, kc_list, tc_vals, weights, neq, name):
    Y = [integrand_hnl(T, kc, tc, weight, neq) for kc, tc, weight in zip(
        kc_list, tc_vals, weights
    )]
    plt.xlabel("kc")
    plt.ylabel("{} integrand".format(name))
    plt.plot(kc_list, Y)
    plt.show()

def plot_integrand_nu(kc_list, tc_vals, weights, name):
    Y = [integrand_nu(kc, tc, weight) for kc, tc, weight in zip(
        kc_list, tc_vals, weights
    )]
    plt.xlabel("kc")
    plt.ylabel("{} integrand".format(name))
    plt.plot(kc_list, Y)
    plt.show()

if __name__ == "__main__":
    T = 1000.0
    #
    # kc_list = [0.5, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1,
    #                 3.3, 3.6, 3.9, 4.2, 4.6, 5.0, 5.7, 6.9, 10.0]

    #yikes
    kc_list_maximal = [0.3, 0.4] + [0.1*kc for kc in range(5,101)]
    # kc_list_maximal = [0.3, 0.4] + [0.1*kc for kc in range(5,101)]
    kc_list = kc_list_maximal

    print("number of modes: ", len(kc_list))

    tc_kc = []

    test_data = path.abspath(path.join(path.dirname(__file__), '../../test_data/'))

    for kc in kc_list:
        fname = 'rates/Int_ModH_MN10E-1_kc{}E-1.dat'.format(int(kc * 10))
        path_rates = path.join(test_data, fname)
        tc_kc.append(
            [tci(T) for tci in get_rate_coefficients(path_rates)]
        )

    path_rates = path.join(test_data, "rates/Int_ModH_MN10E-1_kcAve.dat")
    nugp_avg, nugm_avg, hnlgp_avg, hnlgm_avg, hnlhp_avg, hnlhm_avg, hnlh0_avg, hnldeq_avg = \
        [tci(T) for tci in get_rate_coefficients(path_rates)]

    nugp, nugm, hnlgp, hnlgm, hnlhp, hnlhm, hnlh0, hnldeq = np.array(tc_kc).T

    weights = quad_weights(kc_list)
    neq = (3. * zeta3 * (T ** 3)) / (4 * (np.pi ** 2))

    # HNL AVERAGING
    print("hnlgp quad:", integrate_hnl(T, kc_list, hnlgp, weights, neq))
    print("hnlgp avg:", hnlgp_avg)
    print("-")
    print("hnlgm quad: ", integrate_hnl(T, kc_list, hnlgm, weights, neq))
    print("hnlgm avg: ", hnlgm_avg)
    print("-")
    print("hnlhp quad: ", integrate_hnl(T, kc_list, hnlhp, weights, neq))
    print("hnlhp avg: ", hnlhp_avg)
    print("-")
    print("hnlhm quad: ", integrate_hnl(T, kc_list, hnlhm, weights, neq))
    print("hnlhm avg: ", hnlhm_avg)
    print("-")
    print("hnlh0 quad: ", integrate_hnl(T, kc_list, hnlh0, weights, neq))
    print("hnlh0 avg: ", hnlh0_avg)
    print("-")
    print("hnldeq quad: ", integrate_hnl(T, kc_list, hnldeq, weights, neq))
    print("hnldeq avg: ", hnldeq_avg)
    print("-")
    # NU AVERAGING
    print("nugp quad: ", integrate_nu(kc_list, nugp, weights))
    print("nugp avg: ", nugp_avg)
    print("-")
    print("nugm quad: ", integrate_nu(kc_list, nugm, weights))
    print("nugm avg: ", nugm_avg)
    #
    # plot_rate(kc_list, hnlgp, "hnlgp")
    # plot_rate(kc_list, hnlgm, "hnlgm")
    # plot_rate(kc_list, hnlhp, "hnlhp")
    # plot_rate(kc_list, hnlhm, "hnlhm")
    # plot_rate(kc_list, hnlh0, "hnlh0")
    # plot_rate(kc_list, hnldeq, "hnldeq")
    # plot_rate(kc_list, nugp, "nugp")
    # plot_rate(kc_list, nugm, "nugm")
    # plot_integrand_nu(kc_list, nugp, weights, "nugp")
    # plot_integrand_nu(kc_list, nugm, weights, "nugm")



