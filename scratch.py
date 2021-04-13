from rates import *
from load_precomputed import *
from common import *
import numpy as np

mp = ModelParams(
    M=1.0,
    dM=1e-11,
    Imw=np.log(3),
    Rew=1/4 * np.pi,
    delta= np.pi,
    eta=3/2 * np.pi
)

if __name__ == "__main__":
    kc_list = [1.0]
    # kc_list = [0.3, 0.4] + [0.1 * kc for kc in range(5, 101)]

    test_data = path.abspath(path.join(path.dirname(__file__), 'test_data/'))

    # Load averaged coefficients
    path_rates = path.join(test_data, "rates/Int_ModH_MN10E-1_kcAve.dat")
    tc_avg = get_rate_coefficients(path_rates)

    # Get averaged rates
    rates_avg = get_rates(mp, tc_avg)

    # Get standard rates
    rates_std = []
    test_data = path.abspath(path.join(path.dirname(__file__), 'test_data/'))

    for kc in kc_list:
        fname = 'rates/Int_ModH_MN10E-1_kc{}E-1.dat'.format(int(kc * 10))
        path_rates = path.join(test_data, fname)
        tc = get_rate_coefficients(path_rates)
        rt = get_rates(mp, tc, 1)

        rates_std.append(rt)

    # Get normalised rates
    rates_normed = get_normalised_rates(mp, kc_list, trapezoidal_weights(kc_list), scale_to_avg=True)

    rnames = [
        "GB_nu_a",  # (3)
        "GBt_nu_a",  # (3,2,2)
        "GBt_N_a",  # (3,2,2)
        "HB_N",  # (2,2)
        "GB_N"  # (2,2)
     ]

    # Equilibrium density
    neq = lambda T: (3. * zeta3 * (T ** 3)) / (4 * (np.pi ** 2))

    T = 131.7
    z = np.log(1/T)

    for i, rn in enumerate(rnames):
        print(rn)
        print("averaged")
        print(rates_avg[i](z))

        for j, kc in enumerate(kc_list):
            # print("kc == ", kc, ", std")
            # print(rates_std[j][i](zTest))
            print("kc == ", kc, ", normed")
            print(rates_normed[j][i](z))
        print("===================")
