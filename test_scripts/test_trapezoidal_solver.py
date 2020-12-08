from os import path
from common import ModelParams
from rates import get_rates
from load_precomputed import *

mp = ModelParams(
    M=1.0,
    dM=1e-12,
    Imw=np.log(3),
    Rew=13 / 16 * np.pi,
    delta=29 / 16 * np.pi,
    eta=22 / 16 * np.pi
)

kc_list = [0.5, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1,
                3.3, 3.6, 3.9, 4.2, 4.6, 5.0, 5.7, 6.9, 10.0]

def run_test():

    # Load rates for each momentum mode
    kc_rates = []

    for kc in kc_list:
        fname = 'rates/Int_ModH_MN10E-1_kc{}E-1.dat'.format(int(kc*10))
        test_data = path.abspath(path.join(path.dirname(__file__), '../test_data/'))
        path_rates = path.join(test_data, fname)
        tc = get_rate_coefficients(path_rates)
        kc_rates.append(get_rates(mp, tc))

    # Load standard model data, susceptibility matrix
    path_SMdata = path.join(test_data, "standardmodel.dat")
    path_suscept_data = path.join(test_data, "susceptibility.dat")
    susc = get_susceptibility_matrix(path_suscept_data)
    smdata = get_sm_data(path_SMdata)

