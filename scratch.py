from rates import *
from load_precomputed import *
from common import *
import numpy as np
from os import path
import sys
import time

mp = ModelParams(
    M=1.0,
    dM=1e-11,
    Imw=np.log(3),
    Rew=1/4 * np.pi,
    delta= np.pi,
    eta=3/2 * np.pi
)

if __name__ == "__main__":

    test_data = path.abspath(path.join(path.dirname(__file__), 'test_data/'))

    # Load averaged coefficients
    path_rates = path.join(test_data, "rates/Int_ModH_MN10E-1_kcAve.dat")
    tc_avg = get_rate_coefficients(path_rates)

    for i in range(10000):
        start = time.time()
        print(tc_avg.nugp(Tsph))
        end = time.time()
        print("Time: {}".format(end - start))