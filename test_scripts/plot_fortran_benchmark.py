import numpy as np
from os.path import expanduser
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path_benchmark = expanduser("~/SciCodes/nuMSM/test_data/LA_Evo_ModH.dat")

    #T, Y_e, Y_mu, Y_tau, n_+/s(1,1), n_+/s(2,2), Re[n_+/s(1,2)], Im[n_+/s(1,2)], n_-/s(1,1), n_-/s(2,2), Re[n_-/s(1,2)], Im[n_-/s(1,2)]
    T, Y_e, Y_mu, Y_tau, np11, np22, renp12, imnp12, nm11, nm22, renm12, imnm12 = np.loadtxt(path_benchmark).T

    Y = np.abs(Y_e + Y_mu + Y_tau)
    plt.loglog(T, Y)
    plt.show()