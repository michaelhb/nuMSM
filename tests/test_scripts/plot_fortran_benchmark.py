from os.path import dirname, abspath, join
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from nuMSM_solver.common import *

if __name__ == '__main__':
    # path_benchmark = expanduser("~/SciCodes/nuMSM/test_data/LA_Evo_ModH.dat")
    parent = dirname(dirname(abspath(__file__)))
    path_benchmark = join(parent, "fortran/Evo_LA_momentum_H_Gamma/Evo_OrgH2/LA_Evo_OrgH.dat")

    #T, Y_e, Y_mu, Y_tau, n_+/s(1,1), n_+/s(2,2), Re[n_+/s(1,2)], Im[n_+/s(1,2)], n_-/s(1,1), n_-/s(2,2), Re[n_-/s(1,2)], Im[n_-/s(1,2)]
    T, Y_e, Y_mu, Y_tau, np11, np22, renp12, imnp12, nm11, nm22, renm12, imnm12 = np.loadtxt(path_benchmark).T

    Y = Y_e + Y_mu + Y_tau
    bau_interp = interp1d(T, (28./78.)*Y)

    bau = (28./78.)*Y[-1]
    plt.title("BAU @ Tsph: {:.3e}, BAU final: {:.3e}".format(bau_interp(Tsph), bau))
    plt.loglog(T, np.abs(Y))
    plt.show()

    print("BAU: {}".format(bau))