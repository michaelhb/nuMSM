from load_precomputed import get_rate_coefficients, get_sm_data
from nuMSM_solver.common import *
import numpy as np
import matplotlib.pyplot as plt
from os import path
from leptotools.scantools import leptogenesisScanSetup

mp = ModelParams(M=10.0, dM=1e-12, Imw=3.0, Rew=0.7853981633974483, delta=3.141592653589793, eta=4.71238898038469)

if __name__ == "__main__":
    H = 1

    T0 = 10000.0
    TF = Tsph
    Tlist = np.linspace(TF, T0, 200)

    test_data = path.abspath(path.join(path.dirname(__file__), '../../test_data/'))

    # rate_names = ["nugp", "nugm", "hnlgp", "hnlgm", "hnlhp", "hnlhm", "hnlh0"]
    rate_names = ["hnldeq"]

    path_rates = path.join(test_data, "rates/Int_OrgH_MN{}E-1_kcAve.dat".format(int(mp.M * 10)))
    nugp_avg, nugm_avg, hnlgp_avg, hnlgm_avg, hnlhp_avg, hnlhm_avg, hnlh0_avg, hnldeq_avg = \
        get_rate_coefficients(path_rates)

    path_SMdata = path.join(test_data, "standardmodel.dat")
    smdata = get_sm_data(path_SMdata)

    # rates_old = [nugp_avg, nugm_avg, hnlgp_avg, hnlgm_avg, hnlhp_avg, hnlhm_avg, hnlh0_avg]
    rates_old = [hnldeq_avg]

    ls = leptogenesisScanSetup(mp.M)
    if H == 1:
        ls.set_ordering("NO")
    else:
        ls.set_ordering("IH")

    ls.set_dM(mp.dM)
    ls.set_romega(mp.Rew)
    ls.set_iomega(mp.Imw)
    ls.set_delta(mp.delta)
    ls.set_eta(mp.eta)
    ls.set_xi(1)
    ls.set_CI()
    ls.set_HeffGammaR()

    nugp = lambda T: ls.gammanuP(zT(T, mp.M)) * T
    nugm = lambda T: ls.gammanuM(zT(T, mp.M)) * T
    hnlgp = lambda T: ls.gammaNP(zT(T, mp.M)) * T
    hnlgm = lambda T: ls.gammaNM(zT(T, mp.M)) * T
    hnlhp = lambda T: ls.hNP(zT(T, mp.M)) * T
    hnlhm = lambda T: ls.hNM(zT(T, mp.M)) * T
    hnlh0 = lambda T: ls.hNM0(zT(T, mp.M)) * T
    # hnldeq = lambda T: (ls.rhoN0(zT(T, mp.M)) * T) #/ jacobian(zT(T, mp.M), smdata)
    hnldeq = lambda T: ls.dYdzeta(zT(T, mp.M)) / (jacobian(zT(T, mp.M), mp, smdata))
    # hnldeq = lambda T: ls.dYdzeta(zT(T, mp.M)) / T**3

    # rates_new = [nugp, nugm, hnlgp, hnlgm, hnlhp, hnlhm, hnlh0]
    rates_new = [hnldeq]
    for i, rate in enumerate(rate_names):

        Y1 = np.abs(list(map(rates_old[i], Tlist)))
        Y2 = np.abs(list(map(rates_new[i], Tlist)))
        plt.loglog(Tlist, Y1, label="Old rates", color="red")
        plt.loglog(Tlist, Y2, label="Jurai", color="blue")
        plt.xlabel("T")
        plt.ylabel(rate)
        plt.legend()
        plt.show()

    # nugp_jurai = lambda T: ls.gammanuP(T) * T
    #
    # Y1 = list(map(nugp_avg, Tlist))
    # Y2 = list(map(nugp_jurai, Tlist))
    #
    # plt.plot(Tlist, Y1, label="Old rates", color="red")
    # plt.plot(Tlist, Y2, label="Jurai", color="blue")
    # plt.xlabel("T")
    # plt.ylabel("nugp")
    # plt.legend()
    # plt.show()

