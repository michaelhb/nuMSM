from common import *
import numpy as np
from scipy import integrate
from os import path
from load_precomputed import get_rate_coefficients, get_sm_data
import matplotlib.pyplot as plt
from leptotools.scantools import leptogenesisScanSetup

mp = ModelParams(M=10.0, dM=1e-12, Imw=3.0, Rew=0.7853981633974483, delta=3.141592653589793, eta=4.71238898038469)

# def f_Ndot(kc, T, mp, smdata):
#     E_k = np.sqrt(mp.M**2 + (T**2)*(kc**2))
#     Mpl = MpStar(zT(T, mp.M), mp, smdata)
#     return -1*((T*E_k)/Mpl)*np.exp(E_k/T)/((1 + np.exp(E_k/T))**2)

def Seq(T, mp, smdata):
    I = lambda kc: (kc**2)*(f_Ndot(kc, T, mp, smdata) + (3*(T**2)/MpStar(zT(T, mp.M), mp, smdata))*f_N(T, mp.M, kc))
    return ((T**3)/(2*np.pi**2))*(1.0/smdata.s(T))*integrate.quad(I, 0, np.inf)[0]

if __name__ == "__main__":

    ## Seq from old rates
    rates_dir = path.abspath(path.join(path.dirname(__file__), '../../test_data/'))
    path_rates = path.join(rates_dir,
                           "rates/Int_OrgH_MN{}E-1_kcAve.dat".format(int(mp.M * 10)))
    path_SMdata = path.join(rates_dir, "standardmodel.dat")
    smdata = get_sm_data(path_SMdata)

    nugp_avg, nugm_avg, hnlgp_avg, hnlgm_avg, hnlhp_avg, hnlhm_avg, hnlh0_avg, hnldeq_avg = \
        get_rate_coefficients(path_rates)

    ## Calculated directly
    hnldeq_new = lambda T: Seq(T, mp, smdata)

    Tlist = np.linspace(10,1000,200)

    ## Jurai's rates ##
    ls = leptogenesisScanSetup(mp.M)
    ls.set_ordering("NO")

    ls.set_dM(mp.dM)
    ls.set_romega(mp.Rew)
    ls.set_iomega(mp.Imw)
    ls.set_delta(mp.delta)
    ls.set_eta(mp.eta)
    ls.set_xi(1)
    ls.set_CI()
    ls.set_HeffGammaR()

    hnldeq_jurai = lambda T: ls.dYdzeta(zT(T, mp.M)) / (jacobian(zT(T, mp.M), mp, smdata))
    ##

    Y_avg = np.abs(list(map(hnldeq_avg, Tlist)))
    Y_new = np.abs(list(map(hnldeq_new, Tlist)))
    Y_jurai = np.abs(list(map(hnldeq_jurai, Tlist)))

    print(Y_new)

    plt.loglog(Tlist, Y_avg, label="Fortran", color="red")
    plt.loglog(Tlist, Y_new, label="scipy.integrate", color="blue")
    plt.loglog(Tlist, Y_jurai, label="jurai", color="green")
    plt.xlabel("T")
    plt.ylabel("hnldeq")
    plt.legend()
    plt.show()