from common import *
from os import path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from leptotools.momentumDep import interpHFast, interpFast

def make_plot(title, rate1, label1, rate2, label2, Tlist, yscale="linear"):
    Y1 = np.abs(np.array(list(map(rate1, Tlist))))
    Y2 = np.abs(np.array(list(map(rate2, Tlist))))
    plt.plot(Tlist, Y1, label=label1)
    plt.plot(Tlist, Y2, label=label2)
    plt.xlabel("T")
    plt.ylabel("rate")
    plt.yscale(yscale)
    plt.title(title)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    mp = ModelParams(M=10.0, dM=1e-11, Imw=1.0, Rew=0.7853981633974483,
                     delta=3.141592653589793,  eta=4.71238898038469)

    # kc_list = np.array([1.0])
    kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1,
                3.3, 3.6, 3.9, 4.2, 4.6, 5.0, 5.7, 6.9, 10.0])

    show_mode = kc_list.shape[0] - 1

    T0 = get_T0(mp)
    TF = Tsph
    # TF = 5
    Tlist = np.linspace(TF, T0, 200)

    # Get tcs from Jurai's code
    S = 1
    gP_, gM_ = interpFast(S*mp.M, kc_list)
    hP_, hM_ = interpHFast(S*mp.M, kc_list)

    K = 1
    L = 1

    gP_j = lambda T: gP_(zT(L*T, mp.M))[show_mode] * K * T
    gM_j = lambda T: gM_(zT(L*T, mp.M))[show_mode] * K * T
    hP_j = lambda T: hP_(zT(L*T, mp.M))[show_mode] * K * T
    hM_j = lambda T: hM_(zT(L*T, mp.M))[show_mode] * K * T

    # Get tcs from fortran code
    test_data = path.abspath(path.join(path.dirname(__file__), '../test_data/'))
    path_rates = path.join(test_data,
        "rates/Int_OrgH_MN{}E-1_kc{}E-1.dat".format(int(mp.M * 10), int(kc_list[show_mode] * 10)))

    T, nugp, nugm, hnlgp, hnlgm, hnlhp, hnlhm, hnlh0, hnldeq \
        = np.flipud(np.loadtxt(path_rates)).T

    gP_f = interp1d(T, nugp, fill_value="extrapolate")
    gM_f = interp1d(T, nugm, fill_value="extrapolate")
    hP_f = interp1d(T, hnlhp, fill_value="extrapolate")
    hM_f = interp1d(T, hnlhm, fill_value="extrapolate")

    yscale = "linear"
    # yscale = "log"
    make_plot("gamma+", gP_j, "Jurai", gP_f, "fortran", Tlist, yscale)
    make_plot("gamma-", gM_j, "Jurai", gM_f, "fortran", Tlist, yscale)
    make_plot("h+", hP_j, "Jurai", hP_f, "fortran", Tlist, yscale)
    make_plot("h-", hM_j, "Jurai", hM_f, "fortran", Tlist, yscale)

