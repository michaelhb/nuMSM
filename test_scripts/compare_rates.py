from common import *
from os import path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from leptotools.momentumDep import interpHFast, interpFast

# def make_plot(title,
#               rate1, label1,
#               rate2, label2,
#               rate3, label3,
#               Tlist, yscale="linear"):
#     Y1 = np.abs(np.array(list(map(rate1, Tlist))))
#     Y2 = np.abs(np.array(list(map(rate2, Tlist))))
#     plt.plot(Tlist, Y1, label=label1)
#     plt.plot(Tlist, Y2, label=label2)
#     plt.xlabel("T")
#     plt.ylabel("rate")
#     plt.yscale(yscale)
#     plt.title(title)
#     plt.legend()
#     plt.show()

def make_plot(title, data, Tlist, yscale="linear"):
    # Data: list of tuples (rate_func, label)
    for datum in data:
        rate_func, label = datum
        Y = np.abs(np.array(list(map(rate_func, Tlist))))

        # Hacky
        if label == "Jurai(tot=False)":
            plt.plot(Tlist, Y, label=label, linestyle="dashed")
        else:
            plt.plot(Tlist, Y, label=label)

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

    show_mode = np.where(kc_list == 10.0)[0][0]
    kc = kc_list[show_mode]

    # T0 = get_T0(mp)
    # TF = 100.
    T0 = 200.
    TF = 100.
    Tlist = np.linspace(TF, T0, 200)

    # Get tcs from Jurai's code
    gP_, gM_ = interpFast(mp.M, kc_list)
    hP_, hM_ = interpHFast(mp.M, kc_list)

    gP_j = lambda T: gP_(zT(T, mp.M))[show_mode] * T
    gM_j = lambda T: gM_(zT(T, mp.M))[show_mode] * T
    hP_j = lambda T: hP_(zT(T, mp.M))[show_mode] * T
    hM_j = lambda T: hM_(zT(T, mp.M))[show_mode] * T

    # Same with tot=False
    gP_nt, gM_nt = interpFast(mp.M, kc_list, tot=False)

    gP_j_nt = lambda T: gP_nt(zT(T, mp.M))[show_mode] * T
    gM_j_nt = lambda T: gM_nt(zT(T, mp.M))[show_mode] * T

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

    make_plot("gamma+, kc = {}".format(kc),
              [(gP_j, "Jurai"), (gP_j_nt, "Jurai(tot=False)"), (gP_f, "fortran")], Tlist, yscale="linear")
    make_plot("gamma-, kc = {}".format(kc),
              [(gM_j, "Jurai"), (gM_j_nt, "Jurai(tot=False)"), (gM_f, "fortran")], Tlist, yscale="log")
    make_plot("h+, kc = {}".format(kc),
              [(hP_j, "Jurai"), (hP_f, "fortran")], Tlist, yscale="log")
    make_plot("h-, kc = {}".format(kc),
              [(hM_j, "Jurai"),  (hM_f, "fortran")], Tlist, yscale="symlog")

