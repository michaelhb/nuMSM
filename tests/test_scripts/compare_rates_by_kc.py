from common import *
from os import path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from leptotools.momentumDep import interpHFast, interpFast

def make_plot(title, data, kc_list, yscale="linear"):
    # data: list of tuples (rate_vals, label)
    fig, ax = plt.subplots()

    for datum in data:
        rate_vals, label = datum
        ax.plot(kc_list, rate_vals, label=label)

    ax.set_xlabel("kc")
    ax.set_ylabel("rate")
    ax.set_yscale(yscale)
    fig.suptitle(title)
    ax.legend()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2e'))
    ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%.2e'))
    # ax.yaxis.set_major_locator(ticker.LogLocator(base=2, numticks=10))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    mp = ModelParams(M=1.0, dM=1e-11, Imw=1.0, Rew=0.7853981633974483,
                     delta=3.141592653589793,  eta=4.71238898038469)

    kc_list = np.array([0.1 * kc for kc in range(5, 101)])

    # temps = [100, 200, 300, 500, 1000]

    temps = [100, 110, 120, 130, 140, 150, 160]

    for T in temps:

        # Get tcs from Jurai's code
        gP_j_, gM_j_ = interpFast(mp.M, kc_list)
        gP_j_nt_, gM_j_nt_ = interpFast(mp.M, kc_list, tot=False)
        hP_j_, hM_j_ = interpHFast(mp.M, kc_list)

        gP_j = gP_j_(zT(T, mp.M)) * T
        gM_j = gM_j_(zT(T, mp.M)) * T
        gP_j_nt = gP_j_nt_(zT(T, mp.M)) * T
        gM_j_nt = gM_j_nt_(zT(T, mp.M)) * T
        hP_j = hP_j_(zT(T, mp.M)) * T
        hM_j = hM_j_(zT(T, mp.M)) * T

        # Get tcs from Shintaro's files
        gP_f = []
        gM_f = []
        hP_f = []
        hM_f = []

        test_data = path.abspath(path.join(path.dirname(__file__), '../../test_data/'))
        for kc in kc_list:
            path_rates = path.join(test_data,
                "rates/Int_OrgH_MN{}E-1_kc{}E-1.dat".format(int(mp.M * 10), int(kc * 10)))

            Tlist, nugp, nugm, hnlgp, hnlgm, hnlhp, hnlhm, hnlh0, hnldeq \
                = np.flipud(np.loadtxt(path_rates)).T

            gP_f_ = interp1d(Tlist, nugp, fill_value="extrapolate")
            gM_f_ = interp1d(Tlist, nugm, fill_value="extrapolate")
            hP_f_ = interp1d(Tlist, hnlhp, fill_value="extrapolate")
            hM_f_ = interp1d(Tlist, hnlhm, fill_value="extrapolate")

            gP_f.append(gP_f_(T))
            gM_f.append(gM_f_(T))
            hM_f.append(hM_f_(T))
            hP_f.append(hP_f_(T))

        yscale = "symlog"

        # Make the plots
        make_plot("gamma+, T = {}".format(T),
                  [(gP_j, "Jurai"), (gP_j_nt, "Jurai (tot=False)"), (gP_f, "Shintaro")],
                  kc_list, yscale=yscale)
        make_plot("gamma-, T = {}".format(T),
                  [(gM_j, "Jurai"), (gM_j_nt, "Jurai (tot=False)"), (gM_f, "Shintaro")],
                  kc_list, yscale=yscale)
        make_plot("h+, T = {}".format(T),
                  [(hP_j, "Jurai"), (hP_f, "Shintaro")],
                  kc_list, yscale=yscale)
        make_plot("h-, T = {}".format(T),
                  [(hM_j, "Jurai"), (hM_f, "Shintaro")],
                  kc_list, yscale=yscale)

