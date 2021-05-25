from common import *
from os import path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from leptotools.momentumDep import interpHFast, interpFast
from rates import Rates_Fortran, Rates_Jurai
from solvers import *



mp = ModelParams(M=1.0, dM=1e-11, Imw=3.0, Rew=0.7853981633974483, delta=3.141592653589793,
                 eta=4.71238898038469)
kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.9, 2.5, 3.1, 3.9, 5.0, 10.0])
# kc_list = np.array([0.5, 1.0, 2.0])

TF = Tsph

if __name__ == "__main__":

    rates_f = Rates_Fortran(mp,1)
    rates_j = Rates_Jurai(mp, 1, kc_list)

    solver_f = TrapezoidalSolverCPI(kc_list,
        model_params=mp, rates=rates_f, TF=TF,  H=1, fixed_cutoff=None, eig_cutoff=False,
        method="Radau", ode_pars={'atol' : 1e-15, 'rtol' : 1e-6}, source_term=False)

    solver_j = TrapezoidalSolverCPI(kc_list,
        model_params=mp, rates=rates_j, TF=TF, H=1, fixed_cutoff=None, eig_cutoff=False,
        method="Radau", ode_pars={'atol': 1e-15, 'rtol': 1e-6}, source_term=False)

    print("Solving with Fortran rates")
    solver_f.solve(eigvals=False)

    print("Sovling with Jurai's rates")
    solver_j.solve(eigvals=False)

    fig, axs = plt.subplots(2, sharex=True)

    X = solver_f.get_Tlist()
    bau_f = (28./78.) * np.abs(solver_f.get_total_lepton_asymmetry())
    bau_j = (28./78.) * np.abs(solver_j.get_total_lepton_asymmetry())
    rdiff = np.abs((bau_f - bau_j)/(bau_f + bau_j))

    # Final baus, rdiff
    bau_final_f = (28./78.) * solver_f.get_final_lepton_asymmetry()
    bau_final_j = (28. / 78.) * solver_j.get_final_lepton_asymmetry()
    rdiff_final = np.abs((bau_final_f - bau_final_j)/(bau_final_j + bau_final_f))

    title = "BAU (F): {:.3e}, BAU (J): {:.3e}, RDIFF@TSPH: {:.3f}, {} modes".format(
        bau_final_f, bau_final_j, rdiff_final, kc_list.shape[0]
    )

    # Lepton asymmetry plot
    axs[0].set_yscale("log")
    axs[0].set_xscale("log")
    axs[0].set_ylabel("BAU")
    axs[0].plot(X, bau_f, label="Fortran")
    axs[0].plot(X, bau_j, label="Jurai")
    axs[0].legend()
    
    # Plot rdiff
    axs[1].set_yscale("linear")
    axs[0].set_xscale("log")
    axs[1].set_ylabel("rdiff")
    axs[1].plot(X, rdiff)

    fig.supxlabel("T")
    fig.suptitle(title)

    plt.tight_layout()
    plt.show()


