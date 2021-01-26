import numpy as np
from os import path
import matplotlib.pyplot as plt
from common import *
from solvers import AveragedSolver, TrapezoidalSolver

def total_asymmetry_plots():
    points_per_dim = 10

    M = 1.0
    delta = np.pi
    eta = 3*np.pi/2
    Rew = np.pi/4

    dM_min = -17
    dM_max = -8
    dM_exp = list(range(dM_min, dM_max)) # for file names
    dMs = [10**e for e in range(dM_min, dM_max)]

    Imw_min = -6.9
    Imw_max = 6.9
    Imws = np.linspace(Imw_min, Imw_max, points_per_dim)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})

    output_dir = path.abspath(path.join(path.dirname(__file__), 'output/'))

    plot_num = 1

    for ix, dM in enumerate(dMs):
        for Imw in Imws:
            mp = ModelParams(M, dM,Imw, Rew, delta, eta)
            print(mp)
            T0 = get_T0(mp)
            print("T0 = {}".format(T0))

            avg_solver = AveragedSolver(mp, T0, Tsph)
            avg_solver.solve()
            avg_final = avg_solver.get_final_asymmetry()
            print("Total asymmetry (averaged): {}".format(avg_final))
            asymmetry_avg = avg_solver.get_total_asymmetry()

            trap_solver = TrapezoidalSolver(mp, T0, Tsph)
            trap_solver.solve()
            trap_final = trap_solver.get_final_asymmetry()
            print("Total asymmetry (trapezoidal): {}".format(trap_final))
            asymmetry_trap = trap_solver.get_total_asymmetry()

            Tlist = trap_solver.Tz(np.linspace(trap_solver.zT(T0), trap_solver.zT(Tsph), 200))
            plt.loglog(Tlist, asymmetry_avg, label="Avg: {:.3e}".format(avg_final))
            plt.loglog(Tlist, asymmetry_trap, label="Trap: {:.3e}".format(trap_final))
            plt.legend()
            plt.title("M = {:.3}, $\delta$ = {:.3}, $\eta$ = {:.3}, " \
                      "Re($\omega$) = {:.3}, Im($\omega$) = {:.3}, dM = {:.3e}".format(
                M, delta, eta, Rew, Imw, dM
            ))

            fname = path.join(output_dir, "{}_avg_trap_comp_imw_{:.3}_dM_{:3}.png".format(plot_num, Imw, dM_exp[ix]))
            plt.savefig(fname, dpi=300)
            # plt.show()
            plt.clf()

            plot_num += 1
