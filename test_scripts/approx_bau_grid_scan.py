from os import path
from common import *
from solvers import AveragedSolver, TrapezoidalSolver
import csv

def bau_grid_scan():
    M = 0.7732
    delta = -2.199
    eta = -1.857
    Rew = 2.444

    dM_min = -14
    dM_max = -2
    dMs = [10**e for e in np.linspace(dM_min, dM_max, 50)]

    points_per_dim = 50
    Imw_min = -6.
    Imw_max = 6.
    Imws = np.linspace(Imw_min, Imw_max, points_per_dim)

    output_dir = path.abspath(path.join(path.dirname(__file__), 'output/'))

    plot_no = 0
    plot_max = 50**2

    with open(path.join(output_dir, "bau_avg.csv"), "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        for ix, dM in enumerate(dMs):
            for Imw in Imws:
                mp = ModelParams(M,dM,Imw, Rew, delta, eta)
                print("Plot {} / {}".format(plot_no, plot_max))
                print(mp)
                T0 = get_T0(mp)
                print("T0 = {}".format(T0))

                avg_solver = AveragedSolver(mp, T0, Tsph, 2)
                avg_solver.solve()
                bau = (28./78.)*avg_solver.get_final_asymmetry()
                print("BAU: {}".format(bau))

                writer.writerow([M,dM,Imw, Rew, delta, eta, bau])
                plot_no += 1





