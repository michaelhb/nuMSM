from solvers import *
from multiprocessing import Pool
from os import path
from plots import heatmap_dm_imw, contour_dm_imw
import csv

# kc_list = np.array([0.5, 1.0, 1.3, 1.5,  1.9, 2.5, 3.1, 3.9, 5.0, 10.0])
# kc_list = np.array([0.5, 1.0, 1.5, 2.5, 5.0])
kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1,
                3.3, 3.6, 3.9, 4.2, 4.6, 5.0, 5.7, 6.9, 10.0])
# kc_list = [0.3, 0.4] + [0.1 * kc for kc in range(5, 101)]

# Returns tuples (dm, imw, ModelParams)
def get_scan_points(points_per_dim):
    M = 1.0
    delta = np.pi
    eta = 3/2 * np.pi
    Rew = 1/4 * np.pi

    dM_min = -14
    dM_max = -1
    dMs = [10**e for e in np.linspace(dM_min, dM_max, points_per_dim)]

    print(dMs)

    Imw_min = -6.
    Imw_max = 6.
    Imws = np.linspace(Imw_min, Imw_max, points_per_dim)

    points = []

    for dM in dMs:
        for Imw in Imws:
            points.append([dM, Imw, ModelParams(M, dM, Imw, Rew, delta, eta)])

    return np.array(points)

def get_bau(point):
    dM, Imw, mp = point
    print("Starting point {}".format(mp))
    T0 = get_T0(mp)

    if dM < 1e-9:
        solver = AveragedSolver(mp, T0, Tsph, 1, eig_cutoff=False, ode_pars={'atol' : 1e-9})
    else:
        solver = AveragedSolver(mp, T0, Tsph, 1, eig_cutoff=True, ode_pars={'atol': 1e-9})

    # if dM < 1e-9:
    #     solver = TrapezoidalSolverCPI(mp, T0, Tsph, kc_list, H=1, eig_cutoff=False, method="BDF", ode_pars={'atol' : 1e-11})
    # elif dM < 1e-6:
    #     solver = TrapezoidalSolverCPI(mp, T0, Tsph, kc_list, H=1, eig_cutoff=True, method="BDF", ode_pars={'atol' : 1e-11})
    # else:
    #     solver = TrapezoidalSolverCPI(mp, T0, Tsph, kc_list, H=1, eig_cutoff=True, method="Radau", ode_pars={'atol': 1e-10})

    start = time.time()
    solver.solve(eigvals=False)
    end = time.time()
    bau = (28./78.) * solver.get_final_lepton_asymmetry()
    print("Point {} finished in {} s, BAU = {}".format(mp, end - start, bau))
    return (bau, dM, Imw)

if __name__ == '__main__':
    output_dir = path.abspath(path.join(path.dirname(__file__), 'output/'))
    outfile_data = path.join(output_dir, "grid_scan_dm_imw.csv")

    # axsize = 51
    axsize = 60

    if not path.exists(outfile_data):
        with Pool(8) as p:
            points = get_scan_points(axsize)
            res = p.map(get_bau, points)

        with open(outfile_data, mode="w") as csv_out:
            writer = csv.writer(csv_out, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for line in res:
                writer.writerow(line)
    else:
        with open(outfile_data, mode="r") as csv_in:
            reader = csv.reader(csv_in, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            res = []
            for row in reader:
                res.append(list(map(float, row)))
            res = np.array(res)

    outfile_heatmap = path.join(output_dir, "grid_scan_dm_imw_heatmap.png")
    outfile_contours = path.join(output_dir, "grid_scan_dm_imw_contours.png")

    title = r'$M = 1.0$ GeV'

    heatmap_dm_imw(np.array(res), axsize, title, outfile_heatmap)
    contour_dm_imw(np.array(res), axsize, title, outfile_contours)

    print("Finished!")
