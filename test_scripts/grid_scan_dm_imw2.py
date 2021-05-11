from os import path, environ
environ["MKL_NUM_THREADS"] = "1"
environ["NUMEXPR_NUM_THREADS"] = "1"
environ["OMP_NUM_THREADS"] = "1"
from solvers import *
from multiprocessing import Pool
from plots import heatmap_dm_imw, contour_dm_imw, heatmap_dm_imw_timing
from scandb import ScanDB

kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1,
                3.3, 3.6, 3.9, 4.2, 4.6, 5.0, 5.7, 6.9, 10.0])

def get_scan_points(points_per_dim, M, delta, eta, Rew):
    dM_min = -14
    dM_max = -1
    dMs = [10**e for e in np.linspace(dM_min, dM_max, points_per_dim)]

    Imw_min = -6.
    Imw_max = 6.
    Imws = np.linspace(Imw_min, Imw_max, points_per_dim)

    points = []

    for dM in dMs:
        for Imw in Imws:
            points.append([dM, Imw, ModelParams(M, dM, Imw, Rew, delta, eta)])

    return np.array(points)

def get_bau(point):
    dM, Imw, mp, scan_db_path = point

    print("Starting point {}".format(mp))
    T0 = get_T0(mp)

    if (np.abs(Imw) > 4.0):
        solver = TrapezoidalSolverCPI(mp, T0, Tsph, kc_list, H=1, ode_pars={'atol': 1e-13}, method="Radau")
    else:
        solver = TrapezoidalSolverCPI(mp, T0, Tsph, kc_list, H=1, fixed_cutoff=1e4, ode_pars={'atol': 1e-10},
                                      method="Radau")
    start = time.time()
    solver.solve(eigvals=False)
    end = time.time()
    time_sol = end - start
    bau = (28./78.) * solver.get_final_lepton_asymmetry()
    print("Point {} finished in {} s, BAU = {}".format(mp, time_sol, bau))
    res_db = ScanDB(scan_db_path)
    res_db.save_bau(mp, tag, bau, time_sol)
    res_db.close_conn()
    return (bau, dM, Imw, time_sol)

if __name__ == '__main__':
    M = 10.0
    delta = np.pi
    eta = 3/2 * np.pi
    Rew = 1/4 * np.pi

    output_dir = path.abspath(path.join(path.dirname(__file__), 'output/'))
    db_path = path.join(output_dir, "scan.db")

    db = ScanDB(db_path)

    axsize = 30
    tag = "std"

    points = get_scan_points(axsize, M, delta, eta, Rew)
    res_cache = []
    points_scan = []

    for point in points:
        dm, Imw, mp = point
        bau, time_sol = db.get_bau(mp, tag)
        if bau is None:
            points_scan.append((*point,db_path))
        else:
            print("Got from cache: {}".format(mp))
            res_cache.append((bau, mp.dM, mp.Imw, time_sol))

    db.close_conn() # No longer need this connection

    with Pool(8) as p:
        res_scan = p.map(get_bau, points_scan)

    res = sorted(res_cache + res_scan, key=lambda r: (r[1], r[2]))

    outfile_heatmap = path.join(output_dir, "grid_scan_dm_imw_heatmap.png")
    outfile_contours = path.join(output_dir, "grid_scan_dm_imw_contours.png")
    outfile_heatmap_timings = path.join(output_dir, "grid_scan_dm_imw_timings.png")

    title = r'$M = 1.0$ GeV'

    heatmap_dm_imw(np.array(res), axsize, title, outfile_heatmap)
    heatmap_dm_imw_timing(np.array(res), axsize, title, outfile_heatmap_timings)
    contour_dm_imw(np.array(res), axsize, title, outfile_contours)

    print("Finished!")



