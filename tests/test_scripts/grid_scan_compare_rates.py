from os import environ
environ["MKL_NUM_THREADS"] = "1"
environ["NUMEXPR_NUM_THREADS"] = "1"
environ["OMP_NUM_THREADS"] = "1"
environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")
from nuMSM_solver.solvers import *
from multiprocessing import get_context
from scandb import ScanDB
from rates import Rates_Fortran, Rates_Jurai
from plots import bau_rdiff_heatmap

import warnings
warnings.filterwarnings("ignore")

kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.9, 2.5, 3.1, 3.9, 5.0, 10.0])

def get_scan_points(points_per_dim, M, delta, eta, Rew):
    dM_min = -16
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
    dM, Imw, mp, scan_db_path, tag = point

    print("Starting {} point {}".format(tag, mp))

    if tag == "fortran":
        rates = Rates_Fortran(mp,1)
    elif tag == "jurai":
        rates = Rates_Jurai(mp, 1, kc_list)

    # if (np.abs(Imw) > 5.0):
    #     fixed_cutoff = None
    # else:
    #     fixed_cutoff = 1e4
    fixed_cutoff = 1e4

    solver = QuadratureSolver(kc_list,
                              model_params=mp, rates=rates, TF=Tsph, H=1, fixed_cutoff=fixed_cutoff, eig_cutoff=False,
                              method="Radau", ode_pars={'atol' : 1e-15, 'rtol' : 1e-6}, source_term=False)

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
    # set_start_method("spawn", force=True)
    M = 1.0
    delta = np.pi
    eta = 3/2 * np.pi
    Rew = 1/4 * np.pi

    output_dir = path.abspath(path.join(path.dirname(__file__), 'output/'))
    db_path = path.join(output_dir, "scan_compare_rates.db")
    db = ScanDB(db_path)
    axsize = 15

    points = get_scan_points(axsize, M, delta, eta, Rew)
    res_cache_fortran = []
    res_cache_jurai = []
    points_scan_fortran = []
    points_scan_jurai = []

    for point in points:
        dm, Imw, mp = point
        bau_fortran, time_sol_fortran = db.get_bau(mp, "fortran")
        if bau_fortran is None:
            points_scan_fortran.append((*point, db_path, "fortran"))
        else:
            print("Got from fortran cache: {}".format(mp))
            print("BAU FORTRAN: {}".format(bau_fortran))
            res_cache_fortran.append((bau_fortran, mp.dM, mp.Imw, time_sol_fortran))

        bau_jurai, time_sol_jurai = db.get_bau(mp, "jurai")
        if bau_jurai is None:
            points_scan_jurai.append((*point, db_path, "jurai"))
        else:
            print("Got from jurai cache: {}".format(mp))
            res_cache_jurai.append((bau_jurai, mp.dM, mp.Imw, time_sol_jurai))

    db.close_conn() # No longer need this connection

    with get_context("spawn").Pool() as p:
        res_scan_jurai = p.map(get_bau, points_scan_jurai)
        res_scan_fortran = p.map(get_bau, points_scan_fortran)

    res_fortran = sorted(res_cache_fortran + res_scan_fortran, key=lambda r: (r[1], r[2]))
    res_jurai = sorted(res_cache_jurai + res_scan_jurai, key=lambda r: (r[1], r[2]))

    outfile_comp = path.join(output_dir, "grid_scan_dm_imw_new_rates_rdiff.png")
    bau_rdiff_heatmap(res_fortran, res_jurai, axsize, "Fortran vs. new rates: rdiff", outfile_comp)

    print("DONE!")