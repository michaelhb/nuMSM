from os import path, environ
from sys import argv
import yaml
environ["MKL_NUM_THREADS"] = "1"
environ["NUMEXPR_NUM_THREADS"] = "1"
environ["OMP_NUM_THREADS"] = "1"
environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")
from solvers import *
from multiprocessing import Pool, set_start_method
from plots import heatmap_dm_imw, contour_dm_imw, heatmap_dm_imw_timing, contour_dm_imw_comp
from quadrature import GaussianQuadrature
from scandb import ScanDB
from rates import Rates_Jurai

import warnings
warnings.filterwarnings(
    action='ignore',
    # category=DeprecationWarning,
    module=r'.*leptotools.*'
)

ode_pars={'atol': 1e-20, 'rtol': 1e-4}

# kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1,
#                 3.3, 3.6, 3.9, 4.2, 4.6, 5.0, 5.7, 6.9, 10.0])
kc_min = 0.0
kc_max = 10.0
n_kc = 20
cutoff = 1e4

def get_scan_points(points_per_dim, M, delta, eta, Rew, dM_min, dM_max):
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
    dM, Imw, mp, scan_db_path, tag, H = point

    print("Starting {} point {}".format(tag, mp))
    T0 = get_T0(mp)

    if "avg" in tag:
        kc_list = np.array([1.0]) # Dummy param, remove after refactor
        rates = Rates_Jurai(mp, H, kc_list, tot=True)
        solver = AveragedSolver(model_params=mp, rates_interface=rates, TF=Tsph, H=H, fixed_cutoff=cutoff, eig_cutoff=False,
                                ode_pars=ode_pars, source_term=False)
    else:
        quadrature = GaussianQuadrature(n_kc, kc_min, kc_max, mp, H, tot=True, qscheme="legendre")
        solver = QuadratureSolver(quadrature,
                                  model_params=mp, TF=Tsph, H=H, fixed_cutoff=cutoff, eig_cutoff=False,
                                  method="Radau", ode_pars={'atol': 1e-20, 'rtol': 1e-4}, source_term=True)

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
    # Args: input yaml, db_name, axsize
    yaml_file = argv[1]
    db_name = argv[2]
    axsize = int(argv[3])

    yaml_path = path.abspath(path.join(path.dirname(__file__), yaml_file))
    with open(yaml_path) as file:
        doc = yaml.load(file, Loader=yaml.FullLoader)
        M = doc["M"]
        delta = doc["delta"]
        eta = doc["eta"]
        Rew = doc["rew"]
        H = int(doc["H"])
        tag = doc["tag"]
        dM_min = doc["dm_min"]
        dM_max = doc["dm_max"]

    output_dir = path.abspath(path.join(path.dirname(__file__), 'output/'))
    db_path = path.join(output_dir, db_name)
    db = ScanDB(db_path)

    points = get_scan_points(axsize, M, delta, eta, Rew, dM_min, dM_max)
    res_cache = []
    points_scan = []

    for point in points:
        dm, Imw, mp = point
        bau, time_sol = db.get_bau(mp, tag)
        if bau is None:
            points_scan.append(((*point, db_path, tag, H)))
        else:
            print("Got from cache: {}".format(mp))

    db.close_conn()  # No longer need this connection

    with Pool() as p:
        res_scan = p.map(get_bau, points_scan)

    res = sorted(res_cache + res_scan, key=lambda r: (r[1], r[2]))

    print("Finished!")
    print(res)