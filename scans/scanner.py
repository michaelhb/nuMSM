from common import *
from solvers import *
from rates import Rates_Jurai, Rates_Fortran
from quadrature import GaussianQuadrature
from scandb import *

from multiprocessing import Pool
from os import environ
environ["MKL_NUM_THREADS"] = "1"
environ["NUMEXPR_NUM_THREADS"] = "1"
environ["OMP_NUM_THREADS"] = "1"
environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

import warnings
warnings.filterwarnings(
    action='ignore',
    module=r'.*leptotools.*'
)

n_kc = 10
kc_min = 0
kc_max = 10
cutoff = None

ode_pars={'atol': 1e-20, 'rtol': 1e-4}

def get_bau(point):
    mp, db_path, scan_key, H, avg = point
    print("Starting: {}".format(mp))
    if avg:
        rates = Rates_Jurai(mp, H, np.array([1.0]), tot=True)
        # rates = Rates_Fortran(mp, 1.0)
        solver = AveragedSolver(model_params=mp, rates_interface=rates, TF=Tsph, H=H, fixed_cutoff=cutoff,
                                eig_cutoff=False, method="Radau", ode_pars=ode_pars, source_term=True)
    else:
        quadrature = GaussianQuadrature(n_kc, kc_min, kc_max, mp, H, tot=True, qscheme="legendre")
        solver = QuadratureSolver(quadrature,
                                  model_params=mp, TF=Tsph, H=H, fixed_cutoff=cutoff, eig_cutoff=False,
                                  method="Radau", ode_pars=ode_pars, source_term=True)

    start = time.time()
    solver.solve(eigvals=False)
    end = time.time()
    time_sol = end - start
    bau = (28./78.) * solver.get_final_lepton_asymmetry()
    print("Point {} finished in {} s, BAU = {}".format(mp, time_sol, bau))
    res_db = ScanDB(db_path)
    res_db.save_bau(mp, str(scan_key), bau, time_sol)
    res_db.close_conn()

    return mp, bau


def scan(samples, db_path, scan_key, H=1, avg=False, old_rates=False):
    db = ScanDB(db_path)
    tag = str(scan_key)

    scan_points = []
    res_cache = []

    for mp in samples:
        bau, time_sol = db.get_bau(mp, tag)

        if bau is None:
            scan_points.append((mp, db_path, scan_key, H, avg))
        else:
            print("Got from cache: {}".format(mp))
            res_cache.append((mp, bau))

    db.close_conn()  # No longer need this connection

    with Pool() as p:
        res_scan = p.map(get_bau, scan_points)

    print("Finished!")
    return res_cache + res_scan

