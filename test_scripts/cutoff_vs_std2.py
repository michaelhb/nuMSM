from os import path, environ
environ["MKL_NUM_THREADS"] = "1"
environ["NUMEXPR_NUM_THREADS"] = "1"
environ["OMP_NUM_THREADS"] = "1"
from solvers import *
import time
from multiprocessing import Pool
from collections import namedtuple
import hashlib
import pickle

import sys

Point = namedtuple("Point",
    ["cutoff", "mp"]
)

Result = namedtuple("Result",
    ["cutoff", "dM", "bau", "time"]
)

def run_point(point):
    print("Starting point: {}".format(point))
    T0 = get_T0(point.mp)
    solver = TrapezoidalSolverCPI(point.mp, T0, Tsph, kc_list, H=1, fixed_cutoff=point.cutoff, method="Radau", ode_pars={'atol': 1e-9})
    start = time.time()
    solver.solve()
    end = time.time()
    res = Result(point.cutoff, point.mp.dM, (28./78.) * solver.get_final_lepton_asymmetry(), end - start)
    print("Finished: {}".format(res))
    return res

if __name__ == '__main__':
    # kc_list = np.array([0.5, 1.0, 1.5, 2.5, 5.0])
    kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1,
                3.3, 3.6, 3.9, 4.2, 4.6, 5.0, 5.7, 6.9, 10.0])
    dMs = np.geomspace(1e-13, 1e-8, 10)
    # dMs = np.array([1e-13, 1e-12, 1e-11])
    cutoff = 1e4

    # Fixed parameters
    M = 1.0
    Imw = 0.0
    Rew = 0.7853981633974483
    delta = 3.141592653589793
    eta = 4.71238898038469

    # Hash key for std results
    hash_items = [kc_list,dMs,M,Imw,Rew,delta,eta]
    hash_str = "".join(map(str,hash_items))
    print(hash_str)
    hash = hashlib.md5(hash_str.encode()).hexdigest()
    print("HASH: {}".format(hash))

    # Try to retrieve std results
    cache_dir = path.abspath(path.join(path.dirname(__file__), 'output/cache/'))
    cache_file = path.join(cache_dir, hash)

    if path.exists(cache_file):
        print("Using cache for std results")
        use_cache = True
        results_std = pickle.load(open(cache_file, 'rb'))
    else:
        print("No cached std results found")
        use_cache = False

    points = []

    for dM in dMs:
        mp = ModelParams(M, dM, Imw, Rew, delta, eta)
        T0 = get_T0(mp)

        if not use_cache: points.append(Point(None, mp))
        points.append(Point(cutoff, mp))

    with Pool(8) as p:
        results = p.map(run_point, points)

    if not use_cache:
        results_std = sorted(filter(lambda r: r.cutoff == None, results), key=lambda r: r.dM)
        pickle.dump(results_std, open(cache_file, 'wb'))

    results_cutoff = sorted(filter(lambda r: r.cutoff != None, results), key=lambda r: r.dM)

    title = "{} modes, M = {}, Imw = {}, Cutoff= {}".format(kc_list.shape[0], mp.M, mp.Imw, cutoff)

    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle(title)
    fig.supxlabel("dM")

    # Plot BAUs
    baus_std = np.array([np.abs(r.bau) for r in results_std])
    baus_cutoff = np.array([np.abs(r.bau) for r in results_cutoff])

    print("std")
    print(baus_std)
    print("cutoff")
    print(baus_cutoff)

    axs[0].set_yscale('log')
    axs[0].set_xscale('log')
    axs[0].set_ylabel("BAU")
    axs[0].scatter(dMs, np.abs(baus_cutoff), color="green", alpha=0.5, label="cutoff", s=5)
    axs[0].scatter(dMs, np.abs(baus_std), color="purple", alpha=0.5, label="std picture", s=5)
    axs[0].legend()

    # Plot residuals
    residuals = np.abs(baus_std - baus_cutoff)/baus_std
    axs[1].set_xscale('log')
    axs[1].set_ylabel("abs(std-cutoff)/std")
    axs[1].scatter(dMs, residuals, color="black", s=5)

    # Plot times
    times_std = [r.time for r in results_std]
    times_cutoff = [r.time for r in results_cutoff]
    axs[2].set_xscale('log')
    axs[2].set_ylabel("time (s)")
    axs[2].scatter(dMs, times_cutoff, color="green", alpha=0.5, label="cutoff", s=5)
    axs[2].scatter(dMs, times_std, color="purple", alpha=0.5, label="std picture", s=5)

    plt.tight_layout()

    plt.show()

