from scandb_mp import *
import numpy as np
import sys
from itertools import groupby
from collections import namedtuple

# Because I didn't tag the results properly :((
def group_results(results):

    def my_hash(result):
        sample = result[0]
        return round(sample.M + np.log(sample.dM) + sample.Imw + sample.Rew + sample.delta + sample.eta, 5)

    results = sorted(results, key=my_hash)
    grouped = []

    for k, g in groupby(results, my_hash):
        g = list(g)
        if len(g) == 1:
            print("Missed pair for sample: ", g[0])
        elif len(g) > 2:
            raise Exception("Hash collision!")
        else:
            grouped.append(list(g))

    # Consistent ordering
    for g in grouped:
        if g[0][0].solvername == "QuadratureSolver":
            temp = g[1]
            g[1] = g[0]
            g[0] = temp

    return grouped

PointResult = namedtuple('PointResult', [
    "M", "dM", "Imw", "Rew", "delta", "eta", "bau_avg", "bau_quad", "bau_reldiff_quad"
])
def process_results(grouped):
    res = []

    for g in grouped:
        M = g[0][0].M
        dM = g[0][0].dM
        Imw = g[0][0].Imw
        Rew = g[0][0].Rew
        delta = g[0][0].delta
        eta = g[0][0].eta
        bau_avg = g[0][1]
        bau_quad = g[1][1]

        if bau_avg == 0 or bau_quad == 0:
            print("=== Zero bau here: ===")
            print(g)
            print("===")
            continue

        bau_reldiff_quad = np.abs((bau_quad - bau_avg)/bau_avg)

        res.append(PointResult(
            M=M,dM=dM,Imw=Imw,Rew=Rew,delta=delta,eta=eta,bau_avg=bau_avg,
            bau_quad=bau_quad,bau_reldiff_quad=bau_reldiff_quad))

    return sorted(res, key = lambda r: r.bau_reldiff_quad, reverse=True)

def summarise_results(processed):
    for r in processed:
        print("Reldiff={}, bau_avg={:.2e}, bau_quad={:.2e}, M={}, dM={:.2e}, Imw={:.2f}, Rew={:.2f}, delta={:.2f},eta={:.2f}".format(
            r.bau_reldiff_quad, r.bau_avg, r.bau_quad, r.M, r.dM, r.Imw, r.Rew, r.delta, r.eta
        ))

def get_from_db(db_path):
    db = MPScanDB(input_db, fast_insert=True)
    return process_results(group_results(db.get_all_results()))

if __name__ == '__main__':
    input_db = sys.argv[1]

    # Instantiate DB
    db = MPScanDB(input_db, fast_insert=True)

    # Get results
    results = process_results(group_results(db.get_all_results()))

    # Print em
    summarise_results(results)
