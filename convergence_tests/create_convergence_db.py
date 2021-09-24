from common import *
from scandb_mp import *
import yaml
import sys

"""
Args: output db path
"""

if __name__ == '__main__':
    mp = ModelParams(M=1.0, dM=1e-9, Imw=5.0, Rew=0.785398, delta=3.14159, eta=4.71239)
    kc_maxs = [10, 20]
    n_kcs = list(range(5,51))
    cutoffs = [None, 1e3, 1e4, 1e5]

    db = MPScanDB(sys.argv[1], fast_insert=True)

    for cutoff in cutoffs:
        for kc_max in kc_maxs:
            for n_kc in n_kcs:
                tag = "conv_test_kcmax_{}_cutoff_{}_nkc_{}".format(kc_max, cutoff, n_kc)
                desc = tag
                solver_class = "QuadratureSolver"
                heirarchy = 1
                db.add_sample(mp, tag, desc, solver_class, n_kc, kc_max, heirarchy, cutoff)
