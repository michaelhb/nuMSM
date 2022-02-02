from scandb_mp import *
import sys

"""
Args: output db path
"""

if __name__ == '__main__':

    delta_opt = np.pi
    eta_opt = (3.0*np.pi)/2.0
    Rew_opt = np.pi/4.0

    tag = "cutoff_test"
    n_kc = 15
    kc_max = 10

    cutoffs = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
    dMs = np.logspace(-11,-2,5)
    Imws = np.linspace(0,7,5)

    db = MPScanDB(sys.argv[1], fast_insert=True)

    for dM in dMs:
        for Imw in Imws:
            for cutoff in cutoffs:
                desc = tag
                solver_class = "QuadratureSolver"
                heirarchy = 1
                mp = ModelParams(M=1.0, dM=dM, Imw=Imw, Rew=Rew_opt, delta=delta_opt, eta=eta_opt)
                sample = Sample(**mp._asdict(), tag=tag, description=desc, solvername=solver_class, n_kc=n_kc, kc_max=kc_max,
                                heirarchy=heirarchy, cutoff=cutoff)
                db.add_sample(sample)