"""
Args: yaml file, output db path
"""
from common import *
from scandb_mp import *
import yaml
import sys

if __name__ == '__main__':
    # Load yaml file
    stream = open(sys.argv[1], 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)

    # Instantiate DB
    db = MPScanDB(sys.argv[2], fast_insert=True)

    delta_opt = np.pi
    eta_opt = (3.0*np.pi)/2.0
    Rew_opt = np.pi/4.0

    n_kcs = list(range(15,105,5))

    for tag, attrs in config.items():
        desc = attrs["description"]
        H = int(attrs["heirarchy"])
        M = float(attrs["M"])
        dM = float(attrs["dM"])
        Imw = float(attrs["Imw"])
        kc_max = float(attrs["kc_max"])

        cutoff = attrs["cutoff"]
        if cutoff is not None:
            cutoff = float(cutoff)

        for n_kc in n_kcs:
            mp = ModelParams(M=M, dM=dM, Imw=Imw, Rew=Rew_opt, delta=delta_opt, eta=eta_opt)

            sample = Sample(**mp._asdict(), tag=tag, description=desc, solvername="QuadratureSolver",
                            n_kc=n_kc, kc_max=kc_max, heirarchy=H, cutoff=cutoff)

            db.add_sample(sample)
