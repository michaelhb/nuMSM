"""
Args: yaml file, output db path
"""
from scandb_mp import *
import yaml
import sys

if __name__ == '__main__':

    delta_opt = np.pi
    eta_opt = (3.0*np.pi)/2.0
    rew_opt = np.pi/4.0

    # Load yaml file
    stream = open(sys.argv[1], 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)

    # Instantiate DB
    db = MPScanDB(sys.argv[2], fast_insert=True)

    for tag, attrs in config.items():
        mp = ModelParams(
            M=float(attrs["M"]), dM=float(attrs["dM"]), Imw=float(attrs["Imw"]),
            Rew=rew_opt, delta=delta_opt, eta=eta_opt
        )
        sample = sample(**mp, tag=tag, description=attrs["description"], solvername=attrs["solver_class"],
                        n_kc=attrs["n_kc"], kc_max=attrs["kc_max"], heirarchy=attrs["heirarchy"],
                        cutoff=float(attrs["cutoff"]))
        db.add_sample(sample)