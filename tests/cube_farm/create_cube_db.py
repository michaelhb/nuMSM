"""
Args: axis size, yaml file, output db path
"""
from common import *
from scandb_mp import *
import yaml
import sys

if __name__ == '__main__':
    axis_size = int(sys.argv[1])

    # Load yaml file
    stream = open(sys.argv[2], 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)

    # Instantiate DB
    db = MPScanDB(sys.argv[3], fast_insert=True)

    # Create and load cube samples
    Rew_list = np.linspace(0, 2*np.pi, axis_size)
    delta_list = np.linspace(0, 2*np.pi, axis_size)
    eta_list = np.linspace(0, 2*np.pi, axis_size)

    for tag, attrs in config.items():
        for Rew_ in Rew_list:
            for delta_ in delta_list:
                for eta_ in eta_list:
                    mp = ModelParams(
                        M=float(attrs["M"]), dM=float(attrs["dM"]), Imw=float(attrs["Imw"]),
                        Rew=Rew_, delta=delta_, eta=eta_
                    )
                    sample = Sample(**mp._asdict(), tag=tag, description=attrs["description"], solvername=attrs["solver_class"],
                                    n_kc=attrs["n_kc"], kc_max=attrs["kc_max"], heirarchy=attrs["heirarchy"],
                                    cutoff=float(attrs["cutoff"]))
                    db.add_sample(sample)