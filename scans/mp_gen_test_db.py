from common import *
from os import path
from scandb_mp import MPScanDB

if __name__ == "__main__":
    db_name = "mp_scan_test.db"
    output_dir = path.abspath(path.join(path.dirname(__file__), 'output/'))
    db_path = path.join(output_dir, db_name)

    # Fixed parameters for the test
    H_ = 1
    M_ = 1.0
    dM_ = 1e-9
    Imw_ = 0.5

    # Generate the grid
    axis_size = 30
    Rew_list = np.linspace(0, 2*np.pi, axis_size)
    delta_list = np.linspace(0, 2*np.pi, axis_size)
    eta_list = np.linspace(0, 2*np.pi, axis_size)

    # Create the sample list
    samples = []

    for Rew_ in Rew_list:
        for delta_ in delta_list:
            for eta_ in eta_list:
                samples.append(
                    ModelParams(M=M_, dM=dM_, Imw=Imw_, Rew=Rew_, delta=delta_, eta=eta_)
                )

    # Load DB up with samples
    db = MPScanDB(db_path)

    for mp in samples:
        db.add_sample(mp)