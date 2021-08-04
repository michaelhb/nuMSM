from scanner import scan
from common import *
from os import path
import numpy as np

if __name__ == "__main__":
    db_name = "grid_scan_3d.db"
    output_dir = path.abspath(path.join(path.dirname(__file__), 'output/'))
    db_path = path.join(output_dir, db_name)

    # Fixed parameters for the test
    H_ = 1
    M_ = 1.0
    dM_ = 1e-9
    # Imw_ = imw(1e-5, M_, H=H_)
    Imw_ = 0.5

    # Generate the grid
    axis_size = 30
    Rew_list = np.linspace(0, 2*np.pi, axis_size)
    delta_list = np.linspace(0, 2*np.pi, axis_size)
    eta_list = np.linspace(0, 2*np.pi, axis_size)

    # Create sample list
    samples = []

    for Rew_ in Rew_list:
        for delta_ in delta_list:
            for eta_ in eta_list:
                samples.append(
                    ModelParams(M=M_, dM=dM_, Imw=Imw_, Rew=Rew_, delta=delta_, eta=eta_)
                )

    # Run the scan
    scan(samples, db_path, "grid_scan_3d_avg_x30NH_smallIMW", H=H_, avg=True)
