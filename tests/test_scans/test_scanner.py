import generate_samples
from scanner import scan
from nuMSM_solver.common import *
from os import path
import numpy as np

if __name__ == "__main__":
    db_name = "test.db"
    output_dir = path.abspath(path.join(path.dirname(__file__), 'output/'))
    db_path = path.join(output_dir, db_name)

    fixed_params = ModelParams(M=1,dM=1e-12,Imw=None,Rew=0.25*np.pi,delta=np.pi,eta=(3./2.)*np.pi)
    ranges = ModelParams(M=None,dM=None,Imw=(0,6),Rew=None,delta=None,eta=None)

    num_samples=6
    seed=123

    samples = generate_samples.random(fixed_params, ranges, num_samples, seed)

    res = scan(samples, db_path, "test_avg",H=1,avg=True)
    print(res)