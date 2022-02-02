import generate_samples
from scanner import scan
from nuMSM_solver.common import *
from os import path
import numpy as np

if __name__ == "__main__":
    db_name = "arg_max_phases.db"
    output_dir = path.abspath(path.join(path.dirname(__file__), 'output/'))
    db_path = path.join(output_dir, db_name)

    # Fixed parameters for the test
    M_ = 1.0
    dM_ = 1e-10
    Imw_ = imw(1e-5, M_, H=1)
    fixed_params = ModelParams(M=M_,dM=dM_,Imw=Imw_,Rew=None,delta=None,eta=None)

    # Sampling ranges for free parameters
    rn = (0, 2*np.pi)
    ranges = ModelParams(M=None,dM=None,Imw=None,Rew=rn,delta=rn,eta=rn)

    # Expected phase values for argmax c.f. BAU
    xRew = 0.25*np.pi
    xdelta = np.pi
    xeta = (3./2.)*np.pi

    # Generate the samples
    # n_samples=250
    n_samples=5000
    samples = generate_samples.random(fixed_params, ranges, n_samples, 789)

    # Run the scan
    res = scan(samples, db_path, "arg_max_phases_avg", H=1, avg=True)

    # Maximise the BAU & record phases
    res = sorted(res, key = lambda r: abs(r[1]), reverse=True)

    # Compare to expected phase values & report results
    print("Number of samples: {}".format(n_samples))
    print("Fixed params: M = {}, dM = {}, Imw = {}".format(M_, dM_, Imw_))
    print("Max BAU: {}".format(res[0][1]))
    print("Rew: {}, {}*expected".format(res[0][0].Rew, res[0][0].Rew/xRew))
    print("Delta: {}, {}*expected".format(res[0][0].delta, res[0][0].delta/xdelta))
    print("eta: {}, {}*expected".format(res[0][0].eta, res[0][0].eta/xeta))
