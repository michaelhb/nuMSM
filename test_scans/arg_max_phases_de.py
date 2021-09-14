import generate_samples
from scanner import scan
from common import *
from os import path
from solvers import *
from rates import Rates_Jurai, Rates_Fortran
import numpy as np
from scipy.optimize import differential_evolution
from quadrature import GaussianQuadrature
from os import environ
environ["MKL_NUM_THREADS"] = "1"
environ["NUMEXPR_NUM_THREADS"] = "1"
environ["OMP_NUM_THREADS"] = "1"
environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

import warnings
warnings.filterwarnings(
    action='ignore',
    module=r'.*leptotools.*'
)
warnings.filterwarnings(
    action='ignore',
    module=r'.*common.*'
)

n_kc = 10
kc_min = 0
kc_max = 10

if __name__ == "__main__":
    # Fixed parameters / hyperparameters for the test
    H_ = 2
    M_ = 1.0
    dM_ = 1e-9
    # Imw_ = imw(1e-5, M_, H=H_)
    Imw_ = 0.5
    cutoff = None
    ode_pars = {'atol': 1e-20, 'rtol': 1e-4}
    # ode_pars = {'atol' : 1e-15, 'rtol' : 1e-8}

    # Bounds on parameter space
    bounds = [(0, 2*np.pi),(0, 2*np.pi),(0, 2*np.pi)]

    # Objective function. X = [Rew,delta,eta]
    def obj(x):
        Rew_, delta_, eta_ = x
        print("Starting point: {}".format(x))
        mp = ModelParams(M=M_,dM=dM_,Imw=Imw_,Rew=Rew_,delta=delta_,eta=eta_)
        # rates = Rates_Jurai(mp, H_, np.array([1.0]), tot=True)
        rates = Rates_Fortran(mp, H_)
        # solver = AveragedSolver(model_params=mp, rates_interface=rates, TF=Tsph, H=H_, fixed_cutoff=cutoff,
        #                         eig_cutoff=False, method="Radau", ode_pars=ode_pars, source_term=True)
        quadrature = GaussianQuadrature(n_kc, kc_min, kc_max, mp, H_, tot=True, qscheme="legendre")
        solver = QuadratureSolver(quadrature,
                                  model_params=mp, TF=Tsph, H=H_, fixed_cutoff=cutoff, eig_cutoff=False,
                                  method="Radau", ode_pars=ode_pars, source_term=True)
        solver.solve(eigvals=False)

        # Use -abs(bau) since differential_evolution always minimizes
        bau = (28./78.) * solver.get_final_lepton_asymmetry()
        print("Finished {}, bau = {}".format(x, bau))
        return -np.abs(bau)

    # Expected phases
    if H_ == 1:
        xRew = 0.25*np.pi
        xdelta = np.pi
        xeta = (3./2.)*np.pi
    elif H_ == 2:
        xRew = 0.25*np.pi
        xdelta = 0
        xeta = 0.5*np.pi

    x0_ = [xRew, xdelta, xeta]

    # Run the optimiser in parallel, using all cores
    # result = differential_evolution(obj, bounds, updating='deferred', popsize=30, workers=-1,x0=x0_)
    # result = differential_evolution(obj, bounds, updating='deferred', popsize=30, workers=-1)
    result = differential_evolution(obj, bounds, updating='deferred', popsize=30, mutation=0.3, recombination=0.5, init='halton', workers=-1)

    print(result)

    # Compare to expected phase values
    rRew, rdelta, reta = result.x

    print("Fixed params: M = {}, dM = {}, Imw = {}".format(M_, dM_, Imw_))
    print("Max BAU: {}".format(result.fun))
    print("Rew: {}, {}*expected".format(rRew, rRew/xRew))
    if xdelta != 0:
        print("Delta: {}, {}*expected".format(rdelta, rdelta/xdelta))
    else:
        print("Delta: expected 0, got {}".format(rdelta))
    print("eta: {}, {}*expected".format(reta, reta/xeta))