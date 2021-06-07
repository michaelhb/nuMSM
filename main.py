from os import path, environ
environ["MKL_NUM_THREADS"] = "1"
environ["NUMEXPR_NUM_THREADS"] = "1"
environ["OMP_NUM_THREADS"] = "1"
environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")
from solvers import *
import time
from quadrature import TrapezoidalQuadrature, GaussFermiQuadrature
import cProfile
from rates import Rates_Fortran, Rates_Jurai
# #
mp = ModelParams(
    M=1.0,
    dM=1e-12,
    # dM=0,
    Imw=4.1,
    Rew=1/4 * np.pi,
    delta= np.pi,
    eta=3/2 * np.pi
)

if __name__ == '__main__':
    # kc_list = [0.3, 0.4] + [0.1 * kc for kc in range(5, 101)]
    # kc_list = np.array([0.5, 1.0, 2.0])
    # kc_list = np.array([1.0])
    # kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.9, 2.5, 3.1, 3.9, 5.0, 10.0])
    # kc_list = np.array([0.8, 1.6, 2.4, 3.2, 4. , 4.8, 5.6, 6.4, 7.2, 8.])
    # kc_list = np.array([0.5, 1.0, 1.5, 2.5, 5.0])
    # kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1,
    #             3.3, 3.6, 3.9, 4.2, 4.6, 5.0, 5.7, 6.9, 10.0])

    cutoff = 1e5
    eig = False
    use_source_term = False
    TF = Tsph
    H = 1
    ode_pars = {'atol': 1e-15, 'rtol': 1e-6}
    # ode_pars = {'atol': 1e-13, 'rtol': 1e-4}

    quadrature = GaussFermiQuadrature(5, mp, H, tot=True)

    kc_list = np.array(quadrature.kc_list())
    # rates = Rates_Fortran(mp,1)

    rates = Rates_Jurai(mp, H, kc_list, tot=True)
    quadrature = TrapezoidalQuadrature(kc_list, rates)

    # rates = None


    # solver = AveragedSolver(model_params=mp, rates_interface=rates, TF=TF, H=1, eig_cutoff=False,
    #                         ode_pars=ode_pars, source_term=use_source_term)
    solver = TrapezoidalSolverCPI(quadrature,
        model_params=mp, rates_interface=rates, TF=TF,  H=2, fixed_cutoff=cutoff, eig_cutoff=eig,
        method="Radau", ode_pars=ode_pars, source_term=use_source_term)

    start = time.time()
    solver.solve(eigvals=True)
    end = time.time()
    bau = (28./78.) * solver.get_final_lepton_asymmetry()
    title = "M = {}, dM = {:.3e}, Imw = {:.2f}, n_kc = {}, cutoff = {}, BAU = {:.3e}".format(
        mp.M, mp.dM, mp.Imw, kc_list.shape[0], str(cutoff), bau
    )
    print("Time (solve): {}".format(end - start))
    solver.plot_eigenvalues(title)
    # solver.plot_eigenvalues(title, use_z=True)
    solver.plot_total_lepton_asymmetry(title)
    # solver.plot_total_hnl_asymmetry(title)
    # solver.plot_L_violation()
    print("BAU: {:.3e}".format(bau))

    print(solver.get_total_hnl_asymmetry() / solver.get_total_lepton_asymmetry())
