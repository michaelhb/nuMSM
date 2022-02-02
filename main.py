from os import path, environ
environ["MKL_NUM_THREADS"] = "1"
environ["NUMEXPR_NUM_THREADS"] = "1"
environ["OMP_NUM_THREADS"] = "1"
environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")
from solvers import *
import time
from quadrature import TrapezoidalQuadrature, GaussFermiDiracQuadrature, GaussianQuadrature
import cProfile
from rates import Rates_Fortran, Rates_Jurai
# #


if __name__ == '__main__':

    delta_opt = np.pi
    eta_opt = (3.0*np.pi)/2.0
    rew_opt = np.pi/4.0

    mp = ModelParams(M=1.0, dM=5e-9, Imw=6, Rew=rew_opt, delta=delta_opt, eta=eta_opt)

    n_kc = 15
    kc_max = 10
    kc_list = np.linspace(0, kc_max, n_kc)
    cutoff = None
    eig = False
    use_source_term = True
    TF = 10.
    H = 1
    ode_pars = {'atol': 1e-20, 'rtol': 1e-4}
    # ode_pars = {'atol': 1e-15, 'rtol': 1e-8}

    # quadrature = GaussianQuadrature(10, 0.1, 10, mp, H, tot=True, qscheme="radau")
    quadrature = GaussianQuadrature(n_kc, 0, kc_max, mp, H, tot=True, qscheme="legendre")
    # rates = Rates_Jurai(mp, H, [1.0], tot=True)
    # rates = Rates_Fortran(mp,1)
    # quadrature = TrapezoidalQuadrature(kc_list, rates)

    # quadrature = GaussLegendreQuadrature(20, 0.1, 10, mp, H, tot=True)
    # kc_list = np.array(quadrature.kc_list())

    # solver = AveragedSolver(model_params=mp, rates_interface=rates, TF=TF, H=1, eig_cutoff=False,
    #                          ode_pars=ode_pars, source_term=use_source_term, method="Radau")
    solver = QuadratureSolver(quadrature,
                             model_params=mp, TF=TF, H=H, fixed_cutoff=cutoff, eig_cutoff=eig,
                             method="Radau", ode_pars=ode_pars, source_term=use_source_term)

    start = time.time()
    solver.solve(eigvals=True)
    end = time.time()
    bau = (28./79.) * solver.get_final_lepton_asymmetry()

    title = "M = {}, dM = {:.3e}, Imw = {:.2f}, n_kc = {}, cutoff = {}, BAU = {:.3e}".format(
        mp.M, mp.dM, mp.Imw, n_kc, str(cutoff), bau
    )

    print("Time (solve): {}".format(end - start))

    # solver.print_L_violation()
    # solver.plot_total_L()
    # # solver.plot_everything()
    # solver.plot_eigenvalues(title)
    # # solver.plot_eigenvalues(title, use_z=True)
    solver.plot_total_lepton_asymmetry(title)
    # solver.plot_total_hnl_asymmetry(title)
    # solver.plot_L_violation()
    print("BAU: {:.3e}".format(bau))

    # print(solver.get_total_hnl_asymmetry() / solver.get_total_lepton_asymmetry())
