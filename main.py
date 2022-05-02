from os import environ
environ["MKL_NUM_THREADS"] = "1"
environ["NUMEXPR_NUM_THREADS"] = "1"
environ["OMP_NUM_THREADS"] = "1"
environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

import matplotlib.pyplot as plt

from nuMSM_solver.solvers import *
import time
from nuMSM_solver.quadrature import GaussianQuadrature
from nuMSM_solver.rates import *

if __name__ == '__main__':

    # Optimal phases
    delta_opt = np.pi
    eta_opt = (3.0 * np.pi) / 2.0
    rew_opt = np.pi / 4.0

    # Model params
    mp = ModelParams(M=2.0, dM=1e-9, Imw=4.0, Rew=rew_opt, delta=delta_opt, eta=eta_opt)

    n_kc = 10
    kc_max = 10
    # kc_list = np.linspace(0, kc_max, n_kc)
    cutoff = 1e5
    eig = False
    use_source_term = True
    TF = Tsph
    H = 1
    ode_pars = {'atol': 1e-15, 'rtol': 1e-6}
    # ode_pars = {'atol': 1e-15, 'rtol': 1e-8}

    # quadrature = GaussianQuadrature(10, 0.1, 10, mp, H, tot=True, qscheme="radau")
    quadrature = GaussianQuadrature(n_kc, 0, kc_max, qscheme="legendre")
    rates = Rates_Jurai(mp, H, quadrature.kc_list(), tot=False)
    # rates = Rates_Jurai(mp, H, [1.0], tot=False)
    # rates = Rates_Fortran(mp,1)
    # quadrature = TrapezoidalQuadrature(kc_list, rates)

    # quadrature = GaussLegendreQuadrature(20, 0.1, 10, mp, H, tot=True)
    # kc_list = np.array(quadrature.kc_list())

    # solver = AveragedSolver(model_params=mp, rates_interface=rates, TF=TF, H=1, cutoff=1e5,
    #                          ode_pars=ode_pars, source_term=use_source_term, method="Radau")
    solver = QuadratureSolver(quadrature, rates,
                             model_params=mp, TF=TF, H=H, fixed_cutoff=cutoff, eig_cutoff=eig,
                             method="Radau", ode_pars=ode_pars, source_term=use_source_term)

    start = time.time()
    solver.solve()
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
    # solver.plot_total_lepton_asymmetry(title)
    # solver.plot_total_hnl_asymmetry(title)
    # solver.plot_L_violation()
    print("BAU: {:.3e}".format(bau))

    title = "M = {}, dM = {}, Imw = {}, n_kc = {}, cutoff = {}, BAU = {:.3e}".format(
        mp.M, mp.dM, mp.Imw, kc_list.shape[0], cutoff, bau
    )

    Tlist = solver.get_Tlist()
    plt.loglog(Tlist, (28./79.) * np.abs(solver.get_total_lepton_asymmetry()))
    plt.xlabel("T")
    plt.ylabel("lepton asymmetry")
    plt.title(title, fontsize=10)
    # plt.tight_layout()
    plt.show()

    # print(solver.get_total_hnl_asymmetry() / solver.get_total_lepton_asymmetry())
