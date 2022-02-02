from nuMSM_solver.solvers import *
import matplotlib.pyplot as plt

mp = ModelParams(
    M=1.0,
    dM=1e-6,
    Imw=5.0,
    Rew=1/4 * np.pi,
    delta= np.pi,
    eta=3/2 * np.pi
)

if __name__ == '__main__':
    # kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.9, 2.5, 3.1, 3.9, 5.0, 10.0])
    kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1,
                3.3, 3.6, 3.9, 4.2, 4.6, 5.0, 5.7, 6.9, 10.0])

    T0 = get_T0(mp)
    TF = Tsph
    # TF = 10.

    if mp.dM < 1e-9:
        solver_kdep = QuadratureSolver(mp, T0, TF, kc_list, H=1, eig_cutoff=False, method="BDF", ode_pars={'atol' : 1e-11})
    elif mp.dM < 1e-6:
        solver_kdep = QuadratureSolver(mp, T0, TF, kc_list, H=1, eig_cutoff=True, method="BDF", ode_pars={'atol' : 1e-11})
    else:
        solver_kdep = QuadratureSolver(mp, T0, TF, kc_list, H=1, eig_cutoff=True, method="Radau", ode_pars={'atol': 1e-10})

    if mp.dM < 1e-9:
        solver_avg = AveragedSolver(mp, T0, TF, 1, eig_cutoff=False, ode_pars={'atol' : 1e-9})
    else:
        solver_avg = AveragedSolver(mp, T0, TF, 1, eig_cutoff=True, ode_pars={'atol': 1e-9})

    solver_kdep.solve()
    solver_avg.solve()

    title = "Avg. vs Momentum, dM = {:.3e}, Imw = {:.3f}, n_kc = {}".format(
        mp.dM, mp.Imw, kc_list.shape[0]
    )

    Tlist = solver_avg.get_Tlist()
    plt.loglog(Tlist, np.abs(solver_kdep.get_total_lepton_asymmetry()), color="blue", label="Momentum dependent")
    plt.loglog(Tlist, np.abs(solver_avg.get_total_lepton_asymmetry()), color="red", label="Averaged")

    plt.title(title)
    plt.legend()
    plt.show()
