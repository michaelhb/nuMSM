from solvers import *
import time

if __name__ == '__main__':
    kc_list = np.array([0.5, 1.0, 1.5, 2.5, 5.0])
    # kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1,
    #                 3.3, 3.6, 3.9, 4.2, 4.6, 5.0, 5.7, 6.9, 10.0])
    # dMs = np.array([1e-13, 1e-12, 1e-11])
    # dMs = np.geomspace(1e-13, 1e-8, 20)
    dMs = np.geomspace(1e-13, 1e-8, 10)

    print(dMs)
    # dMs = np.array([1e-12, 1e-11, 1e-10])

    baus_std = []
    times_std = []
    baus_interaction = []
    times_interacton = []
    baus_cutoff = []
    times_cutoff = []
    baus_avg = []
    times_avg = []
    baus_avg_cutoff = []
    times_avg_cutoff = []

    cutoff = "max_eig"

    for dM in dMs:
        print(dM)
        mp = ModelParams(M=1.0, dM=dM, Imw=0.0, Rew=0.7853981633974483, delta=3.141592653589793, eta=4.71238898038469)
        T0 = get_T0(mp)

        # solver_interaction = TrapezoidalSolverCPI(mp, T0, Tsph, kc_list, H=1, cutoff=None, interaction=True, ode_pars={'atol' : 1e-11})
        # start = time.time()
        # solver_interaction.solve()
        # end = time.time()
        # baus_interaction.append((28./78.)*solver_interaction.get_final_lepton_asymmetry())
        # times_interacton.append(end - start)

        solver_std = TrapezoidalSolverCPI(mp, T0, Tsph, kc_list, H=1, cutoff=None, method="BDF", ode_pars={'atol' : 1e-11})
        start = time.time()
        solver_std.solve()
        end = time.time()
        baus_std.append((28./78.)*solver_std.get_final_lepton_asymmetry())
        times_std.append(end - start)

        if cutoff == "max_eig":
            solver_cutoff = TrapezoidalSolverCPI(mp, T0, Tsph, kc_list, H=1, eig_cutoff=True, method="BDF", ode_pars={'atol' : 1e-11})
        else:
            solver_cutoff = TrapezoidalSolverCPI(mp, T0, Tsph, kc_list, H=1, fixed_cutoff=cutoff, method="BDF", ode_pars={'atol' : 1e-11})
        start = time.time()
        solver_cutoff.solve()
        end = time.time()
        baus_cutoff.append((28./78.)*solver_cutoff.get_final_lepton_asymmetry())
        times_cutoff.append(end - start)

        # solver_avg = AveragedSolver(mp, T0, Tsph, H=1, eig_cutoff=False, ode_pars={'atol' : 1e-11})
        # start = time.time()
        # solver_avg.solve()
        # end = time.time()
        # baus_avg.append((28./78.)*solver_avg.get_final_lepton_asymmetry())
        # times_avg.append(end - start)
        #
        # if cutoff == "max_eig":
        #     solver_avg_cutoff = AveragedSolver(mp, T0, Tsph, H=1, eig_cutoff=True, ode_pars={'atol' : 1e-11})
        # else:
        #     solver_avg_cutoff = AveragedSolver(mp, T0, Tsph, H=1, fixed_cutoff=cutoff, ode_pars={'atol': 1e-11})
        # start = time.time()
        # solver_avg_cutoff.solve()
        # end = time.time()
        # baus_avg_cutoff.append((28./78.)*solver_avg.get_final_lepton_asymmetry())
        # times_avg_cutoff.append(end - start)

    title = "BAU & timing comparaison, {} modes, M = {}, Imw = {}, Cutoff= {}".format(kc_list.shape[0], mp.M, mp.Imw, cutoff)

    # Plot BAUs
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.scatter(dMs, np.abs(baus_cutoff), color="green", alpha=0.5, label="cutoff", s=5)
    ax.scatter(dMs, np.abs(baus_std), color="purple", alpha=0.5, label="std picture", s=5)
    # ax.scatter(dMs, np.abs(baus_interaction), color="blue", alpha=0.5, label="interaction picture", s=5)
    # ax.scatter(dMs, np.abs(baus_avg), color="red", alpha=0.5, label="momentum averaged", s=5)
    # ax.scatter(dMs, np.abs(baus_avg_cutoff), color="orange", alpha=0.5, label="momentum averaged (eig cutoff)", s=5)
    ax.legend()
    plt.title(title)
    plt.xlabel("dM")
    plt.ylabel("BAU")
    plt.show(block=False)

    # Plot times
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.scatter(dMs, times_cutoff, color="green", alpha=0.5, label="cutoff", s=5)
    # ax.scatter(dMs, times_interacton, color="blue", alpha=0.5, label="interaction picture",s=5)
    ax.scatter(dMs, times_std, color="purple", alpha=0.5, label="std picture", s=5)
    # ax.scatter(dMs, times_avg, color="red", alpha=0.5, label="momentum averaged", s=5)
    # ax.scatter(dMs, times_avg_cutoff, color="orange", alpha=0.5, label="momentum averaged (eig cutoff)", s=5)
    ax.legend()
    plt.title(title)
    plt.xlabel("dM")
    plt.ylabel("Time (s)")
    plt.show(block=False)


