from solvers import *
import time
import cProfile
#
# mp = ModelParams(
#     M=1.0,
#     dM=1e-8,
#     # dM=0,
#     Imw=1.0,
#     Rew=1/4 * np.pi,
#     delta= np.pi,
#     eta=3/2 * np.pi
# )
mp = ModelParams(M=10.0, dM=1e-2, Imw=0.0, Rew=0.7853981633974483, delta=3.141592653589793,
                 eta=4.71238898038469)
# mp = ModelParams(
#     M=1.0,
#     dM=1e-9,
#     # Imw=1.0,
#     Imw=4.1,
#     Rew=np.pi/4,
#     delta= np.pi,
#     eta=1.5*np.pi
# )
# mp = ModelParams(M=10.0, dM=5.6234132519034906e-11, Imw=-6.0, Rew=0.7853981633974483, delta=3.141592653589793, eta=4.71238898038469)
# MN = 1.0 # HNLs mass
# dM = 1e-12 # mass difference
#
# imw = np.log(3)
# rew = 13/16*pi
# delta = 29/16*pi
# eta = 22/16*pi

# MN = 1.0 # HNLs mass
# dM = 1e-12 # mass difference
# imw = 0.5
# rew = 0.3*pi
# delta = pi
# eta = 3/2*pi

# mp = ModelParams(M=1.0, dM=1e-12, Imw=0.5, Rew=0.8*np.pi, delta=np.pi, eta=3/2*np.pi)

if __name__ == '__main__':
    # kc_list = [0.3, 0.4] + [0.1 * kc for kc in range(5, 101)]
    # kc_list = np.array([0.5, 1.0, 2.0])
    # kc_list = np.array([1.0])
    # kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.9, 2.5, 3.1, 3.9, 5.0, 10.0])
    # kc_list = np.array([0.5, 1.0, 1.5, 2.5, 5.0])
    kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1,
                3.3, 3.6, 3.9, 4.2, 4.6, 5.0, 5.7, 6.9, 10.0])

    T0 = get_T0(mp)
    print(T0)
    cutoff = 1e4
    eig = False
    TF = Tsph
    # max_step = np.abs((T0 - TF)/500)
    # solver = AveragedSolver(mp, T0, TF, H=1, eig_cutoff=False, ode_pars={'atol' : 1e-12})
    solver = TrapezoidalSolverCPI(
        mp, T0, TF, kc_list, H=1, fixed_cutoff=cutoff, eig_cutoff=eig, method="BDF", ode_pars={'atol' : 1e-10})
    # solver = TrapezoidalSolverCPI(mp, T0, Tsph, kc_list, H=1, method="BDF", ode_pars={'atol': 1e-11})
    # solver = TrapezoidalSolverCPI(mp, T0, Tsph, kc_list, H = 1, eig_cutoff=True, method="Radau")

    start = time.time()
    solver.solve(eigvals=True)
    end = time.time()
    bau = (28./78.) * solver.get_final_lepton_asymmetry()
    title = "M = {}, dM = {}, Imw = {}, n_kc = {}, cutoff = {}, BAU = {:.3e}".format(
        mp.M, mp.dM, mp.Imw, kc_list.shape[0], cutoff, bau
    )
    print("Time (solve): {}".format(end - start))
    solver.plot_eigenvalues(title)
    solver.plot_total_lepton_asymmetry(title)
    # solver.plot_total_hnl_asymmetry(title)
    # solver.plot_L_violation()
    print("BAU: {:.3e}".format(bau))

    print(solver.get_total_hnl_asymmetry() / solver.get_total_lepton_asymmetry())
