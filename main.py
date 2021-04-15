from solvers import *
import time
import cProfile
#
# mp = ModelParams(
#     M=1.0,
#     dM=1e-12,
#     # dM=0,
#     Imw=np.log(3),
#     Rew=1/4 * np.pi,
#     delta= np.pi,
#     eta=3/2 * np.pi
# )
#
mp = ModelParams(
    M=1.0,
    dM=1e-11,
    Imw=np.log(3),
    Rew=13/16*np.pi,
    delta= 29/16*np.pi,
    eta=22/16*np.pi
)

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
    # kc_list = [0.5, 1.0, 2.0]
    # kc_list = [1.0]
    kc_list = np.array([0.5, 1.0, 1.3, 1.5,  1.9, 2.5, 3.1, 3.9, 5.0, 10.0])

    T0 = get_T0(mp)
    print(T0)
    # solver = AveragedSolver(mp, T0, Tsph, 1, {'rtol' : 1e-7, 'atol' : 1e-17})
    # solver = TrapezoidalSolverCPI(mp, T0, Tsph, kc_list, 1, {'rtol': 1e-7, 'atol': 1e-17})
    solver = TrapezoidalSolverCPI(mp, T0, Tsph, kc_list, 1)
    start = time.time()
    solver.solve(eigvals=False)
    end = time.time()
    print("Time (solve): {}".format(end - start))
    # solver.plot_eigenvalues()
    solver.plot_total_lepton_asymmetry()
    solver.plot_total_hnl_asymmetry()
    solver.plot_L_violation()
    print("BAU: {:.3e}".format((28./78.) * solver.get_final_lepton_asymmetry()))

    print(solver.get_total_hnl_asymmetry() / solver.get_total_lepton_asymmetry())
