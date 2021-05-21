from solvers import *
import time
import cProfile
from rates import Rates_Fortran, Rates_Jurai
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
mp = ModelParams(M=1.0, dM=1e-11, Imw=3.0, Rew=0.7853981633974483, delta=3.141592653589793,
                 eta=4.71238898038469)
# mp = ModelParams(M=10.0, dM=0.004520353656360241, Imw=-4.344827586206897, Rew=0.7853981633974483, delta=3.141592653589793,
#             eta=4.71238898038469)
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
    kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.9, 2.5, 3.1, 3.9, 5.0, 10.0])
    # kc_list = np.array([0.5, 1.0, 1.5, 2.5, 5.0])
    # kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1,
    #             3.3, 3.6, 3.9, 4.2, 4.6, 5.0, 5.7, 6.9, 10.0])

    cutoff = None
    eig = False
    use_source_term = False
    TF = Tsph

    rates = Rates_Fortran(mp,1)
    # rates = Rates_Jurai(mp, 1, kc_list)

    # solver = AveragedSolver(model_params=mp, rates=rates, TF=TF, H=1, eig_cutoff=False,
    #                         ode_pars={'atol' : 1e-15, 'rtol' : 1e-6}, source_term=use_source_term)
    solver = TrapezoidalSolverCPI(kc_list,
        model_params=mp, rates=rates, TF=TF,  H=1, fixed_cutoff=cutoff, eig_cutoff=eig,
        method="Radau", ode_pars={'atol' : 1e-15, 'rtol' : 1e-6}, source_term=use_source_term)

    start = time.time()
    solver.solve(eigvals=True)
    end = time.time()
    bau = (28./78.) * solver.get_final_lepton_asymmetry()
    title = "M = {}, dM = {}, Imw = {}, n_kc = {}, cutoff = {}, BAU = {:.3e}".format(
        mp.M, mp.dM, mp.Imw, kc_list.shape[0], cutoff, bau
    )
    print("Time (solve): {}".format(end - start))
    solver.plot_eigenvalues(title)
    # solver.plot_eigenvalues(title, use_z=True)
    solver.plot_total_lepton_asymmetry(title)
    # solver.plot_total_hnl_asymmetry(title)
    # solver.plot_L_violation()
    print("BAU: {:.3e}".format(bau))

    print(solver.get_total_hnl_asymmetry() / solver.get_total_lepton_asymmetry())
