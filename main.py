from solvers import *
from test_scripts.total_asymmetry_plots import total_asymmetry_plots
from test_scripts.approx_bau_grid_scan import bau_grid_scan
from test_scripts.bau_grid_plot import bau_grid_plot
# from scipy.integrate.odepack import ODEintWarning
# import warnings
# warnings.simplefilter("error", ODEintWarning)

# mp = ModelParams(
#     M=1.0,
#     dM=1e-12,
#     Imw=np.log(3),
#     Rew=1/4 * np.pi,
#     delta= np.pi,
#     eta=3/2 * np.pi
# )

# MN = 1.0 # HNLs mass
# dM = 1e-12 # mass difference
#
# imw = 0.5
# rew = 0.3*pi
# delta = pi
# eta = 3/2*pi

mp = ModelParams(M=1.0, dM=1e-12, Imw=0.5, Rew=0.3*np.pi, delta=np.pi, eta=3/2*np.pi)

if __name__ == '__main__':

    # T0 = get_T0(mp)
    # print(T0)
    # solver = TrapezoidalSolver(mp, T0, 10., 1, {'rtol' : 1e-7, 'atol' : 1e-20})
    # solver.solve()
    # solver.plot_total_asymmetry()
    # print("BAU: {:.3e}".format((28./78.)*solver.get_final_asymmetry()))

    # # total_asymmetry_plots()
    # # bau_grid_scan("avg", AveragedSolver)
    # # bau_grid_scan("trap_15", TrapezoidalSolver, 15, 1e-20)
    bau_grid_scan("trap_30", TrapezoidalSolver, 30, 1e-20)
    # # bau_grid_scan("avg_15", AveragedSolver, 15, 1e-15)
    # bau_grid_plot("trap_15")