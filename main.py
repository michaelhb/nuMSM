from solvers import *
from test_scripts.total_asymmetry_plots import total_asymmetry_plots
from test_scripts.approx_bau_grid_scan import bau_grid_scan

# mp = ModelParams(
#     M=1.0,
#     dM=1e-11,
#     Imw=np.log(3),
#     Rew=1/4 * np.pi,
#     delta= np.pi,
#     eta=3/2 * np.pi
# )
mp = ModelParams(M=0.7732, dM=1.5264179671752364e-11, Imw=6.0, Rew=2.444, delta=-2.199, eta=-1.857)

if __name__ == '__main__':

    T0 = get_T0(mp)
    solver = AveragedSolver(mp, T0, Tsph, 2,{'rtol' : 1e-6, 'atol' : 1e-13})
    solver.solve()
    solver.plot_total_asymmetry()
    print("BAU: {:.3e}".format((28./78.)*solver.get_final_asymmetry()))

    # total_asymmetry_plots()
    # bau_grid_scan()