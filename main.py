from solvers import *

mp = ModelParams(
    M=1.0,
    dM=1e-11,
    Imw=np.log(3),
    Rew=1/4 * np.pi,
    delta= np.pi,
    eta=3/2 * np.pi
)

if __name__ == '__main__':

    T0 = get_T0(mp)
    # solver = AveragedSolver(mp, T0, 10.)
    solver = TrapezoidalSolver(mp, T0, 10.)
    solver.solve()
    solver.plot_total_asymmetry()