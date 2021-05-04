from solvers import *

mp = ModelParams(
    M=1.0,
    dM=1e-12,
    Imw=np.log(3),
    Rew=1/4 * np.pi,
    delta= np.pi,
    eta=3/2 * np.pi
)

if __name__ == '__main__':
    kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.9, 2.5, 3.1, 3.9, 5.0, 10.0])

    T0 = get_T0(mp)
    TF = Tsph
    T_sample = np.linspace(T0, TF, 10)

    solver = TrapezoidalSolverCPI(mp, T0, Tsph, kc_list, H=1, ode_pars={'atol': 1e-11})
    solver.solve(eigvals=False)
