from solvers import *
from multiprocessing import Pool

kc_list = np.array([0.5, 1.0, 1.3, 1.5,  1.9, 2.5, 3.1, 3.9, 5.0, 10.0])

def get_scan_points(points_per_dim):
    M = 1.0
    delta = -2.199
    eta = -1.857
    Rew = 2.444

    dM_min = -14
    dM_max = -2
    dMs = [10**e for e in np.linspace(dM_min, dM_max, points_per_dim)]

    Imw_min = -6.
    Imw_max = 6.
    Imws = np.linspace(Imw_min, Imw_max, points_per_dim)

    points = []

    for dM in dMs:
        for Imw in Imws:
            points.append(ModelParams(M, dM, Imw, Rew, delta, eta))

    return points

def get_bau(mp):
    print("Starting point {}".format(mp))
    T0 = get_T0(mp)
    solver = TrapezoidalSolverCPI(mp, T0, Tsph, kc_list, 1)
    start = time.time()
    solver.solve(eigvals=False)
    end = time.time()
    bau = (28./78.) * solver.get_final_lepton_asymmetry()
    print("Point {} finished in {} s, BAU = {}".format(mp, end - start, bau))
    return bau


if __name__ == '__main__':
    with Pool() as p:
        points = get_scan_points(5)
        baus = p.map(get_bau, points)

    print("Finished!")
    print("BAUs: {}".format(baus))