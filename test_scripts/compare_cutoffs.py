from solvers import *
from plots import lepton_asymmetry_comp

if __name__ == '__main__':
    # kc_list = np.array([1.0])
    # kc_list = np.array([0.5, 1.0, 1.5, 2.5, 5.0])
    kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1,
                    3.3, 3.6, 3.9, 4.2, 4.6, 5.0, 5.7, 6.9, 10.0])
    osc_cutoffs = [1e-13, 1e-11, 1e-10, 1e-9, 1e-8]
    # osc_cutoffs = [1e-13, 1e-11, 1e-10]

    n_kc = kc_list.shape[0]

    mp = ModelParams(M=1.0, dM=1e-8, Imw=2.0, Rew=0.7853981633974483, delta=3.141592653589793, eta=4.71238898038469)
    T0 = get_T0(mp)

    data = []

    # Cutoff, normal picture
    for cutoff in osc_cutoffs:
        solver = TrapezoidalSolverCPI(mp, T0, Tsph, kc_list, H = 1, cutoff=cutoff)
        start = time.time()
        solver.solve()
        end = time.time()
        tag = "cutoff = {}, t = {:.2f} s".format(cutoff, end - start)
        print(tag)
        data.append((tag, solver.get_Tlist(), solver.get_total_lepton_asymmetry()))

    # No cutoff, interaction picture
    solver = TrapezoidalSolverCPI(mp, T0, Tsph, kc_list, H=1, cutoff=None, interaction=True)
    start = time.time()
    solver.solve()
    end = time.time()
    tag = "Interaction / no cutoff, t = {:.2f} s".format(end - start)
    print(tag)
    data.append((tag, solver.get_Tlist(), solver.get_total_lepton_asymmetry()))

    lepton_asymmetry_comp(np.array(data), "Modifying fast osc cutoff, n_kc = {}".format(len(kc_list)),
                          "output/fast_mode_cutoff_comp_{}modes.png".format(n_kc))