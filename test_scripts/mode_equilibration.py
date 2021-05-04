from solvers import *
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

mp = ModelParams(
    M=1.0,
    dM=1e-6,
    Imw=5.0,
    Rew=1/4 * np.pi,
    delta= np.pi,
    eta=3/2 * np.pi
)

def get_N1_density(sol, mp, smdata, kc_list, Tlist):
    res = []

    # Tlist = Tlist[::-1]

    for i, kc in enumerate(kc_list):
        base_col = 3 + 8*i
        n_k = sol[:, base_col] + 0.5*sol[:, base_col + 4]

        for j, T in enumerate(Tlist):
            n_k[j] /= f_N(T, mp.M, kc)
            n_k[j] *= (smdata.s(T)/(T**3))

        res.append(interp1d(Tlist, n_k, fill_value="extrapolate"))

    return res

if __name__ == '__main__':
    # kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.9, 2.5, 3.1, 3.9, 5.0, 10.0])
    kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1,
                        3.3, 3.6, 3.9, 4.2, 4.6, 5.0, 5.7, 6.9, 10.0])

    T0 = get_T0(mp)
    TF = 10.
    T_sample = np.geomspace(T0, TF, 30)
    print(T_sample)

    if mp.dM < 1e-9:
        solver = TrapezoidalSolverCPI(mp, T0, TF, kc_list, H=1, eig_cutoff=False, method="BDF", ode_pars={'atol' : 1e-11})
    elif mp.dM < 1e-6:
        solver = TrapezoidalSolverCPI(mp, T0, TF, kc_list, H=1, eig_cutoff=True, method="BDF", ode_pars={'atol' : 1e-11})
    else:
        solver = TrapezoidalSolverCPI(mp, T0, TF, kc_list, H=1, eig_cutoff=True, method="Radau", ode_pars={'atol': 1e-10})

    Tlist = solver.get_Tlist()
    print(Tlist)

    solver.solve(eigvals=False)
    sol = solver.get_full_solution()

    densities = get_N1_density(sol, mp, solver.smdata, kc_list, Tlist)

    kmin = TF*kc_list[0]
    kmax = T0*kc_list[-1]

    # cm_idx = lambda i: (i + 1)/float(T_sample.shape[0])
    normalize = colors.LogNorm(vmin=TF, vmax=T0)
    colormap = cm.plasma

    plt.rcParams.update({"text.usetex": False})

    for i, T in enumerate(T_sample):
        k_list_T = T*kc_list
        densities_T = []

        for density in densities:
            densities_T.append(density(T))

        plt.plot(kc_list, densities_T, label="T = {}".format(T), color=colormap(normalize(T)))

    scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
    scalarmappaple.set_array(T_sample)
    clb = plt.colorbar(scalarmappaple)
    clb.ax.set_title("T")

    plt.xlabel(r'$k_c$')
    plt.ylabel(r'$(n_{N_1} - f_N)/f_N$')
    plt.title("N_1 equilibration, dM = {:.3e}, Imw = {:.2f}, n_kc={}".format(mp.dM, mp.Imw, kc_list.shape[0]))
    # plt.legend()
    plt.show()