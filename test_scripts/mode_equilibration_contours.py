from solvers import *
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.colors as colors
import matplotlib.cm as cm

mp = ModelParams(
    M=1.0,
    dM=1e-12,
    Imw=1.0,
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
    kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.9, 2.5, 3.1, 3.9, 5.0, 10.0])
    # kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1,
    #                     3.3, 3.6, 3.9, 4.2, 4.6, 5.0, 5.7, 6.9, 10.0])

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

    Z = []

    for T in T_sample:
        row = []
        for i in range(kc_list.shape[0]):
            row.append(densities[i](T))
        Z.append(row)

    Z = np.array(Z)
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.xticks(ticks=np.arange(kc_list.shape[0]), labels=kc_list)
    plt.yticks(ticks=np.arange(T_sample.shape[0]), labels=map(lambda f: "{:.2f}".format(f), T_sample))
    hm = plt.imshow(Z, cmap='cool', interpolation='nearest')
    clb = plt.colorbar(hm)
    clb.ax.set_title(r'$(n_{N_1} - f_N)/f_N$')
    ax.set_aspect("auto")
    plt.xlabel("k_c")
    plt.ylabel("T")
    plt.title("N_1 equilibration, dM = {:.3e}, Imw = {:.2f}, n_kc={}".format(mp.dM, mp.Imw, kc_list.shape[0]))
    # plt.tight_layout()
    plt.show()