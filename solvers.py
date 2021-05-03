from abc import ABC, abstractmethod
from scipy.integrate import odeint, solve_ivp
import matplotlib as mpl
import matplotlib.pyplot as plt
from common import *
from common import trapezoidal_weights
from load_precomputed import *
from rates import get_rates
from os import path
from scipy.linalg import block_diag
from scipy.linalg import eig as speig
from scipy.sparse.linalg import eigs as sparse_eig
import time

# ode_par_defaults = {'rtol' : 1e-6, 'atol' : 1e-15}
ode_par_defaults = {}
mpl.rcParams['figure.dpi'] = 300

class Solver(ABC):

    def __init__(self, model_params, T0, TF, H = 1, ode_pars = ode_par_defaults):
        self.T0 = T0
        self.TF = TF
        self.mp = model_params
        self._total_lepton_asymmetry = None
        self._total_hnl_asymmetry = None
        self._eigenvalues = None
        self.H = H
        self.ode_pars = ode_pars
        super().__init__()

    @abstractmethod
    def solve(self, eigvals=False):
        """
        Solve the kinetic equations
        :param TF: Lower temperature bound
        :return: None
        """
        pass

    def plot_total_lepton_asymmetry(self, title=None):
        Tlist = self.get_Tlist()
        plt.loglog(Tlist, np.abs(self._total_lepton_asymmetry))
        plt.xlabel("T")
        plt.ylabel("lepton asymmetry")
        plt.title(title)
        plt.show()

    def plot_total_hnl_asymmetry(self, title=None):
        Tlist = self.get_Tlist()
        plt.loglog(Tlist, np.abs(self._total_hnl_asymmetry))
        plt.xlabel("T")
        plt.ylabel("hnl asymmetry")
        plt.title(title)
        plt.show()

    def plot_L_violation(self, title=None):
        Tlist = self.get_Tlist()
        plt.loglog(Tlist, np.abs(self._total_hnl_asymmetry + self._total_lepton_asymmetry))
        plt.xlabel("T")
        plt.ylabel("L violation")
        plt.title(title)
        plt.show()

    def plot_eigenvalues(self, title=None):
        Tlist = self.get_Tlist()
        n_eig = len(self._eigenvalues[0])

        plt.xlabel("T")
        plt.ylabel("Eigenvalues")
        if title:
            plt.title(title)

        for i in range(n_eig):
            y = []
            for j in range(len(Tlist)):
                y.append(np.abs(self._eigenvalues[j][i]))
            plt.loglog(Tlist, y, '.', markersize=1)

        plt.show()

    def get_final_lepton_asymmetry(self):
        return self._total_lepton_asymmetry[-1]

    def get_total_lepton_asymmetry(self):
        return self._total_lepton_asymmetry

    def get_total_hnl_asymmetry(self):
        return self._total_hnl_asymmetry

    def get_Tlist(self):
        return Tz(np.linspace(zT(self.T0, self.mp.M), zT(self.TF, self.mp.M), 200), self.mp.M)

class AveragedSolver(Solver):

    def __init__(self, model_params, T0, TF, H = 1, ode_pars = ode_par_defaults):
        super().__init__(model_params, T0, TF, H, ode_pars)

        # Load precomputed data files
        test_data = path.abspath(path.join(path.dirname(__file__), 'test_data/'))
        path_rates = path.join(test_data, "rates/Int_ModH_MN10E-1_kcAve.dat")
        path_SMdata = path.join(test_data, "standardmodel.dat")
        path_suscept_data = path.join(test_data, "susceptibility.dat")

        self.tc = get_rate_coefficients(path_rates)
        self.susc = get_susceptibility_matrix(path_suscept_data)
        self.smdata = get_sm_data(path_SMdata)
        self.rates = get_rates(self.mp, self.tc, self.H)
        self._eigenvalues = []

    def solve(self, eigvals=False):

        # Get initial conditions
        initial_state = self.get_initial_state()

        # Integration bounds
        z0 = zT(self.T0, self.mp.M)
        zF = zT(self.TF, self.mp.M)

        # Output grid
        zlist = np.linspace(z0, zF, 200)

        # Construct system of equations
        def f_state(x, z):
            sysmat = self.coefficient_matrix(z)
            jac = jacobian(z, self.mp, self.smdata)

            if eigvals:
                eig = speig((jac*sysmat), right=False, left=True)
                self._eigenvalues.append(eig[0])

            res = jac*(np.dot(sysmat, x) + self.inhomogeneous_part(z))
            # res = jac * (np.dot(sysmat, x)) # without source term

            return res

        def jac(x, z):
            return jacobian(z, self.mp, self.smdata)*self.coefficient_matrix(z)

        # Solve them
        sol = odeint(f_state, initial_state, zlist, Dfun=jac, full_output=True, **self.ode_pars)
        self._total_lepton_asymmetry = sol[0][:, 0] + sol[0][:, 1] + sol[0][:, 2]
        self._total_hnl_asymmetry = sol[0][:, 7] + sol[0][:, 8]

    def get_initial_state(self):
        #equilibrium number density for a relativistic fermion, T = T0
        neq = (3.*zeta3*(self.T0**3))/(4*(np.pi**2))

        #entropy at T0
        s = self.smdata.s(self.T0)

        n_plus0 = -np.identity(2)*neq/s
        r_plus0 = np.einsum('kij,ji->k',tau,n_plus0)

        res = np.concatenate([[0,0,0],r_plus0,[0,0,0,0]])
        return np.real(res)

    def gamma_omega(self, z):
        """
        :param z: integration coordinate z = ln(M/T)
        :param rt: see common.ModelParams
        :param smdata: see common.SMData
        :return: (3,3) matrix which mixes the n_delta in the upper left corner of the evolution matrix,
        proportional to T^2/6: -Re(GammaBar_nu_alpha).omega_alpha_beta (no sum over alpha)
        """
        GB_nu_a = self.rates.GB_nu_a(z)
        return (GB_nu_a.T * self.susc(Tz(z, self.mp.M))).T

    def Yr(self, z):
        """
        :param z: integration coordinate z = ln(M/T)
        :param rt: see common.ModelParams
        :return: (4,3) matrix appearing in the (3,1) block of the evolution matrix
        """
        susc = self.susc(Tz(z, self.mp.M))
        reGBt_nu_a = np.real(self.rates.GBt_N_a(z))
        return np.einsum('kij,aji,ab->kb',tau,reGBt_nu_a,susc)

    def Yi(self, z):
        """
        :param z: integration coordinate z = ln(M/T)
        :param rt: see common.ModelParams
        :return: (4,3) matrix appearing in the (2,1) block of the evolution matrix
        """
        susc = self.susc(Tz(z, self.mp.M))
        imGBt_nu_a = np.imag(self.rates.GBt_N_a(z))
        return np.einsum('kij,aji,ab->kb',tau,imGBt_nu_a,susc)

    def inhomogeneous_part(self, z):
        """
        :param rt: see common.IntegratedRates
        :return: inhomogeneous part of the ODE system
        """
        Seq = self.rates.Seq(z)
        seq = np.einsum('kij,ji->k', tau, Seq)
        return np.real(np.concatenate([[0, 0, 0], -seq, [0, 0, 0, 0]]))

    def coefficient_matrix(self, z):
        '''
        :param z: integration coordinate z = ln(M/T)
        :param rt: see common.IntegratedRates
        :param mp: see common.ModelParams
        :param suscT: Suscepibility matrix (2,2) (function of T)
        :param smdata: see common.SMData
        :return: z-dependent coefficient matrix of the ODE system
        '''

        T = Tz(z, self.mp.M)
        GB_nu_a, GBt_nu_a, GBt_N_a, HB_N, GB_N, Seq, *_ = [R(z) for R in self.rates]
        susc = self.susc(T)

        b11 = -self.gamma_omega(z)*(T**2)/6.
        b12 = 2j*tr_h(np.imag(GBt_nu_a))
        b13 = -tr_h(np.real(GBt_nu_a))
        b21 = -(1j/2.)*self.Yi(z)*(T**2)/6.
        b22 = -1j*Ch(np.real(HB_N)) - (1./2.)*Ah(np.real(GB_N))
        b23 = (1./2.)*Ch(np.imag(HB_N)) - (1j/4.)*Ah(np.imag(GB_N))
        b31 = -self.Yr(z)*(T**2)/6.
        b32 = 2*Ch(np.imag(HB_N)) - 1j*Ah(np.imag(GB_N))
        b33 = -1j*Ch(np.real(HB_N)) - (1./2.)*Ah(np.real(GB_N))

        res = np.block([
            [b11,b12,b13],
            [b21,b22,b23],
            [b31,b32,b33]
        ])
        return np.real(res)

"""
Data type to describe a quadrature scheme, incl. the list of 
momenta points, weights and rates.
"""
QuadratureInputs = namedtuple("QuadratureInputs", [
    "kc_list", "weights", "rates"
])

class TrapezoidalSolverCPI(Solver):

    def __init__(self, model_params, T0, TF, kc_list,
                 cutoff = None, H = 1, ode_pars = ode_par_defaults, eig_cutoff = False, method=None):

        self.kc_list = kc_list
        self.mp = model_params
        self.cutoff = cutoff
        self.eig_cutoff = eig_cutoff
        self.method = method

        self.rates = []
        test_data = path.abspath(path.join(path.dirname(__file__), 'test_data/'))

        for kc in self.kc_list:
            fname = 'rates/Int_ModH_MN10E-1_kc{}E-1.dat'.format(int(kc * 10))
            path_rates = path.join(test_data, fname)
            tc = get_rate_coefficients(path_rates)
            rt = get_rates(self.mp, tc, H)

            self.rates.append(rt)

        test_data = path.abspath(path.join(path.dirname(__file__), 'test_data/'))

        # Load standard model data, susceptibility matrix
        path_SMdata = path.join(test_data, "standardmodel.dat")
        path_suscept_data = path.join(test_data, "susceptibility.dat")
        self.susc = get_susceptibility_matrix(path_suscept_data)
        self.smdata = get_sm_data(path_SMdata)

        Solver.__init__(self, model_params, T0, TF, H, ode_pars)

    # Initial condition
    def get_initial_state(self):
        x0 = [0, 0, 0]

        for kc in self.kc_list:
            rho_plus_0 = -1 * (self.T0 ** 3) * f_N(self.T0, self.mp.M, kc) * np.identity(2) / self.smdata.s(self.T0)
            r_plus_0 = np.einsum('kij,ji->k', tau, rho_plus_0)
            x0.extend(r_plus_0)
            x0.extend([0, 0, 0, 0])

        return np.real(x0)

    def gamma_omega(self, z, quad):
        T = Tz(z, self.mp.M)

        # Integrate rate
        g_int = np.zeros(3, dtype="complex128")
        for wi, rt, kc in zip(quad.weights, quad.rates, quad.kc_list):
            g_int += wi*(kc**2)*rt.GB_nu_a(z)*f_nu(kc)*(1 - f_nu(kc))

        g_int *= (T**2)/(np.pi**2)

        # Contract with susc matrix
        return (g_int.T * self.susc(T)).T

    def gamma_N(self, z, kc, rt, imag=False):
        T = Tz(z, self.mp.M)

        G_N = np.imag(rt.GBt_N_a(z)) if imag else np.real(rt.GBt_N_a(z))

        return 2.0 * T**2 * f_nu(kc) * (1 - f_nu(kc)) * np.einsum('ab,kij,aji->kb', self.susc(T), tau,
                                                                                      G_N)
    def coefficient_matrix(self, z, quad):
        T = Tz(z, self.mp.M)

        # Top left block, only part that doesn't depend on kc.
        g_nu = -self.gamma_omega(z, quad)

        top_row = []
        left_col = []
        diag_H0 = []
        diag_HI = []

        n_kc = len(quad.kc_list)

        jac = jacobian(z, self.mp, self.smdata)

        for i in range(n_kc):
            kc = quad.kc_list[i]
            rt = quad.rates[i]
            w_i = quad.weights[i]

            GB_nu_a, GBt_nu_a, GBt_N_a, HB_N, GB_N, Seq, H_I = [R(z) for R in rt]

            W = (1.0 / (2 * (np.pi ** 2))) * w_i * (kc ** 2)

            # Top row
            top_row.append(2j*W*tr_h(np.imag(GBt_nu_a)))
            top_row.append(-W*tr_h(np.real(GBt_nu_a)))

            # Left col
            left_col.append(-0.5j*self.gamma_N(z, kc, rt, imag=True))
            left_col.append(-1*self.gamma_N(z, kc, rt, imag=False))

            H_0 = HB_N - H_I

            # Diag blocks
            b11_HI = -1j*Ch(np.real(H_I)) - 0.5*Ah(np.real(GB_N))
            b12_HI = 0.5*Ch(np.imag(H_I)) - 0.25j*Ah(np.imag(GB_N))
            b21_HI = 2*Ch(np.imag(H_I)) - 1j*Ah(np.imag(GB_N))
            diag_HI.append(np.block([[b11_HI, b12_HI], [b21_HI, b11_HI]]))

            b11_H0 = -1j*Ch(np.real(H_0))
            b12_H0 = 0.5*Ch(np.imag(H_0))
            b21_H0 = 2*Ch(np.imag(H_0))
            diag_H0.append(np.block([[b11_H0, b12_H0], [b21_H0, b11_H0]]))

        res = np.real(np.block([
            [g_nu, np.hstack(top_row)],
            [np.vstack(left_col), block_diag(*diag_HI) + block_diag(*diag_H0)]
        ]))

        if self.eig_cutoff:
            Aprime = np.real(np.block([
                [np.zeros((3,3)), np.zeros((3,8*n_kc))],
                [np.zeros((8*n_kc,3)), block_diag(*diag_H0)]
            ]))
            cutoff = 1.5*np.abs(sparse_eig(block_diag(*diag_H0), k=1, which="LM", return_eigenvectors=False)[0])
            res = np.dot(res, np.linalg.inv(-Aprime/cutoff + np.eye(3 + 8*n_kc)))

        return jac*res

        # # Average out fast modes...
        # if self.cutoff:
        #     return np.dot(res, np.linalg.inv(-res/self.cutoff + np.eye(3 + 8*n_kc)))
        # else:
        #     return res

    def calc_hnl_asymmetry(self, sol, zlist, quad):
        res = []

        for i, z in enumerate(zlist):
            res_i = 0
            T = Tz(z, self.mp.M)

            for j, kc in enumerate(quad.kc_list):
                kc = quad.kc_list[j]
                weight = quad.weights[j]
                base_col = 3 + 8*j
                rminus_11 = sol[i, base_col + 4]
                rminus_22 = sol[i, base_col + 5]
                res_i += (1.0 / (2 * (np.pi ** 2))) * (kc ** 2) * weight * (rminus_11 + rminus_22)

            res.append(res_i)
        return np.array(res)

    def calc_lepton_asymmetry(self, sol, zlist):

        res = []

        for i, z in enumerate(zlist):
            res.append((sol[i, 0] + sol[i, 1] + sol[i, 2]))

        return np.array(res)

    def solve(self, eigvals=False):
        quad = QuadratureInputs(self.kc_list, trapezoidal_weights(self.kc_list), self.rates)
        initial_state = self.get_initial_state()

        # Integration bounds
        z0 = zT(self.T0, self.mp.M)
        zF = zT(self.TF, self.mp.M)

        # Output grid
        zlist = np.linspace(z0, zF, 200)

        # Eigenvalues
        self._eigenvalues = []

        # Construct system of equations
        def f_state(z, x):
            sysmat = self.coefficient_matrix(z, quad)
            res = (sysmat.dot(x))

            if eigvals:
                # eig = speig((jac*sysmat), right=False, left=True)
                eig = speig((sysmat), right=False, left=True)
                # ix_small = np.argmin(np.abs(eig[0]))
                # eval_small = np.abs(eig[0][ix_small])
                # evec_small = np.abs(eig[1][:,ix_small])
                # print("Smallest Eval: ", eval_small)
                # print("Corresponding left Evec: ", evec_small)
                # print("Normalized: ", evec_small / evec_small[0])
                self._eigenvalues.append(eig[0])

            return res

        def jac(z, x):
            return self.coefficient_matrix(z, quad)

        # Solve them
        if self.method == None or self.method == "LSODE":
            # Default to old odeint / LSODE integrator
            sol = odeint(f_state, initial_state, zlist, Dfun=jac, full_output=True, tfirst=True, **self.ode_pars)
            self._total_lepton_asymmetry = self.calc_lepton_asymmetry(sol[0], zlist)
            self._total_hnl_asymmetry = self.calc_hnl_asymmetry(sol[0], zlist, quad)
        else:
            # Otherwise try solve_ivp
            sol = solve_ivp(f_state, (z0, zF), y0=initial_state, jac=jac, t_eval=zlist, method=self.method, **self.ode_pars)
            self._total_lepton_asymmetry = self.calc_lepton_asymmetry(sol.y.T, zlist)
            self._total_hnl_asymmetry = self.calc_hnl_asymmetry(sol.y.T, zlist, quad)