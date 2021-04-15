from abc import ABC, abstractmethod
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from common import *
from common import trapezoidal_weights
from load_precomputed import *
# from rates import get_rates, get_normalised_rates
from rates import get_rates
from os import path
from scipy.sparse import coo_matrix, csr_matrix
from scipy.linalg import block_diag
from scipy.linalg import eig as speig

ode_par_defaults = {'rtol' : 1e-6, 'atol' : 1e-15}

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

    def plot_total_lepton_asymmetry(self):
        Tlist = Tz(np.linspace(zT(self.T0, self.mp.M), zT(self.TF, self.mp.M), 200), self.mp.M)
        plt.loglog(Tlist, np.abs(self._total_lepton_asymmetry))
        plt.xlabel("T")
        plt.ylabel("lepton asymmetry")
        plt.show()

    def plot_total_hnl_asymmetry(self):
        Tlist = Tz(np.linspace(zT(self.T0, self.mp.M), zT(self.TF, self.mp.M), 200), self.mp.M)
        plt.loglog(Tlist, np.abs(self._total_hnl_asymmetry))
        plt.xlabel("T")
        plt.ylabel("hnl asymmetry")
        plt.show()

    def plot_L_violation(self):
        Tlist = Tz(np.linspace(zT(self.T0, self.mp.M), zT(self.TF, self.mp.M), 200), self.mp.M)
        plt.loglog(Tlist, np.abs(self._total_hnl_asymmetry + self._total_lepton_asymmetry))
        plt.xlabel("T")
        plt.ylabel("L violation")
        plt.show()

    def plot_eigenvalues(self, title=None):
        Tlist = Tz(np.linspace(zT(self.T0, self.mp.M), zT(self.TF, self.mp.M), 200), self.mp.M)
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
        initial_state = self.get_initial_state(self.T0, self.smdata)

        # Integration bounds
        z0 = zT(self.T0, self.mp.M)
        zF = zT(self.TF, self.mp.M)

        # Output grid
        zlist = np.linspace(z0, zF, 200)

        # Construct system of equations
        def f_state(x, z):
            sysmat = self.coefficient_matrix(z, self.rates, self.mp, self.susc)
            jac = jacobian(z, self.mp, self.smdata)

            if eigvals:
                eig = speig((jac*sysmat), right=False, left=True)
                self._eigenvalues.append(eig[0])

            res = jac*(np.dot(sysmat, x) + self.inhomogeneous_part(z, self.rates))
            # res = jac * (np.dot(sysmat, x)) # without source term

            return res

        def jac(x, z):
            return jacobian(z, self.mp, self.smdata)*self.coefficient_matrix(z, self.rates, self.mp, self.susc)

        # Solve them
        sol = odeint(f_state, initial_state, zlist, Dfun=jac, full_output=True, **self.ode_pars)
        self._total_lepton_asymmetry = sol[0][:, 0] + sol[0][:, 1] + sol[0][:, 2]
        self._total_hnl_asymmetry = sol[0][:, 7] + sol[0][:, 8]

    def get_initial_state(self, T0, smdata):
        #equilibrium number density for a relativistic fermion, T = T0
        neq = (3.*zeta3*(T0**3))/(4*(np.pi**2))

        #entropy at T0
        s = smdata.s(T0)

        n_plus0 = -np.identity(2)*neq/s
        r_plus0 = np.einsum('kij,ji->k',tau,n_plus0)

        res = np.concatenate([[0,0,0],r_plus0,[0,0,0,0]])
        return np.real(res)

    def gamma_omega(self, z, rt, susc):
        """
        :param z: integration coordinate z = ln(M/T)
        :param rt: see common.ModelParams
        :param smdata: see common.SMData
        :return: (3,3) matrix which mixes the n_delta in the upper left corner of the evolution matrix,
        proportional to T^2/6: -Re(GammaBar_nu_alpha).omega_alpha_beta (no sum over alpha)
        """
        GB_nu_a = rt.GB_nu_a(z)
        return (GB_nu_a.T * susc).T

    def Yr(self, z, rt, susc):
        """
        :param z: integration coordinate z = ln(M/T)
        :param rt: see common.ModelParams
        :return: (4,3) matrix appearing in the (3,1) block of the evolution matrix
        """
        reGBt_nu_a = np.real(rt.GBt_N_a(z))
        return np.einsum('kij,aji,ab->kb',tau,reGBt_nu_a,susc)

    def Yi(self, z, rt, susc):
        """
        :param z: integration coordinate z = ln(M/T)
        :param rt: see common.ModelParams
        :return: (4,3) matrix appearing in the (2,1) block of the evolution matrix
        """
        imGBt_nu_a = np.imag(rt.GBt_N_a(z))
        return np.einsum('kij,aji,ab->kb',tau,imGBt_nu_a,susc)

    def inhomogeneous_part(self, z, rt):
        """
        :param rt: see common.IntegratedRates
        :return: inhomogeneous part of the ODE system
        """
        Seq = rt.Seq(z)
        seq = np.einsum('kij,ji->k', tau, Seq)
        return np.real(np.concatenate([[0, 0, 0], -seq, [0, 0, 0, 0]]))

    def coefficient_matrix(self, z, rt, mp, suscT):
        '''
        :param z: integration coordinate z = ln(M/T)
        :param rt: see common.IntegratedRates
        :param mp: see common.ModelParams
        :param suscT: Suscepibility matrix (2,2) (function of T)
        :param smdata: see common.SMData
        :return: z-dependent coefficient matrix of the ODE system
        '''

        T = Tz(z, self.mp.M)
        GB_nu_a, GBt_nu_a, GBt_N_a, HB_N, GB_N, Seq, *_ = [R(z) for R in rt]
        susc = suscT(T)

        b11 = -self.gamma_omega(z, rt, susc)*(T**2)/6.
        b12 = 2j*tr_h(np.imag(GBt_nu_a))
        b13 = -tr_h(np.real(GBt_nu_a))
        b21 = -(1j/2.)*self.Yi(z, rt, susc)*(T**2)/6.
        b22 = -1j*Ch(np.real(HB_N)) - (1./2.)*Ah(np.real(GB_N))
        b23 = (1./2.)*Ch(np.imag(HB_N)) - (1j/4.)*Ah(np.imag(GB_N))
        b31 = -self.Yr(z, rt, susc)*(T**2)/6.
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

    def __init__(self, model_params, T0, TF, H = 1, ode_pars = ode_par_defaults):
        # self.kc_list = [0.3, 0.4] + [0.1 * kc for kc in range(5, 101)]
        self.kc_list = np.array([0.5, 1.0, 1.3, 1.5,  1.9, 2.5, 3.1, 3.9, 5.0, 10.0])
        # self.kc_list = [0.5, 1.0, 2.0]
        # self.kc_list = [1.0]
        print("Using {} modes".format(len(self.kc_list)))

        self.mp = model_params
        # self.Tscale = T0/((2*(np.pi**2))**(1/3))
        # self.Tscale = T0
        self.Tscale = 1.0

        self.rates = []
        test_data = path.abspath(path.join(path.dirname(__file__), 'test_data/'))

        for kc in self.kc_list:
            fname = 'rates/Int_ModH_MN10E-1_kc{}E-1.dat'.format(int(kc * 10))
            path_rates = path.join(test_data, fname)
            tc = get_rate_coefficients(path_rates)
            rt = get_rates(self.mp, tc, H)

            self.rates.append(rt)

        # self.rates = get_normalised_rates(model_params, self.kc_list, trapezoidal_weights(self.kc_list), scale_to_avg=True)
        test_data = path.abspath(path.join(path.dirname(__file__), 'test_data/'))

        # Load standard model data, susceptibility matrix
        path_SMdata = path.join(test_data, "standardmodel.dat")
        path_suscept_data = path.join(test_data, "susceptibility.dat")
        self.susc = get_susceptibility_matrix(path_suscept_data)
        self.smdata = get_sm_data(path_SMdata)

        Solver.__init__(self, model_params, T0, TF, H, ode_pars)

    def U(self, mp, smdata, kc, T):
        k = T*kc

        Mstar = 1.22e19 * np.sqrt(45.0 / (4 * (np.pi ** 3) * smdata.geff(T)))
        E_N = np.sqrt(k**2 + mp.M**2)

        I = -1*((mp.dM * Mstar)/(2*(T**2)))*(E_N/mp.M - ((k/mp.M)**2)*np.arcsinh((mp.M/k)))
        # I = -1*((mp.dM*mp.M*Mstar)/(3*k*(T**2)))*(1 - (3./10.)*((mp.M/k)**2)) # approx

        res = np.array([
            [np.cos(I), -1j*np.sin(I)],
            [-1j*np.sin(I), np.cos(I)]
        ])
        return res

    # Initial condition
    def get_initial_state(self, T0, mp, smdata, kc_list):
        x0 = [0, 0, 0]

        for kc in kc_list:
            # rho_plus_0 = -1*f_N(T0, mp, kc)*np.identity(2)/smdata.s(T0)
            # rho_plus_0 = -1 * (self.Tscale**3) * f_nu(kc) * np.identity(2) / smdata.s(T0)
            rho_plus_0 = -1 * (T0 ** 3) * f_N(T0, mp, kc) * np.identity(2) / smdata.s(T0)
            r_plus_0 = np.einsum('kij,ji->k', tau, rho_plus_0)
            x0.extend(r_plus_0)
            x0.extend([0, 0, 0, 0])
        np.set_printoptions(precision=4, linewidth=500, threshold=np.inf)
        print(x0)
        return np.real(x0)

    def gamma_omega(self, z, susc, quad):
        T = Tz(z, self.mp.M)

        # Integrate rate
        g_int = np.zeros(3, dtype="complex128")
        for wi, rt, kc in zip(quad.weights, quad.rates, quad.kc_list):
            g_int += wi*(kc**2)*rt.GB_nu_a(z)*f_nu(kc)*(1 - f_nu(kc))

        g_int *= (T**2)/(np.pi**2)

        # Contract with susc matrix
        return (g_int.T * susc(T)).T

    def hat(self, A, U):
        return np.dot(np.conj(U.T), np.dot(A, U))

    def gamma_N(self, z, kc, rt, mp, susc, U, imag=False):
        T = Tz(z, self.mp.M)
        G_Nhat = np.array([
            self.hat(np.imag(g), U) if imag else self.hat(np.real(g), U) for g in rt.GBt_N_a(z)])
        # return (self.Tscale**3)*(2.0 / T) * f_nu(kc) * (1 - f_nu(kc)) * np.einsum('ab,kij,aji->kb', susc(T), tau, G_Nhat)
        return ((T**3))*(self.Tscale ** 3) * (2.0 / T) * f_nu(kc) * (1 - f_nu(kc)) * np.einsum('ab,kij,aji->kb', susc(T), tau,
                                                                                      G_Nhat)

    def coefficient_matrix(self, z, quad, mp, susc, smdata):
        T = Tz(z, self.mp.M)

        # Top left block, only part that doesn't depend on kc.
        g_nu = -self.gamma_omega(z, susc, quad)

        top_row = []
        left_col = []
        diag = []

        n_kc = len(quad.kc_list)

        for i in range(n_kc):
            kc = quad.kc_list[i]
            rt = quad.rates[i]
            w_i = quad.weights[i]

            GB_nu_a, GBt_nu_a, GBt_N_a, HB_N, GB_N, Seq, H_I = [R(z) for R in rt]

            U = self.U(mp, smdata, kc, T)
            # U = np.identity(2)
            # W = (((T/self.Tscale)**3)/(2*(np.pi**2)))*w_i*(kc**2)
            W = (((1.0 / self.Tscale) ** 3) / (2 * (np.pi ** 2))) * w_i * (kc ** 2)

            # Top row
            GBTI_nu_a_hat = [self.hat(np.imag(A), U) for A in GBt_nu_a]
            GBTR_nu_a_hat = [self.hat(np.real(A), U) for A in GBt_nu_a]
            top_row.append(2j*W*tr_h(GBTI_nu_a_hat))
            top_row.append(-W*tr_h(GBTR_nu_a_hat))

            # Left col
            left_col.append(-0.5j*self.gamma_N(z, kc, rt, mp, susc, U, imag=True))
            left_col.append(-1*self.gamma_N(z, kc, rt, mp, susc, U, imag=False))

            # Diag blocks
            # H_I = HB_N
            H_IhatR = self.hat(np.real(H_I), U)
            H_IhatI = self.hat(np.imag(H_I), U)
            GB_NhatI = self.hat(np.imag(GB_N), U)
            GB_NhatR = self.hat(np.real(GB_N), U)
            b11 = -1j*Ch(H_IhatR) - 0.5*Ah(GB_NhatR)
            b12 = 0.5*Ch(H_IhatI) - 0.25j*Ah(GB_NhatI)
            b21 = 2*Ch(H_IhatI) - 1j*Ah(GB_NhatI)

            diag.append(np.block([[b11, b12], [b21, b11]]))

        res = np.block([
            [g_nu, np.hstack(top_row)],
            [np.vstack(left_col), block_diag(*diag)]
        ])
        np.set_printoptions(precision=4, linewidth=500, threshold=np.inf)
        return np.real(res)

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
                # res_i += (1.0/(self.Tscale**3))*((T**3)/(2*(np.pi**2)))*(kc**2)*weight*(rminus_11 + rminus_22)
                res_i += (1.0 / (self.Tscale ** 3)) * (1.0 / (2 * (np.pi ** 2))) * (kc ** 2) * weight * (
                            rminus_11 + rminus_22)

            res.append(res_i)
        return np.array(res)

    def calc_lepton_asymmetry(self, sol, zlist):

        res = []

        for i, z in enumerate(zlist):
            T = Tz(z, self.mp.M)
            res.append((sol[i, 0] + sol[i, 1] + sol[i, 2]))

        return np.array(res)

    def solve(self, eigvals=False):
        quad = QuadratureInputs(self.kc_list, trapezoidal_weights(self.kc_list), self.rates)
        initial_state = self.get_initial_state(self.T0, self.mp, self.smdata, self.kc_list)

        # Integration bounds
        z0 = zT(self.T0, self.mp.M)
        zF = zT(self.TF, self.mp.M)

        # Output grid
        zlist = np.linspace(z0, zF, 200)

        # Eigenvalues
        self._eigenvalues = []

        # Construct system of equations
        def f_state(x, z):
            sysmat = self.coefficient_matrix(z, quad, self.mp, self.susc, self.smdata)
            jac = jacobian(z, self.mp, self.smdata)
            res = jac * (sysmat.dot(x))

            if eigvals:
                eig = speig((jac*sysmat), right=False, left=True)
                ix_small = np.argmin(np.abs(eig[0]))
                eval_small = np.abs(eig[0][ix_small])
                evec_small = np.abs(eig[1][:,ix_small])
                print("Smallest Eval: ", eval_small)
                print("Corresponding left Evec: ", evec_small)
                print("Normalized: ", evec_small / evec_small[0])
                self._eigenvalues.append(eig[0])

            return res

        def jac(x, z):
            return jacobian(z, self.mp, self.smdata) * \
                   self.coefficient_matrix(z, quad, self.mp, self.susc, self.smdata)

        # Solve them
        sol = odeint(f_state, initial_state, zlist, Dfun=jac, full_output=True, **self.ode_pars)
        self._total_lepton_asymmetry = self.calc_lepton_asymmetry(sol[0], zlist)
        self._total_hnl_asymmetry = self.calc_hnl_asymmetry(sol[0], zlist, quad)