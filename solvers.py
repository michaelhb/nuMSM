from abc import ABC, abstractmethod
from scipy.integrate import odeint
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from common import *
from load_precomputed import *
from rates import get_rates
from os import path

ode_par_defaults = {'rtol' : 1e-6, 'atol' : 1e-13}

class Solver(ABC):

    def __init__(self, model_params, T0, TF, H = 1, ode_pars = ode_par_defaults):
        self.T0 = T0
        self.TF = TF
        self.mp = model_params
        self._total_asymmetry = None
        self.H = H
        self.ode_pars = ode_pars
        super().__init__()

    @abstractmethod
    def solve(self):
        """
        Solve the kinetic equations
        :param TF: Lower temperature bound
        :return: None
        """
        pass

    def plot_total_asymmetry(self):
        Tlist = self.Tz(np.linspace(self.zT(self.T0), self.zT(self.TF), 200))
        plt.loglog(Tlist, self._total_asymmetry)
        plt.show()

    def get_final_asymmetry(self):
        return self._total_asymmetry[-1]

    def get_total_asymmetry(self):
        return self._total_asymmetry

    # Change of variables
    def zT(self, T):
        return np.log(self.mp.M/T)

    def Tz(self, z):
        return self.mp.M*np.exp(-z)

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

    def solve(self):

        # Get initial conditions
        initial_state = self.get_initial_state_avg(self.T0, self.smdata)

        # Integration bounds
        z0 = self.zT(self.T0)
        zF = self.zT(self.TF)

        # Output grid
        zlist = np.linspace(z0, zF, 200)
        tlist = self.Tz(zlist)

        # Construct system of equations
        def f_state(x_dot, z):
            return jacobian(z, self.mp, self.smdata)*(
                np.dot(self.coefficient_matrix(z, self.rates, self.mp, self.susc), x_dot) +
                self.inhomogeneous_part(z, self.rates)
            )

        def jac(x_dot, z):
            return jacobian(z, self.mp, self.smdata)*self.coefficient_matrix(z, self.rates, self.mp, self.susc)

        # Solve them
        sol = odeint(f_state, initial_state, zlist, Dfun=jac, full_output=True, **self.ode_pars)
        self._total_asymmetry = np.abs(sol[0][:,0] + sol[0][:,1] + sol[0][:,2])

    def get_initial_state_avg(self, T0, smdata):
        #equilibrium number density for a relativistic fermion, T = T0
        neq = (3.*zeta3*(T0**3))/(4*(np.pi**2))

        #entropy at T0
        s = smdata.s(T0)

        n_plus0 = -np.identity(2)*neq/s
        r_plus0 = np.einsum('kij,ji->k',tau,n_plus0)

        return np.real(np.concatenate([[0,0,0],r_plus0,[0,0,0,0]]))

    def gamma_omega(self, z, rt, susc):
        """
        :param z: integration coordinate z = ln(M/T)
        :param rt: see common.ModelParams
        :param smdata: see common.SMData
        :return: (3,3) matrix which mixes the n_delta in the upper left corner of the evolution matrix,
        proportional to T^2/6: -Re(GammaBar_nu_alpha).omega_alpha_beta (no sum over alpha)
        """
        GB_nu_a = rt.GB_nu_a(z)
        return (np.real(GB_nu_a).T*susc).T

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

        T = mp.M * np.exp(-z)
        GB_nu_a, GBt_nu_a, GBt_N_a, HB_N, GB_N, Seq = [R(z) for R in rt]
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

        return np.real(np.block([
            [b11,b12,b13],
            [b21,b22,b23],
            [b31,b32,b33]
        ]))

"""
Data type to describe a quadrature scheme, incl. the list of 
momenta points, weights and rates.
"""
QuadratureInputs = namedtuple("QuadratureInputs", [
    "kc_list", "weights", "rates"
])

class QuadratureSolver:
    """
    Helper class for solvers implementing quadrature over momentum.
    TODO: Move more things from TrapezoidalSolver into this class...
    """
    # Initial condition
    def get_initial_state_quad(self, T0, mp, smdata, kc_list):
        x0 = [0, 0, 0]
        s = smdata.s(T0)

        for kc in kc_list:
            rho0 = -1*f_N(T0, mp, kc)*np.identity(2)/s
            r0 = np.einsum('kij,ji->k', tau, rho0)
            x0.extend(r0)
            x0.extend(r0)

        return np.real(x0)

    # same as in the averaged solver
    def gamma_omega(self, z, rt, susc):
        """
        :param z: integration coordinate z = ln(M/T)
        :param rt: see common.ModelParams
        :param smdata: see common.SMData
        :return: (3,3) matrix which mixes the n_delta in the upper left corner of the evolution matrix,
        proportional to T^2/6: -Re(GammaBar_nu_alpha).omega_alpha_beta (no sum over alpha)
        """
        GB_nu_a = rt.GB_nu_a(z)
        return (np.real(GB_nu_a).T*susc).T

    def gamma_N(self, z, kc, rt, mp, susc, conj=False):
        T = mp.M * np.exp(-z)
        G_N = np.conj(rt.GBt_N_a(z)) if conj else rt.GBt_N_a(z)
        return (1.0/T)*f_nu(kc)*(1-f_nu(kc))*np.einsum('kij,aji,ab->kb', tau, G_N, susc(T))

    def coefficient_matrix(self, z, quad, rt_avg, mp, susc):
        T = mp.M * np.exp(-z)

        # Top left block, only part that doesn't depend on kc.
        g_nu = -self.gamma_omega(z, rt_avg, susc(T))*(T**2)/6.

        top_row = []
        left_col = []
        diag = []

        n_kc = len(quad.kc_list)

        for i in range(n_kc):
            kc = quad.kc_list[i]
            rt = quad.rates[i]
            w_i = quad.weights[i]

            GB_nu_a, GBt_nu_a, GBt_N_a, HB_N, GB_N, Seq = [R(z) for R in rt]

            # Top row
            top_row.append(-w_i*(T**3/(2.0*(np.pi**2)))*tr_h(np.conj(GBt_nu_a)))
            top_row.append(w_i*(T**3/(2.0*(np.pi**2)))*tr_h(GBt_nu_a))

            # Left column
            g_N = self.gamma_N(z, kc, rt, mp, susc)
            g_Nc = self.gamma_N(z, kc, rt, mp, susc, conj=True)
            left_col.append(-g_N)
            left_col.append(g_Nc)

            # Diagonal
            diag.append(-1j*Ch(HB_N) - 0.5*Ah(GB_N))
            diag.append(-1j*Ch(np.conj(HB_N)) - 0.5*Ah(np.conj(GB_N)))

        # Construct the full matrix. Not using a sparse format yet, but should think
        # about it...

        return np.real(np.block([
            [g_nu, np.hstack(top_row)],
            [np.vstack(left_col), block_diag(*diag)]
        ]))

class TrapezoidalSolver(Solver, QuadratureSolver):

    def __init__(self, model_params, T0, TF, H = 1, ode_pars = ode_par_defaults):
        self.kc_list = [0.5, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1,
                3.3, 3.6, 3.9, 4.2, 4.6, 5.0, 5.7, 6.9, 10.0]

        self.mp = model_params
        self.rates = []

        test_data = path.abspath(path.join(path.dirname(__file__), 'test_data/'))

        for kc in self.kc_list:
            fname = 'rates/Int_ModH_MN10E-1_kc{}E-1.dat'.format(int(kc * 10))
            path_rates = path.join(test_data, fname)
            tc = get_rate_coefficients(path_rates)
            self.rates.append(get_rates(self.mp, tc, H))

        # Load averaged rates for lepton-lepton part
        path_rates = path.join(test_data, "rates/Int_ModH_MN10E-1_kcAve.dat")
        self.tc_avg = get_rate_coefficients(path_rates)
        self.rt_avg = get_rates(self.mp, self.tc_avg)

        # Load standard model data, susceptibility matrix
        path_SMdata = path.join(test_data, "standardmodel.dat")
        path_suscept_data = path.join(test_data, "susceptibility.dat")
        self.susc = get_susceptibility_matrix(path_suscept_data)
        self.smdata = get_sm_data(path_SMdata)

        Solver.__init__(self, model_params, T0, TF, H, ode_pars)

    def quad_weights(self, points):
        weights = [0.5 * (points[1] - points[0])]

        for i in range(1, len(points) - 1):
            weights.append(0.5 * (points[i + 1] - points[i - 1]))

        weights.append(0.5 * (points[-1] - points[-2]))
        return weights

    def solve(self):

        quad = QuadratureInputs(self.kc_list, self.quad_weights(self.kc_list), self.rates)
        initial_state = self.get_initial_state_quad(self.T0, self.mp, self.smdata, self.kc_list)

        # Integration bounds
        z0 = self.zT(self.T0)
        zF = self.zT(self.TF)

        # Output grid
        zlist = np.linspace(z0, zF, 200)

        # Construct system of equations
        def f_state(x_dot, z):
            res = jacobian(z, self.mp, self.smdata) * (
                np.dot(self.coefficient_matrix(z, quad, self.rt_avg, self.mp, self.susc), x_dot))
            return res

        def jac(x_dot, z):
            return jacobian(z, self.mp, self.smdata) * \
                   self.coefficient_matrix(z, quad, self.rt_avg, self.mp, self.susc)

        # Solve them
        # sol = odeint(f_state, initial_state, zlist, Dfun=jac, rtol=1e-7, atol=1e-13, full_output=True)
        sol = odeint(f_state, initial_state, zlist, Dfun=jac, full_output=True, **self.ode_pars)

        self._total_asymmetry = np.abs(sol[0][:, 0] + sol[0][:, 1] + sol[0][:, 2])

