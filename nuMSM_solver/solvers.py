from abc import ABC, abstractmethod
from os import path

import matplotlib as mpl
from scipy import integrate
from scipy.integrate import odeint, solve_ivp
from scipy.linalg import block_diag
from scipy.sparse.linalg import eigs as sparse_eig

from nuMSM_solver.common import *
from nuMSM_solver.load_precomputed import *

mpl.rcParams['figure.dpi'] = 300


class SolverInterface(ABC):

    @abstractmethod
    def solve(self):
        """
        Solve the kinetic equations
        :param TF: Lower temperature bound
        :return: None
        """
        pass

    @abstractmethod
    def get_full_solution(self):
        pass

    @abstractmethod
    def get_final_lepton_asymmetry(self):
        pass

    @abstractmethod
    def get_total_lepton_asymmetry(self):
        pass

    @abstractmethod
    def get_Tlist(self):
        pass


class Solver(SolverInterface):

    def __init__(self, model_params=None, TF=Tsph, H = 1, eig_cutoff = False, cutoff = None,
                 output_grid_size = 200, ode_pars = ode_par_defaults, method="Radau", source_term=True):

        self.TF = TF
        self.mp = model_params
        self.T0 = get_T0(self.mp)
        self.zlist = np.linspace(zT(self.T0, self.mp.M), zT(self.TF, self.mp.M), output_grid_size)
        self.Tlist = Tz(self.zlist, self.mp.M)

        self._total_lepton_asymmetry = None
        self._total_hnl_asymmetry = None

        self.H = H
        self.ode_pars = ode_pars
        self.eig_cutoff = eig_cutoff
        self.fixed_cutoff = cutoff
        self.method = method
        self.use_source_term = source_term

        if eig_cutoff and (cutoff is not None):
            raise Exception("Cannot use fixed and dynamic cutoff at the same time")

        super().__init__()

    @abstractmethod
    def solve(self):
        """
        Solve the kinetic equations
        :param TF: Lower temperature bound
        :return: None
        """
        pass

    def get_full_solution(self):
        return self._full_solution

    def get_final_lepton_asymmetry(self):
        return self._total_lepton_asymmetry[-1]

    def get_total_lepton_asymmetry(self):
        return self._total_lepton_asymmetry

    def get_total_hnl_asymmetry(self):
        return self._total_hnl_asymmetry

    def get_Tlist(self):
        return self.Tlist


class AveragedSolver(Solver):

    def __init__(self, rates_interface, **kwargs):
        super(AveragedSolver, self).__init__(**kwargs)

        # Load rates
        self.rates = rates_interface.get_averaged_rates()

        # Load standard model data
        test_data = path.abspath(path.join(path.dirname(__file__), '../test_data/'))
        path_SMdata = path.join(test_data, "standardmodel.dat")
        path_suscept_data = path.join(test_data, "susceptibility.dat")
        self.susc = get_susceptibility_matrix(path_suscept_data)
        self.smdata = get_sm_data(path_SMdata)

    def solve(self):

        # Get initial conditions
        initial_state = self.get_initial_state()

        # Integration bounds
        z0 = zT(self.T0, self.mp.M)
        zF = zT(self.TF, self.mp.M)

        # Output grid
        zlist = np.linspace(z0, zF, 200)

        # Construct system of equations
        def f_state(z, x):
            sysmat = self.coefficient_matrix(z)
            jac = jacobian(z, self.mp, self.smdata)

            if self.use_source_term:
                res = (np.dot(sysmat, x) - self.source_term(z))
            else:
                res = np.dot(sysmat, x)

            return res

        def jac(z, x):
            return self.coefficient_matrix(z)

        # Solve them
        if self.method == None or self.method == "LSODE":
            # Default to old odeint / LSODE integrator
            sol = odeint(f_state, initial_state, zlist, Dfun=jac, full_output=True, tfirst=True, **self.ode_pars)
            self._total_lepton_asymmetry = sol[0][:, 0] + sol[0][:, 1] + sol[0][:, 2]
            self._total_hnl_asymmetry = sol[0][:, 7] + sol[0][:, 8]
            self._full_solution = sol[0]
        else:
            # Otherwise try solve_ivp
            sol = solve_ivp(f_state, (z0, zF), y0=initial_state, jac=jac, t_eval=zlist, method=self.method, **self.ode_pars)
            self._total_lepton_asymmetry = sol.y.T[:, 0] + sol.y.T[:, 1] + sol.y.T[:, 2]
            self._total_hnl_asymmetry = sol.y.T[:, 7] + sol.y.T[:, 8]
            self._full_solution = sol.y.T

    def get_initial_state(self):
        #equilibrium number density for a relativistic neutrino, T = T0
        neq = (3.*zeta3*(self.T0**3))/(4*(np.pi**2))
        # neq = (3. * zeta3 * (self.T0 ** 3)) / (2 * (np.pi ** 2)) # don't forget g!

        #entropy at T0
        s = self.smdata.s(self.T0)

        n_plus0 = -np.identity(2)*neq/s
        # n_plus0 = -((self.T0**3)*np.identity(2) * neq) / s
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
        GB_nu_a = self.rates.Gamma_nu_a(z)
        return (GB_nu_a.T * self.susc(Tz(z, self.mp.M))).T

    def Yr(self, z):
        """
        :param z: integration coordinate z = ln(M/T)
        :param rt: see common.ModelParams
        :return: (4,3) matrix appearing in the (3,1) block of the evolution matrix
        """
        T = Tz(z, self.mp.M)
        susc = self.susc(T)
        reGBt_N_a = np.real(self.rates.GammaTilde_N_a(z)) #* (T**3)
        return np.einsum('kij,aji,ab->kb',tau,reGBt_N_a,susc)

    def Yi(self, z):
        """
        :param z: integration coordinate z = ln(M/T)
        :param rt: see common.ModelParams
        :return: (4,3) matrix appearing in the (2,1) block of the evolution matrix
        """
        T = Tz(z, self.mp.M)
        susc = self.susc(T)
        imGBt_N_a = np.imag(self.rates.GammaTilde_N_a(z)) #* (T**3)
        return np.einsum('kij,aji,ab->kb',tau,imGBt_N_a,susc)

    def source_term(self, z):
        """
        :param rt: see common.IntegratedRates
        :return: inhomogeneous part of the ODE system
        """
        T = Tz(z, self.mp.M)
        jac = jacobian(z, self.mp, self.smdata)

        I = lambda kc: (kc**2)*(f_Ndot(kc, T, self.mp, self.smdata) + \
                                (3*(T**2)/MpStar(z, self.mp, self.smdata))*f_N(T, self.mp.M, kc))
        Seq = ((T**3)/(2*np.pi**2))*(1.0/self.smdata.s(T))*integrate.quad(I, 0, np.inf)[0]

        return jac*np.array([0, 0, 0, Seq, Seq, 0, 0, 0, 0, 0, 0])

        # Seq = self.rates.Seq(z)
        # seq = np.einsum('kij,ji->k', tau, Seq)
        # return jac*np.real(np.concatenate([[0, 0, 0], seq, [0, 0, 0, 0]]))

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
        GB_nu_a, GBt_nu_a, GBt_N_a, HB_N, HB_I,  GB_N, Seq = [R(z) for R in self.rates]

        jac = jacobian(z, self.mp, self.smdata)

        HB_0 = HB_N - HB_I

        # Top row
        b11 = -self.gamma_omega(z)*(T**2)/6.
        b12 = 2j*tr_h(np.imag(GBt_nu_a))
        b13 = -tr_h(np.real(GBt_nu_a))

        # Left col
        b21 = -(1j/2.)*self.Yi(z)*(T**2)/6. #* (T**3)
        b31 = -self.Yr(z)*(T**2)/6. #* (T**3)

        # Inner block
        b22_HI = -1j * Ch(np.real(HB_I)) - (1. / 2.) * Ah(np.real(GB_N))
        b23_HI = (1. / 2.) * Ch(np.imag(HB_I)) - (1j / 4.) * Ah(np.imag(GB_N))
        b32_HI = 2 * Ch(np.imag(HB_I)) - 1j * Ah(np.imag(GB_N))
        b33_HI = b22_HI

        b22_H0 = -1j * Ch(np.real(HB_0))
        b23_H0 = (1. / 2.) * Ch(np.imag(HB_0))
        b32_H0 = 2 * Ch(np.imag(HB_0))
        b33_H0 = b22_H0

        res = np.block([
            [b11,b12,b13],
            [b21,b22_H0 + b22_HI,b23_H0 + b23_HI],
            [b31,b32_H0 + b32_HI,b33_H0 + b33_HI]
        ])

        if self.eig_cutoff or (self.fixed_cutoff is not None):
            H0_inner = np.real(np.block([
                [b22_H0, b23_H0],
                [b32_H0, b33_H0]
            ]))

            if self.eig_cutoff:
                evs = np.abs(np.linalg.eigvals(H0_inner))
                cutoff = 1.5*np.sort(evs)[-1]
            else:
                cutoff = self.fixed_cutoff

            Aprime = np.real(np.block([
                [np.zeros((3,3)), np.zeros((3,8))],
                [np.zeros((8,3)), H0_inner]
            ]))
            return np.real(np.dot(jac*res, np.linalg.inv(-jac*Aprime/cutoff + np.eye(11))))
        else:
            return jac*np.real(res)

    def get_densities(self):
        densities = []

        for i_T, T in enumerate(self.get_Tlist()):
            density = [T, self.smdata.s(T).tolist()]
            density += self._full_solution[i_T][3:11].tolist()
            densities.append(density)

        return densities

class QuadratureSolver(Solver):

    def __init__(self, rates_interface, quadrature, **kwargs):
        Solver.__init__(self, **kwargs)

        self.kc_list = quadrature.kc_list()
        self.weights = quadrature.weights()

        self.rates = []
        for kc in self.kc_list:
            self.rates.append(rates_interface.get_rates(kc))

        test_data = path.abspath(path.join(path.dirname(__file__), '../test_data/'))

        # Load standard model data, susceptibility matrix
        path_SMdata = path.join(test_data, "standardmodel.dat")
        path_suscept_data = path.join(test_data, "susceptibility.dat")
        self.susc = get_susceptibility_matrix(path_suscept_data)
        self.smdata = get_sm_data(path_SMdata)

    # Initial condition
    def get_initial_state(self):
        x0 = [0, 0, 0]

        for kc in self.kc_list:
            rho_plus_0 = -1 * (self.T0 ** 3) * f_N(self.T0, self.mp.M, kc) * np.identity(2) / self.smdata.s(self.T0)
            r_plus_0 = np.einsum('kij,ji->k', tau, rho_plus_0)
            x0.extend(r_plus_0)
            x0.extend([0, 0, 0, 0])

        return np.real(x0)

    def gamma_omega(self, z):
        T = Tz(z, self.mp.M)

        # Integrate rate
        g_int = np.zeros(3, dtype="complex128")
        for wi, rt, kc in zip(self.weights, self.rates, self.kc_list):
            g_int += wi*(kc**2)*rt.Gamma_nu_a(z)*f_nu(kc)*(1 - f_nu(kc))

        g_int *= (T**2)/(np.pi**2)

        # Contract with susc matrix
        return (g_int.T * self.susc(T)).T

    def gamma_N(self, z, kc, rt, imag=False):
        T = Tz(z, self.mp.M)

        G_N = np.imag(rt.GammaTilde_N_a(z)) if imag else np.real(rt.GammaTilde_N_a(z))

        return 2.0 * T**2 * f_nu(kc) * (1 - f_nu(kc)) * np.einsum('ab,kij,aji->kb', self.susc(T), tau,
                                                                                      G_N)

    def source_term(self, z):
        T = Tz(z, self.mp.M)
        jac = jacobian(z, self.mp, self.smdata)
        Svec = [0,0,0]

        for j, kc in enumerate(self.kc_list):
            kc = self.kc_list[j]
            St = -1*(T**3)*(1.0/self.smdata.s(T))*(
                f_Ndot(kc, T, self.mp, self.smdata) + \
                3.0*((T**2)/MpStar(z, self.mp, self.smdata))*f_N(T, self.mp.M, kc))
            Svec += [St, St, 0, 0, 0, 0, 0, 0]

        return jac * np.array(Svec)

    def coefficient_matrix(self, z):
        T = Tz(z, self.mp.M)

        # Top left block, only part that doesn't depend on kc.
        g_nu = -self.gamma_omega(z)

        top_row = []
        left_col = []
        diag_H0 = []
        diag_HI = []

        n_kc = len(self.kc_list)

        jac = jacobian(z, self.mp, self.smdata)

        for i in range(n_kc):
            kc = self.kc_list[i]
            rt = self.rates[i]
            w_i = self.weights[i]

            G_nu_a, Gt_nu_a, Gt_N_a, H_N, H_I, G_N, Seq = [R(z) for R in rt]

            W = (1.0 / (2 * (np.pi ** 2))) * w_i * (kc ** 2)

            # Top row
            top_row.append(2j*W*tr_h(np.imag(Gt_nu_a)))
            top_row.append(-W*tr_h(np.real(Gt_nu_a)))

            # Left col
            left_col.append(-0.5j*self.gamma_N(z, kc, rt, imag=True))
            left_col.append(-1*self.gamma_N(z, kc, rt, imag=False))

            H_0 = H_N - H_I

            # Diag blocks

            # Part involving the interaction Hamiltonian
            b11_HI = -1j*Ch(np.real(H_I)) - 0.5*Ah(np.real(G_N))
            b12_HI = 0.5*Ch(np.imag(H_I)) - 0.25j*Ah(np.imag(G_N))
            b21_HI = 2*Ch(np.imag(H_I)) - 1j*Ah(np.imag(G_N))
            diag_HI.append(np.block([[b11_HI, b12_HI], [b21_HI, b11_HI]]))

            # Everything else
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
            cutoff = K*np.abs(jac*sparse_eig(block_diag(*diag_H0), k=1, which="LM", return_eigenvectors=False)[0])
            return np.dot(jac*res, np.linalg.inv(-jac*Aprime / cutoff + np.eye(3 + 8 * n_kc)))
        elif self.fixed_cutoff is not None:
            Aprime = np.real(np.block([
                [np.zeros((3,3)), np.zeros((3,8*n_kc))],
                [np.zeros((8*n_kc,3)), block_diag(*diag_H0)]
            ]))
            cutoff = self.fixed_cutoff
            return np.dot(jac*res, np.linalg.inv(-jac*Aprime/cutoff + np.eye(3 + 8*n_kc)))
        else:
            return jac * res

    def calc_hnl_asymmetry(self, sol, zlist):
        res = []

        for i, z in enumerate(zlist):
            res_i = 0

            for j, kc in enumerate(self.kc_list):
                kc = self.kc_list[j]
                weight = self.weights[j]
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

    def solve(self):
        initial_state = self.get_initial_state()

        # Integration bounds
        z0 = zT(self.T0, self.mp.M)
        zF = zT(self.TF, self.mp.M)

        # Output grid
        # zlist = [zT(T, self.mp.M) for T in self.get_Tlist()]
        # zlist = np.linspace(zT(self.T0, self.mp.M), zT(self.TF, self.mp.M), self.output_grid_size)

        # Construct system of equations
        def f_state(z, x):
            sysmat = self.coefficient_matrix(z)
            res = (sysmat.dot(x))

            if self.use_source_term:
                return res + self.source_term(z)
            else:
                return res

        def jac(z, x):
            return self.coefficient_matrix(z)

        # Solve them
        if self.method == None or self.method == "LSODE":
            # Default to old odeint / LSODE integrator
            sol = odeint(f_state, initial_state, self.zlist, Dfun=jac, full_output=True, tfirst=True, **self.ode_pars)
            self._total_lepton_asymmetry = self.calc_lepton_asymmetry(sol[0], self.zlist)
            self._total_hnl_asymmetry = self.calc_hnl_asymmetry(sol[0], self.zlist)
            self._full_solution = sol[0]
        else:
            # Otherwise try solve_ivp
            sol = solve_ivp(f_state, (z0, zF), y0=initial_state, jac=jac, t_eval=self.zlist, method=self.method, **self.ode_pars)
            print(sol.message)
            print(sol.success)
            self._total_lepton_asymmetry = self.calc_lepton_asymmetry(sol.y.T, self.zlist)
            self._total_hnl_asymmetry = self.calc_hnl_asymmetry(sol.y.T, self.zlist)
            self._full_solution = sol.y.T

    def get_densities(self):
        densities = []

        for i_T, T in enumerate(self.get_Tlist()):
            for i_kc, kc in enumerate(self.kc_list):
                ix_start = 3 + 8*i_kc
                density = [T, kc, self.weights[i_kc], self.smdata.s(T).tolist()]
                density += self._full_solution[i_T][ix_start : ix_start + 8].tolist()
                densities.append(density)

        return densities