import numpy as np
from os.path import expanduser

from scipy.integrate import odeint
import matplotlib.pyplot as plt

from common import *
from load_precomputed import *
from integrated_rates import *
from initial_conditions import *
from averaged_equations import *


"""
For now we'll just try to recreate the benchmark from Inar's notebook,
with a single momentum mode (AFAIK this corresponds to the averaged system)
"""

if __name__ == '__main__':

    # Model parameters
    M = 1.0  # HNLs mass
    dM = 1e-12  # mass difference

    Imw = np.log(3)
    Rew = 13 / 16 * np.pi
    delta = 29 / 16 * np.pi
    eta = 22 / 16 * np.pi

    mp = ModelParams(M, dM, Imw, Rew, delta, eta)

    # Load precomputed data files
    path_rates = expanduser("~/SciCodes/nuMSM/test_data/Int_OrgH_MN10E-1_kcAve.dat")
    path_SMdata = expanduser("~/SciCodes/nuMSM/test_data/standardmodel.dat")
    path_suscept_data = expanduser("~/SciCodes/nuMSM/test_data/susceptibility.dat")

    tc = get_rate_coefficients(path_rates)
    susc = get_susceptibility_matrix(path_suscept_data)
    smdata = get_sm_data(path_SMdata)

    # Change of variables
    def zT(T):
        return np.log(M/T)

    def Tz(z):
        return mp.M*np.exp(-z)

    # Get integrated rates
    rates = get_integrated_rates(mp, tc)

    # Get initial conditions
    T0 = get_T0(mp)
    initial_state = get_initial_state(T0, smdata)

    # Integration bounds
    z0 = zT(T0)
    zF = zT(10.)

    # Output grid
    zlist = np.linspace(z0, zF, 200)

    # Construct system of equations
    # f_state = lambda x_dot, z: np.dot(coefficient_matrix(z, rates, mp, susc, sm_data), x_dot)
    def f_state(x_dot, z):
        return jacobian(z, mp, smdata)*(
            np.dot(coefficient_matrix(z, rates, mp, susc, smdata), x_dot) +
            inhomogeneous_part(z, rates)
        )

    def jac(x_dot, z):
        return coefficient_matrix(z, rates, mp, susc, smdata)

    # Solve them
    sol = odeint(f_state, initial_state, zlist, Dfun=jac, rtol=1e-7, atol=1e-13, full_output=True)

    # Plot stuff
    Tlist = Tz(zlist)
    # plt.loglog(Tlist, np.abs(sol[0][:,0] + sol[0][:,1] + sol[0][:,2]))
    plt.loglog(Tlist, np.abs(sol[0][:, 7]))
    plt.show()
