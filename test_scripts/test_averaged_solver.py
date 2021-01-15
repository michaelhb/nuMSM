from os import path
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from common import *
from load_precomputed import *
from rates import *
from averaged_equations import *

"""
For now we'll just try to recreate the benchmark from Inar's notebook,
with a single momentum mode (AFAIK this corresponds to the averaged system)
"""

benchmarks = [
    # From Inar's notebook
    ModelParams(
        M=1.0,
        dM=1e-12,
        Imw = np.log(3),
        Rew = 13 / 16 * np.pi,
        delta = 29 / 16 * np.pi,
        eta = 22 / 16 * np.pi
    ),
    # From Table 4
    ModelParams(
        M=5.0000e-01,
        dM=5.9794e-09,
        Imw=5.3897e+00,
        Rew=6.4803e-01*np.pi,
        delta=1.8599e+00*np.pi,
        eta=3.5291e-01*np.pi
    ),
    ModelParams(
        M=1.0000e+00,
        dM=5.3782e-09,
        Imw=5.2607e+00,
        Rew=8.3214e-01 * np.pi,
        delta=1.2708e+00 * np.pi,
        eta=1.7938e+00 * np.pi
    ),
    ModelParams(
        M=2.0000e+00,
        dM=3.0437e-09,
        Imw=5.5435e+00,
        Rew=1.6514e+00 * np.pi,
        delta=1.6384e+00 * np.pi,
        eta=7.5733e-01 * np.pi
    ),
    ModelParams(
        M=5.0000e+00,
        dM=1.7945e-09,
        Imw=5.229e+00,
        Rew=1.7753e+00 * np.pi,
        delta=1.4481e+00 * np.pi,
        eta=1.2070e+00 * np.pi              
    ),
    ModelParams(
        M=1.0000e+01,
        dM=2.7660e-09,
        Imw=4.4442e+00,
        Rew=8.4146e-01 * np.pi,
        delta=1.7963e+00 * np.pi,
        eta=9.2261e-01 * np.pi
    ),
]

def run_test():
    # Model parameters
    mp = benchmarks[0]

    # Load precomputed data files
    test_data = path.abspath(path.join(path.dirname(__file__), '../test_data/'))
    path_rates = path.join(test_data, "rates/Int_ModH_MN10E-1_kcAve.dat")
    path_SMdata = path.join(test_data, "standardmodel.dat")
    path_suscept_data = path.join(test_data, "susceptibility.dat")

    tc = get_rate_coefficients(path_rates)
    susc = get_susceptibility_matrix(path_suscept_data)
    smdata = get_sm_data(path_SMdata)

    # Change of variables
    def zT(T):
        return np.log(mp.M/T)

    def Tz(z):
        return mp.M*np.exp(-z)

    # Get integrated rates
    rates = get_rates(mp, tc)

    # Get initial conditions
    T0 = get_T0(mp)
    initial_state = get_initial_state_avg(T0, smdata)

    # Integration bounds
    z0 = zT(T0)
    zF = zT(10.)

    # Output grid
    zlist = np.linspace(z0, zF, 200)

    # Construct system of equations
    def f_state(x_dot, z):
        return jacobian(z, mp, smdata)*(
            np.dot(coefficient_matrix(z, rates, mp, susc), x_dot) +
            inhomogeneous_part(z, rates)
        )

    def jac(x_dot, z):
        return jacobian(z, mp, smdata)*coefficient_matrix(z, rates, mp, susc)

    # Solve them
    sol = odeint(f_state, initial_state, zlist, Dfun=jac, rtol=1e-6, atol=1e-13, full_output=True)
    print(sol)

    # Plot stuff
    Tlist = Tz(zlist)
    plt.loglog(Tlist, np.abs(sol[0][:,0] + sol[0][:,1] + sol[0][:,2]))
    # plt.loglog(Tlist, np.abs(sol[0][:, 7]))
    plt.show()


