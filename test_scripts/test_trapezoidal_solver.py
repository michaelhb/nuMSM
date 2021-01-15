from os import path
import sys
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from rates import get_rates
from quadrature_equations import *
from load_precomputed import *
from common import get_T0

# mp = ModelParams(
#     M=1.0,
#     dM=1e-12,
#     Imw=np.log(3),
#     Rew=13 / 16 * np.pi,
#     delta=29 / 16 * np.pi,
#     eta=22 / 16 * np.pi
# )
mp = ModelParams(
    M=1.0,
    dM=1e-11,
    Imw=np.log(3),
    Rew=1/4 * np.pi,
    delta= np.pi,
    eta=3/2 * np.pi
)

kc_list = [0.5, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1,
           3.3, 3.6, 3.9, 4.2, 4.6, 5.0, 5.7, 6.9, 10.0]
# kc_list = [0.5, 1.0, 2.5, 5.0, 10.0]
# kc_list = [2.1, 3.1]

# Change of variables
def zT(T):
    return np.log(mp.M / T)

def Tz(z):
    return mp.M*np.exp(-z)

def quad_weights(points):
    weights = [0.5 * (points[1] - points[0])]

    for i in range(1, len(points) - 1):
        weights.append(0.5 * (points[i + 1] - points[i - 1]))

    weights.append(0.5 * (points[-1] - points[-2]))
    return weights

def run_test():
    # Load rates for each momentum mode
    rates = []
    test_data = path.abspath(path.join(path.dirname(__file__), '../test_data/'))

    for kc in kc_list:
        fname = 'rates/Int_ModH_MN10E-1_kc{}E-1.dat'.format(int(kc * 10))
        path_rates = path.join(test_data, fname)
        tc = get_rate_coefficients(path_rates)
        rates.append(get_rates(mp, tc))

    # Load averaged rates for lepton-lepton part
    path_rates = path.join(test_data, "rates/Int_ModH_MN10E-1_kcAve.dat")
    tc_avg = get_rate_coefficients(path_rates)
    rt_avg = get_rates(mp, tc_avg)

    # Load standard model data, susceptibility matrix
    path_SMdata = path.join(test_data, "standardmodel.dat")
    path_suscept_data = path.join(test_data, "susceptibility.dat")
    susc = get_susceptibility_matrix(path_suscept_data)
    smdata = get_sm_data(path_SMdata)

    quad = QuadratureInputs(kc_list, quad_weights(kc_list), rates)

    # Set up initial conditions
    T0 = get_T0(mp)
    # initial_state = [0]*(3 + 8*len(kc_list))
    initial_state = get_initial_state_quad(T0, mp, smdata, kc_list)

    # Integration bounds
    z0 = zT(T0)
    zF = zT(10.)

    # For testing: plot geff over temp range
    # tic = np.linspace(z0, zF, 200)
    # geff_x = [Tz(ztick) for ztick in tic]
    # geff_plot = [smdata.geff(T) for T in geff_x]
    # plt.plot(geff_x, geff_plot)
    # plt.show()

    # Output grid
    zlist = np.linspace(z0, zF, 200)

    # Construct system of equations
    def f_state(x_dot, z):
        # print(inhomogeneous_part(z, quad, mp, smdata))
        # res = jacobian(z, mp, smdata)*(
        #     np.dot(coefficient_matrix(z, quad, rt_avg, mp, susc), x_dot) -
        #         inhomogeneous_part(z, quad, mp, smdata)
        # )

        res = jacobian(z, mp, smdata) * (
                np.dot(coefficient_matrix(z, quad, rt_avg, mp, susc), x_dot))

        return res

    def jac(x_dot, z):
        return jacobian(z, mp, smdata)*coefficient_matrix(z, quad, rt_avg, mp, susc)

    # Solve them
    sol = odeint(f_state, initial_state, zlist, Dfun=jac, rtol=1e-7, atol=1e-13, full_output=True)
    print(sol)

    # Plot stuff
    Tlist = Tz(zlist)

    plt.loglog(Tlist, np.abs(sol[0][:,0] + sol[0][:,1] + sol[0][:,2]))
    plt.show()

    for i in range(3):
        plt.loglog(Tlist, np.abs(sol[0][:, i]))
    plt.show()
    #
    for i in range(len(2*kc_list)):
        plt.loglog(Tlist, np.abs(sol[0][:, 3 + i]))
    plt.show()



