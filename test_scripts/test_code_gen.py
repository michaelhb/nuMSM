import numpy as np
from os.path import expanduser

from scipy.integrate import odeint
import matplotlib.pyplot as plt

from common import *
from load_precomputed import *
from rates import *
from initial_conditions import *
from averaged_equations import *

"""
Print out the coefficient matrix, inhomogeneous vector, T-> Z jacobian, and integrated rates
at some fixed temperature so that it can be compared to what is generated in Mathematica 
"""
T_test = 80.

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
    rates = get_rates(mp, tc)

    # Print rates at T_test
    for k, v in rates._asdict().items():
        print(k)
        print(repr(v(zT(T_test))))

    # Print T->z jacobian
    print("T->z jacobian")
    print(jacobian(zT(T_test), mp, smdata))

    # Print inhomogeneous vector
    print("Inhomogeneous part")
    print(inhomogeneous_part(zT(T_test), rates))

    # Print susceptibility matrix
    print("Susceptibility matrix")
    print(repr(susc(T_test)))

    # Print coefficient matrix
    print("Coefficient matrix")
    print(repr(coefficient_matrix(zT(T_test), rates, mp, susc, smdata)))