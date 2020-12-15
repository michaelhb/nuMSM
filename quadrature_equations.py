from collections import namedtuple
import sys
from common import *
import numpy as np
from scipy.linalg import block_diag

# Quadrature: provide the values of kc + associated weights and rates.
QuadratureInputs = namedtuple("QuadratureInputs", [
    "kc_list", "weights", "rates"
])

# Equilibrium distribution for neutrinos
def f_nu(kc):
    return 1.0/(np.exp(kc) + 1)

# same as in the averaged solver
def gamma_omega(z, rt, susc):
    """
    :param z: integration coordinate z = ln(M/T)
    :param rt: see common.ModelParams
    :param smdata: see common.SMData
    :return: (3,3) matrix which mixes the n_delta in the upper left corner of the evolution matrix,
    proportional to T^2/6: -Re(GammaBar_nu_alpha).omega_alpha_beta (no sum over alpha)
    """
    GB_nu_a = rt.GB_nu_a(z)
    return (np.real(GB_nu_a).T*susc).T

def gamma_N(z, kc, rt, mp, susc, conj=False):
    T = mp.M * np.exp(-z)
    G_N = np.conj(rt.GBt_N_a(z)) if conj else rt.GBt_N_a(z)
    return (1.0/T)*f_nu(kc)*(1-f_nu(kc))*np.einsum('kij,aji,ab->kb', tau, G_N, susc(T))

def f_N(T, mp, kc):
    return 1.0/(np.exp(np.sqrt(mp.M**2 + ((T**2)*(kc**2)))/T) + 1.0)

def inhomogeneous_part(z, quad, mp, smdata):
    T = mp.M * np.exp(-z)
    s = smdata.s(T)

    b = [np.array([0, 0, 0])]
    n_kc = len(quad.kc_list)

    for i in range(n_kc):
        kc = quad.kc_list[i]
        rt = quad.rates[i]
        f_Nkc = f_N(T, mp, kc)
        r_eq = np.array([f_Nkc, f_Nkc, 0, 0])
        g_N = rt.GB_N(z)
        b.append(0.5*np.dot(Ah(g_N), r_eq))
        b.append(0.5*np.dot(Ah(np.conj(g_N)), r_eq))

    return np.real(np.concatenate(b))

def coefficient_matrix(z, quad, rt_avg, mp, susc):
    T = mp.M * np.exp(-z)

    # Top left block, only part that doesn't depend on kc.
    # g_nu = gamma_nu(z, quad, mp, susc)
    g_nu = -gamma_omega(z, rt_avg, susc(T))*(T**2)/6.

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
        g_N = gamma_N(z, kc, rt, mp, susc)
        g_Nc = gamma_N(z, kc, rt, mp, susc, conj=True)
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












