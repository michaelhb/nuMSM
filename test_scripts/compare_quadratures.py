from os import path, environ
environ["MKL_NUM_THREADS"] = "1"
environ["NUMEXPR_NUM_THREADS"] = "1"
environ["OMP_NUM_THREADS"] = "1"
environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")
from solvers import *
from quadrature import TrapezoidalQuadrature, GaussFermiDiracQuadrature
from rates import Rates_Jurai
from multiprocessing import Pool
import matplotlib.pyplot as plt
from collections import namedtuple

mp = ModelParams(
    M=1.0,
    dM=1e-12,
    # dM=0,
    Imw=4.1,
    Rew=1/4 * np.pi,
    delta= np.pi,
    eta=3/2 * np.pi
)

# Can't pickle these, but follow this format :/
# point = namedtuple("Point", ["mp", "quad_tag", "n_kc"])
# result = namedtuple("Result", ["point", "bau"])

cutoff = None

trap_kc_min = 0.1
trap_kc_max = 20

def get_bau(point):
    mp, quad_tag, n_kc = point

    T0 = get_T0(mp)

    if quad_tag == "GaussFermiDirac":
        quad_ = GaussFermiDiracQuadrature(n_kc, mp, H=1, tot=True)
    elif quad_tag == "Trapezoidal":
        kc_list = np.linspace(trap_kc_min, trap_kc_max, n_kc)
        rates = Rates_Jurai(mp, 1, kc_list, tot=True)
        quad_ = TrapezoidalQuadrature(kc_list, rates)
    else:
        raise Exception("Unknown quadrature type")

    solver = QuadratureSolver(
        quad_,
        model_params=mp, TF=Tsph, H=1, eig_cutoff=None, fixed_cutoff = cutoff,
        ode_pars={'atol': 1e-15, 'rtol': 1e-6}, method="Radau"
    )

    solver.solve(eigvals=False)
    return (quad_tag, n_kc, (28./78.) * solver.get_final_lepton_asymmetry())

if __name__ == "__main__":
    points = []

    # kc_counts = list(range(5, 41))
    # kc_counts = list(range(5, 21))
    # kc_counts = list(range(5, 8))
    kc_counts = np.array([5 + 5*i for i in range(20)])

    # Set up FD quadrature points
    for n_kc in kc_counts:
        points.append((mp, "GaussFermiDirac", n_kc))

    # Trap points will be linspaced in hardcoded global range
    for n_kc in kc_counts:
        points.append((mp, "Trapezoidal", n_kc))

    with Pool() as p:
        res = p.map(get_bau, points)

    res_GFD = [r[2] for r in sorted(filter(lambda r: r[0] == "GaussFermiDirac", res), key=lambda r: r[1])]
    res_trap = [r[2] for r in sorted(filter(lambda r: r[0] == "Trapezoidal", res), key=lambda r: r[1])]

    fig, ax = plt.subplots()

    fig.suptitle("Convergence, GFD vs. Trapezoidal")

    ax.scatter(kc_counts, res_GFD, color="green", label="GaussFermiDirac", s=5)
    ax.scatter(kc_counts, res_trap, color="red", label="Trapezoidal", s=5)
    ax.set_xlabel("n_kc")
    ax.set_ylabel("bau")
    ax.set_yscale("log")
    ax.set_xticks(kc_counts)
    ax.legend()

    output_dir = path.abspath(path.join(path.dirname(__file__), 'output/'))
    outfile = path.join(output_dir, "compare_quadratures.png")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()
