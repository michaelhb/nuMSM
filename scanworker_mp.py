from common import *
import yaml
import sys
import solvers
import time
from rates import Rates_Jurai
from quadrature import GaussianQuadrature
from scandb_mp import MPScanDB
from mpi4py import MPI

def get_solver(mp, solver_class, n_kc, kc_max, H, cutoff, ode_atol, ode_rtol):
    ode_pars = {'atol': ode_atol, 'rtol': ode_rtol}

    # Kludge city
    if solver_class == "AveragedSolver":
        rates = Rates_Jurai(mp, H, np.array([1.0]), tot=True)
        solver = solvers.AveragedSolver(
            model_params=mp, rates_interface=rates, TF=Tsph, H=H, fixed_cutoff=cutoff,
            eig_cutoff=False, method="Radau", ode_pars=ode_pars, source_term=True)

    elif solver_class == "QuadratureSolver":
        quadrature = GaussianQuadrature(
            n_kc, 0.0, kc_max, mp, H, tot=True, qscheme="legendre")
        solver = solvers.QuadratureSolver(quadrature,
            model_params=mp, TF=Tsph, H=H, fixed_cutoff=cutoff, eig_cutoff=False,
            method="Radau", ode_pars=ode_pars, source_term=True)
    else:
        raise Exception("Unknown solver class!")

    return solver

stream = yaml.open(sys.argv[1])
config = yaml.load(stream)

db_path = config["db_path"]
solver_class = config["solver_class"]
n_kc = int(config["n_kc"])
kc_max = int(config["kc_max"])
H = int(config["hierarchy"])
cutoff = float(config["cutoff"])
ode_atol = float(config["ode_atol"])
ode_rtol = float(config["ode_rtol"])

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_proc = comm.Get_size()

# Keep track of number of finished workers;
# terminate program when n_finished = n_proc
n_finished = 0

if rank == 0: # sample dispatcher / result recorder
    # Instantiate DB connection
    db = MPScanDB(db_path)

    # Purge any hung samples
    db.purge_hung_samples()

    # Message listen loop
    while True:
        message = comm.recv()

        # Message type 1: request for sample
        if message[0] == "sample_please":
            worker_rank = message[1]
            mp = db.get_and_lock(1)[0]

            # Samples are exhausted; shut down worker
            if mp == None:
                n_finished += 1
                comm.send("terminate_worker", dest=worker_rank)

                # This was the last worker, terminate program
                if n_finished == n_proc:
                    print("Finished scan!")
                    db.close_conn()
                    sys.exit(0)

            # Send sample as requested
            else:
                comm.send(mp, dest=worker_rank)

        # Message type 2: return of result
        else:
            mp, bau, time = message
            db.save_result(mp, bau, time)


else: # worker process
    while True:
        # Request a sample
        comm.send(("sample_please", rank), dest=0)

        # Wait for the sample
        message = comm.recv(source=0)

        # Check for a terminate message
        if message == "terminate_worker":
            sys.exit(0)
        # Otherwise, process the sample
        else:
            mp = message
            solver = get_solver(mp, solver_class, n_kc, kc_max, H, cutoff, ode_atol, ode_rtol)
            start = time.time()
            solver.solve(eigvals=False)
            end = time.time()
            time_sol = end - start
            bau = (28. / 78.) * solver.get_final_lepton_asymmetry()
            print("Point {} finished in {} s, BAU = {}".format(mp, time_sol, bau))

            # Send result back to process 0
            comm.send((mp, bau, time_sol), dest=0)



