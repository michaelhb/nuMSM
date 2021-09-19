from common import *
import sys
import solvers
import time
from rates import Rates_Jurai
from quadrature import GaussianQuadrature
from scandb_mp import *
from mpi4py import MPI

# Ensure numpy, scipy, etc don't spawn extra threads
from os import environ
environ["MKL_NUM_THREADS"] = "1"
environ["NUMEXPR_NUM_THREADS"] = "1"
environ["OMP_NUM_THREADS"] = "1"
environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")

# Boldly ignore warnings that I probably shouldn't ignore
import warnings
warnings.filterwarnings(
    action='ignore',
    module=r'.*leptotools.*'
)
warnings.filterwarnings(
    action='ignore',
    module=r'.*common.*'
)

import logging
logging.basicConfig(level=logging.INFO)
ode_atol = 1e-20
ode_rtol = 1e-4

def get_solver(sample):
    ode_pars = {'atol': ode_atol, 'rtol': ode_rtol}

    # Kludge city
    if sample.solvername == "AveragedSolver":
        rates = Rates_Jurai(sample, sample.heirarchy, np.array([1.0]), tot=True)
        solver = solvers.AveragedSolver(
            model_params=mp, rates_interface=rates, TF=Tsph, H=sample.heirarchy, fixed_cutoff=sample.cutoff,
            eig_cutoff=False, method="Radau", ode_pars=ode_pars, source_term=True)

    elif sample.solvername == "QuadratureSolver":
        quadrature = GaussianQuadrature(
            sample.n_kc, 0.0, sample.kc_max, mp, sample.heirarchy, tot=True, qscheme="legendre")
        solver = solvers.QuadratureSolver(quadrature,
            model_params=mp, TF=Tsph, H=sample.heirarchy, fixed_cutoff=sample.cutoff, eig_cutoff=False,
            method="Radau", ode_pars=ode_pars, source_term=True)
    else:
        raise Exception("Unknown solver class!")

    return solver

db_path = sys.argv[1]

if len(sys.argv) > 2:
    tag = sys.argv[2]
else:
    tag = None

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_workers = comm.Get_size() - 1

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
            logging.info("proc 0: got sample request")

            worker_rank = message[1]
            sample = db.get_and_lock(tag)

            # Samples are exhausted; shut down worker
            if sample is None:
                n_finished += 1
                logging.info("proc 0: samples exhausted, terminating worker")
                comm.send("terminate_worker", dest=worker_rank)

                # This was the last worker, terminate program
                if n_finished == n_workers:
                    print("Finished scan!")
                    db.close_conn()
                    sys.exit(0)

            # Send sample as requested
            else:
                logging.info("proc 0: sending sample to worker {}".format(worker_rank))
                comm.send(sample, dest=worker_rank)

        # Message type 2: return of result
        else:
            mp, bau, time, worker_rank = message
            logging.info("proc 0: got results from worker {}, writing to DB".format(worker_rank))
            db.save_result(mp, tag, bau, time)

else: # worker process
    while True:
        # Request a sample
        logging.info("worker {}, sending request for sample".format(rank))
        comm.send(("sample_please", rank), dest=0)

        # Wait for the sample
        message = comm.recv(source=0)

        # Check for a terminate message
        if message == "terminate_worker":
            logging.info("worker {}, terminating as requested".format(rank))
            sys.exit(0)
        # Otherwise, process the sample
        else:
            sample = message
            logging.info("worker {}: recieved sample for processing".format(rank))
            solver = get_solver(sample)
            start = time.time()
            solver.solve(eigvals=False)
            end = time.time()
            time_sol = end - start
            bau = (28. / 78.) * solver.get_final_lepton_asymmetry()
            logging.info("Point {} finished in {} s, BAU = {}".format(sample, time_sol, bau))

            # Send result back to process 0
            logging.info("worker {}: sending results to proc 0".format(rank))
            comm.send((sample, bau, time_sol, rank), dest=0)



