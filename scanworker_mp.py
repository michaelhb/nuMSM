from common import *
import sys
import solvers
import time
from rates import Rates_Jurai
from quadrature import GaussianQuadrature
from scandb_mp import *
from mpi4py import MPI
import argparse

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
            model_params=sample, rates_interface=rates, TF=Tsph, H=sample.heirarchy, fixed_cutoff=sample.cutoff,
            eig_cutoff=False, method="Radau", ode_pars=ode_pars, source_term=True)

    elif sample.solvername == "QuadratureSolver":
        quadrature = GaussianQuadrature(
            sample.n_kc, 0.0, sample.kc_max, sample, sample.heirarchy, tot=True, qscheme="legendre")
        solver = solvers.QuadratureSolver(quadrature,
            model_params=sample, TF=Tsph, H=sample.heirarchy, fixed_cutoff=sample.cutoff, eig_cutoff=False,
            method="Radau", ode_pars=ode_pars, source_term=True)
    else:
        raise Exception("Unknown solver class!")

    return solver


# Command line interface
parser = argparse.ArgumentParser(description="MPI-enabled BAU scan tool")

parser.add_argument("--output-db", action="store", type=str, required=True)
parser.add_argument("--tag", action="store", type=str, default=None)
parser.add_argument("--save-solutions", action="store_true", default=False)

args = parser.parse_args()

db_path = args.output_db
tag = args.tag
save_solutions = args.save_solutions

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_workers = comm.Get_size() - 1

# Keep track of number of finished workers;
# terminate program when n_finished = n_proc
n_finished = 0

if rank == 0: # sample dispatcher / result recorder
    # Instantiate DB connection
    db = MPScanDB(db_path, fast_insert=True, save_solutions=save_solutions)

    # Purge any hung samples
    db.purge_hung_samples()

    # Message listen loop
    while True:
        message = comm.recv()

        # Message type 1: request for sample
        if message[0] == "sample_please":
            logging.info("proc 0: got sample request")

            worker_rank = message[1]
            start_get = time.time()
            sample = db.get_and_lock(tag)
            end_get = time.time()
            logging.info("got sample from db in {}s".format(end_get - start_get))

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
            if save_solutions:
                sample, bau, proc_time, worker_rank, solution = message
            else:
                sample, bau, proc_time, worker_rank = message

            logging.info("proc 0: got results from worker {}, writing to DB".format(worker_rank))
            start_wait = time.time()
            db.save_result(sample, bau, proc_time)

            if save_solutions:
                db.save_solution(sample, solution)

            end_wait = time.time()
            logging.info("db write took {}s".format(end_wait - start_wait))

else: # worker process
    while True:
        # Request a sample
        logging.info("worker {}, sending request for sample".format(rank))
        comm.send(("sample_please", rank), dest=0)

        # Wait for the sample
        start_wait = time.time()
        message = comm.recv(source=0)
        end_wait = time.time()
        time_wait = end_wait - start_wait

        # Check for a terminate message
        if message == "terminate_worker":
            logging.info("worker {}, terminating as requested".format(rank))
            sys.exit(0)
        # Otherwise, process the sample
        else:
            sample = message
            logging.info("worker {}: received sample for processing, waited {}s".format(rank, time_wait))
            solver = get_solver(sample)
            start = time.time()

            try:
                solver.solve(eigvals=False)
                end = time.time()
                time_sol = end - start
                bau = (28. / 78.) * solver.get_final_lepton_asymmetry()
                logging.info("Point {} finished in {} s, BAU = {}".format(sample, time_sol, bau))

                # Send result back to process 0
                logging.info("worker {}: sending results to proc 0".format(rank))

                if save_solutions:
                    Tlist = solver.get_Tlist()
                    baus = solver.get_total_lepton_asymmetry()
                    solution = list(zip(Tlist, baus))
                    comm.send((sample, bau, time_sol, rank, solution), dest=0)
                else:
                    comm.send((sample, bau, time_sol, rank), dest=0)

            except Exception as e:
                print("point failed: {}".format(sample))
                print(e)





