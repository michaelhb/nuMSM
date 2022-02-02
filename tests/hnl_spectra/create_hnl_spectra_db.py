from scandb_mp import *
import yaml
import argparse

if __name__ == '__main__':
    # Command line interface
    parser = argparse.ArgumentParser(description="Create DB for convergence scan (varying number of modes)")
    parser.add_argument("--yaml", action="store", type=str, required=True)
    parser.add_argument("--db", action="store", type=str, required=True)

    args = parser.parse_args()

    # Load yaml file
    stream = open(args.yaml, 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)

    # Instantiate DB
    db = MPScanDB(args.db, fast_insert=True)

    delta_opt = np.pi
    eta_opt = (3.0*np.pi)/2.0
    Rew_opt = np.pi/4.0

    for tag, attrs in config.items():
        desc = attrs["description"]
        H = int(attrs["heirarchy"])
        M = float(attrs["M"])
        dM = float(attrs["dM"])
        Imw = float(attrs["Imw"])
        solver_class = attrs["solver_class"]

        if solver_class == "AveragedSolver":
            quadscheme = None
            kc_min = -1
            kc_max = -1
            n_kc = -1
        elif solver_class == "QuadratureSolver":
            quadscheme = attrs["quadscheme"]
            kc_min = float(attrs["kc_min"])
            kc_max = float(attrs["kc_max"])
            n_kc = int(attrs["n_kc"])
        else:
            raise Exception("Unknown solver!")


        cutoff = attrs["cutoff"]
        if cutoff is not None:
            cutoff = float(cutoff)

        mp = ModelParams(M=M, dM=dM, Imw=Imw, Rew=Rew_opt, delta=delta_opt, eta=eta_opt)

        sample = Sample(**mp._asdict(), tag=tag, description=desc, solvername=solver_class,
                        n_kc=n_kc, kc_min=kc_min, kc_max=kc_max, quadscheme=quadscheme, heirarchy=H, cutoff=cutoff)

        db.add_sample(sample)

