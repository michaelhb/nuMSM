from scandb_mp import *
import yaml
import argparse

if __name__ == '__main__':
    # Command line interface
    parser = argparse.ArgumentParser(description="Create DB for convergence scan (varying number of modes)")
    parser.add_argument("--yaml", action="store", type=str, required=True)
    parser.add_argument("--db", action="store", type=str, required=True)
    parser.add_argument("--nkc-list", nargs="+", type=int)
    parser.add_argument("--nkc-min", action="store", type=int, required=False)
    parser.add_argument("--nkc-max", action="store", type=int, required=False)
    parser.add_argument("--nkc-step", action="store", type=int, required=False)

    args = parser.parse_args()

    # Load yaml file
    stream = open(args.yaml, 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)

    # Instantiate DB
    db = MPScanDB(args.db, fast_insert=True)

    delta_opt = np.pi
    eta_opt = (3.0*np.pi)/2.0
    Rew_opt = np.pi/4.0

    if args.nkc_list is not None:
        n_kcs = args.nkc_list
    else:
        n_kcs = []

    if None not in set([args.nkc_min, args.nkc_max, args.nkc_step]):
        n_kcs += list(range(args.nkc_min, args.nkc_max + args.nkc_step, args.nkc_step))

    for tag, attrs in config.items():
        desc = attrs["description"]
        H = int(attrs["heirarchy"])
        M = float(attrs["M"])
        dM = float(attrs["dM"])
        Imw = float(attrs["Imw"])
        kc_min = float(attrs["kc_min"])
        kc_max = float(attrs["kc_max"])
        quadscheme = attrs["quadscheme"]

        cutoff = attrs["cutoff"]
        if cutoff is not None:
            cutoff = float(cutoff)

        for n_kc in n_kcs:
            mp = ModelParams(M=M, dM=dM, Imw=Imw, Rew=Rew_opt, delta=delta_opt, eta=eta_opt)

            sample = Sample(**mp._asdict(), tag=tag, description=desc, solvername="QuadratureSolver",
                            n_kc=n_kc, kc_min=kc_min, kc_max=kc_max, quadscheme=quadscheme, heirarchy=H, cutoff=cutoff)

            db.add_sample(sample)