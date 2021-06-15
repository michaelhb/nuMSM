from os import path, environ
from sys import argv
import yaml
environ["MKL_NUM_THREADS"] = "1"
environ["NUMEXPR_NUM_THREADS"] = "1"
environ["OMP_NUM_THREADS"] = "1"
environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                           "intra_op_parallelism_threads=1")
from solvers import *
from multiprocessing import Pool, set_start_method
from plots import heatmap_dm_imw, contour_dm_imw, heatmap_dm_imw_timing, contour_dm_imw_comp
from scandb import ScanDB

kc_list = np.array([0.5, 1.0, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1,
                3.3, 3.6, 3.9, 4.2, 4.6, 5.0, 5.7, 6.9, 10.0])

def get_scan_points(points_per_dim, M, delta, eta, Rew):
    dM_min = -16
    dM_max = -1
    dMs = [10**e for e in np.linspace(dM_min, dM_max, points_per_dim)]

    Imw_min = -6.
    Imw_max = 6.
    Imws = np.linspace(Imw_min, Imw_max, points_per_dim)

    points = []

    for dM in dMs:
        for Imw in Imws:
            points.append([dM, Imw, ModelParams(M, dM, Imw, Rew, delta, eta)])

    return np.array(points)

def get_bau(point):
    dM, Imw, mp, scan_db_path, tag = point

    print("Starting {} point {}".format(tag, mp))
    T0 = get_T0(mp)


    if tag == "std":
        if (np.abs(Imw) > 5.0):
            print("Cutoff: None")
            solver = QuadratureSolver(mp, T0, Tsph, kc_list, H=1, ode_pars={'atol': 1e-13}, method="Radau")
        else:
            print("Cutoff: 1e4")
            solver = QuadratureSolver(mp, T0, Tsph, kc_list, H=1, fixed_cutoff=1e4, ode_pars={'atol': 1e-10},
                                      method="Radau")
    elif tag == "avg":
        if (np.abs(Imw) > 5.0):
            print("Cutoff: None")
            solver = AveragedSolver(mp, T0, Tsph, H=1, ode_pars={'atol': 1e-13}, method="Radau")
        else:
            print("Cutoff: 1e4")
            solver = AveragedSolver(mp, T0, Tsph, H=1, fixed_cutoff=1e4, ode_pars={'atol': 1e-10},
                                          method="Radau")
    else:
        raise Exception("unknown tag")

    start = time.time()
    solver.solve(eigvals=False)
    end = time.time()
    time_sol = end - start
    bau = (28./78.) * solver.get_final_lepton_asymmetry()
    print("Point {} finished in {} s, BAU = {}".format(mp, time_sol, bau))
    res_db = ScanDB(scan_db_path)
    res_db.save_bau(mp, tag, bau, time_sol)
    res_db.close_conn()
    return (bau, dM, Imw, time_sol)

if __name__ == '__main__':
    # Args: input yaml, db path, axsize
    yaml_file = argv[1]
    db_name = argv[2]
    axsize = argv[3]

    yaml_file = path.abspath(path.join(path.dirname(__file__), yaml_file))
    with open(yaml_file) as file:
        doc = yaml.load(file, Loader=yaml.FullLoader)
        M = doc["M"]
        delta = doc["delta"]
        eta = doc["eta"]
        Rew = doc["rew"]
        avg = doc["avg"]
        tag = doc["tag"]

    output_dir = path.abspath(path.join(path.dirname(__file__), 'output/'))
    db_path = path.join(output_dir, db_name)
    db = ScanDB(db_path)

    points = get_scan_points(axsize, M, delta, eta, Rew)
    res_cache = []
    points_scan = []

    for point in points:
        dm, Imw, mp = point
        bau, time_sol = db.get_bau(mp, tag)
        if bau is None:
            points_scan.append(((*point, db_path, tag)))
        else:
            print("Got from cache: {}".format(mp))

    db.close_conn()  # No longer need this connection

    with Pool() as p:
        res_scan = p.map(get_bau, points_scan)

    res = sorted(res_cache + res_scan, key=lambda r: (r[1], r[2]))

    # points = get_scan_points(axsize, M, delta, eta, Rew)
    # res_cache_std = []
    # res_cache_avg = []
    # points_scan_std = []
    # points_scan_avg = []
    #
    # for point in points:
    #     dm, Imw, mp = point
    #     bau_std, time_sol_std = db.get_bau(mp, "std")
    #     if bau_std is None:
    #         points_scan_std.append((*point, db_path, "std"))
    #     else:
    #         print("Got from std cache: {}".format(mp))
    #         res_cache_std.append((bau_std, mp.dM, mp.Imw, time_sol_std))
    #
    #     bau_avg, time_sol_avg = db.get_bau(mp, "avg")
    #     if bau_avg is None:
    #         points_scan_avg.append((*point, db_path, "avg"))
    #     else:
    #         print("Got from avg cache: {}".format(mp))
    #         res_cache_avg.append((bau_avg, mp.dM, mp.Imw, time_sol_avg))
    #
    # db.close_conn() # No longer need this connection
    #
    # with Pool(8) as p:
    #     res_scan_avg = p.map(get_bau, points_scan_avg)
    #     res_scan_std = p.map(get_bau, points_scan_std)
    #
    # res_std = sorted(res_cache_std + res_scan_std, key=lambda r: (r[1], r[2]))
    # res_avg = sorted(res_cache_avg + res_scan_avg, key=lambda r: (r[1], r[2]))
    #
    # # outfile_heatmap = path.join(output_dir, "grid_scan_dm_imw_heatmap.png")
    # # outfile_contours = path.join(output_dir, "grid_scan_dm_imw_contours.png")
    #
    # outfile_heatmap_timings_std = path.join(output_dir, "grid_scan_dm_imw_timings_std.png")
    # title_timings_std = "M = {:.2f}, std".format(M)
    # outfile_heatmap_timings_avg = path.join(output_dir, "grid_scan_dm_imw_timings_avg.png")
    # title_timings_avg = "M = {:.2f}, avg".format(M)
    #
    # heatmap_dm_imw_timing(np.array(res_std), axsize, title_timings_std, outfile_heatmap_timings_std)
    # heatmap_dm_imw_timing(np.array(res_avg), axsize, title_timings_avg, outfile_heatmap_timings_avg)
    #
    # # Format data for comparison plot
    # # Data should be list of tuples [(tag, [[dm, imw, bau],...]),...]
    # #(bau, dM, Imw, time_sol)
    # res_comp_std = []
    # res_comp_avg = []
    # for r in res_std:
    #     res_comp_std.append([r[0], r[1], r[2]])
    # for r in res_avg:
    #     res_comp_avg.append([r[0], r[1], r[2]])
    #
    # res_comp = [("std", np.array(res_comp_std)), ("avg", np.array(res_comp_avg))]
    # title_comp = "M = {:.2f}, avg (red) vs std (blue)".format(M)
    # outfile_comp = path.join(output_dir, "grid_scan_dm_imw_std_v_avg.png")
    # contour_dm_imw_comp(res_comp, axsize, title_comp, outfile_comp)
    #
    # print("Finished!")



