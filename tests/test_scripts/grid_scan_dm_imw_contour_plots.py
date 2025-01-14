from scandb import ScanDB
from sys import argv
import yaml
from common import *
from plots import *

def get_scan_points(points_per_dim, M, delta, eta, Rew, dM_min, dM_max):
    dMs = [10**e for e in np.linspace(dM_min, dM_max, points_per_dim)]

    Imw_min = -6.
    Imw_max = 6.
    Imws = np.linspace(Imw_min, Imw_max, points_per_dim)

    points = []

    for dM in dMs:
        for Imw in Imws:
            points.append([dM, Imw, ModelParams(M, dM, Imw, Rew, delta, eta)])

    return np.array(points)

if __name__ == "__main__":
    # #Args: yaml file, db file, axsize, outfile
    # yaml_file = argv[1]
    # db_name = argv[2]
    # axsize = int(argv[3])

    # Args: db file, axsize, outfile, title, [yaml files...]
    db_name = argv[1]
    axsize = int(argv[2])
    outfile = argv[3]
    title = argv[4]
    yaml_files = argv[5:]

    print(outfile)

    output_dir = path.abspath(path.join(path.dirname(__file__), 'output/'))
    db_path = path.join(output_dir, db_name)
    db = ScanDB(db_path)

    res_plots = []

    for yaml_file in yaml_files:
        yaml_path = path.abspath(path.join(path.dirname(__file__), yaml_file))
        with open(yaml_path) as file:
            doc = yaml.load(file, Loader=yaml.FullLoader)
            M = float(doc["M"])
            delta = float(doc["delta"])
            eta = float(doc["eta"])
            Rew = float(doc["rew"])
            H = int(doc["H"])
            tag = doc["tag"]
            dM_min = int(doc["dm_min"])
            dM_max = int(doc["dm_max"])

        res_plot = []
        points = get_scan_points(axsize, M, delta, eta, Rew, dM_min, dM_max)

        for point in points:
            dm, Imw, mp = point
            bau, time_sol = db.get_bau(mp, tag)
            if bau is None:
                raise Exception("Missing point! {}".format(mp))
            res_plot.append([bau, dm, Imw, time_sol])

        res_plots.append((tag, np.array(res_plot)))

    outfile_bau_plot = path.join(output_dir, outfile)
    contour_dm_imw_comp(res_plots, axsize, title, outfile_bau_plot)

    # yaml_path = path.abspath(path.join(path.dirname(__file__), yaml_file))
    # with open(yaml_path) as file:
    #     doc = yaml.load(file, Loader=yaml.FullLoader)
    #     M = doc["M"]
    #     delta = doc["delta"]
    #     eta = doc["eta"]
    #     Rew = doc["rew"]
    #     avg = doc["avg"]
    #     H = int(doc["H"])
    #     tag = doc["tag"]
    #     dM_min = doc["dm_min"]
    #     dM_max = doc["dm_max"]
    #
    # # def get_scan_points(points_per_dim, M, delta, eta, Rew):
    # points = get_scan_points(axsize, M, delta, eta, Rew, dM_min, dM_max)
    #
    # output_dir = path.abspath(path.join(path.dirname(__file__), 'output/'))
    # db_path = path.join(output_dir, db_name)
    # db = ScanDB(db_path)
    #
    # res_plot = []
    # res_time = []
    #
    # for point in points:
    #     dm, Imw, mp = point
    #     bau, time_sol = db.get_bau(mp, tag)
    #     if bau is None:
    #         raise Exception("Missing point! {}".format(mp))
    #     res_plot.append([bau, dm, Imw, time_sol])
    #
    # outfile_bau_plot = path.join(output_dir, "grid_scan_dm_imw_{}.png".format(tag))
    # outfile_timing_plot = path.join(output_dir, "grid_scan_dm_imw_timing.png".format(tag))
    # title = "M = {}".format(M)
    #
    # contour_dm_imw_comp([(tag, np.array(res_plot))], axsize, title, outfile_bau_plot)
    # heatmap_dm_imw_timing(np.array(res_plot), axsize, title, outfile_timing_plot)

