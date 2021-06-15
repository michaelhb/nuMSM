import numpy as np
import matplotlib.pyplot as plt
from scandb import ScanDB
from os import path
from sys import argv
from common import *

dM_min = -16
dM_max = -1
Imw_min = -6.
Imw_max = 6.

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

def contour_dm_imw_comp(data, axsize, title, outfile):
    plt.clf()
    # Data should be list of tuples [(tag, [[dm, imw, bau],...]),...]

    colors = ["green", "purple", "red", "midnightblue"]

    for datum in data:
        color = colors.pop()

        tag, grid = datum
        bau = grid[:, 0]
        dm = grid[:,1]
        imw = grid[:,2]

        bau = np.reshape(np.array(bau), (axsize, axsize))
        dm = list(dict.fromkeys(dm))
        imw = list(dict.fromkeys(imw))
        plt.contour(imw, dm, bau, levels=[-1e-10, 1e-10], linestyles="-", label=tag, colors=color)

    plt.ylabel(r'$\Delta M/M$')
    plt.yscale('log')
    plt.xlabel(r'$Im \omega$')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    print("SAVING")
    plt.savefig(outfile)

if __name__ == "__main__":
    #Args: yaml file, db file, axsize, tag
    yaml_file = argv[1]
    db_name = argv[2]
    axsize = int(argv[3])
    tag = argv[4]

    output_dir = path.abspath(path.join(path.dirname(__file__), 'output/'))
    db_path = path.join(output_dir, db_name)
    db = ScanDB(db_path)
