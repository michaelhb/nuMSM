import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import numpy as np
from os import path
import csv

mpl.rcParams['figure.dpi'] = 300

def lepton_asymmetry_comp(data, title, outfile):
    # data: list of (tag, Tlist, lepton asymmetry) tuples

    cm_idx = lambda i: (i + 1)/float(len(data))

    for i, datum in enumerate(data):
        tag, Tlist, Y = datum
        plt.loglog(Tlist, np.abs(Y), label=tag, color=plt.cm.cool(cm_idx(i)))

    plt.xlabel("T")
    plt.ylabel("lepton asymmetry")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)

def bau_rdiff_heatmap(data1, data2, axsize, title, outfile):
    plt.clf()

    bau1 = data1[:, 0]
    bau2 = data2[:, 0]
    rdiff = np.abs((bau1 - bau2) / (bau1 + bau2))
    rdiff_min = min(rdiff)
    rdiff_max = max(rdiff)

    dm = data1[:, 1]
    imw = data1[:, 2]

    rdiff = np.reshape(np.array(rdiff), (axsize, axsize))
    dm = list(dict.fromkeys(dm))
    imw = list(dict.fromkeys(imw))

    ax_ticks = 7
    imw_ticks = np.linspace(min(imw),max(imw),ax_ticks)
    dm_ticks = np.geomspace(min(dm),max(dm),ax_ticks)
    rdiff_cb = np.geomspace(rdiff_min, rdiff_max, ax_ticks)

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(rdiff, cmap=plt.cm.viridis, norm=colors.LogNorm(vmin=rdiff_min, vmax=rdiff_max))

    ax.set_xticks(np.linspace(0, len(imw), ax_ticks))
    ax.set_xticklabels(imw_ticks)
    ax.set_yticks(np.linspace(0, len(dm), ax_ticks))
    ax.set_yticklabels(["{:.3e}".format(y) for y in dm_ticks])
    ax.set_xlabel("$Im(\omega)$ / GeV")
    ax.set_ylabel("$dM / GeV$")

    cb = plt.colorbar(heatmap, shrink=1, orientation='horizontal')
    cb.set_ticks(rdiff_cb)
    cb.set_ticklabels(["{:.2e}".format(z) for z in rdiff_cb])
    cb.set_label("rdiff")
    # plt.yscale('log')
    plt.tight_layout()
    plt.title(title)

    plt.savefig(outfile)


def contour_dm_imw_comp(data, axsize, title, outfile):
    plt.clf()
    # Data should be list of tuples [(tag, [[dm, imw, bau],...]),...]

    colors = ["green", "purple", "red", "midnightblue"]

    leg_lines = []
    leg_tags = []

    for datum in data:
        color = colors.pop()

        tag, grid = datum
        bau = grid[:, 0]
        dm = grid[:,1]
        imw = grid[:,2]

        leg_lines.append(Line2D([0], [0], color=color))
        leg_tags.append(tag)

        bau = np.reshape(np.array(bau), (axsize, axsize))
        dm = list(dict.fromkeys(dm))
        imw = list(dict.fromkeys(imw))
        plt.contour(imw, dm, bau, levels=[-1e-10, 1e-10], linestyles="-", label=tag, colors=color)

    plt.ylabel(r'$\Delta M/M$')
    plt.yscale('log')
    plt.xlabel(r'$Im \omega$')
    plt.title(title)
    plt.legend(leg_lines, leg_tags)
    plt.tight_layout()
    print("SAVING")
    plt.savefig(outfile, format="png")

def contour_dm_imw(data, axsize, title, outfile):
    plt.clf()

    # Data should be np array [[dm, imw, bau],...]
    # bau = np.abs(data[:,0])
    bau = data[:, 0]
    dm = data[:,1]
    imw = data[:,2]

    # Make grid
    bau = np.reshape(np.array(bau), (axsize, axsize))
    dm = list(dict.fromkeys(dm))
    imw = list(dict.fromkeys(imw))

    plt.contourf(imw, dm, bau, levels=[-1e-10, 1e-10], colors="lightblue")
    plt.contour(imw, dm, bau, levels=[-1e-10, 1e-10], colors='midnightblue', linestyles="-")
    plt.ylabel(r'$\Delta M/M$')
    plt.yscale('log')
    plt.xlabel(r'$Im \omega$')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile)

def heatmap_dm_imw(data, axsize, title, outfile):
    plt.clf()

    # Data should be np array [[bau, dm, imw],...]
    bau = np.abs(data[:,0])
    dm = data[:,1]
    imw = data[:,2]

    bau_min = min(bau)
    bau_max = max(bau)

    ax_ticks = 7

    bau_cb = np.geomspace(bau_min, bau_max, ax_ticks)

    # Make grid
    bau = np.reshape(np.array(bau), (axsize, axsize))
    dm = list(dict.fromkeys(dm))
    imw = list(dict.fromkeys(imw))

    imw_ticks = np.linspace(min(imw),max(imw),ax_ticks)
    dm_ticks = np.geomspace(min(dm),max(dm),ax_ticks)

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(bau, cmap=plt.cm.viridis, norm=colors.LogNorm(vmin=bau_min, vmax=bau_max))

    ax.set_xticks(np.linspace(0, len(imw), ax_ticks))
    ax.set_xticklabels(imw_ticks)
    ax.set_yticks(np.linspace(0, len(dm), ax_ticks))
    ax.set_yticklabels(["{:.3e}".format(y) for y in dm_ticks])
    ax.set_xlabel("$Im(\omega)$ / GeV")
    ax.set_ylabel("$dM / GeV$")

    cb = plt.colorbar(heatmap, shrink=1, orientation='horizontal')
    cb.set_ticks(bau_cb)
    cb.set_ticklabels(["{:.2e}".format(z) for z in bau_cb])
    cb.set_label("BAU")
    # plt.yscale('log')
    plt.tight_layout()
    plt.title(title)

    plt.savefig(outfile)

def heatmap_dm_imw_timing(data, axsize, title, outfile):
    plt.clf()

    # Data should be np array [[dm, imw, bau],...]
    time = np.abs(data[:,3])
    dm = data[:,1]
    imw = data[:,2]

    time_min = min(time)
    time_max = max(time)

    ax_ticks = 7

    time_cb = np.linspace(time_min, time_max, ax_ticks)

    # Make grid
    time = np.reshape(np.array(time), (axsize, axsize))
    dm = list(dict.fromkeys(dm))
    imw = list(dict.fromkeys(imw))

    imw_ticks = np.linspace(min(imw),max(imw),ax_ticks)
    dm_ticks = np.geomspace(min(dm),max(dm),ax_ticks)

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(time, cmap=plt.cm.viridis)#, norm=colors.LogNorm(vmin=time_min, vmax=time_max))

    ax.set_xticks(np.linspace(0, len(imw), ax_ticks))
    ax.set_xticklabels(imw_ticks)
    ax.set_yticks(np.linspace(0, len(dm), ax_ticks))
    ax.set_yticklabels(["{:.3e}".format(y) for y in dm_ticks])
    ax.set_xlabel("$Im(\omega)$ / GeV")
    ax.set_ylabel("$dM / GeV$")

    cb = plt.colorbar(heatmap, shrink=1, orientation='horizontal')
    cb.set_ticks(time_cb)
    cb.set_ticklabels(["{:.2f}".format(z) for z in time_cb])
    cb.set_label("Time (s)")
    # plt.yscale('log')
    plt.tight_layout()
    plt.title(title)
    plt.savefig(outfile)

# Various __main__ functions - custom plots while experimenting with code.
# Don't use this approach for production plots!

# Comparing contour plots
if __name__ == '__main__':
    output_dir = path.abspath(path.join(path.dirname(__file__), 'output/'))
    file_kdep = path.join(output_dir, "20mode_60_full/grid_scan_dm_imw.csv")
    file_avg = path.join(output_dir, "avg_60_full/grid_scan_dm_imw.csv")

    res_kdep = []
    res_avg = []

    with open(file_kdep, mode="r") as csv_in:
        reader = csv.reader(csv_in, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            res_kdep.append(list(map(float, row)))
        res_kdep = np.array(res_kdep)

    with open(file_avg, mode="r") as csv_in:
        reader = csv.reader(csv_in, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            res_avg.append(list(map(float, row)))
        res_avg = np.array(res_avg)

    data = [
        ("kdep (20)", res_kdep),
        ("avg", res_avg)
    ]

    contour_dm_imw_comp(data, 60, r'$M = 1.0$ GeV, kdep (20 modes, blue) avg (red)', "output/grid_scan_dm_imw_contours_kdep_vs_avg.png")
