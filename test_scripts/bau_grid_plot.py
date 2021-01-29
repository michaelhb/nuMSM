from os import path
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import numpy as np
from common import *

plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})

# Fetch results in row order
fetch_qry = """select dm,imw,bau from bau_scan ORDER BY dm ASC, imw ASC;"""
axsize_qry = """select count(distinct dm) from bau_scan;"""

def bau_grid_plot(name):
    output_dir = path.abspath(path.join(path.dirname(__file__), 'output/'))
    conn = sqlite3.connect(path.join(output_dir, "bau_{}.db".format(name)))
    c = conn.cursor()

    c.execute(axsize_qry)
    axsize = c.fetchall()[0][0]

    c.execute(fetch_qry)
    data = c.fetchall()
    dm, imw, bau = list(zip(*data))

    bau_min = min(bau)
    bau_max = max(bau)

    ax_ticks = 7

    bau_cb = np.geomspace(bau_min, bau_max, ax_ticks)
    bau_ticks = np.arange(ax_ticks)

    bau = np.reshape(np.array(bau), (axsize, axsize))
    dm = list(dict.fromkeys(dm))
    imw = list(dict.fromkeys(imw))

    imw_ticks = np.linspace(min(imw),max(imw),ax_ticks)
    dm_ticks = np.geomspace(min(dm),max(dm),ax_ticks)

    fig, ax = plt.subplots()
    # heatmap = ax.pcolor(bau, cmap=plt.cm.gist_heat, norm=colors.LogNorm(vmin=bau_min, vmax=bau_max))
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
    plt.tight_layout()
    plt.show()