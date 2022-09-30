import platform
from typing import Optional
if platform.system() != 'Windows':
    import matplotlib as mpl
    mpl.use('Agg')
import numpy as np
from ._dataClass import RateMap, TrajectoryData, SpikeData
from matplotlib.figure import Figure
from matplotlib import pyplot

__all__ = ["plot_map2d", "plot_trajectory"]


# ----------------------------- Visualizer module ----------------------------- #
def plot_map2d(obj_fig: Figure, ax_idx: tuple, ratemap: RateMap, ith, colormap=pyplot.cm.jet, annot=True):
    ax = obj_fig.add_subplot(*ax_idx)
    ax.pcolor(ratemap.X, ratemap.Y, ratemap.rate[ith], cmap=colormap)
    if annot is True:
        ax.annotate('#%d' % ith, xy=(0.05, ratemap.Y[-1, 0] - 0.2), color='w', size=15)
        hz_str = '%.1fHz' % np.nanmax(ratemap.rate[ith])
        ax.annotate(hz_str, xy=(ratemap.X[0, -1] - len(hz_str) * 0.1, 0.05), color='k', size=15)
    ax.tick_params(labelbottom=False, bottom=False)
    ax.tick_params(labelleft=False, left=False)
    ax.set_xticklabels([])
    ax.axis('equal')
    return ax


def plot_trajectory(obj_fig: Figure, ax_idx: tuple, trajectory: TrajectoryData, spike: Optional[SpikeData], ith: Optional[int], ms=1, alpha=1):
    ax = obj_fig.add_subplot(*ax_idx)
    ax.plot(trajectory.x, trajectory.y, "-k")
    if ith is not None:
        idx = np.where(spike.cell_id == spike.member[ith])[0]
        t = (spike.spike_timing[idx] * 1000).astype(np.int)
        ax.plot(trajectory.x[t], trajectory.y[t], ".r", ms=ms, alpha=alpha)
        ax.annotate('#%d' % ith, xy=(0.05, 0.2), color='green', size=15)
    ax.tick_params(labelbottom=False, bottom=False)
    ax.tick_params(labelleft=False, left=False)
    ax.set_xticklabels([])
    ax.axis('equal')
    return ax
