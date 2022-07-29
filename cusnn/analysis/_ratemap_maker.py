import numpy as np
from scipy import ndimage
from ._dataClass import *


__all__ = ["calc_ratemap2d_div", "calc_firing_rate", "make_xy_mesh2d", "kernel_smoothing", "calc_ratemap2d"]


def calc_ratemap2d_div(spikes: SpikeData, trajectory: TrajectoryData, time_window: list, space_bin_size=0.025, sigma=2, no_nan=False):
    """
    :param spikes:
    :param trajectory:
    :param time_window: 解析対象の時間帯 [sec]
    :param space_bin_size: [m]
    :param sigma: [bins]
    :return:
    """
    # make trajectory histogram
    X, Y = make_xy_mesh2d(trajectory.env_size, space_bin_size)
    traj_x, traj_y = meter2bin(trajectory.x, space_bin_size), meter2bin(trajectory.y, space_bin_size)
    traj_hist = make_trajectory2d(trajectory, space_bin_size, time_window)

    # data array
    heatmap = np.zeros((spikes.n_cells, X.shape[0], X.shape[1]))
    n_spike = np.zeros(spikes.n_cells, dtype=np.int)

    if no_nan is False:
        nan_idx = ndimage.filters.gaussian_filter(traj_hist, sigma=sigma, mode='constant') == 0.0
    traj_hist[traj_hist == 0] = 1

    print('\rCalculating Firing Rate Heatmap')
    for i in range(spikes.n_cells):
        # search cell's spike data
        index = np.where(spikes.cell_id == spikes.member[i])[0]
        cell_spike = spikes.spike_timing[index]
        cell_spike = cell_spike[(time_window[0] < cell_spike) & (cell_spike <= time_window[1])]
        # make 2D-space histogram of spikes
        freq = X * 0
        for t in np.searchsorted(trajectory.t, cell_spike, side='right') - 1:
            freq[traj_y[t], traj_x[t]] += 1
        heatmap[i] = ndimage.filters.gaussian_filter(freq/traj_hist, sigma=sigma, mode='constant')  # frequency to Hz -> smoothing
        if no_nan is False:
            heatmap[i][nan_idx] = np.nan
        n_spike[i] = cell_spike.size

    return RateMap(heatmap, X, Y, n_spike, bin_size=space_bin_size)


def calc_ratemap2d(spikes: SpikeData, trajectory: TrajectoryData, time_window: list, space_bin_size=0.025, sigma=3, truncate=4):
    """
    :param spikes:
    :param trajectory:
    :param time_window: 解析対象の時間帯 [sec]
    :param space_bin_size: [m]
    :param sigma: [bins]
    :return:
    """
    # make trajectory histogram
    X, Y = make_xy_mesh2d(trajectory.env_size, space_bin_size)
    traj_x, traj_y = meter2bin(trajectory.x, space_bin_size), meter2bin(trajectory.y, space_bin_size)
    traj_hist = make_trajectory2d(trajectory, space_bin_size, time_window)
    traj_hist = ndimage.filters.gaussian_filter(traj_hist, sigma=sigma, truncate=truncate, mode='constant')
    traj_hist[traj_hist == 0] = np.nan

    # data array
    heatmap = np.zeros((spikes.n_cells, X.shape[0], X.shape[1]))
    n_spike = np.zeros(spikes.n_cells, dtype=np.int)

    print('\rCalculating Firing Rate Heatmap')
    for i in range(spikes.n_cells):
        # search cell's spike data
        index = np.where(spikes.cell_id == spikes.member[i])[0]
        cell_spike = spikes.spike_timing[index]
        cell_spike = cell_spike[(time_window[0] < cell_spike) & (cell_spike <= time_window[1])]
        # make 2D-space histogram of spikes
        freq = X * 0
        for t in np.searchsorted(trajectory.t, cell_spike, side='right') - 1:
            freq[traj_y[t], traj_x[t]] += 1
        heatmap[i] = ndimage.filters.gaussian_filter(freq, sigma=sigma, truncate=truncate, mode='constant') / traj_hist  # frequency to Hz -> smoothing
        n_spike[i] = cell_spike.size

    return RateMap(heatmap, X, Y, n_spike, bin_size=space_bin_size)


def calc_firing_rate(spikes: SpikeData, time_window):
    """
    :param spikes:
    :param time_window: 解析対象の時間帯 [sec]
    :return:
    """
    firing_rate = np.zeros(spikes.n_cells)
    print('\rCalculating Firing Rate')
    for i in range(spikes.n_cells):
        # search cell's spike data
        cell_spike = spikes.spike_timing[np.where(spikes.cell_id == spikes.member[i])[0]]
        cell_spike = cell_spike[(time_window[0] < cell_spike) & (cell_spike <= time_window[1])]
        firing_rate[i] = len(cell_spike)
    firing_rate /= (time_window[1] - time_window[0])  # frequency to Hz
    return firing_rate


def kernel_smoothing(ratemap: RateMap, sigma=2) -> RateMap:
    for i in range(ratemap.n_cells):
        ratemap.rate[i] = ndimage.filters.gaussian_filter(ratemap.rate[i], sigma=sigma, mode='constant')
    return ratemap


def make_xy_mesh2d(env_size, space_bin_size):
    """環境サイズに合わせた空間メッシュを作成
    :param env_size: 環境サイズ [m]
    :param space_bin_size: 空間のビンサイズ [m]
    :return:
    """
    x, y = np.arange(0, env_size[2] + space_bin_size, space_bin_size), np.arange(0, env_size[3] + space_bin_size, space_bin_size)
    return np.meshgrid(x, y)


def make_trajectory2d(trajectory, space_bin_size, time_window):
    """
    :param trajectory:
    :param space_bin_size: [m]
    :param time_window: 解析対象の時間帯 [sec]
    :return:
    """
    X, Y = make_xy_mesh2d(trajectory.env_size, space_bin_size)
    trajectory_hist = X * 0
    ids = (time_window[0] < trajectory.t) & (trajectory.t <= time_window[1])
    for x, y in zip(meter2bin(trajectory.x[ids], space_bin_size), meter2bin(trajectory.y[ids], space_bin_size)):
        trajectory_hist[y, x] += 1
    trajectory_hist *= (trajectory.t[1] - trajectory.t[0])  # frequency to sec
    return trajectory_hist


def meter2bin(meter, space_bin_size):
    """座標データをメーター単位からビン単位へ変換
    :param meter:
    :param space_bin_size: [m]
    :return:
    """
    return (meter / space_bin_size).astype(np.int32)
