import cv2
import numpy as np
from scipy.spatial.distance import correlation
from scipy.stats import skew
from progressbar import ProgressBar
from ._dataClass import *


__all__ = ["calc_gridness", "calc_center", "calc_spike_skewness_x_axis", "calc_spatial_information",
           "find_place_cell_by_field_number", "find_place_cell_by_spatial_information", "find_place_cell_by_peak_rate",
           "calc_place_field_number", "calc_place_field_area"]


def calc_gridness(ratemap: RateMap, gc_thresh=0.1):
    """gridnessを計算
    :param ratemap:
    :param gc_thresh:
    :return:
    """
    space_bin_size = ratemap.X[0, 1] - ratemap.X[0, 0]
    gc_corr = np.zeros((ratemap.n_cells, ratemap.X.shape[0] * 2, ratemap.X.shape[1] * 2))  # 2D auto-correlation function
    gc_gridness = np.zeros(ratemap.n_cells)
    gc_spatial_freq = np.zeros(ratemap.n_cells)

    print('\rCalculating ACF')
    p = ProgressBar(min=0, max_value=ratemap.n_cells)
    for cell_id in range(ratemap.n_cells):
        p.update(cell_id)
        gc_corr[cell_id] = calc_acf(ratemap.rate[cell_id])

    print('\rCalculating Gridness Score')
    p = ProgressBar(min=0, max_value=ratemap.n_cells)
    for cell_id in range(ratemap.n_cells):
        p.update(cell_id)
        corr = gc_corr[cell_id].copy()

        # make mesh
        fig_cent = int(len(corr) / 2)
        mesh = np.arange(0, len(corr), dtype=np.int), np.arange(0, len(corr), dtype=np.int)
        x, y = np.meshgrid(mesh[0], mesh[1])
        dist_mesh = np.sqrt(np.power(x - fig_cent, 2) + np.power(y - fig_cent, 2))

        # make binary map
        binary = corr.copy() * 0
        binary[corr > gc_thresh] = 255
        binary = binary.astype(np.uint8)

        # labeling of small regions
        ret, markers = cv2.connectedComponents(binary)

        # if 7 or more peaks were found
        if ret < 8:
            gc_gridness[cell_id] = np.nan
            gc_spatial_freq[cell_id] = np.nan
        else:
            # calc center
            centers = np.zeros((ret - 1, 2))
            for i in range(1, ret):  # (index 0 is background)
                temp = markers == i
                x_sum, y_sum = np.sum(temp, axis=0), np.sum(temp, axis=1)
                x_index, y_index = np.where(x_sum > 0)[0], np.where(y_sum > 0)[0]
                centers[i - 1, 0] = int((np.max(x_index) - np.min(x_index)) / 2 + np.min(x_index))
                centers[i - 1, 1] = int((np.max(y_index) - np.min(y_index)) / 2 + np.min(y_index))

            # calc distance between center of small region and center of figure
            dist = np.zeros(ret - 1)
            for i in range(ret - 1):
                dist[i] = np.sqrt((centers[i, 0] - fig_cent) ** 2 + (centers[i, 1] - fig_cent) ** 2)

            # find 6 peaks close to the center of figure (closest one is excluded)
            closest = np.argmin(dist)
            six_neighbors = np.argsort(dist)[1:7]

            # calc spatial frequency
            gc_spatial_freq[cell_id] = 1 / np.mean(dist[six_neighbors] * space_bin_size)

            # calc inner circle and outer circle
            inner_cir = np.max(dist_mesh[markers == (closest + 1)])
            outer_cir = 0
            for i in range(6):
                if np.max(dist_mesh[markers == (six_neighbors[i] + 1)]) > outer_cir:
                    outer_cir = np.max(dist_mesh[markers == (six_neighbors[i] + 1)])

            # calc gridness
            deg_list = [30, 60, 90, 120, 150]
            gridness_masked = np.zeros(len(deg_list))
            rows, cols = gc_corr[cell_id].shape
            mask = (inner_cir <= dist_mesh) & (dist_mesh < outer_cir)
            for deg in deg_list:
                m = cv2.getRotationMatrix2D((cols / 2, rows / 2), deg, 1)
                base = corr * mask
                rotated = cv2.warpAffine(base, m, (cols, rows))
                gridness_masked[deg_list.index(deg)] = 1 - correlation(base.flatten(), rotated.flatten())

            gc_gridness[cell_id] = np.min(gridness_masked[[1, 3]]) - np.max(gridness_masked[[0, 2, 4]])

    return gc_gridness, gc_spatial_freq


def calc_spatial_information(ratemap: RateMap):
    si = np.zeros(ratemap.n_cells)
    mean_rate = np.mean(np.mean(ratemap.rate, axis=1), axis=1)
    idx = mean_rate > 0
    si[idx] = np.nansum(np.nansum(ratemap.rate[idx] * np.log2(ratemap.rate[idx] / mean_rate[idx].reshape(np.sum(idx), 1, 1)), axis=1), axis=1) \
              / (mean_rate[idx] * ratemap.X.shape[0] * ratemap.X.shape[1])

    return si


def calc_acf(base):
    """2次元の自己相関を計算（ピアソン相関）
    :param base:
    :return:
    """
    d_l = len(base)
    r = np.zeros((d_l * 2, d_l * 2))
    lags = np.arange(0, 2 * d_l, dtype=np.int)
    for tau_x in lags:
        for tau_y in lags:
            if tau_x <= d_l and tau_y <= d_l:
                map1 = base[:tau_x, :tau_y]
                map2 = base[d_l - tau_x:, d_l - tau_y:]
            elif tau_y <= d_l < tau_x:
                map1 = base[tau_x - d_l:, :tau_y]
                map2 = base[:d_l * 2 - tau_x, d_l - tau_y:]
            elif tau_x <= d_l < tau_y:
                map1 = base[:tau_x, tau_y - d_l:]
                map2 = base[d_l - tau_x:, :d_l * 2 - tau_y]
            else:
                map1 = base[tau_x - d_l:, tau_y - d_l:]
                map2 = base[:d_l * 2 - tau_x, :d_l * 2 - tau_y]
            n = map1.shape[0] * map1.shape[1]
            if n < 20:
                continue
            sum1, sum2 = np.sum(map1), np.sum(map2)
            a = n * np.sum(map1 * map2)
            b = sum1 * sum2
            c = np.sqrt(n * np.sum(np.power(map1, 2)) - np.power(sum1, 2))
            d = np.sqrt(n * np.sum(np.power(map2, 2)) - np.power(sum2, 2))
            if c * d != 0:
                r[tau_x, tau_y] = (a - b) / (c * d)
    return r


def calc_center(ratemap: RateMap):
    """発火マップの中心とピーク発火率を計算
    :param ratemap:
    :return:
    """
    space_bin_size = ratemap.X[0, 1] - ratemap.X[0, 0]
    pc_xy_hz = np.zeros((ratemap.n_cells, 3))  # [x[m], y[m]]
    for i, rate in enumerate(ratemap.rate):
        if np.nanmax(rate) == 0:
            pc_xy_hz[i, 0] = np.nan
            pc_xy_hz[i, 1] = np.nan
            pc_xy_hz[i, 2] = np.nan
        else:
            i_pos = np.unravel_index(np.nanargmax(rate), (rate.shape[0], rate.shape[1]))
            pc_xy_hz[i, 0] = i_pos[1] * space_bin_size + space_bin_size * 0.5
            pc_xy_hz[i, 1] = i_pos[0] * space_bin_size + space_bin_size * 0.5
            pc_xy_hz[i, 2] = np.nanmax(rate)
    return pc_xy_hz


def calc_spike_skewness_x_axis(spikes: SpikeData, trajectory: TrajectoryData, time_window: list):
    skew_list, width_list = [], []
    for i in range(spikes.n_cells):
        # search cell's spike data
        index = np.where(spikes.cell_id == spikes.member[i])[0]
        cell_spike = np.floor(spikes.spike_timing[index]).astype(np.int32)  # ms
        # 解析対象の時間窓が与えられている場合は時間外の発火は除外する
        cell_spike = cell_spike[(time_window[0] < cell_spike) & (cell_spike <= time_window[1])]
        if cell_spike.size:
            x = trajectory.x[np.searchsorted(trajectory.t, cell_spike)]
            skew_list.append(skew(x))
            width_list.append(x[-1] - x[0])
    return skew_list, width_list


def find_place_cell_by_peak_rate(rate_map: RateMap, thresh_rate):
    """
    ピーク発火頻度で場所細胞を探す関数
    :param rate_map: 探索対象の発火頻度マップ
    :param thresh_rate: ピーク発火頻度のしきい値 [Hz]
    :return: 場所細胞のインデックス
    """
    return np.where(np.max(np.max(rate_map.rate, axis=1), axis=1) > thresh_rate)[0]


def find_place_cell_by_spatial_information(rate_map: RateMap, p=0.3, si_thresh=None):
    """
    空間情報量で場所細胞を探す関数
    :param rate_map: 探索対象の発火頻度マップ
    :param p: 上位何割を場所細胞とするか [0~1]
    :param si_thresh: 空間情報量のしきい値
    :return: 場所細胞のインデックス
    """
    si = calc_spatial_information(rate_map)
    # しきい値が指定されない場合は上位p%に基づいてしきい値を決める
    if si_thresh is None:
        si_thresh = np.sort(si)[int(rate_map.n_cells*(1 - p))]
    return np.where(si > si_thresh)[0]


def find_place_cell_by_field_number(rate_map: RateMap, thresh_rate, n_field_thresh=1):
    """
    場所受容野の数で場所細胞を探す関数
    :param rate_map: 探索対象の発火頻度マップ
    :param thresh_rate: 場所受容野判定のしきい値発火頻度 [Hz]
    :param n_field_thresh: 場所細胞判定の受容野数しきい値
    :return: 場所細胞のインデックス
    """
    index = calc_place_field_number(rate_map, thresh_rate)
    return np.where((0 < index) & (index <= n_field_thresh))[0]


def calc_place_field_number(rate_map: RateMap, thresh_rate):
    """
    場所受容野の数で場所細胞を探す関数
    :param rate_map: 探索対象の発火頻度マップ
    :param thresh_rate: 場所受容野判定のしきい値発火頻度 [Hz]
    :return: 場所受容野数の配列
    """
    # make binary map
    binary = rate_map.rate.copy() * 0
    binary[rate_map.rate > thresh_rate] = 255
    binary = binary.astype(np.uint8)

    # labeling of regions
    index = np.zeros(rate_map.n_cells)
    for i in range(rate_map.n_cells):
        index[i], _ = cv2.connectedComponents(binary[i])
    index -= 1
    return index


def calc_place_field_area(rate_map: RateMap, p=0.3, thresh_rate=None):
    """
    場所受容野の面積を計算する関数
    :param rate_map: 解析対象の発火頻度マップ
    :param p: ピークから何%までを受容野とするか
    :param thresh_rate: 受容野のしきい値 [Hz]
    :return: 面積の配列 [cm^2]
    """
    # calc mean area of the place fields
    areas = np.zeros(rate_map.n_cells)
    if thresh_rate is None:     # 可変しきい値
        for i, r_map in enumerate(rate_map.rate):
            peak = np.max(r_map)
            if peak > 0.0:
                areas[i] = (np.sum(r_map > peak * p))  # p%まで減衰する領域（連続していなくともよい）
    else:   # 固定しきい値
        for i, r_map in enumerate(rate_map.rate):
            peak = np.max(r_map)
            if peak > thresh_rate:
                areas[i] = (np.sum(r_map > thresh_rate))  # p%まで減衰する領域（連続していなくともよい）
    areas = np.array(areas) * np.power(rate_map.bin_size * 100, 2)  # pixel to cm2
    return areas
