import numpy as np
from scipy import signal
from scipy.ndimage.filters import gaussian_filter1d

from ._dataClass import *


__all__ = ["calc_group_firing_rate", "find_ripple", "calc_bandpass", "calc_ripple_envelope", "RippleEvent"]


class RippleEvent:
    def __init__(self, envelope: TimeSignal, filtered_rate: TimeSignal, intervals: list):
        self.envelope = envelope
        self.filtered_rate = filtered_rate
        self.intervals = intervals


def calc_group_firing_rate(spikes: SpikeData, time_window: list, time_bin_size=0.01):
    """平均発火頻度の時間変化を計算する
    :param spikes: ラスターデータ
    :param time_window: シミュレーション時間 [sec]
    :param time_bin_size: 時間ビンサイズ [sec] 1/samplerate
    :return mean_firing_rate: 1D ndarray 平均発火頻度 [Hz]
    :return t_vec: 1D ndarray 時刻 [sec]
    """
    t_vec = np.arange(time_window[0], time_window[1], time_bin_size)
    mean_firing_rate, _ = np.histogram(spikes.spike_timing, bins=t_vec)
    mean_firing_rate = mean_firing_rate / (spikes.n_cells * time_bin_size)
    t_vec = t_vec[:-1] + time_bin_size / 2
    return TimeSignal(mean_firing_rate, t_vec)


def calc_bandpass(lfp: TimeSignal, time_bin_size, band=(150, 250), g_pass=3, g_stop=40):
    """時間信号にバンドパスフィルタを適用する
    :param lfp:
    :param time_bin_size: [sec]
    :param band: int 通過域端周波数 [Hz]
    :param g_pass: int 通過域端最大損失 [dB]
    :param g_stop: int 阻止域端最小損失 [dB]
    :return: 1D ndarray フィルタを適用した信号
    """
    fp = np.array(band)
    fs = np.array([band[0] / 2, band[1] * 2])  # 阻止域端周波数[Hz]
    return TimeSignal(val=bandpass(lfp.val, 1 / time_bin_size, fp, fs, g_pass, g_stop), t_vec=lfp.t_vec)


def calc_ripple_envelope(lfp: TimeSignal, sd=0.0125):
    """リップル周波数帯の包絡線を求める
    :param lfp:
    :param sd:
    :return:
    """
    time_bin_size = lfp.t_vec[1] - lfp.t_vec[0]
    filtered_rate = calc_bandpass(lfp, time_bin_size, band=(150, 250), g_pass=3, g_stop=40)  # band pass for the ripple 150~250Hz
    filtered_envelope = np.abs(signal.hilbert(filtered_rate.val))  # hilbert transform
    ripple_envelope_smoothed = gaussian_filter1d(filtered_envelope, sd / time_bin_size)  # gaussian kernel smoothing
    return TimeSignal(val=ripple_envelope_smoothed, t_vec=lfp.t_vec), TimeSignal(val=filtered_rate.val, t_vec=lfp.t_vec)


def find_ripple(lfp: TimeSignal, sd=0.0125, sd_lim=3, abs_lim=0.1):
    """リップル区間を求める
    :param lfp:
    :param sd:
    :param sd_lim:
    :param abs_lim:
    :return:
    """
    time_bin_size = lfp.t_vec[1] - lfp.t_vec[0]
    ripple_envelope_smoothed, filtered_rate = calc_ripple_envelope(lfp, sd)
    # find above 3SD
    ripple_intervals = []
    last_below_mu, flag_sd = 0, False
    ripple_sd, ripple_mu = np.std(ripple_envelope_smoothed.val), np.mean(ripple_envelope_smoothed.val)
    for i, val in enumerate(ripple_envelope_smoothed.val):
        if flag_sd:
            if val < ripple_mu:
                ripple_intervals.append((last_below_mu * time_bin_size + lfp.t_vec[0], i * time_bin_size + lfp.t_vec[0]))
                flag_sd = False
                last_below_mu = i
        else:
            if (val >= ripple_sd * sd_lim + ripple_mu) and (val >= abs_lim + ripple_mu):
                flag_sd = True
            elif val < ripple_mu:
                last_below_mu = i
    return RippleEvent(ripple_envelope_smoothed, filtered_rate, ripple_intervals)


def bandpass(x, sample_rate, fp, fs, g_pass, g_stop):
    """
    :param x:
    :param sample_rate: サンプリング周波数 [Hz]
    :param fp: 通過域端周波数 [Hz]
    :param fs: 阻止域端周波数 [Hz]
    :param g_pass: int 通過域端最大損失 [dB]
    :param g_stop: int 阻止域端最小損失 [dB]
    :return:
    """
    fn = sample_rate / 2  # ナイキスト周波数
    wp = fp / fn  # ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn  # ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, g_pass, g_stop)  # オーダーとバターワースの正規化周波数を計算
    [b, a] = signal.butter(N, Wn, "band")  # フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)  # 信号に対してフィルタをかける
    return y
