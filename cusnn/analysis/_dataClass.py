import numpy as np

__all__ = ["SpikeData", "TrajectoryData", "TimeSignal", "RateMap"]


# ----------------------------- Data Class ----------------------------- #
class SpikeData:
    def __init__(self, spike_timing: np.ndarray, cell_id: np.ndarray, member: list):
        """スパイクデータクラス
        :param spike_timing: 発火時刻 [sec]
        :param cell_id: 細胞番号
        :param member: データに含まれる細胞番号
        """
        self.spike_timing = spike_timing.astype(np.float)
        self.cell_id = cell_id.astype(np.int)
        self.member = member
        self.n_cells = len(member)


class TrajectoryData:
    def __init__(self, t: np.ndarray, x: np.ndarray, y: np.ndarray, env_size: list):
        """軌道データクラス
        :param t: 時刻 [sec]
        :param x: x座標 [m]
        :param y: y座標 [m]
        :param env_size: (x_min, y_min, x_max, y_max) 環境サイズ [m]
        """
        self.t = t
        self.x = x
        self.y = y
        self.env_size = env_size


class TimeSignal:
    def __init__(self, val: np.ndarray, t_vec: np.ndarray):
        """1次元時間信号データ構造体
        :param val: 値
        :param t_vec: 時刻 [sec]
        """
        self.val = val  # value of signal
        self.t_vec = t_vec  # time vector [sec]


class RateMap:
    def __init__(self, rate: np.ndarray, mesh_x: np.ndarray, mesh_y: np.ndarray, n_spike: np.ndarray, bin_size: float):
        """2次元空間上での発火頻度データ
        :param rate: (n_cells, y, x)　発火頻度 [Hz]
        :param mesh_x: (y, x) x座標 [m]
        :param mesh_y: (y, x) y座標 [m]
        """
        self.rate = rate
        self.X = mesh_x
        self.Y = mesh_y
        self.bin_size = bin_size
        self.n_cells = rate.shape[0]
        self.n_spike = n_spike
