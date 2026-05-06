import numpy as np
from scipy import signal


class Abnormal_Vibration_Filter():
    def __init__(self, fs=50, nfft=2048, nperseg=1024, noverlap=512):
        """
        初始化改进能量集中系数(MECC)振动分类滤波器

        :param fs:       采样频率（Hz）
        :param nfft:     Welch 法 FFT 点数
        :param nperseg:  Welch 法每段长度
        :param noverlap: Welch 法段间重叠长度
        """
        self.fs       = fs
        self.nfft     = nfft
        self.nperseg  = nperseg
        self.noverlap = noverlap

    def __compute_mecc(self, data, f0, k):
        """
        核心方法：计算改进能量集中系数MECC（剔除临近峰值干扰）
        基于拉索基频构建k阶物理区间，排除区间内邻近模态峰值的计算干扰

        :param data: 拉索加速度时序振动数据
        :param f0:   拉索基频
        :param k:    临近峰扩展区间倍数（物理意义：基频k阶）
        :return:     mecc - 改进能量集中系数, f_major - 功率谱主导模态频率
        """
        fx, Pxx = signal.welch(
            data,
            fs=self.fs,
            nfft=self.nfft,
            nperseg=self.nperseg,
            noverlap=self.noverlap,
        )

        E1      = np.max(Pxx)
        f_major = fx[np.argmax(Pxx)]

        left  = f_major - k * f0
        right = f_major + k * f0

        Pxx_out = Pxx[(fx < left) | (fx > right)]
        Ek      = np.max(Pxx_out) if len(Pxx_out) > 0 else 0

        mecc = Ek / E1 if E1 != 0 else 1.0
        return mecc, f_major

    def classify_vibration(self, data, f0,
                           sigma_0=0.1,
                           freq_min=1,
                           k_viv=2,
                           C_viv=0.3,
                           return_score=False,
                           **kwargs,
                           ):
        """
        拉索振动二分类主函数（基于改进MECC准则，仅区分一般振动与涡激共振）

        判定逻辑：
        1. RMS < sigma_0              → 一般振动（未达起振条件）
        2. f_major < freq_min         → 一般振动（主频过低，不在 VIV 频率范围内）
        3. MECC < C_viv               → 涡激共振（主频能量高度集中）
        4. 其余情况                   → 一般振动

        :param data:         拉索加速度时序数据
        :param f0:           拉索基频（Hz）
        :param sigma_0:      RMS 起振阈值（m/s²）
        :param freq_min:     主频参与 VIV 判定的最低频率（Hz）
        :param k_viv:        MECC 临近峰剔除区间倍数
        :param C_viv:        MECC 判定阈值
        :param return_score: 若为 True，额外返回 (mecc_score, f_major)
                             完整返回值为 (label, mecc_score, f_major)；
                             mecc_score=1.0 / f_major=None 表示因 RMS 不足提前返回
        :return:             return_score=False → int（0 或 1）
                             return_score=True  → (int, float, float|None)
        """
        rms = np.sqrt(np.mean((data - np.mean(data)) ** 2))
        if rms < sigma_0:
            if return_score:
                return 0, 1.0, None
            return 0

        mecc, f_major = self.__compute_mecc(data, f0, k_viv)

        if f_major < freq_min:
            if return_score:
                return 0, mecc, f_major
            return 0

        if mecc < C_viv:
            if return_score:
                return 1, mecc, f_major
            return 1

        if return_score:
            return 0, mecc, f_major
        return 0
