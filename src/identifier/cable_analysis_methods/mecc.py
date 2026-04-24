import numpy as np
from scipy import signal

class Abnormal_Vibration_Filter():
    def __init__(self, fs=50, rms_thd=0.3):
        """
        初始化改进能量集中系数(MECC)振动分类滤波器
        :param fs: 数据采样率，单位Hz
        :param rms_thd: 兼容旧版参数，无实际调用
        """
        self.fs = fs

    def __compute_mecc(self, data, f0, k):
        """
        核心方法：计算改进能量集中系数MECC（剔除临近峰值干扰）
        基于拉索基频构建k阶物理区间，排除区间内邻近模态峰值的计算干扰
        :param data: 拉索加速度时序振动数据
        :param f0: 拉索基频
        :param k: 临近峰扩展区间倍数（物理意义：基频k阶）
        :return: mecc - 改进能量集中系数, f_major - 功率谱主导模态频率
        """
        # 采用Welch法计算信号功率谱密度PSD
        fx, Pxx = signal.welch(data, fs=self.fs, nfft=512, nperseg=512, noverlap=256)
        
        # 提取功率谱主峰值及对应主导模态频率
        E1 = np.max(Pxx)
        f_major = fx[np.argmax(Pxx)]
        
        # 构建临近峰剔除物理区间 [f_major - k*f0, f_major + k*f0]
        left = f_major - k * f0
        right = f_major + k * f0
        
        # 提取区间外频谱数据，计算次大峰值
        Pxx_out = Pxx[(fx < left) | (fx > right)]
        Ek = np.max(Pxx_out) if len(Pxx_out) > 0 else 0
        
        # 计算改进能量集中系数 MECC = 次大峰值 / 主峰值
        mecc = Ek / E1 if E1 != 0 else 1.0
        return mecc, f_major

    def classify_vibration(self, data, f0,
                          sigma_0=0.1,   # RMS 起振阈值（m/s²）
                          freq_min=0.1,  # 主频下限（Hz）；低于此值不参与 VIV 判定
                          k_viv=2,       # MECC 计算时主峰临近区间倍数
                          C_viv=0.3,     # MECC 判定阈值；mecc < C_viv → VIV
                          **kwargs       # 忽略旧版 RWIV 参数（k_rmw/C_rmw/freq_threshold）
                          ):
        """
        拉索振动二分类主函数（基于改进MECC准则，仅区分一般振动与涡激共振）

        判定逻辑：
        1. RMS < sigma_0              → 一般振动（未达起振条件）
        2. f_major < freq_min         → 一般振动（主频过低，不在 VIV 频率范围内）
        3. MECC < C_viv               → 涡激共振（主频能量高度集中）
        4. 其余情况                   → 一般振动

        :param data:     拉索加速度时序数据
        :param f0:       拉索基频（Hz）
        :param sigma_0:  RMS 起振阈值（m/s²）
        :param freq_min: 主频参与 VIV 判定的最低频率（Hz）
        :param k_viv:    MECC 临近峰剔除区间倍数
        :param C_viv:    MECC 判定阈值
        :return:         0=一般振动，1=涡激共振
        """
        # 步骤1：RMS 起振条件
        rms = np.sqrt(np.mean((data - np.mean(data)) ** 2))
        if rms < sigma_0:
            return 0

        # 步骤2：计算 MECC 及主导频率
        mecc, f_major = self.__compute_mecc(data, f0, k_viv)

        # 步骤3：主频下限掩码
        if f_major < freq_min:
            return 0

        # 步骤4：MECC 准则判定
        if mecc < C_viv:
            return 1

        return 0