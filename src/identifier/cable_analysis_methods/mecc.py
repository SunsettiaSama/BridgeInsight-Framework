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
                          sigma_0 = 0.1, # 振动起振阈值
                          k_viv=2,  C_viv=0.1,   # 低频区间：涡激共振专属判定参数
                          k_rmw=3, C_rmw=0.25,  # 高频区间：风雨振专属判定参数
                          freq_threshold = 5 # 高低频分割阈值，单位Hz，由统计计算确定
                          ): 
        """
        拉索异常振动分类主函数（基于改进MECC准则）
        判定逻辑：
        1. 数据RMS小于起振阈值 → 判定为一般振动
        2. 计算主导模态频率，依据频率阈值划分高低频区间
        3. 低频区间：采用涡激共振参数判定 → 满足条件判定为涡激共振
        4. 高频区间：采用风雨振参数判定 → 满足条件判定为风雨振
        5. 不满足所有异常判定条件 → 统一判定为一般振动
        :param data: 拉索加速度时序数据
        :param f0: 拉索基频
        :param sigma_0: 振动起振阈值
        :param k_viv: 低频区间临近峰扩展倍数（涡激共振专用）
        :param C_viv: 低频区间MECC判定阈值（涡激共振专用）
        :param k_rmw: 高频区间临近峰扩展倍数（风雨振专用）
        :param C_rmw: 高频区间MECC判定阈值（风雨振专用）
        :param freq_threshold: 高低频分割阈值
        :return: 振动类型标签 0=一般振动, 1=涡激共振, 2=风雨振
        """
        # 步骤1：振动起振条件判定
        rms = np.sqrt(np.mean((data - np.mean(data)) ** 2))
        if rms < sigma_0:
            return 0  # 未达到起振阈值，判定为一般振动

        # 步骤2：计算主导模态频率，用于高低频区间划分
        _, f_major = self.__compute_mecc(data, f0, k=k_viv)
        
        # 步骤3：基于主导频率分区间执行MECC判定
        if f_major < freq_threshold:
            # 低频区间：涡激共振判定
            mecc, _ = self.__compute_mecc(data, f0, k_viv)
            if mecc < C_viv:
                return 1  # 满足MECC准则，判定为涡激共振
        else:
            # 高频区间：风雨振判定
            mecc, _ = self.__compute_mecc(data, f0, k_rmw)
            if mecc < C_rmw:
                return 2  # 满足MECC准则，判定为风雨振

        # 步骤4：不满足所有异常振动条件，判定为一般振动
        return 0