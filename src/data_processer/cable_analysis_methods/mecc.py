import numpy as np
from scipy import signal

class Abnormal_Vibration_Filter():


    def __init__(self):


        return 


    def VIV_Filter(self, data, f0 = 1, f0times = 2):
        """
        利用MECC准则，筛选VIV
        1. 振动加速度的最大值应大于0.01
        2. 单峰准则k，f0：k为能量峰值底部的区间长度，f0为基频
        3. 高阶控制n, 应当为基频的3阶及以上
        """

        rms = np.sqrt(np.mean((data - np.mean(data)) ** 2))
        # 准则4 RMS控制
        if rms < 0.3:
            return False
        

        fx, Pxx_den = signal.welch(data, fs = 50, 
                                   nfft = 512,
                                   nperseg = 512,
                                   noverlap = 256)

        # 准则一：振幅控制
        if np.max(data) < 0.01:
            return False
        E1 = np.max(Pxx_den)

        index = np.where(Pxx_den == E1)
        # 主导模态
        f_major = fx[index]

        # 准则三：主导模态
        if f_major < 3 * f0:
            return False

        
        f_major_left, f_major_right = f_major - f0times * f0, f_major + f0times * f0
        # 主导模态 2倍基频以内不管
        # 2倍基频以外寻找最大值
        Pxx_den_left = Pxx_den[fx < f_major_left]
        Pxx_den_right = Pxx_den[fx > f_major_right]
        
        Ek = np.sort(np.hstack((Pxx_den_left, Pxx_den_right)))[-1]

        # 准则4：rms阈值
        rms = np.mean(np.square(data - np.mean(data)))
        if rms < 0.03:
            return False

        # 准则二：单峰
        if Ek / E1 < 0.1:
            return True

