import numpy as np
from scipy.stats import kurtosis as _kurtosis, skew as _skew


def compute_time_stats(signal: np.ndarray) -> dict:
    """
    计算单路信号的时域统计特征。

    返回字段
    --------
    rms              : 均方根值，振幅的直接度量
    kurtosis         : 峭度（Fisher 定义，正态分布≈0），高值表示冲击成分
    skewness         : 偏度，分布不对称程度
    crest_factor     : 波峰因子 = max(|x|) / RMS，冲击类事件的指标
    zero_crossing_rate : 过零率 = 符号变化次数 / (N-1)，与主频近似正相关
    """
    x = np.asarray(signal, dtype=np.float64)
    n = len(x)

    rms = float(np.sqrt(np.mean(x ** 2)))
    kurt = float(_kurtosis(x, fisher=True, bias=False))
    skewness = float(_skew(x, bias=False))
    peak = float(np.max(np.abs(x)))
    crest = peak / rms if rms > 1e-12 else float("nan")
    sign_changes = int(np.sum(np.diff(np.sign(x)) != 0))
    zcr = sign_changes / (n - 1) if n > 1 else float("nan")

    return {
        "rms":                rms,
        "kurtosis":           kurt,
        "skewness":           skewness,
        "crest_factor":       crest,
        "zero_crossing_rate": zcr,
    }
