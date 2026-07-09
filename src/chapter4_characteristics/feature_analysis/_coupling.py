import numpy as np
from scipy.signal import csd, welch


def compute_cross_coupling(
    in_signal: np.ndarray,
    out_signal: np.ndarray,
    fs: float,
    nperseg: int = 2048,
) -> dict:
    """
    计算面内-面外振动信号的耦合特征。

    返回字段
    --------
    cross_correlation   : Pearson 互相关系数（时域），[-1, 1]
                          风雨振耦合运动时趋向较高正/负值
    ellipticity         : 轨迹椭圆率 = 1 - 短轴/长轴，由 PCA 估计
                          0 → 线性运动，1 → 圆形运动
    dominant_coherence  : 在面内主频处的互谱相干值，[0, 1]
                          1 → 两方向在该频率完全相干
    phase_difference_deg: 在面内主频处的相位差（°），
                          ±90° 对应椭圆轨迹，0/180° 对应线性轨迹
    """
    x = np.asarray(in_signal,  dtype=np.float64)
    y = np.asarray(out_signal, dtype=np.float64)

    # 互相关系数
    corr = float(np.corrcoef(x, y)[0, 1])

    # 轨迹椭圆率（PCA）
    traj = np.column_stack([x - x.mean(), y - y.mean()])
    cov  = np.cov(traj.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.maximum(eigvals, 0.0)
    eigvals_sorted = np.sort(eigvals)[::-1]
    if eigvals_sorted[0] > 1e-12:
        ellipticity = float(1.0 - np.sqrt(eigvals_sorted[1] / eigvals_sorted[0]))
    else:
        ellipticity = float("nan")

    # 互谱相干 + 相位差
    nperseg = min(nperseg, len(x))
    f_xx, psd_xx = welch(x, fs=fs, nperseg=nperseg)
    f_xy, csd_xy = csd(x, y, fs=fs, nperseg=nperseg)

    # 在面内 PSD 峰值处评估
    dom_idx = int(np.argmax(psd_xx))

    psd_yy_at_dom = None
    f_yy, psd_yy = welch(y, fs=fs, nperseg=nperseg)
    if dom_idx < len(psd_yy):
        psd_yy_at_dom = float(psd_yy[dom_idx])

    cxy_dom  = csd_xy[dom_idx]
    pxx_dom  = float(psd_xx[dom_idx])
    pyy_dom  = psd_yy_at_dom if psd_yy_at_dom is not None else float("nan")

    denom = pxx_dom * pyy_dom
    if denom > 1e-30:
        coherence = float(np.abs(cxy_dom) ** 2 / denom)
        coherence = min(coherence, 1.0)
    else:
        coherence = float("nan")

    phase_deg = float(np.degrees(np.angle(cxy_dom)))

    return {
        "cross_correlation":    corr,
        "ellipticity":          ellipticity,
        "dominant_coherence":   coherence,
        "phase_difference_deg": phase_deg,
    }
