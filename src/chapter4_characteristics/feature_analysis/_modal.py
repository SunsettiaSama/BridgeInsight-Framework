import numpy as np
from scipy.signal import welch, find_peaks


def compute_psd_top_modes(
    signal: np.ndarray,
    fs: float = 50.0,
    n_modes: int = 10,
    nperseg: int = 2048,
    min_peak_distance_hz: float = 0.1,
) -> dict:
    """
    PSD 前 N 阶主导模态（频率 + 功率），按频率升序返回。
    """
    f, psd = _compute_psd(signal, fs, nperseg)
    freq_res = f[1] - f[0] if len(f) > 1 else 1.0
    min_distance = max(1, int(min_peak_distance_hz / freq_res))

    peaks, _ = find_peaks(psd, distance=min_distance)
    if len(peaks) == 0:
        return {"frequencies": [], "powers": []}

    top_idx   = np.argsort(psd[peaks])[::-1][:n_modes]
    top_peaks = peaks[top_idx]
    top_peaks = top_peaks[np.argsort(f[top_peaks])]

    return {
        "frequencies": f[top_peaks].tolist(),
        "powers":      psd[top_peaks].tolist(),
    }


def compute_spectral_features(
    signal: np.ndarray,
    fs: float = 50.0,
    nperseg: int = 2048,
    min_peak_distance_hz: float = 0.1,
    n_modes: int = 10,
) -> dict:
    """
    频域综合统计特征。

    返回字段
    --------
    spectral_entropy        : 谱熵，越低越窄带（涡激），越高越宽带（随机）
    spectral_centroid_hz    : 谱质心（加权平均频率）
    spectral_bandwidth_hz   : 谱带宽（质心的加权标准差）
    top_modes_energy_ratio  : 前 N 个峰值能量之和 / 总能量
    dominant_mode_energy_ratio : 第一主频能量 / 总能量
    """
    f, psd = _compute_psd(signal, fs, nperseg)
    total_power = float(np.sum(psd))

    # 谱熵
    if total_power > 1e-30:
        p_norm = psd / total_power
        p_norm = np.where(p_norm > 0, p_norm, 1e-30)
        spectral_entropy = float(-np.sum(p_norm * np.log(p_norm)))
    else:
        spectral_entropy = float("nan")

    # 谱质心 / 谱带宽
    if total_power > 1e-30:
        centroid = float(np.sum(f * psd) / total_power)
        bandwidth = float(np.sqrt(np.sum((f - centroid) ** 2 * psd) / total_power))
    else:
        centroid = float("nan")
        bandwidth = float("nan")

    # 主导模态能量占比
    freq_res = f[1] - f[0] if len(f) > 1 else 1.0
    min_distance = max(1, int(min_peak_distance_hz / freq_res))
    peaks, _ = find_peaks(psd, distance=min_distance)

    if len(peaks) > 0 and total_power > 1e-30:
        peak_powers = psd[peaks]
        top_idx     = np.argsort(peak_powers)[::-1][:n_modes]
        top_energy  = float(np.sum(peak_powers[top_idx]))
        dom_energy  = float(peak_powers[top_idx[0]])
        top_ratio   = top_energy / total_power
        dom_ratio   = dom_energy / total_power
    else:
        top_ratio = float("nan")
        dom_ratio = float("nan")

    return {
        "spectral_entropy":           spectral_entropy,
        "spectral_centroid_hz":       centroid,
        "spectral_bandwidth_hz":      bandwidth,
        "top_modes_energy_ratio":     top_ratio,
        "dominant_mode_energy_ratio": dom_ratio,
    }


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _compute_psd(
    signal: np.ndarray,
    fs: float,
    nperseg: int,
):
    nperseg = min(nperseg, len(signal))
    f, psd = welch(signal, fs=fs, nperseg=nperseg, scaling="density")
    return f, psd
