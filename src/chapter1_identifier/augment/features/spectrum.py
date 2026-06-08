from __future__ import annotations

import numpy as np
from scipy import signal


def welch_psd(
    x: np.ndarray,
    fs: float = 50.0,
    nfft: int = 2048,
    freq_max_hz: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    f, psd = signal.welch(
        x,
        fs=fs,
        nperseg=int(nfft / 2),
        noverlap=int(nfft / 4),
        nfft=nfft,
    )
    mask = f <= freq_max_hz
    return f[mask], psd[mask].astype(np.float32)


def psd_bin_count(
    fs: float = 50.0,
    nfft: int = 2048,
    freq_max_hz: float = 25.0,
) -> int:
    # 与 window_size=3000 的真实样本一致，避免短 dummy 导致 nperseg > len(x)
    dummy = np.zeros(3000, dtype=np.float64)
    f, _ = welch_psd(dummy, fs=fs, nfft=nfft, freq_max_hz=freq_max_hz)
    return len(f)


def compute_psd_vector(
    x: np.ndarray,
    fs: float = 50.0,
    nfft: int = 2048,
    freq_max_hz: float = 25.0,
) -> np.ndarray:
    _, psd = welch_psd(x, fs=fs, nfft=nfft, freq_max_hz=freq_max_hz)
    return psd
