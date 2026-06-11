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
    n = len(x)
    if n < 4:
        f = np.linspace(0.0, freq_max_hz, max(2, nfft // 8))
        return f, np.zeros(len(f), dtype=np.float32)

    nperseg = min(int(nfft / 2), n)
    noverlap = min(int(nfft / 4), max(0, nperseg - 1))
    if noverlap >= nperseg:
        noverlap = nperseg - 1
    fft_size = max(nfft, nperseg)

    f, psd = signal.welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=fft_size,
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
