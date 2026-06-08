from __future__ import annotations

import io
from collections import OrderedDict
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.chapter1_identifier.augment.features.context_window import extract_context_window
from src.chapter1_identifier.augment.features.spectrum import welch_psd
from src.data_processer.preprocess.get_data_vib import VICWindowExtractor
from src.figure_paintings.figs_for_thesis.config import get_viridis_color_map

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

_WINDOW_SIZE = 3000
_FS = 50.0
_NFFT = 2048
_FREQ_MAX = 25.0


class ContextFigureCache:
    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self._cache: OrderedDict[str, Tuple[bytes, bytes]] = OrderedDict()

    def get(self, key: str) -> Optional[Tuple[bytes, bytes]]:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, timeseries_png: bytes, spectrogram_png: bytes) -> None:
        self._cache[key] = (timeseries_png, spectrogram_png)
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)


_context_cache = ContextFigureCache()


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_long_timeseries(
    long_signal: np.ndarray,
    fs: float,
    current_start_s: float,
    current_end_s: float,
    title: str,
) -> bytes:
    t = np.arange(len(long_signal)) / fs
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, long_signal, color="#333333", linewidth=0.8)
    ax.axvspan(current_start_s, current_end_s, color="#4f8ef7", alpha=0.25, label="当前窗口")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("加速度 (m/s²)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _fig_to_bytes(fig)


def plot_long_spectrogram(
    long_signal: np.ndarray,
    fs: float,
    segment_s: float,
    current_start_s: float,
    current_end_s: float,
    title: str,
    nfft: int = 2048,
    freq_max: float = 25.0,
) -> bytes:
    seg_len = int(segment_s * fs)
    n_segments = max(1, len(long_signal) // seg_len)
    psd_list = []
    for i in range(n_segments):
        seg = long_signal[i * seg_len: (i + 1) * seg_len]
        if len(seg) < 4:
            continue
        f, psd = welch_psd(seg, fs=fs, nfft=nfft, freq_max_hz=freq_max)
        psd_list.append(psd)
    if not psd_list:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "无法生成时频谱", ha="center", va="center")
        return _fig_to_bytes(fig)

    spec = np.array(psd_list)
    total_s = n_segments * segment_s
    cmap = get_viridis_color_map(start_gray=0.2)
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(
        spec,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=[0, freq_max, 0, total_s],
        interpolation="bilinear",
    )
    ax.axhline(current_start_s, color="white", linestyle="--", linewidth=1)
    ax.axhline(current_end_s, color="white", linestyle="--", linewidth=1)
    ax.set_xlabel("频率 (Hz)")
    ax.set_ylabel("时间 (s)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="PSD")
    fig.tight_layout()
    return _fig_to_bytes(fig)


def render_context_figures(
    file_path: str,
    window_index: int,
    direction: str,
    sensor_id: str,
    before: int = 3,
    after: int = 3,
    segment_s: float = 2.0,
    cache_size: int = 30,
) -> Tuple[bytes, bytes]:
    global _context_cache
    if _context_cache.max_size != cache_size:
        _context_cache = ContextFigureCache(cache_size)

    key = f"{file_path}:{window_index}:{direction}:{before}:{after}"
    cached = _context_cache.get(key)
    if cached is not None:
        return cached

    ctx = extract_context_window(
        file_path=file_path,
        window_index=window_index,
        window_size=_WINDOW_SIZE,
        fs=_FS,
        before=before,
        after=after,
    )
    title = f"{sensor_id} @ win{window_index} ({direction}) 长窗口上下文"
    ts_png = plot_long_timeseries(
        ctx.signal, ctx.fs, ctx.current_start_s, ctx.current_end_s, title + " - 时程"
    )
    sp_png = plot_long_spectrogram(
        ctx.signal,
        ctx.fs,
        segment_s,
        ctx.current_start_s,
        ctx.current_end_s,
        title + " - 时频谱",
    )
    _context_cache.put(key, ts_png, sp_png)
    return ts_png, sp_png
