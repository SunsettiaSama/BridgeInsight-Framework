from __future__ import annotations

import io
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.chapter3_identifier.augment.features.context_window import extract_context_window
from src.chapter3_identifier.augment.features.spectrum import welch_psd
from src.chapter3_identifier.augment.figures.layout import context_fig_size
from src.chapter3_identifier.augment.figures.render.titles import format_sample_title
from src.chapter3_identifier.augment.settings import load_config
from src.figure_paintings.figs_for_thesis.config import get_viridis_color_map

plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

_WINDOW_SIZE = 3000
_FS = 50.0
_NFFT = 2048
_FREQ_MAX = 25.0
_CFG = load_config()
_LONG_FIG_DPI = int(_CFG.get("figure_context_dpi", 96))
_USE_TIGHT_BBOX = bool(_CFG.get("figure_export_tight_bbox", False))


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
    if _USE_TIGHT_BBOX:
        fig.savefig(buf, format="png", dpi=_LONG_FIG_DPI, bbox_inches="tight")
    else:
        fig.savefig(buf, format="png", dpi=_LONG_FIG_DPI)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def plot_long_timeseries(
    long_signal: np.ndarray,
    fs: float,
    current_start_s: float,
    current_end_s: float,
    title: str,
    layout_profile: str = "wide_fill_v3",
    discontinuity_note: Optional[str] = None,
) -> bytes:
    t = np.arange(len(long_signal)) / fs
    fig, ax = plt.subplots(figsize=context_fig_size(layout_profile, "timeseries"))
    ax.plot(t, long_signal, color="#333333", linewidth=0.8)
    ax.axvspan(current_start_s, current_end_s, color="#4f8ef7", alpha=0.25, label="当前窗口")
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel(r"加速度 ($m/s^2$)")
    plot_title = title
    if discontinuity_note:
        plot_title = f"{title}\n{discontinuity_note}"
    ax.set_title(plot_title, fontsize=8)
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
    layout_profile: str = "wide_fill_v3",
    nfft: int = 2048,
    freq_max: float = 25.0,
) -> bytes:
    seg_len = int(segment_s * fs)
    n_segments = max(1, len(long_signal) // seg_len)
    usable = n_segments * seg_len
    if usable < 4:
        fig, ax = plt.subplots(figsize=context_fig_size(layout_profile, "spectrogram"))
        ax.text(0.5, 0.5, "无法生成时频谱", ha="center", va="center")
        return _fig_to_bytes(fig)

    segments = np.asarray(long_signal[:usable], dtype=np.float32).reshape(n_segments, seg_len)
    segments = segments - np.mean(segments, axis=1, keepdims=True)
    window = np.hanning(seg_len).astype(np.float32)
    fft_size = max(int(nfft), seg_len)
    spectrum = np.fft.rfft(segments * window[None, :], n=fft_size, axis=1)
    scale = max(float(fs) * float(np.sum(window ** 2)), 1e-12)
    psd = (np.abs(spectrum) ** 2 / scale).astype(np.float32)
    f = np.fft.rfftfreq(fft_size, d=1.0 / fs)
    mask = f <= freq_max
    spec = psd[:, mask]
    total_s = n_segments * segment_s
    cmap = get_viridis_color_map(start_gray=0.2)
    fig, ax = plt.subplots(figsize=context_fig_size(layout_profile, "spectrogram"))
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
    ax.set_title(title, fontsize=8)
    fig.colorbar(im, ax=ax, label="PSD")
    fig.tight_layout()
    return _fig_to_bytes(fig)


@dataclass
class ContextRenderData:
    signal: np.ndarray
    fs: float
    current_start_s: float
    current_end_s: float
    title: str
    discontinuity_note: Optional[str] = None


def build_context_render_data(
    file_path: str,
    window_index: int,
    direction: str,
    sensor_id: str,
    before: int = 3,
    after: int = 3,
) -> ContextRenderData:
    ctx = extract_context_window(
        file_path=file_path,
        window_index=window_index,
        window_size=_WINDOW_SIZE,
        fs=_FS,
        before=before,
        after=after,
    )
    title = format_sample_title(
        "面内" if direction == "inplane" else "面外",
        sensor_id,
        file_path,
        window_index,
    )
    return ContextRenderData(
        signal=ctx.signal,
        fs=ctx.fs,
        current_start_s=ctx.current_start_s,
        current_end_s=ctx.current_end_s,
        title=title,
        discontinuity_note=ctx.discontinuity_note,
    )


def render_context_part_from_data(
    render_data: ContextRenderData,
    part: str,
    segment_s: float = 2.0,
    layout_profile: str = "wide_fill_v3",
) -> bytes:
    if part == "timeseries":
        return plot_long_timeseries(
            render_data.signal,
            render_data.fs,
            render_data.current_start_s,
            render_data.current_end_s,
            render_data.title + " · 长时程",
            layout_profile=layout_profile,
            discontinuity_note=render_data.discontinuity_note,
        )
    if part == "spectrogram":
        return plot_long_spectrogram(
            render_data.signal,
            render_data.fs,
            segment_s,
            render_data.current_start_s,
            render_data.current_end_s,
            render_data.title + " · 长时频谱",
            layout_profile=layout_profile,
        )
    raise ValueError(f"未知上下文图类型: {part}")


def render_context_figures(
    file_path: str,
    window_index: int,
    direction: str,
    sensor_id: str,
    before: int = 3,
    after: int = 3,
    segment_s: float = 2.0,
    cache_size: int = 30,
    layout_profile: str = "wide_fill_v3",
) -> Tuple[bytes, bytes]:
    global _context_cache
    if _context_cache.max_size != cache_size:
        _context_cache = ContextFigureCache(cache_size)

    key = f"{file_path}:{window_index}:{direction}:{before}:{after}:{layout_profile}"
    cached = _context_cache.get(key)
    if cached is not None:
        return cached

    render_data = build_context_render_data(
        file_path=file_path,
        window_index=window_index,
        direction=direction,
        sensor_id=sensor_id,
        before=before,
        after=after,
    )
    ts_png = render_context_part_from_data(
        render_data,
        part="timeseries",
        segment_s=segment_s,
        layout_profile=layout_profile,
    )
    sp_png = render_context_part_from_data(
        render_data,
        part="spectrogram",
        segment_s=segment_s,
        layout_profile=layout_profile,
    )
    _context_cache.put(key, ts_png, sp_png)
    return ts_png, sp_png
