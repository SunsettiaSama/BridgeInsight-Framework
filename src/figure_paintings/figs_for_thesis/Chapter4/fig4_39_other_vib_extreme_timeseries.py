"""图4-39：其他振动 95% 分位以上极端时程样本。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_processer.io_unpacker import UNPACK
from src.figure_paintings.figs_for_thesis.Chapter4.fig4_39_other_vib_timeseries import (
    Config as BaseConfig,
    _prepare_plot_window,
    load_other_samples_for_figure,
    plot_other_vib_timeseries_grid,
)
from src.visualize_tools.web_dashboard import push as web_push


class Config:
    QUANTILE = 0.95
    DISPLAY_MIN_PEAK = 2.0
    MIN_DOMINANT_FREQ_HZ = 2.0
    NUM_SAMPLES_TO_PLOT = 20
    METRICS_PATH = (
        BaseConfig.SAMPLE_COPY_PATH.parent
        / "other_round9_annotation_2023_peak_spectrum_metrics.json"
    )
    WEB_PAGE = BaseConfig.WEB_PAGE


def _dominant_freq(signal: np.ndarray) -> float:
    nperseg = min(256, len(signal))
    f, psd = welch(
        np.asarray(signal, dtype=np.float64),
        fs=BaseConfig.FS,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="density",
    )
    mask = f > 0
    f = f[mask]
    psd = psd[mask]
    if len(psd) == 0:
        return 0.0
    return float(f[int(np.argmax(psd))])


def _window_metrics(raw: np.ndarray, window_idx: int) -> tuple[float, float, float] | None:
    start = int(window_idx) * BaseConfig.WINDOW_SIZE
    end = start + BaseConfig.WINDOW_SIZE
    if end > len(raw):
        return None
    win = raw[start:end]
    full_peak = float(np.max(np.abs(win)))
    display_window = _prepare_plot_window(win)
    display_peak = float(np.max(np.abs(display_window)))
    dominant_freq = _dominant_freq(display_window)
    return full_peak, display_peak, dominant_freq


def compute_peak_metrics(samples: list[dict], refresh: bool) -> list[dict]:
    if Config.METRICS_PATH.exists() and not refresh:
        with open(Config.METRICS_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        metrics = payload.get("metrics", [])
        print(f"  读取峰值指标缓存：{Config.METRICS_PATH}  n={len(metrics)}")
        return metrics

    print(f"  计算峰值指标：候选样本 {len(samples)} 个")
    by_path: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for sample_i, sample in enumerate(samples):
        by_path[str(sample["inplane_file_path"])].append((sample_i, int(sample["window_idx"])))
        by_path[str(sample["outplane_file_path"])].append((sample_i, int(sample["window_idx"])))

    full_peaks = np.full(len(samples), np.nan, dtype=np.float64)
    display_peaks = np.full(len(samples), np.nan, dtype=np.float64)
    display_dom_freqs = np.full(len(samples), np.nan, dtype=np.float64)
    unpacker = UNPACK(init_path=False)
    n_files = len(by_path)
    for file_i, (path, items) in enumerate(by_path.items(), start=1):
        if not Path(path).exists():
            continue
        raw = np.asarray(unpacker.VIC_DATA_Unpack(path), dtype=np.float64)
        for sample_i, window_idx in items:
            metrics = _window_metrics(raw, window_idx)
            if metrics is None:
                continue
            full_peak, display_peak, dominant_freq = metrics
            if not np.isfinite(full_peaks[sample_i]) or full_peak > full_peaks[sample_i]:
                full_peaks[sample_i] = full_peak
            if not np.isfinite(display_peaks[sample_i]) or display_peak > display_peaks[sample_i]:
                display_peaks[sample_i] = display_peak
                display_dom_freqs[sample_i] = dominant_freq
        if file_i % 50 == 0 or file_i == n_files:
            print(f"    已扫描文件 {file_i}/{n_files}")

    metrics: list[dict] = []
    for sample, full_peak, display_peak, dominant_freq in zip(
        samples, full_peaks, display_peaks, display_dom_freqs
    ):
        if (
            not np.isfinite(full_peak)
            or not np.isfinite(display_peak)
            or not np.isfinite(dominant_freq)
        ):
            continue
        item = dict(sample)
        item["full_peak_amp"] = float(full_peak)
        item["display_peak_amp"] = float(display_peak)
        item["display_dom_freq_hz"] = float(dominant_freq)
        item["peak_amp"] = float(display_peak)
        metrics.append(item)
    if not metrics:
        raise ValueError("未计算到有效峰值指标")

    payload = {
        "version": "other_peak_spectrum_metrics_v1",
        "metric": (
            "max(abs(inplane), abs(outplane)) over 60s window and displayed 0-20s window; "
            "dominant frequency of the side with displayed peak"
        ),
        "n_samples": len(metrics),
        "metrics": metrics,
    }
    Config.METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(Config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  写出峰值指标缓存：{Config.METRICS_PATH}  n={len(metrics)}")
    return metrics


def select_extreme_samples(
    metrics: list[dict],
    filter_low_freq: bool,
    only_cable: str | None,
) -> tuple[list[dict], float, int]:
    full_peaks = np.asarray([float(item["full_peak_amp"]) for item in metrics], dtype=np.float64)
    threshold = float(np.quantile(full_peaks, Config.QUANTILE))
    extreme_all = [item for item in metrics if float(item["full_peak_amp"]) >= threshold]
    display_extreme = [
        item
        for item in extreme_all
        if float(item["display_peak_amp"]) >= Config.DISPLAY_MIN_PEAK
    ]
    if filter_low_freq:
        extreme = [
            item
            for item in display_extreme
            if float(item["display_dom_freq_hz"]) >= Config.MIN_DOMINANT_FREQ_HZ
        ]
    else:
        extreme = display_extreme
    if only_cable:
        needle = only_cable.strip()
        before = len(extreme)
        extreme = [
            item
            for item in extreme
            if needle in str(item.get("inplane_sensor_id", ""))
            or needle in str(item.get("outplane_sensor_id", ""))
        ]
        print(f"  测点过滤 {needle}：{len(extreme)}/{before} 个")
    extreme.sort(key=lambda item: float(item["display_peak_amp"]), reverse=True)
    selected = extreme[: Config.NUM_SAMPLES_TO_PLOT]
    freq_msg = (
        f"；主导频率>={Config.MIN_DOMINANT_FREQ_HZ:g} Hz 后 {len(extreme)} 个"
        if filter_low_freq
        else ""
    )
    print(
        f"  60s 95% 分位阈值 |A|max={threshold:.3f} m/s^2；"
        f"阈值以上 {len(extreme_all)} 个；"
        f"显示窗口 |A|max>={Config.DISPLAY_MIN_PEAK:g} m/s^2 后 {len(display_extreme)} 个"
        f"{freq_msg}；"
        f"取显示窗口峰值最大 {len(selected)} 个"
    )
    return selected, threshold, len(display_extreme)


def main() -> None:
    parser = argparse.ArgumentParser(description="图4-39 其他振动极端时程")
    parser.add_argument("--refresh-sample-copy", action="store_true")
    parser.add_argument("--refresh-metrics", action="store_true")
    parser.add_argument("--filter-low-freq", action="store_true")
    parser.add_argument("--only-cable", type=str, default=None)
    args = parser.parse_args()

    print("=" * 80)
    print("图4-39 其他振动 95% 分位以上极端时程")
    print("=" * 80)

    print("\n[步骤1] 加载其他振动样本池...")
    samples = load_other_samples_for_figure(args.refresh_sample_copy)

    print("\n[步骤2] 计算/加载峰值指标...")
    metrics = compute_peak_metrics(samples, refresh=args.refresh_metrics)

    print("\n[步骤3] 选择极端样本...")
    selected, threshold, n_display_extreme = select_extreme_samples(
        metrics,
        filter_low_freq=args.filter_low_freq,
        only_cable=args.only_cable,
    )

    print("\n[步骤4] 绘制并推送极端时程图...")
    unpacker = UNPACK(init_path=False)
    figure = plot_other_vib_timeseries_grid(selected, unpacker)
    slot = 2 if args.filter_low_freq else 1
    title = (
        f"其他振动 极端时程（显示窗≥{Config.DISPLAY_MIN_PEAK:g}，"
        f"主导频率≥{Config.MIN_DOMINANT_FREQ_HZ:g} Hz）"
        if args.filter_low_freq
        else (
            f"其他振动 95%分位以上极端时程"
            f"（60s阈值 {threshold:.2f} m/s²，显示窗≥{Config.DISPLAY_MIN_PEAK:g}）"
        )
    )
    if args.only_cable:
        title = f"{title}｜{args.only_cable}"
    web_push(
        figure,
        page=Config.WEB_PAGE,
        slot=slot,
        title=title,
        page_cols=1,
    )
    plt.close(figure)
    print(
        f"已推送到 WebUI：{Config.WEB_PAGE} / slot={slot} "
        f"（显示窗极端候选 {n_display_extreme} 个）"
    )


if __name__ == "__main__":
    main()
