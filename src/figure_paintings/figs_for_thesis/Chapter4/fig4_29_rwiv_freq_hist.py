"""图4-29：风雨振主频分布直方图。

样式对齐 fig4_19。样本池与 fig4_25 共用（合并副本或仅 DL）；
主频取 Welch PSD（与 fig4_27 同参）峰值频率。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_processer.io_unpacker import UNPACK
from src.chapter4_characteristics._bootstrap import ensure_paths

ensure_paths()
from src.figure_paintings.figs_for_thesis.Chapter4._rwiv_pipeline import (
    RWIV_SAMPLE_COPY_PATH,
    USE_MERGED_DATASET,
    add_dataset_switch_args,
    load_rwiv_samples_for_figures,
    resolve_use_merged,
)
from src.figure_paintings.figs_for_thesis.Chapter4.fig4_25_rwiv_timeseries import (
    Config as SharedConfig,
)
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT,
    FONT_SIZE,
    REC_FIG_SIZE,
    VIV_INPLANE_COLOR,
    VIV_OUTPLANE_COLOR,
)
from src.visualize_tools.web_dashboard import push as web_push


class Config:
    FS = SharedConfig.FS
    WINDOW_SIZE = SharedConfig.WINDOW_SIZE

    # 与 fig4_27 频谱图一致
    NFFT = 2048
    FREQ_LIMIT = 25.0

    N_BINS = 80
    FREQ_X_PERCENTILE = 100.0

    FIG_SIZE = REC_FIG_SIZE
    GRID_COLOR = "gray"
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = "--"

    INPLANE_COLOR = VIV_INPLANE_COLOR
    OUTPLANE_COLOR = VIV_OUTPLANE_COLOR
    BAR_ALPHA = 0.72

    SNAPSHOT_DIR = project_root / "results" / "chapter4_characteristics" / "figure_snapshots"
    SNAPSHOT_PATH = SNAPSHOT_DIR / "fig4_29_rwiv_freq_hist.npz"
    WEB_PAGE = "fig4_29 风雨振主频分布"


def _snapshot_config(use_merged: bool) -> dict:
    return {
        "figure": "fig4_29_rwiv_freq_hist",
        "use_merged": bool(use_merged),
        "window_size": int(Config.WINDOW_SIZE),
        "fs": float(Config.FS),
        "nfft": int(Config.NFFT),
        "freq_limit": float(Config.FREQ_LIMIT),
        "sample_copy": str(RWIV_SAMPLE_COPY_PATH) if use_merged else "dl_only",
    }


def _welch(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    f, psd = scipy_signal.welch(
        data,
        fs=Config.FS,
        nperseg=Config.NFFT // 2,
        noverlap=Config.NFFT // 4,
        nfft=Config.NFFT,
        scaling="density",
    )
    mask = f <= Config.FREQ_LIMIT
    return f[mask], psd[mask]


def _dominant_freq(data: np.ndarray) -> float:
    f, psd = _welch(data)
    if len(f) == 0:
        raise ValueError("Welch 频谱为空，无法提取主频")
    return float(f[int(np.argmax(psd))])


def compute_dom_freq_from_samples(samples: list[dict]) -> dict:
    unpacker = UNPACK(init_path=False)
    cache: dict[str, np.ndarray] = {}
    dom_freq_in: list[float] = []
    dom_freq_out: list[float] = []

    n = len(samples)
    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0 or i == 0 or i + 1 == n:
            print(f"  计算主频：{i + 1}/{n}")

        in_path = sample["inplane_file_path"]
        out_path = sample["outplane_file_path"]
        win = int(sample["window_idx"])

        if in_path not in cache:
            cache[in_path] = np.asarray(unpacker.VIC_DATA_Unpack(in_path), dtype=np.float64)
        if out_path not in cache:
            cache[out_path] = np.asarray(unpacker.VIC_DATA_Unpack(out_path), dtype=np.float64)

        start = win * Config.WINDOW_SIZE
        end = start + Config.WINDOW_SIZE
        raw_in = cache[in_path]
        raw_out = cache[out_path]
        if end > len(raw_in) or end > len(raw_out):
            raise ValueError(
                f"窗口越界：idx={sample.get('idx')} win={win} "
                f"len_in={len(raw_in)} len_out={len(raw_out)}"
            )

        dom_freq_in.append(_dominant_freq(raw_in[start:end]))
        dom_freq_out.append(_dominant_freq(raw_out[start:end]))

    return {
        "dom_freq_in": np.asarray(dom_freq_in, dtype=np.float64),
        "dom_freq_out": np.asarray(dom_freq_out, dtype=np.float64),
    }


def load_snapshot(use_merged: bool, force_refresh: bool) -> dict | None:
    path = Config.SNAPSHOT_PATH
    if force_refresh or not path.exists():
        return None

    payload = np.load(path, allow_pickle=True)
    saved_cfg = json.loads(str(payload["config_json"]))
    if saved_cfg != _snapshot_config(use_merged):
        print(f"  快照参数不匹配，将重新计算：{path}")
        return None

    required = ("dom_freq_in", "dom_freq_out")
    for key in required:
        if key not in payload:
            print(f"  快照缺少字段 {key}，将重新计算：{path}")
            return None

    print(f"  读取结果快照：{path}")
    print(f"    created_at={payload['created_at']}  n={len(payload['dom_freq_in'])}")
    return {key: np.asarray(payload[key], dtype=np.float64) for key in required}


def save_snapshot(data: dict, use_merged: bool) -> None:
    path = Config.SNAPSHOT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        created_at=np.asarray(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        config_json=np.asarray(json.dumps(_snapshot_config(use_merged), ensure_ascii=False)),
        dom_freq_in=np.asarray(data["dom_freq_in"], dtype=np.float64),
        dom_freq_out=np.asarray(data["dom_freq_out"], dtype=np.float64),
    )
    print(f"  写出结果快照：{path}")


def load_dominant_freq_data(
    use_merged: bool,
    force_refresh: bool,
    refresh_sample_copy: bool,
) -> dict:
    cached = load_snapshot(use_merged=use_merged, force_refresh=force_refresh)
    if cached is not None:
        return cached

    print("  未命中快照，加载风雨振样本并计算主频 ...")
    if use_merged:
        print(f"  副本路径：{RWIV_SAMPLE_COPY_PATH}")
    samples = load_rwiv_samples_for_figures(
        use_merged=use_merged,
        force_refresh=refresh_sample_copy,
    )
    print(f"  配对样本：{len(samples)}")
    data = compute_dom_freq_from_samples(samples)
    save_snapshot(data, use_merged=use_merged)
    return data


def _apply_grid(ax) -> None:
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA, linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)


def _add_legend(ax) -> None:
    leg = ax.legend(fontsize=FONT_SIZE - 2, framealpha=0.9, prop=CN_FONT)
    for text in leg.get_texts():
        text.set_fontproperties(CN_FONT)


def plot_dominant_freq_histogram(data: dict) -> plt.Figure:
    freq_in = data["dom_freq_in"]
    freq_out = data["dom_freq_out"]
    combined = np.concatenate([freq_in, freq_out])
    x_max = float(np.percentile(combined, Config.FREQ_X_PERCENTILE))
    x_max = max(x_max, 1e-6)

    bins = np.linspace(0, x_max, Config.N_BINS + 1)
    width = bins[1] - bins[0]
    centers = (bins[:-1] + bins[1:]) / 2
    counts_in, _ = np.histogram(freq_in[freq_in <= x_max], bins=bins)
    counts_out, _ = np.histogram(freq_out[freq_out <= x_max], bins=bins)

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    half = width * 0.46
    ax.bar(
        centers - half / 2,
        counts_in,
        width=half,
        color=Config.INPLANE_COLOR,
        alpha=Config.BAR_ALPHA,
        label="面内",
    )
    ax.bar(
        centers + half / 2,
        counts_out,
        width=half,
        color=Config.OUTPLANE_COLOR,
        alpha=Config.BAR_ALPHA,
        label="面外",
    )

    ax.set_xlim(0, x_max)
    ax.set_xlabel("主频 (Hz)", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel("样本数（个）", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 2)
    _add_legend(ax)
    _apply_grid(ax)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.96)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="图4-29 风雨振主频分布")
    add_dataset_switch_args(parser)
    parser.add_argument(
        "--refresh-snapshot",
        action="store_true",
        help="强制重算主频并覆盖快照",
    )
    args = parser.parse_args()
    use_merged = resolve_use_merged(args.use_merged)

    print("=" * 80)
    print("图4-29 风雨振主频分布直方图")
    print(f"  默认开关 USE_MERGED_DATASET={USE_MERGED_DATASET}  → 本次 use_merged={use_merged}")
    print("=" * 80)

    print("\n[步骤1] 加载风雨振主频 ...")
    print(f"  Welch：nfft={Config.NFFT}，freq≤{Config.FREQ_LIMIT:g} Hz（与 fig4_27 一致）")
    data = load_dominant_freq_data(
        use_merged=use_merged,
        force_refresh=args.refresh_snapshot,
        refresh_sample_copy=args.refresh_sample_copy,
    )
    print(f"[OK] 面内有效样本：{len(data['dom_freq_in'])}，面外有效样本：{len(data['dom_freq_out'])}")
    print(f"  面内主频 median={float(np.median(data['dom_freq_in'])):.4f} Hz")
    print(f"  面外主频 median={float(np.median(data['dom_freq_out'])):.4f} Hz")
    print(
        f"  面内 p95={float(np.percentile(data['dom_freq_in'], 95)):.4f} Hz  "
        f"面外 p95={float(np.percentile(data['dom_freq_out'], 95)):.4f} Hz"
    )

    print("\n[步骤2] 绘制图像...")
    fig = plot_dominant_freq_histogram(data)
    web_push(fig, page=Config.WEB_PAGE, slot=0, title="风雨振主频分布", page_cols=1)
    plt.close(fig)
    print(f"[OK] 已推送到 WebUI：{Config.WEB_PAGE}")


if __name__ == "__main__":
    main()
