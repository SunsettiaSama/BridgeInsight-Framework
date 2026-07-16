"""
图4-11 随机振动前50阶主频累积能量分布

计算说明
--------
- 「阶」= 全谱 Welch 频点按 PSD 幅值降序后的第 k 强频点（非找峰模态）。
- 总功率 = 全谱 PSD 线性求和（等价于令 Δf=1 的矩形积分），不用 trapz。
- 统一 nfft=128；总功率 = 全谱 PSD 线性求和。
- enriched(2048) 只作样本索引，只读不写。

快照说明（重要）
----------------
从原始 VIC 解包 + Welch 很慢。nfft=128 结果追加到独立快照，
不覆盖 enriched，也不覆盖其它分辨率快照（如 nfft256）。

快照路径：
  results/chapter4_characteristics/figure_snapshots/
    fig4_11_normal_vib_energy_cumsum_nfft128.npz

强制重算：
  python .../fig4_11_normal_vib_energy_cumsum.py --refresh-cache
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_processer.io_unpacker import UNPACK
from src.chapter4_characteristics.feature_analysis.entry import ensure_enriched_for_figures
from src.visualize_tools.web_dashboard import push as web_push
from src.figure_paintings.figs_for_thesis.Chapter4._data_loader import get_enriched_class_dir, iter_enriched_json_files
from src.figure_paintings.figs_for_thesis.Chapter4._viv_pipeline import (
    _WINDOW_SIZE,
    psd_ranked_energy_cumsum,
)
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT, FONT_SIZE, REC_FIG_SIZE,
    get_blue_color_map,
)


class Config:
    N_MODES = 50
    # 统一 nfft=128；快照按 nfft 分文件追加，不动 2048 enriched
    NFFT_LIST = (128,)
    MAX_SAMPLES = 20_000
    SAMPLE_SEED = 42

    FIG_SIZE = REC_FIG_SIZE
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = '--'
    SHADE_ALPHA = 0.18
    LINE_WIDTH = 2.2

    _palette = get_blue_color_map(style='discrete', start_map_index=1, end_map_index=5).colors
    INPLANE_COLOR  = _palette[2]
    OUTPLANE_COLOR = _palette[3]

    FEATURE_BATCH_SIZE = 512
    ENRICHED_STATS_DIR = get_enriched_class_dir(0)
    WEB_PAGE = "fig4_11 累积能量"

    # 快照根目录：按 nfft 分文件，避免两组分辨率互相覆盖
    SNAPSHOT_DIR = (
        project_root
        / "results"
        / "chapter4_characteristics"
        / "figure_snapshots"
    )


def _snapshot_path(nfft: int) -> Path:
    return Config.SNAPSHOT_DIR / f"fig4_11_normal_vib_energy_cumsum_nfft{nfft}.npz"


def _snapshot_config(nfft: int) -> dict:
    """写入快照的配置指纹；任一字段变化则判定快照失效并重算。"""
    return {
        "figure": "fig4_11_normal_vib_energy_cumsum",
        "nfft": int(nfft),
        "n_modes": Config.N_MODES,
        "max_samples": Config.MAX_SAMPLES,
        "sample_seed": Config.SAMPLE_SEED,
        "window_size": _WINDOW_SIZE,
        "energy_definition": "ranked_psd_bin_linear_sum",
        "enriched_stats_dir": str(Config.ENRICHED_STATS_DIR),
    }


def load_snapshot(nfft: int, force_refresh: bool) -> dict | None:
    """
    读取指定 nfft 的累积能量快照。
    不存在 / 强制刷新 / 配置指纹不一致 → 返回 None，触发重算。
    """
    path = _snapshot_path(nfft)
    if force_refresh or not path.exists():
        return None

    payload = np.load(path, allow_pickle=True)
    saved_cfg = json.loads(str(payload["config_json"]))
    if saved_cfg != _snapshot_config(nfft):
        print(f"  快照参数不匹配，将重新计算：{path}")
        return None

    cumsum_in = np.asarray(payload["cumsum_in"], dtype=np.float64)
    cumsum_out = np.asarray(payload["cumsum_out"], dtype=np.float64)
    print(f"  读取结果快照：{path}")
    print(f"    created_at={payload['created_at']}  "
          f"n_in={cumsum_in.shape[0]}  n_out={cumsum_out.shape[0]}")
    return {
        "cumsum_in":  [row for row in cumsum_in],
        "cumsum_out": [row for row in cumsum_out],
        "nfft": int(nfft),
    }


def save_snapshot(data: dict) -> None:
    """将某一 nfft 的曲线矩阵落盘，供下次直接出图。"""
    nfft = int(data["nfft"])
    path = _snapshot_path(nfft)
    path.parent.mkdir(parents=True, exist_ok=True)

    mat_in = np.stack(data["cumsum_in"]).astype(np.float64) if data["cumsum_in"] else np.empty((0, Config.N_MODES))
    mat_out = np.stack(data["cumsum_out"]).astype(np.float64) if data["cumsum_out"] else np.empty((0, Config.N_MODES))

    np.savez_compressed(
        path,
        created_at=np.asarray(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        config_json=np.asarray(json.dumps(_snapshot_config(nfft), ensure_ascii=False)),
        nfft=np.asarray(nfft, dtype=np.int32),
        cumsum_in=mat_in,
        cumsum_out=mat_out,
    )
    print(f"  写出结果快照：{path}")


def _collect_jobs(json_files: list[Path]) -> list[tuple[str, str, int]]:
    jobs: list[tuple[str, str, int]] = []
    for json_file in json_files:
        print(f"  索引：{json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sample in data["samples"]:
            in_path = sample.get("inplane_file_path")
            out_path = sample.get("outplane_file_path")
            window_idx = sample.get("window_idx")
            if in_path is None or out_path is None or window_idx is None:
                continue
            jobs.append((in_path, out_path, int(window_idx)))
    return jobs


def _subsample_jobs(jobs: list[tuple[str, str, int]]) -> list[tuple[str, str, int]]:
    n = len(jobs)
    if n <= Config.MAX_SAMPLES:
        return jobs
    rng = np.random.default_rng(Config.SAMPLE_SEED)
    chosen = rng.choice(n, size=Config.MAX_SAMPLES, replace=False)
    chosen.sort()
    print(f"  全量样本 {n}，随机抽样 {Config.MAX_SAMPLES}")
    return [jobs[i] for i in chosen.tolist()]


def _slice_window(raw: np.ndarray, window_idx: int) -> np.ndarray | None:
    start = window_idx * _WINDOW_SIZE
    end = start + _WINDOW_SIZE
    if end > len(raw):
        return None
    return raw[start:end]


def compute_psd_modes(nfft: int) -> dict:
    """
    从原始信号计算指定 nfft 的累积能量曲线（昂贵；结果应写入快照）。
    按 VIC 文件分组：每个文件只解包一次，再切窗口做 Welch。
    """
    ensure_enriched_for_figures(class_id=0, batch_size=Config.FEATURE_BATCH_SIZE)
    stats_dir = Config.ENRICHED_STATS_DIR
    if not stats_dir.exists():
        raise FileNotFoundError(f"enriched_stats 目录不存在：{stats_dir}")

    json_files = iter_enriched_json_files(stats_dir)
    if not json_files:
        raise FileNotFoundError(f"目录下无 JSON 文件：{stats_dir}")

    jobs = _subsample_jobs(_collect_jobs(json_files))
    n = len(jobs)
    curves_in:  list[np.ndarray | None] = [None] * n
    curves_out: list[np.ndarray | None] = [None] * n

    by_path: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for i, (in_path, out_path, window_idx) in enumerate(jobs):
        by_path[in_path].append((window_idx, i, "in"))
        by_path[out_path].append((window_idx, i, "out"))

    unpacker = UNPACK(init_path=False)
    n_files = len(by_path)
    print(f"  [nfft={nfft}] 待处理样本：{n}，唯一 VIC 文件：{n_files}")

    for file_i, (path, items) in enumerate(by_path.items(), start=1):
        if not Path(path).exists():
            continue
        raw = np.asarray(unpacker.VIC_DATA_Unpack(str(path)), dtype=np.float64)
        for window_idx, sample_i, side in items:
            sig = _slice_window(raw, window_idx)
            if sig is None:
                continue
            curve = psd_ranked_energy_cumsum(sig, n_modes=Config.N_MODES, nfft=nfft)
            if curve is None:
                continue
            if side == "in":
                curves_in[sample_i] = curve
            else:
                curves_out[sample_i] = curve

        if file_i % 200 == 0 or file_i == n_files:
            print(f"    [nfft={nfft}] 已解包文件 {file_i}/{n_files}")

    return {
        "cumsum_in":  [c for c in curves_in  if c is not None],
        "cumsum_out": [c for c in curves_out if c is not None],
        "nfft": nfft,
    }


def load_or_compute(nfft: int, force_refresh: bool) -> dict:
    """优先读快照；缺失或 --refresh-cache 时重算并落盘。"""
    cached = load_snapshot(nfft, force_refresh=force_refresh)
    if cached is not None:
        return cached

    print(f"  未命中快照，开始重算 nfft={nfft} ...")
    data = compute_psd_modes(nfft=nfft)
    save_snapshot(data)
    return data


def _aggregate(curves: list[np.ndarray], n_modes: int) -> dict:
    mat = np.full((len(curves), n_modes), np.nan)
    for i, c in enumerate(curves):
        length = min(len(c), n_modes)
        mat[i, :length] = c[:length]

    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat,  axis=0)
    return {"mean": mean, "std": std, "n": len(curves)}


def _apply_grid(ax):
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA,
            linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)


def _add_legend(ax):
    leg = ax.legend(fontsize=FONT_SIZE - 2, framealpha=0.9, prop=CN_FONT)
    for t in leg.get_texts():
        t.set_fontproperties(CN_FONT)


def _auto_ylim(mean_in: np.ndarray, std_in: np.ndarray,
               mean_out: np.ndarray, std_out: np.ndarray) -> float:
    upper = max(
        float(np.nanmax(mean_in + std_in)),
        float(np.nanmax(mean_out + std_out)),
    )
    return min(1.05, max(0.35, upper * 1.08))


def plot_energy_cumsum(data: dict) -> plt.Figure:
    n = Config.N_MODES
    nfft = int(data["nfft"])
    stats_in  = _aggregate(data["cumsum_in"],  n)
    stats_out = _aggregate(data["cumsum_out"], n)

    x = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)

    mean_in = stats_in["mean"]
    std_in  = stats_in["std"]
    ax.plot(x, mean_in, color=Config.INPLANE_COLOR,
            linewidth=Config.LINE_WIDTH, marker='o', markersize=4,
            label=f'面内（n={stats_in["n"]:,}）')
    ax.fill_between(x,
                    np.clip(mean_in - std_in, 0, None),
                    mean_in + std_in,
                    color=Config.INPLANE_COLOR, alpha=Config.SHADE_ALPHA)

    mean_out = stats_out["mean"]
    std_out  = stats_out["std"]
    ax.plot(x, mean_out, color=Config.OUTPLANE_COLOR,
            linewidth=Config.LINE_WIDTH, marker='s', markersize=4,
            label=f'面外（n={stats_out["n"]:,}）')
    ax.fill_between(x,
                    np.clip(mean_out - std_out, 0, None),
                    mean_out + std_out,
                    color=Config.OUTPLANE_COLOR, alpha=Config.SHADE_ALPHA)

    ax.axhline(y=1.0, color='gray', linewidth=1.0, linestyle='--', alpha=0.6)

    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0, _auto_ylim(mean_in, std_in, mean_out, std_out))
    ax.set_xticks(np.arange(0, n + 1, 5))
    ax.set_xlabel('主频阶序', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel('累积能量占比', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title(
        f'前{n}阶主频累积能量分布（nfft={nfft}）',
        fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14,
    )
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)
    _add_legend(ax)
    _apply_grid(ax)
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="图4-11 前50阶累积能量（支持快照）")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="忽略已有快照，强制从原始 VIC 重算并覆盖快照",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("图4-11 随机振动前50阶主频累积能量分布（nfft=128）")
    print("=" * 80)
    print(f"  样本索引目录：{Config.ENRICHED_STATS_DIR}")
    print(f"  分辨率：{Config.NFFT_LIST}（enriched 2048 只读索引，结果追加快照）")
    print(f"  快照目录：{Config.SNAPSHOT_DIR}")
    if args.refresh_cache:
        print("  模式：--refresh-cache（强制重算 nfft128 快照）")
    else:
        print("  模式：优先读快照，缺失才重算")

    for slot, nfft in enumerate(Config.NFFT_LIST):
        print("\n" + "-" * 80)
        print(f"[分辨率 nfft={nfft}]")
        print("-" * 80)
        print(f"\n[步骤1] 加载/计算 PSD 累积能量（全谱 PSD 线性求和，nfft={nfft}）...")
        data = load_or_compute(nfft=nfft, force_refresh=args.refresh_cache)

        n_in  = len(data["cumsum_in"])
        n_out = len(data["cumsum_out"])
        print(f"面内有效样本：{n_in}，面外有效样本：{n_out}")

        stats_in  = _aggregate(data["cumsum_in"],  Config.N_MODES)
        stats_out = _aggregate(data["cumsum_out"], Config.N_MODES)
        print(f"  面内  第1阶占比均值：{stats_in['mean'][0]:.4f}  "
              f"前50阶累积均值：{stats_in['mean'][-1]:.4f}")
        print(f"  面外  第1阶占比均值：{stats_out['mean'][0]:.4f}  "
              f"前50阶累积均值：{stats_out['mean'][-1]:.4f}")

        print(f"\n[步骤2] 绘制图像（nfft={nfft}）...")
        fig = plot_energy_cumsum(data)
        web_push(
            fig,
            page=Config.WEB_PAGE,
            slot=slot,
            title=f"前50阶累积能量 nfft={nfft}",
            page_cols=1,
        )
        plt.close(fig)
        print(f"已推送到 WebUI：{Config.WEB_PAGE} / slot={slot}")

    print("=" * 80)
    print("完成")


if __name__ == "__main__":
    main()
