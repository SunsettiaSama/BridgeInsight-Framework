"""
图4-5 Copula 特征截断阶数：精度（ERR）与效率（T_norm）权衡

分辨率与数据策略（重要）
------------------------
- 统一使用 Welch nfft=128（与 fig4_10/11 一致），总功率 = 全谱 PSD 线性求和。
- enriched（历史 nperseg=2048）只作样本索引（路径 + window_idx），只读不写、不覆盖。
- nfft=128 的计算结果追加到独立快照，不改动任何 2048 enriched JSON，
  也不覆盖旧快照 fig4_5_copula_err_efficiency.json / nfft256 快照。

快照路径（追加）：
  results/chapter4_characteristics/figure_snapshots/
    fig4_5_copula_err_efficiency_nfft128.json

强制重算：
  python .../fig4_5_copula_err_efficiency.py --refresh-cache
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.signal import find_peaks, welch

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_processer.io_unpacker import UNPACK
from src.chapter4_characteristics.feature_analysis.entry import ensure_enriched_for_figures
from src.chapter4_characteristics.statistics.copula import fit_copula, sample_from_copula
from src.figure_paintings.figs_for_thesis.Chapter4._data_loader import (
    get_enriched_class_dir,
    iter_enriched_json_files,
)
from src.figure_paintings.figs_for_thesis.Chapter4._viv_pipeline import _WINDOW_SIZE
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT,
    FONT_SIZE,
    REC_FIG_SIZE,
    SQUARE_FIG_SIZE,
    get_full_color_map,
)
from src.visualize_tools.web_dashboard import push as web_push


class Config:
    CLASS_ID = 0
    FEATURE_BATCH_SIZE = 512
    MAX_K = 10
    MODEL_SAMPLE_SIZE = 20_000
    MONTE_CARLO_SAMPLES = 100_000

    FS = 50.0
    WINDOW_SIZE = _WINDOW_SIZE
    FREQ_LIMIT_HZ = 25.0
    NFFT = 128
    MIN_PEAK_DISTANCE_HZ = 0.1

    RANDOM_SEED = 42
    COPULA_TYPE = "gaussian"
    ERR_THRESHOLD = 85.0
    DELTA_ERR_THRESHOLD = 2.0

    ENERGY_REFERENCE = "welch_nfft128_peak_linear_sum_relative_to_topk"

    FIG_SIZE = REC_FIG_SIZE
    TIME_FIG_SIZE = SQUARE_FIG_SIZE
    LINEWIDTH = 2.6
    MARKER_SIZE = 8
    BAR_WIDTH = 0.62
    BAR_ALPHA = 0.78
    GRID_COLOR = "#d0d0d0"
    GRID_ALPHA = 0.55
    GRID_LINESTYLE = "-"

    ENRICHED_STATS_DIR = get_enriched_class_dir(CLASS_ID)
    SNAPSHOT_DIR = (
        project_root
        / "results"
        / "chapter4_characteristics"
        / "figure_snapshots"
    )
    # 追加 nfft128 快照；旧文件 / nfft256 快照保持不动
    SNAPSHOT_PATH = SNAPSHOT_DIR / "fig4_5_copula_err_efficiency_nfft128.json"

    _full = get_full_color_map(style="discrete").colors
    ERR_COLOR = _full[0]
    DELTA_COLOR = _full[7]
    TIME_COLOR = _full[9]
    GUIDE_COLOR = "#555555"


def _snapshot_config() -> dict:
    return {
        "class_id": Config.CLASS_ID,
        "max_k": Config.MAX_K,
        "model_sample_size": Config.MODEL_SAMPLE_SIZE,
        "monte_carlo_samples": Config.MONTE_CARLO_SAMPLES,
        "fs": Config.FS,
        "window_size": Config.WINDOW_SIZE,
        "freq_limit_hz": Config.FREQ_LIMIT_HZ,
        "nfft": Config.NFFT,
        "min_peak_distance_hz": Config.MIN_PEAK_DISTANCE_HZ,
        "random_seed": Config.RANDOM_SEED,
        "copula_type": Config.COPULA_TYPE,
        "err_threshold": Config.ERR_THRESHOLD,
        "delta_err_threshold": Config.DELTA_ERR_THRESHOLD,
        "energy_reference": Config.ENERGY_REFERENCE,
        "data_source": "raw_vic_indexed_by_enriched",
        "enriched_stats_dir": str(Config.ENRICHED_STATS_DIR),
    }


def load_snapshot(force_refresh: bool) -> dict | None:
    if force_refresh or not Config.SNAPSHOT_PATH.exists():
        return None

    with open(Config.SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if payload.get("config") != _snapshot_config():
        print(f"  快照参数不匹配，将重新计算：{Config.SNAPSHOT_PATH}")
        return None

    print(f"  读取 nfft={Config.NFFT} 快照：{Config.SNAPSHOT_PATH}")
    return payload


def save_snapshot(
    err: np.ndarray,
    delta_err: np.ndarray,
    timings: np.ndarray,
    n_model: int,
    abs_err_at_maxk: float,
) -> None:
    """追加写入 nfft128 快照；不触碰 enriched，也不覆盖旧 2048 / nfft256 相关快照。"""
    Config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": _snapshot_config(),
        "n_model": int(n_model),
        "k_star": int(select_k_star(err, delta_err)),
        "abs_err_at_maxk_pct": float(abs_err_at_maxk),
        "err": err.tolist(),
        "delta_err": delta_err.tolist(),
        "timings_sec": timings.tolist(),
        "t_norm": (timings / timings[0]).tolist(),
        "note": (
            f"nfft={Config.NFFT}；ERR=相对前 MAX_K 阶保有率；"
            "enriched(2048) 仅作索引，本文件为追加结果"
        ),
    }
    with open(Config.SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  追加写出快照：{Config.SNAPSHOT_PATH}")


def _load_index_samples() -> list[dict]:
    """只读 enriched 索引，绝不写回。"""
    ensure_enriched_for_figures(class_id=Config.CLASS_ID, batch_size=Config.FEATURE_BATCH_SIZE)
    json_files = iter_enriched_json_files(Config.ENRICHED_STATS_DIR)
    if not json_files:
        raise FileNotFoundError(f"目录下无 JSON 文件：{Config.ENRICHED_STATS_DIR}")

    samples: list[dict] = []
    for json_file in json_files:
        print(f"  索引：{json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        samples.extend(payload.get("samples", []))
    return samples


def _sample_jobs(samples: list[dict]) -> list[tuple[str, int]]:
    jobs: list[tuple[str, int]] = []
    for sample in samples:
        path = sample.get("inplane_file_path")
        window_idx = sample.get("window_idx")
        if path is None or window_idx is None:
            continue
        jobs.append((str(path), int(window_idx)))

    n = len(jobs)
    if n <= Config.MODEL_SAMPLE_SIZE:
        return jobs

    rng = np.random.default_rng(Config.RANDOM_SEED)
    chosen = rng.choice(n, size=Config.MODEL_SAMPLE_SIZE, replace=False)
    chosen.sort()
    print(f"  全量索引 {n}，随机抽样 {Config.MODEL_SAMPLE_SIZE}")
    return [jobs[i] for i in chosen.tolist()]


def _slice_window(raw: np.ndarray, window_idx: int) -> np.ndarray | None:
    start = window_idx * Config.WINDOW_SIZE
    end = start + Config.WINDOW_SIZE
    if end > len(raw):
        return None
    return raw[start:end]


def _extract_welch_modes(signal: np.ndarray, max_k: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Welch nfft=128：峰顶功率 / 全谱线性总功率，按峰高取前 max_k 阶。"""
    sig = np.asarray(signal, dtype=np.float64).ravel()
    if sig.size < Config.NFFT:
        return None

    f, psd = welch(
        sig,
        fs=Config.FS,
        nperseg=Config.NFFT // 2,
        noverlap=Config.NFFT // 4,
        nfft=Config.NFFT,
        scaling="density",
    )
    mask = f <= Config.FREQ_LIMIT_HZ
    f = f[mask]
    psd = psd[mask]
    if len(psd) < max_k + 2:
        return None

    total_power = float(np.sum(psd))
    if total_power <= 0:
        return None

    freq_res = float(f[1] - f[0]) if len(f) > 1 else Config.FS / Config.NFFT
    min_distance = max(1, int(Config.MIN_PEAK_DISTANCE_HZ / freq_res))
    peaks, _ = find_peaks(psd, distance=min_distance)
    if len(peaks) < max_k:
        return None

    peak_order = peaks[np.argsort(psd[peaks])[::-1][:max_k]]
    f_modes = f[peak_order]
    e_modes = psd[peak_order] / total_power

    if np.any(f_modes <= 0) or np.any(e_modes <= 0):
        return None
    if not np.all(np.isfinite(f_modes)) or not np.all(np.isfinite(e_modes)):
        return None
    return f_modes, e_modes


def build_mode_arrays(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    enriched 仅提供 (path, window_idx)；按文件分组解包 VIC，用 nfft=128 现算模态。
    """
    jobs = _sample_jobs(samples)
    print(f"  待处理窗口：{len(jobs)}（nfft={Config.NFFT}）")

    by_path: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i, (path, window_idx) in enumerate(jobs):
        by_path[path].append((window_idx, i))

    freq_rows: list[np.ndarray | None] = [None] * len(jobs)
    energy_rows: list[np.ndarray | None] = [None] * len(jobs)
    unpacker = UNPACK(init_path=False)
    n_files = len(by_path)

    for file_i, (path, items) in enumerate(by_path.items(), start=1):
        if not Path(path).exists():
            continue
        raw = np.asarray(unpacker.VIC_DATA_Unpack(str(path)), dtype=np.float64)
        for window_idx, job_i in items:
            sig = _slice_window(raw, window_idx)
            if sig is None:
                continue
            extracted = _extract_welch_modes(sig, Config.MAX_K)
            if extracted is None:
                continue
            freqs, energies = extracted
            freq_rows[job_i] = freqs
            energy_rows[job_i] = energies

        if file_i % 200 == 0 or file_i == n_files:
            print(f"    已解包文件 {file_i}/{n_files}")

    freq_ok = [r for r in freq_rows if r is not None]
    energy_ok = [r for r in energy_rows if r is not None]
    if not freq_ok:
        raise ValueError("未提取到有效面内 Welch(nfft=128) 模态样本")

    print(f"  有效模态样本：{len(freq_ok)}")
    return np.asarray(freq_ok, dtype=np.float64), np.asarray(energy_ok, dtype=np.float64)


def _feature_matrix(freq_matrix: np.ndarray, energy_matrix: np.ndarray, k: int) -> np.ndarray:
    columns: list[np.ndarray] = []
    for i in range(k):
        columns.append(freq_matrix[:, i])
        columns.append(energy_matrix[:, i])
    return np.column_stack(columns)


def compute_err_curve(energy_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    cumulative = np.cumsum(energy_matrix, axis=1)
    topk_total = cumulative[:, -1:]
    relative = cumulative / np.clip(topk_total, 1e-30, None)
    err = np.mean(relative, axis=0) * 100.0
    delta = np.diff(np.insert(err, 0, 0.0))
    abs_err_at_maxk = float(np.mean(topk_total) * 100.0)
    return err, delta, abs_err_at_maxk


def _empirical_pit(matrix: np.ndarray) -> np.ndarray:
    u = np.empty_like(matrix, dtype=np.float64)
    for j in range(matrix.shape[1]):
        u[:, j] = stats.rankdata(matrix[:, j], method="average") / (matrix.shape[0] + 1.0)
    return np.clip(u, 1e-10, 1.0 - 1e-10)


def _empirical_inverse(u_samples: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    reconstructed = np.empty_like(u_samples, dtype=np.float64)
    q_grid = (np.arange(matrix.shape[0], dtype=np.float64) + 0.5) / matrix.shape[0]
    for j in range(matrix.shape[1]):
        sorted_col = np.sort(matrix[:, j])
        reconstructed[:, j] = np.interp(u_samples[:, j], q_grid, sorted_col)
    return reconstructed


def _fit_and_sample_copula(matrix: np.ndarray, rng: np.random.Generator) -> None:
    u_matrix = _empirical_pit(matrix)
    copula_result = fit_copula(u_matrix, copula_type=Config.COPULA_TYPE)
    u_samples = sample_from_copula(copula_result, Config.MONTE_CARLO_SAMPLES, rng)
    _empirical_inverse(u_samples, matrix)


def run_copula_timing(freq_matrix: np.ndarray, energy_matrix: np.ndarray) -> np.ndarray:
    timings: list[float] = []
    rng = np.random.default_rng(Config.RANDOM_SEED)

    for k in range(1, Config.MAX_K + 1):
        matrix = _feature_matrix(freq_matrix, energy_matrix, k)
        print(f"  K={k:02d}  d={matrix.shape[1]:02d}  Gaussian Copula 拟合 + {Config.MONTE_CARLO_SAMPLES:,} 次抽样...")
        t0 = time.perf_counter()
        _fit_and_sample_copula(matrix, rng)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        print(f"    T_calc={elapsed:.3f} s")

    return np.asarray(timings, dtype=np.float64)


def select_k_star(err: np.ndarray, delta_err: np.ndarray) -> int:
    hit = np.flatnonzero(err >= Config.ERR_THRESHOLD) + 1
    if len(hit):
        return int(hit[0])

    candidates = np.flatnonzero(delta_err < Config.DELTA_ERR_THRESHOLD) + 1
    candidates = candidates[candidates > 1]
    if len(candidates):
        return int(candidates[0])
    return int(Config.MAX_K)


def _apply_grid(ax) -> None:
    ax.grid(True, axis="y", color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA, linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)


def plot_tradeoff_figure(err: np.ndarray, delta_err: np.ndarray, timings: np.ndarray) -> plt.Figure:
    k_values = np.arange(1, Config.MAX_K + 1)
    t_norm = timings / timings[0]
    k_star = select_k_star(err, delta_err)

    fig, ax_err = plt.subplots(1, 1, figsize=Config.FIG_SIZE)
    ax_time = ax_err.twinx()

    bars = ax_err.bar(
        k_values,
        delta_err,
        width=Config.BAR_WIDTH,
        color=Config.DELTA_COLOR,
        alpha=Config.BAR_ALPHA,
        edgecolor="white",
        linewidth=0.6,
        label=r"$\Delta$ERR（边际精度）",
        zorder=2,
    )
    err_line, = ax_err.plot(
        k_values, err, marker="o", markersize=Config.MARKER_SIZE,
        color=Config.ERR_COLOR, linewidth=Config.LINEWIDTH,
        label="ERR（累积精度）", zorder=4,
    )
    time_line, = ax_time.plot(
        k_values, t_norm, marker="s", markersize=Config.MARKER_SIZE - 1,
        color=Config.TIME_COLOR, linewidth=Config.LINEWIDTH, linestyle="--",
        label=r"$T_{\mathrm{norm}}$（计算代价）", zorder=4,
    )

    ax_err.axhline(Config.ERR_THRESHOLD, color=Config.GUIDE_COLOR, linewidth=1.3, linestyle="--", zorder=3)
    ax_err.axvline(k_star, color=Config.GUIDE_COLOR, linewidth=1.5, linestyle="-.", zorder=3)
    # K* 标注放在曲线右侧空白区，避开图例与柱顶
    ax_err.annotate(
        rf"$K^*={k_star}$",
        xy=(k_star, err[k_star - 1]),
        xytext=(k_star + 1.15, min(98.0, err[k_star - 1] + 8.0)),
        arrowprops={
            "arrowstyle": "->",
            "color": Config.GUIDE_COLOR,
            "lw": 1.2,
            "connectionstyle": "arc3,rad=-0.15",
        },
        color=Config.GUIDE_COLOR,
        fontsize=FONT_SIZE - 4,
        fontproperties=CN_FONT,
        ha="left",
        va="bottom",
        zorder=5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": Config.GUIDE_COLOR, "alpha": 0.92},
    )

    ax_err.set_xlabel("截断阶数 K", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax_err.set_ylabel("能量保有率 ERR (%)", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax_time.set_ylabel(r"归一化耗时 $T_{\mathrm{norm}}$", labelpad=12, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax_err.set_xticks(k_values)
    ax_err.set_xlim(0.4, Config.MAX_K + 0.6)
    ax_err.set_ylim(0, 108)
    ax_time.set_ylim(0, max(float(np.max(t_norm)) * 1.18, 1.3))
    _apply_grid(ax_err)

    # 图例置于图外上方横排，避免遮挡左侧高柱与上升段
    handles = [err_line, time_line, bars]
    legend = ax_err.legend(
        handles,
        [h.get_label() for h in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        framealpha=0.95,
        prop=CN_FONT,
        fontsize=FONT_SIZE - 5,
        borderpad=0.35,
        columnspacing=1.2,
        handlelength=1.8,
    )
    for text in legend.get_texts():
        text.set_fontproperties(CN_FONT)

    for ax in (ax_err, ax_time):
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 4)
        ax.spines["top"].set_visible(False)

    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.14, top=0.86)
    return fig


def plot_time_figure(timings: np.ndarray, err: np.ndarray, delta_err: np.ndarray) -> plt.Figure:
    k_values = np.arange(1, Config.MAX_K + 1)
    t_norm = timings / timings[0]
    k_star = select_k_star(err, delta_err)

    fig, ax = plt.subplots(1, 1, figsize=Config.TIME_FIG_SIZE)
    ax.plot(
        k_values, t_norm, marker="s", markersize=Config.MARKER_SIZE,
        color=Config.TIME_COLOR, linewidth=Config.LINEWIDTH, label=r"$T_{\mathrm{norm}}$",
    )
    ax.axvline(k_star, color=Config.GUIDE_COLOR, linewidth=1.5, linestyle="-.")
    ax.annotate(
        rf"$K^*={k_star}$",
        xy=(k_star, t_norm[k_star - 1]),
        xytext=(k_star - 2.2 if k_star > 3 else k_star + 0.4, t_norm[k_star - 1] * 0.72),
        arrowprops={"arrowstyle": "->", "color": Config.GUIDE_COLOR, "lw": 1.1},
        color=Config.GUIDE_COLOR,
        fontsize=FONT_SIZE - 4,
        fontproperties=CN_FONT,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": Config.GUIDE_COLOR, "alpha": 0.9},
    )
    ax.set_xlabel("截断阶数 K", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel(r"归一化计算耗时 $T_{\mathrm{norm}}$", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_xticks(k_values)
    ax.set_ylim(0, max(float(np.max(t_norm)) * 1.15, 1.2))
    _apply_grid(ax)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend = ax.legend(loc="upper left", framealpha=0.95, prop=CN_FONT)
    for text in legend.get_texts():
        text.set_fontproperties(CN_FONT)
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 4)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.14, top=0.95)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="图4-5 Copula 截断阶数精度–效率权衡（nfft=128）")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="忽略 nfft128 快照并重算（不改 enriched / 不改旧快照）",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("图4-5 Copula 特征截断阶数性能对比（nfft=128）")
    print("=" * 80)
    print(f"  能量定义：{Config.ENERGY_REFERENCE}")
    print(f"  nfft={Config.NFFT}；enriched(2048) 只读索引，结果追加到独立快照")
    print(f"  快照：{Config.SNAPSHOT_PATH}")

    print("\n[步骤1] 检查 nfft=128 快照...")
    snapshot = load_snapshot(force_refresh=args.refresh_cache)
    if snapshot is not None:
        err = np.asarray(snapshot["err"], dtype=np.float64)
        delta_err = np.asarray(snapshot["delta_err"], dtype=np.float64)
        timings = np.asarray(snapshot["timings_sec"], dtype=np.float64)
        n_model = int(snapshot["n_model"])
        abs_err = float(snapshot.get("abs_err_at_maxk_pct", float("nan")))
        print(f"  使用快照：n={n_model}，K*={snapshot.get('k_star')}，绝对保有率@MAX_K≈{abs_err:.1f}%")
    else:
        print("\n[步骤2] 读取 enriched 索引（不修改 2048 结果）...")
        samples = _load_index_samples()
        print(f"  索引样本：{len(samples)}")

        print(f"\n[步骤3] 从原始 VIC 按 nfft={Config.NFFT} 提取前 K 阶...")
        freq_matrix, energy_matrix = build_mode_arrays(samples)
        n_model = int(freq_matrix.shape[0])
        print(f"  有效建模样本：{n_model}")

        print("\n[步骤4] 计算相对 ERR 与边际增益...")
        err, delta_err, abs_err = compute_err_curve(energy_matrix)
        print(f"  相对 ERR(K={Config.MAX_K})={err[-1]:.2f}%")
        print(f"  绝对保有率(K={Config.MAX_K} vs 全谱)={abs_err:.2f}%")
        print(f"  K*={select_k_star(err, delta_err)}")

        print("\n[步骤5] Copula 拟合与抽样计时...")
        timings = run_copula_timing(freq_matrix, energy_matrix)
        print(f"  T_norm(K={Config.MAX_K})={timings[-1] / timings[0]:.2f}")
        save_snapshot(err, delta_err, timings, n_model, abs_err)

    print("\n[步骤6] 绘制并推送图像...")
    fig_trade = plot_tradeoff_figure(err, delta_err, timings)
    fig_time = plot_time_figure(timings, err, delta_err)
    web_push(fig_trade, page="fig4_5 Copula性能对比", slot=0, title="精度–效率权衡 nfft=128", page_cols=2)
    web_push(fig_time, page="fig4_5 Copula性能对比", slot=1, title="归一化计算耗时 nfft=128")
    plt.close(fig_trade)
    plt.close(fig_time)
    print("OK 已推送到 WebUI：fig4_5 Copula性能对比")


if __name__ == "__main__":
    main()
