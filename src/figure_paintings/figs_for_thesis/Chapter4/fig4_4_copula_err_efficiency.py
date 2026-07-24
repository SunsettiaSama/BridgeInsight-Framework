"""
图4-4 Copula 特征截断阶数：精度（ERR）与效率（T_norm）权衡

分辨率与数据策略（重要）
------------------------
- 统一使用 Welch nfft=128（与 fig4_12 累积能量一致）。
- 阶定义：全谱频点按 PSD 幅值降序；能量 = 频点 PSD / 全谱线性总功率。
- ERR(K) = 前 K 阶能量累积均值（相对全谱，不再相对 top-K 归一）。
- enriched（历史 nperseg=2048）只作样本索引（路径 + window_idx），只读不写、不覆盖。

快照 / 缓存：
  results/chapter4_characteristics/figure_snapshots/
    fig4_4_copula_err_efficiency_nfft128.json
    fig4_4_copula_modes_nfft128.npz

计时：warmup 后对每个 K 重复取中位数，避免 K=1 冷启动伪影。
抽样：按 VIC 文件集中取窗，减少解包文件数。

  python .../fig4_4_copula_err_efficiency.py
  python .../fig4_4_copula_err_efficiency.py --refresh-timing
  python .../fig4_4_copula_err_efficiency.py --refresh-cache
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
from scipy.signal import welch

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
    MAX_K = 50
    XTICK_STEP = 5
    MODEL_SAMPLE_SIZE = 20_000
    MONTE_CARLO_SAMPLES = 100_000

    FS = 50.0
    WINDOW_SIZE = _WINDOW_SIZE
    FREQ_LIMIT_HZ = 25.0
    NFFT = 128

    RANDOM_SEED = 42
    COPULA_TYPE = "gaussian"
    ERR_THRESHOLD = 85.0
    DELTA_ERR_THRESHOLD = 2.0

    # 计时：先 warmup 再重复取中位数，避免 K=1 吞掉冷启动开销
    TIMING_WARMUP = 1
    TIMING_REPEATS = 3
    TIMING_PROTOCOL = "warmup1_median3"

    ENERGY_REFERENCE = "welch_nfft128_ranked_bin_linear_sum_relative_to_full_spectrum"

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
    SNAPSHOT_PATH = SNAPSHOT_DIR / "fig4_4_copula_err_efficiency_nfft128.json"
    MODES_PATH = SNAPSHOT_DIR / "fig4_4_copula_modes_nfft128.npz"

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
        "random_seed": Config.RANDOM_SEED,
        "copula_type": Config.COPULA_TYPE,
        "err_threshold": Config.ERR_THRESHOLD,
        "delta_err_threshold": Config.DELTA_ERR_THRESHOLD,
        "energy_reference": Config.ENERGY_REFERENCE,
        "timing_protocol": Config.TIMING_PROTOCOL,
        "timing_warmup": Config.TIMING_WARMUP,
        "timing_repeats": Config.TIMING_REPEATS,
        "sample_strategy": "file_concentrated",
        "data_source": "raw_vic_indexed_by_enriched",
        "enriched_stats_dir": str(Config.ENRICHED_STATS_DIR),
    }


def _modes_config() -> dict:
    cfg = _snapshot_config()
    return {
        "max_k": cfg["max_k"],
        "model_sample_size": cfg["model_sample_size"],
        "nfft": cfg["nfft"],
        "random_seed": cfg["random_seed"],
        "energy_reference": cfg["energy_reference"],
        "sample_strategy": cfg["sample_strategy"],
        "freq_limit_hz": cfg["freq_limit_hz"],
        "window_size": cfg["window_size"],
        "fs": cfg["fs"],
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
    """写出 nfft128 全谱 ERR 快照；不触碰 enriched。"""
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
            f"nfft={Config.NFFT}；ERR=前K阶排序频点能量/全谱总功率；"
            f"计时={Config.TIMING_PROTOCOL}；enriched(2048) 仅作索引"
        ),
    }
    with open(Config.SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  写出快照：{Config.SNAPSHOT_PATH}")


def save_modes(freq_matrix: np.ndarray, energy_matrix: np.ndarray) -> None:
    Config.SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        Config.MODES_PATH,
        config_json=np.asarray(json.dumps(_modes_config(), ensure_ascii=False)),
        freq_matrix=np.asarray(freq_matrix, dtype=np.float64),
        energy_matrix=np.asarray(energy_matrix, dtype=np.float64),
    )
    print(f"  写出模态缓存：{Config.MODES_PATH}")


def load_modes() -> tuple[np.ndarray, np.ndarray] | None:
    if not Config.MODES_PATH.exists():
        return None
    data = np.load(Config.MODES_PATH, allow_pickle=False)
    saved = json.loads(str(data["config_json"]))
    if saved != _modes_config():
        print(f"  模态缓存参数不匹配，忽略：{Config.MODES_PATH}")
        return None
    freq_matrix = np.asarray(data["freq_matrix"], dtype=np.float64)
    energy_matrix = np.asarray(data["energy_matrix"], dtype=np.float64)
    print(f"  读取模态缓存：{Config.MODES_PATH}（n={freq_matrix.shape[0]}）")
    return freq_matrix, energy_matrix


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
    """按文件集中抽样：优先抽满同一 VIC，减少解包文件数。"""
    by_path: dict[str, list[int]] = defaultdict(list)
    for sample in samples:
        path = sample.get("inplane_file_path")
        window_idx = sample.get("window_idx")
        if path is None or window_idx is None:
            continue
        by_path[str(path)].append(int(window_idx))

    n_total = sum(len(v) for v in by_path.values())
    if n_total == 0:
        return []
    if n_total <= Config.MODEL_SAMPLE_SIZE:
        return [(path, w) for path, windows in by_path.items() for w in windows]

    rng = np.random.default_rng(Config.RANDOM_SEED)
    paths = list(by_path.keys())
    rng.shuffle(paths)
    jobs: list[tuple[str, int]] = []
    for path in paths:
        windows = list(by_path[path])
        rng.shuffle(windows)
        for window_idx in windows:
            jobs.append((path, window_idx))
            if len(jobs) >= Config.MODEL_SAMPLE_SIZE:
                print(
                    f"  全量索引 {n_total}，按文件集中抽样 {Config.MODEL_SAMPLE_SIZE}"
                    f"（涉及 {len({p for p, _ in jobs})} 个 VIC）"
                )
                return jobs
    return jobs


def _slice_window(raw: np.ndarray, window_idx: int) -> np.ndarray | None:
    start = window_idx * Config.WINDOW_SIZE
    end = start + Config.WINDOW_SIZE
    if end > len(raw):
        return None
    return raw[start:end]


def _extract_welch_modes(signal: np.ndarray, max_k: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Welch nfft=128：全谱频点按 PSD 降序，能量 / 全谱线性总功率。"""
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
    if len(psd) < max_k:
        return None

    total_power = float(np.sum(psd))
    if total_power <= 0:
        return None

    # 与累积能量图一致：按全谱 bin 能量排序；Copula 特征跳过 f=0
    order = np.argsort(psd)[::-1]
    f_modes: list[float] = []
    e_modes: list[float] = []
    for idx in order:
        freq = float(f[idx])
        energy = float(psd[idx] / total_power)
        if freq <= 0.0 or energy <= 0.0:
            continue
        if not np.isfinite(freq) or not np.isfinite(energy):
            continue
        f_modes.append(freq)
        e_modes.append(energy)
        if len(f_modes) >= max_k:
            break

    if len(f_modes) < max_k:
        return None
    return np.asarray(f_modes, dtype=np.float64), np.asarray(e_modes, dtype=np.float64)


def build_mode_arrays(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    enriched 仅提供 (path, window_idx)；按文件分组解包 VIC，用 nfft=128 现算全谱排序模态。
    """
    jobs = _sample_jobs(samples)
    print(f"  待处理窗口：{len(jobs)}（nfft={Config.NFFT}，全谱排序频点）")

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
        raise ValueError("未提取到有效面内 Welch(nfft=128) 全谱排序模态样本")

    print(f"  有效模态样本：{len(freq_ok)}")
    return np.asarray(freq_ok, dtype=np.float64), np.asarray(energy_ok, dtype=np.float64)


def _feature_matrix(freq_matrix: np.ndarray, energy_matrix: np.ndarray, k: int) -> np.ndarray:
    columns: list[np.ndarray] = []
    for i in range(k):
        columns.append(freq_matrix[:, i])
        columns.append(energy_matrix[:, i])
    return np.column_stack(columns)


def compute_err_curve(energy_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """ERR(K)=前K阶排序频点能量累积 / 全谱；energy_matrix 已是相对全谱占比。"""
    cumulative = np.cumsum(energy_matrix, axis=1)
    err = np.mean(cumulative, axis=0) * 100.0
    delta = np.diff(np.insert(err, 0, 0.0))
    abs_err_at_maxk = float(err[-1])
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
    """
    对每个 K：先 warmup 丢弃冷启动，再重复计时取中位数。
    T_norm 仍相对 K=1 的中位耗时归一。
    """
    timings: list[float] = []
    warmup_rng = np.random.default_rng(Config.RANDOM_SEED)
    warm_matrix = _feature_matrix(freq_matrix, energy_matrix, 1)
    print(
        f"  计时协议：{Config.TIMING_PROTOCOL}"
        f"（warmup={Config.TIMING_WARMUP}, repeats={Config.TIMING_REPEATS}）"
    )
    for _ in range(Config.TIMING_WARMUP):
        _fit_and_sample_copula(warm_matrix, warmup_rng)

    for k in range(1, Config.MAX_K + 1):
        matrix = _feature_matrix(freq_matrix, energy_matrix, k)
        print(
            f"  K={k:02d}  d={matrix.shape[1]:02d}  Gaussian Copula 拟合 + "
            f"{Config.MONTE_CARLO_SAMPLES:,} 次抽样 ×{Config.TIMING_REPEATS}..."
        )
        elapsed_runs: list[float] = []
        for rep in range(Config.TIMING_REPEATS):
            rng = np.random.default_rng(Config.RANDOM_SEED + 10_000 * k + rep)
            t0 = time.perf_counter()
            _fit_and_sample_copula(matrix, rng)
            elapsed_runs.append(time.perf_counter() - t0)
        elapsed = float(np.median(elapsed_runs))
        timings.append(elapsed)
        print(
            f"    T_calc={elapsed:.3f} s"
            f"（runs={[round(t, 3) for t in elapsed_runs]}）"
        )

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

    ax_err.set_xlabel("截断阶数", labelpad=8, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax_err.set_ylabel("能量保有率 ERR (%)", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax_time.set_ylabel(r"归一化耗时 $T_{\mathrm{norm}}$", labelpad=12, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax_err.set_xticks(np.arange(0, Config.MAX_K + 1, Config.XTICK_STEP))
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
    ax.set_xlabel("截断阶数", labelpad=8, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel(r"归一化计算耗时 $T_{\mathrm{norm}}$", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_xticks(np.arange(0, Config.MAX_K + 1, Config.XTICK_STEP))
    ax.set_xlim(0.4, Config.MAX_K + 0.6)
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


def _compute_from_modes(
    freq_matrix: np.ndarray,
    energy_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    n_model = int(freq_matrix.shape[0])
    print("\n[步骤4] 计算全谱 ERR 与边际增益...")
    err, delta_err, abs_err = compute_err_curve(energy_matrix)
    print(f"  ERR(K={Config.MAX_K} vs 全谱)={err[-1]:.2f}%")
    print(f"  K*={select_k_star(err, delta_err)}")

    print("\n[步骤5] Copula 拟合与抽样计时...")
    timings = run_copula_timing(freq_matrix, energy_matrix)
    print(f"  T_norm(K={Config.MAX_K})={timings[-1] / timings[0]:.2f}")
    save_snapshot(err, delta_err, timings, n_model, abs_err)
    return err, delta_err, timings, n_model, abs_err


def main() -> None:
    parser = argparse.ArgumentParser(description="图4-4 Copula 截断阶数精度–效率权衡（nfft=128，全谱）")
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="忽略快照与模态缓存并重算（不改 enriched）",
    )
    parser.add_argument(
        "--refresh-timing",
        action="store_true",
        help="保留模态缓存，仅按新计时协议重测 T_norm 并出图",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("图4-4 Copula 特征截断阶数性能对比（nfft=128，全谱）")
    print("=" * 80)
    print(f"  能量定义：{Config.ENERGY_REFERENCE}")
    print(f"  计时协议：{Config.TIMING_PROTOCOL}")
    print(f"  nfft={Config.NFFT}；enriched(2048) 只读索引")
    print(f"  快照：{Config.SNAPSHOT_PATH}")

    print("\n[步骤1] 检查 nfft=128 快照...")
    snapshot = None
    if not args.refresh_cache and not args.refresh_timing:
        snapshot = load_snapshot(force_refresh=False)

    if snapshot is not None:
        err = np.asarray(snapshot["err"], dtype=np.float64)
        delta_err = np.asarray(snapshot["delta_err"], dtype=np.float64)
        timings = np.asarray(snapshot["timings_sec"], dtype=np.float64)
        n_model = int(snapshot["n_model"])
        abs_err = float(snapshot.get("abs_err_at_maxk_pct", float("nan")))
        print(f"  使用快照：n={n_model}，K*={snapshot.get('k_star')}，ERR@MAX_K≈{abs_err:.1f}%")
    else:
        modes = None
        if not args.refresh_cache:
            print("\n[步骤2] 检查模态缓存...")
            modes = load_modes()

        if modes is None:
            print("\n[步骤2] 读取 enriched 索引（不修改 2048 结果）...")
            samples = _load_index_samples()
            print(f"  索引样本：{len(samples)}")

            print(f"\n[步骤3] 从原始 VIC 按 nfft={Config.NFFT} 提取全谱排序前 K 阶...")
            freq_matrix, energy_matrix = build_mode_arrays(samples)
            save_modes(freq_matrix, energy_matrix)
        else:
            freq_matrix, energy_matrix = modes

        err, delta_err, timings, n_model, abs_err = _compute_from_modes(
            freq_matrix, energy_matrix
        )
        print(f"  有效建模样本：{n_model}，ERR@MAX_K≈{abs_err:.1f}%")

    print("\n[步骤6] 绘制并推送图像...")
    fig_trade = plot_tradeoff_figure(err, delta_err, timings)
    fig_time = plot_time_figure(timings, err, delta_err)
    web_push(fig_trade, page="fig4_4 Copula性能对比", slot=0, title="精度–效率权衡 全谱", page_cols=2)
    web_push(fig_time, page="fig4_4 Copula性能对比", slot=1, title="归一化计算耗时 全谱")
    plt.close(fig_trade)
    plt.close(fig_time)
    print("OK 已推送到 WebUI：fig4_4 Copula性能对比")


if __name__ == "__main__":
    main()
