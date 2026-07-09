import sys
import time
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy import stats

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_processer.preprocess.get_data_vib import VICWindowExtractor
from src.chapter4_characteristics.feature_analysis.entry import ensure_enriched_for_figures
from src.chapter4_characteristics.statistics.copula import fit_copula, sample_from_copula
from src.figure_paintings.figs_for_thesis.Chapter4._data_loader import (
    get_enriched_class_dir,
    iter_enriched_json_files,
)
from src.figure_paintings.figs_for_thesis.config import (
    ANNOTATION_COLOR,
    CN_FONT,
    DEFAULT_COLOR,
    FONT_SIZE,
    SQUARE_FIG_SIZE,
    get_blue_color_map,
)
from src.visualize_tools.web_dashboard import push as web_push


class Config:
    CLASS_ID = 0
    FEATURE_BATCH_SIZE = 512
    MAX_K = 10
    MODEL_SAMPLE_SIZE = 100_000
    MONTE_CARLO_SAMPLES = 100_000
    FS = 50.0
    WINDOW_SIZE = 3000
    MIN_PEAK_DISTANCE_HZ = 0.1
    PEAK_HALF_BAND_HZ = 0.05
    RANDOM_SEED = 42
    COPULA_TYPE = "gaussian"
    DELTA_ERR_THRESHOLD = 2.0
    FORCE_RECOMPUTE = False
    ENERGY_REFERENCE = "raw_fft_full_spectrum_peak_band"

    FIG_SIZE = SQUARE_FIG_SIZE
    LINEWIDTH = 2.4
    BAR_ALPHA = 0.35
    GRID_COLOR = "gray"
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = "--"

    ENRICHED_STATS_DIR = get_enriched_class_dir(CLASS_ID)
    SNAPSHOT_PATH = (
        project_root
        / "results"
        / "chapter4_characteristics"
        / "figure_snapshots"
        / "fig4_5_copula_err_efficiency.json"
    )
    _palette = get_blue_color_map(style="discrete", start_map_index=1, end_map_index=5).colors
    ERR_COLOR = _palette[3]
    DELTA_COLOR = _palette[1]
    TIME_COLOR = DEFAULT_COLOR


def _snapshot_config() -> dict:
    return {
        "class_id": Config.CLASS_ID,
        "max_k": Config.MAX_K,
        "model_sample_size": Config.MODEL_SAMPLE_SIZE,
        "monte_carlo_samples": Config.MONTE_CARLO_SAMPLES,
        "fs": Config.FS,
        "window_size": Config.WINDOW_SIZE,
        "min_peak_distance_hz": Config.MIN_PEAK_DISTANCE_HZ,
        "peak_half_band_hz": Config.PEAK_HALF_BAND_HZ,
        "random_seed": Config.RANDOM_SEED,
        "copula_type": Config.COPULA_TYPE,
        "delta_err_threshold": Config.DELTA_ERR_THRESHOLD,
        "energy_reference": Config.ENERGY_REFERENCE,
        "enriched_stats_dir": str(Config.ENRICHED_STATS_DIR),
    }


def load_snapshot() -> dict | None:
    if Config.FORCE_RECOMPUTE or not Config.SNAPSHOT_PATH.exists():
        return None

    with open(Config.SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if payload.get("config") != _snapshot_config():
        print(f"  快照参数不匹配，将重新计算：{Config.SNAPSHOT_PATH}")
        return None

    print(f"  读取结果快照：{Config.SNAPSHOT_PATH}")
    return payload


def save_snapshot(err: np.ndarray, delta_err: np.ndarray, timings: np.ndarray, n_model: int) -> None:
    Config.SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": _snapshot_config(),
        "n_model": int(n_model),
        "k_star": int(select_k_star(delta_err)),
        "err": err.tolist(),
        "delta_err": delta_err.tolist(),
        "timings_sec": timings.tolist(),
        "t_norm": (timings / timings[0]).tolist(),
    }
    with open(Config.SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  写出结果快照：{Config.SNAPSHOT_PATH}")


def _load_samples() -> list[dict]:
    ensure_enriched_for_figures(class_id=Config.CLASS_ID, batch_size=Config.FEATURE_BATCH_SIZE)
    json_files = iter_enriched_json_files(Config.ENRICHED_STATS_DIR)
    if not json_files:
        raise FileNotFoundError(f"目录下无 JSON 文件：{Config.ENRICHED_STATS_DIR}")

    samples: list[dict] = []
    for json_file in json_files:
        print(f"  加载：{json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        samples.extend(payload.get("samples", []))
    return samples


def _valid_records(samples: list[dict]) -> list[dict]:
    return [
        sample
        for sample in samples
        if sample.get("inplane_file_path") and sample.get("window_idx") is not None
    ]


def _sample_records(samples: list[dict]) -> list[dict]:
    records = _valid_records(samples)
    n = len(records)
    if n <= Config.MODEL_SAMPLE_SIZE:
        return records

    rng = np.random.default_rng(Config.RANDOM_SEED)
    idx = rng.choice(n, size=Config.MODEL_SAMPLE_SIZE, replace=False)
    return [records[i] for i in idx]


def _extract_fft_modes(window: np.ndarray, max_k: int) -> tuple[np.ndarray, np.ndarray] | None:
    signal = np.asarray(window, dtype=np.float64).ravel()
    if signal.size < 2:
        return None

    signal = signal - float(np.mean(signal))
    coeff = np.fft.rfft(signal)
    energy = np.abs(coeff) ** 2
    if energy.size > 2:
        energy[1:-1] *= 2.0

    freqs = np.fft.rfftfreq(signal.size, d=1.0 / Config.FS)
    freqs = freqs[1:]
    energy = energy[1:]
    total_energy = float(np.sum(energy))
    if total_energy <= 0 or not np.isfinite(total_energy):
        return None

    freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else Config.FS / signal.size
    min_distance = max(1, int(Config.MIN_PEAK_DISTANCE_HZ / freq_res))
    peaks, _ = find_peaks(energy, distance=min_distance)
    if len(peaks) < max_k:
        return None

    peak_order = peaks[np.argsort(energy[peaks])[::-1][:max_k]]
    f_modes = freqs[peak_order]
    e_modes = np.empty(max_k, dtype=np.float64)
    used = np.zeros_like(energy, dtype=bool)
    for i, peak_idx in enumerate(peak_order):
        band = np.abs(freqs - freqs[peak_idx]) <= Config.PEAK_HALF_BAND_HZ
        band = band & ~used
        e_modes[i] = float(np.sum(energy[band]) / total_energy)
        used |= band

    if not np.all(np.isfinite(f_modes)) or not np.all(np.isfinite(e_modes)):
        return None
    if np.any(f_modes <= 0) or np.any(e_modes <= 0):
        return None
    return f_modes, e_modes


def build_mode_arrays(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    records = _sample_records(samples)
    print(f"  抽样原始窗口记录：{len(records)}")

    freq_rows: list[np.ndarray] = []
    energy_rows: list[np.ndarray] = []

    grouped: dict[str, list[dict]] = {}
    for sample in records:
        grouped.setdefault(str(sample["inplane_file_path"]), []).append(sample)

    extractor = VICWindowExtractor(enable_denoise=False)
    for file_idx, (file_path, file_samples) in enumerate(grouped.items(), start=1):
        print(f"  [{file_idx}/{len(grouped)}] 读取原始文件：{Path(file_path).name}，窗口数={len(file_samples)}")
        vic_data = extractor.load_file(file_path)
        for sample in file_samples:
            window = extractor.extract_window_from_data(
                vic_data,
                int(sample["window_idx"]),
                Config.WINDOW_SIZE,
                metadata=None,
                file_path=file_path,
            )
            if window is None:
                continue
            extracted = _extract_fft_modes(window, Config.MAX_K)
            if extracted is None:
                continue
            freqs, energies = extracted
            freq_rows.append(freqs)
            energy_rows.append(energies)
        del vic_data

    if not freq_rows:
        raise ValueError("未提取到有效面内 FFT 模态样本")

    freq_matrix = np.asarray(freq_rows, dtype=np.float64)
    energy_matrix = np.asarray(energy_rows, dtype=np.float64)
    return freq_matrix, energy_matrix


def _feature_matrix(freq_matrix: np.ndarray, energy_matrix: np.ndarray, k: int) -> tuple[np.ndarray, list[str]]:
    columns: list[np.ndarray] = []
    var_names: list[str] = []
    for i in range(k):
        columns.append(freq_matrix[:, i])
        columns.append(energy_matrix[:, i])
        var_names.extend([f"freq_{i + 1}", f"energy_{i + 1}"])
    return np.column_stack(columns), var_names


def compute_err_curve(energy_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cumulative = np.cumsum(energy_matrix, axis=1)
    err = np.mean(cumulative, axis=0) * 100.0
    delta = np.diff(np.insert(err, 0, 0.0))
    return err, delta


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
        matrix, var_names = _feature_matrix(freq_matrix, energy_matrix, k)

        print(f"  K={k:02d}  d={matrix.shape[1]:02d}  Gaussian Copula 拟合 + {Config.MONTE_CARLO_SAMPLES:,} 次抽样...")
        t0 = time.perf_counter()
        _fit_and_sample_copula(matrix, rng)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        print(f"    T_calc={elapsed:.3f} s")

    return np.asarray(timings, dtype=np.float64)


def select_k_star(delta_err: np.ndarray) -> int:
    candidates = np.flatnonzero(delta_err < Config.DELTA_ERR_THRESHOLD) + 1
    candidates = candidates[candidates > 1]
    if len(candidates):
        return int(candidates[0])
    return int(Config.MAX_K)


def _apply_grid(ax) -> None:
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA, linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)


def plot_err_figure(err: np.ndarray, delta_err: np.ndarray) -> plt.Figure:
    k_values = np.arange(1, Config.MAX_K + 1)
    k_star = select_k_star(delta_err)

    fig, ax_err = plt.subplots(1, 1, figsize=Config.FIG_SIZE)

    ax_delta = ax_err.twinx()
    bars = ax_delta.bar(
        k_values,
        delta_err,
        color=Config.DELTA_COLOR,
        alpha=Config.BAR_ALPHA,
        label=r"$\Delta$ERR",
    )
    err_line, = ax_err.plot(
        k_values,
        err,
        marker="o",
        color=Config.ERR_COLOR,
        linewidth=Config.LINEWIDTH,
        label="ERR",
    )
    ax_err.axhline(85.0, color=ANNOTATION_COLOR, linewidth=1.4, linestyle="--", label="85% 阈值")
    ax_delta.axhline(Config.DELTA_ERR_THRESHOLD, color=Config.TIME_COLOR, linewidth=1.2, linestyle=":", label="2% 边际阈值")
    ax_err.axvline(k_star, color=ANNOTATION_COLOR, linewidth=1.4, linestyle="-.")
    ax_err.annotate(
        rf"$K^*={k_star}$",
        xy=(k_star, err[k_star - 1]),
        xytext=(k_star + 0.25, min(98.0, err[k_star - 1] + 6.0)),
        arrowprops={"arrowstyle": "->", "color": ANNOTATION_COLOR, "linewidth": 1.0},
        color=ANNOTATION_COLOR,
        fontsize=FONT_SIZE - 3,
    )

    ax_err.set_xlabel("截断阶数 K", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax_err.set_ylabel("能量保有率 ERR (%)", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax_delta.set_ylabel(r"边际增益 $\Delta$ERR (%)", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax_err.set_xticks(k_values)
    ax_err.set_ylim(0, 105)
    ax_delta.set_ylim(0, max(float(np.max(delta_err)) * 1.15, Config.DELTA_ERR_THRESHOLD * 2.0))
    _apply_grid(ax_err)

    handles = [err_line, bars, *ax_err.lines[1:2], *ax_delta.lines]
    labels = [h.get_label() for h in handles]
    legend = ax_err.legend(handles, labels, loc="center right", framealpha=0.9, prop=CN_FONT)
    for text in legend.get_texts():
        text.set_fontproperties(CN_FONT)

    for ax in (ax_err, ax_delta):
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 3)

    fig.subplots_adjust(left=0.15, right=0.86, bottom=0.14, top=0.96)
    return fig


def plot_time_figure(timings: np.ndarray, delta_err: np.ndarray) -> plt.Figure:
    k_values = np.arange(1, Config.MAX_K + 1)
    t_norm = timings / timings[0]
    k_star = select_k_star(delta_err)

    fig, ax_time = plt.subplots(1, 1, figsize=Config.FIG_SIZE)
    ax_time.plot(
        k_values,
        t_norm,
        marker="s",
        color=Config.TIME_COLOR,
        linewidth=Config.LINEWIDTH,
        label=r"$T_{\mathrm{norm}}$",
    )
    ax_time.axvline(k_star, color=ANNOTATION_COLOR, linewidth=1.4, linestyle="-.")
    ax_time.set_xlabel("截断阶数 K", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax_time.set_ylabel(r"归一化计算耗时 $T_{\mathrm{norm}}$", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax_time.set_xticks(k_values)
    ax_time.set_ylim(0, max(float(np.max(t_norm)) * 1.15, 1.2))
    _apply_grid(ax_time)

    legend = ax_time.legend(loc="upper left", framealpha=0.9, prop=CN_FONT)
    for text in legend.get_texts():
        text.set_fontproperties(CN_FONT)
    ax_time.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 3)

    fig.subplots_adjust(left=0.17, right=0.96, bottom=0.14, top=0.96)
    return fig


def main() -> None:
    print("=" * 80)
    print("图4-5 Copula 特征截断阶数性能对比（ERR + 计算效率）")
    print("=" * 80)
    print("\n[步骤1] 检查结果快照...")
    snapshot = load_snapshot()
    if snapshot is not None:
        err = np.asarray(snapshot["err"], dtype=np.float64)
        delta_err = np.asarray(snapshot["delta_err"], dtype=np.float64)
        timings = np.asarray(snapshot["timings_sec"], dtype=np.float64)
        n_model = int(snapshot["n_model"])
        print(f"  使用快照数据：n={n_model}，K*={snapshot.get('k_star')}")
    else:
        print("\n[步骤2] 加载随机振动 enriched 原始样本...")
        samples = _load_samples()
        print(f"  原始样本记录：{len(samples)}")

        print("\n[步骤3] 提取前 K 阶主频/能量特征...")
        freq_matrix, energy_matrix = build_mode_arrays(samples)
        n_model = int(freq_matrix.shape[0])
        print(f"  有效建模样本：{n_model}，最大阶数：{Config.MAX_K}")

        print("\n[步骤4] 计算 ERR 与边际增益...")
        err, delta_err = compute_err_curve(energy_matrix)
        print(f"  ERR(K={Config.MAX_K})={err[-1]:.2f}%")

        print("\n[步骤5] Copula 拟合与 10^5 次抽样计时...")
        timings = run_copula_timing(freq_matrix, energy_matrix)
        print(f"  T_norm(K={Config.MAX_K})={timings[-1] / timings[0]:.2f}")
        save_snapshot(err, delta_err, timings, n_model)

    print("\n[步骤6] 绘制并推送图像...")
    fig_err = plot_err_figure(err, delta_err)
    fig_time = plot_time_figure(timings, delta_err)
    web_push(fig_err, page="fig4_5 Copula性能对比", slot=0, title="ERR 与边际增益", page_cols=2)
    web_push(fig_time, page="fig4_5 Copula性能对比", slot=1, title="归一化计算耗时")
    plt.close(fig_err)
    plt.close(fig_time)
    print("OK 已推送到 WebUI：fig4_5 Copula性能对比")


if __name__ == "__main__":
    main()
