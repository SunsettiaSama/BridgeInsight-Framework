import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import orjson
from scipy import signal as scipy_signal

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chapter4_characteristics.feature_analysis.entry import ensure_enriched_for_figures
from src.data_processer.io_unpacker import UNPACK
from src.figure_paintings.figs_for_thesis.Chapter4._data_loader import (
    get_enriched_class_dir,
    iter_enriched_json_files,
)
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT,
    ENG_FONT,
    FONT_SIZE,
    VIV_INPLANE_COLOR,
    VIV_OUTPLANE_COLOR,
    VIV_VIB_COLOR,
)
from src.visualize_tools.web_dashboard import push as web_push


class Config:
    VIV_CLASS_ID = 1
    FEATURE_BATCH_SIZE = 512
    FREQ_THRESHOLD = 7.0

    FS = 50.0
    WINDOW_SIZE = 3000
    TRIM_START_SECOND = 0
    TRIM_END_SECOND = 20
    NFFT = 2048
    FREQ_LIMIT = 25.0

    NUM_SAMPLES_PER_GROUP = 20
    RANDOM_SEED = 42

    FIG_SIZE = (18, 11)
    N_ROWS = 4
    N_COLS = 5
    LINEWIDTH = 0.75
    INPLANE_COLOR = VIV_INPLANE_COLOR
    OUTPLANE_COLOR = VIV_OUTPLANE_COLOR
    TRAJ_COLOR = VIV_VIB_COLOR
    SCATTER_SIZE = 3
    SCATTER_ALPHA = 0.28
    GRID_COLOR = "gray"
    GRID_ALPHA = 0.25
    GRID_LINESTYLE = "--"
    GRID_LINEWIDTH = 0.5

    ENRICHED_STATS_DIR = get_enriched_class_dir(1)
    INFERENCE_RESULT_PATH = project_root / "results" / "chapter4_characteristics" / "inference" / "inference.json"
    SNAPSHOT_DIR = project_root / "results" / "chapter4_characteristics" / "figure_snapshots"
    SNAPSHOT_PATH = SNAPSHOT_DIR / "fig4_x_viv_low_freq_exploration_selection.json"
    WEB_PAGE = "fig4_x VIV低频样本探索"


GROUPS = {
    "inplane_only": "仅面内涡激共振",
    "outplane_only": "仅面外涡激共振",
    "both": "面内外同时涡激共振",
}


def _dominant_frequency(psd: dict) -> float | None:
    freqs = psd.get("frequencies")
    powers = psd.get("powers")
    if not freqs or not powers:
        return None
    dom_idx = int(np.argmax(powers))
    if dom_idx >= len(freqs):
        return None
    return float(freqs[dom_idx])


def _is_low_freq(psd: dict) -> bool:
    dom_freq = _dominant_frequency(psd)
    return dom_freq is not None and dom_freq < Config.FREQ_THRESHOLD


def _sample_group(sample: dict, pred_in: dict[int, int], pred_out: dict[int, int]) -> str | None:
    sample_idx = sample.get("sample_idx")
    if sample_idx is None:
        return None

    idx = int(sample_idx)
    low_in = _is_low_freq(sample.get("psd_inplane") or {})
    low_out = _is_low_freq(sample.get("psd_outplane") or {})

    is_in = pred_in.get(idx, 0) == Config.VIV_CLASS_ID
    is_out = pred_out.get(idx, 0) == Config.VIV_CLASS_ID
    if is_in and not is_out and low_in:
        return "inplane_only"
    if is_out and not is_in and low_out:
        return "outplane_only"
    if is_in and is_out and (low_in or low_out):
        return "both"
    return None


def _snapshot_config() -> dict:
    return {
        "inference_result_path": str(Config.INFERENCE_RESULT_PATH),
        "inference_result_mtime": Config.INFERENCE_RESULT_PATH.stat().st_mtime,
        "enriched_stats_dir": str(Config.ENRICHED_STATS_DIR),
        "freq_threshold": Config.FREQ_THRESHOLD,
        "num_samples_per_group": Config.NUM_SAMPLES_PER_GROUP,
        "random_seed": Config.RANDOM_SEED,
    }


def _read_selection_snapshot(candidate_ids: set[int]) -> dict | None:
    if not Config.SNAPSHOT_PATH.exists():
        return None

    with open(Config.SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if payload.get("config") != _snapshot_config():
        return None

    side_predictions = payload.get("side_predictions") or {}
    if not candidate_ids.issubset({int(k) for k in side_predictions}):
        return None

    print(f"  读取筛选快照：{Config.SNAPSHOT_PATH}")
    return payload


def _write_selection_snapshot(
    side_predictions: dict[int, tuple[int, int]],
    groups: dict[str, list[dict]],
    sampled_groups: dict[str, list[dict]],
) -> None:
    Config.SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": _snapshot_config(),
        "side_predictions": {
            str(idx): {"inplane_prediction": int(preds[0]), "outplane_prediction": int(preds[1])}
            for idx, preds in sorted(side_predictions.items())
        },
        "group_sample_ids": {
            group: [int(sample["sample_idx"]) for sample in samples]
            for group, samples in groups.items()
        },
        "selected_sample_ids": {
            group: [int(sample["sample_idx"]) for sample in samples]
            for group, samples in sampled_groups.items()
        },
    }
    with open(Config.SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  写出筛选快照：{Config.SNAPSHOT_PATH}")


def _iter_record_bytes(path: Path):
    key = b'"records"'
    chunk_size = 1024 * 1024 * 8
    buffer = b""
    found_records = False
    in_object = False
    in_string = False
    escape = False
    depth = 0
    obj = bytearray()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            if not found_records:
                buffer += chunk
                key_pos = buffer.find(key)
                if key_pos < 0:
                    buffer = buffer[-len(key):]
                    continue
                array_pos = buffer.find(b"[", key_pos)
                if array_pos < 0:
                    buffer = buffer[key_pos:]
                    continue
                data = buffer[array_pos + 1:]
                buffer = b""
                found_records = True
            else:
                data = chunk

            for byte in data:
                if not in_object:
                    if byte == 123:
                        in_object = True
                        in_string = False
                        escape = False
                        depth = 1
                        obj = bytearray([byte])
                    elif byte == 93:
                        return
                    continue

                obj.append(byte)
                if in_string:
                    if escape:
                        escape = False
                    elif byte == 92:
                        escape = True
                    elif byte == 34:
                        in_string = False
                    continue

                if byte == 34:
                    in_string = True
                elif byte == 123:
                    depth += 1
                elif byte == 125:
                    depth -= 1
                    if depth == 0:
                        yield bytes(obj)
                        in_object = False


def _load_side_predictions(candidate_ids: set[int]) -> dict[int, tuple[int, int]]:
    snapshot = _read_selection_snapshot(candidate_ids)
    if snapshot is not None:
        return {
            int(idx): (int(preds["inplane_prediction"]), int(preds["outplane_prediction"]))
            for idx, preds in snapshot["side_predictions"].items()
            if int(idx) in candidate_ids
        }

    if not Config.INFERENCE_RESULT_PATH.exists():
        raise FileNotFoundError(f"分向识别结果不存在：{Config.INFERENCE_RESULT_PATH}")

    print(f"  扫描分向识别结果：{Config.INFERENCE_RESULT_PATH.name}")
    side_predictions: dict[int, tuple[int, int]] = {}
    for i, raw_record in enumerate(_iter_record_bytes(Config.INFERENCE_RESULT_PATH), start=1):
        record = orjson.loads(raw_record)
        sample_idx = int(record["sample_idx"])
        if sample_idx in candidate_ids:
            side_predictions[sample_idx] = (
                int(record["inplane_prediction"]),
                int(record["outplane_prediction"]),
            )
            if len(side_predictions) == len(candidate_ids):
                break
        if i % 500_000 == 0:
            print(f"    已扫描 records：{i:,}，命中：{len(side_predictions):,}/{len(candidate_ids):,}")

    missing = len(candidate_ids) - len(side_predictions)
    if missing > 0:
        print(f"  注意：有 {missing} 个低频候选未在 inference 中匹配到分向预测")
    return side_predictions


def _load_low_freq_candidates() -> dict[int, dict]:
    ensure_enriched_for_figures(class_id=Config.VIV_CLASS_ID, batch_size=Config.FEATURE_BATCH_SIZE)
    json_files = iter_enriched_json_files(Config.ENRICHED_STATS_DIR)
    if not json_files:
        raise FileNotFoundError(f"目录下无 JSON 文件：{Config.ENRICHED_STATS_DIR}")

    candidates: dict[int, dict] = {}
    for json_file in json_files:
        print(f"  加载：{json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for sample in data["samples"]:
            sample_idx = sample.get("sample_idx")
            if sample_idx is None:
                continue
            if _is_low_freq(sample.get("psd_inplane") or {}) or _is_low_freq(sample.get("psd_outplane") or {}):
                candidates[int(sample_idx)] = sample

    print(f"  enriched 低频候选：{len(candidates)} 个")
    return candidates


def load_low_freq_groups() -> tuple[dict[str, list[dict]], dict[int, tuple[int, int]]]:
    candidates = _load_low_freq_candidates()
    side_predictions = _load_side_predictions(set(candidates))
    pred_in = {idx: preds[0] for idx, preds in side_predictions.items()}
    pred_out = {idx: preds[1] for idx, preds in side_predictions.items()}

    groups = {name: [] for name in GROUPS}
    for sample_idx, sample in candidates.items():
        group = _sample_group(sample, pred_in, pred_out)
        if group is not None:
            groups[group].append(sample)

    for group, samples in groups.items():
        print(f"  {GROUPS[group]}：{len(samples)} 个低频样本")
    return groups, side_predictions


def random_sample(samples: list[dict], group: str) -> list[dict]:
    n = len(samples)
    k = min(Config.NUM_SAMPLES_PER_GROUP, n)
    if k == 0:
        return []
    rng = np.random.default_rng(Config.RANDOM_SEED + list(GROUPS).index(group))
    chosen = sorted(rng.choice(n, size=k, replace=False).tolist())
    print(f"  {GROUPS[group]} 抽样索引：{chosen}（共 {n} 个中选 {k} 个）")
    return [samples[i] for i in chosen]


def _load_window(file_path: str, window_idx: int, unpacker: UNPACK, cache: dict[str, np.ndarray]) -> np.ndarray:
    if file_path not in cache:
        cache[file_path] = np.asarray(unpacker.VIC_DATA_Unpack(file_path), dtype=np.float64)
    raw = cache[file_path]
    start = window_idx * Config.WINDOW_SIZE
    end = start + Config.WINDOW_SIZE
    if end > len(raw):
        raise ValueError(f"窗口越界：window_idx={window_idx}, data_len={len(raw)}")
    return raw[start:end]


def load_sample_pair(sample: dict, unpacker: UNPACK, cache: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    in_data = _load_window(sample["inplane_file_path"], int(sample["window_idx"]), unpacker, cache)
    out_data = _load_window(sample["outplane_file_path"], int(sample["window_idx"]), unpacker, cache)
    return in_data, out_data


def _trim(data: np.ndarray) -> np.ndarray:
    start = int(Config.TRIM_START_SECOND * Config.FS)
    end = int(Config.TRIM_END_SECOND * Config.FS)
    return data[start:end]


def _format_title(sample: dict, sample_idx: int) -> str:
    timestamp = sample.get("timestamp", [])
    time_str = "-".join(str(t) for t in timestamp) if timestamp else ""
    sensor_id = sample.get("inplane_sensor_id", "")
    cable_id = sensor_id.replace("ST-VIC-", "").rsplit("-", 1)[0]
    f_in = _dominant_frequency(sample.get("psd_inplane") or {})
    f_out = _dominant_frequency(sample.get("psd_outplane") or {})
    freq_str = f"fi={f_in:.2f}, fo={f_out:.2f}" if f_in is not None and f_out is not None else ""
    return f"{sample_idx + 1}. {cable_id} {time_str} {freq_str}"


def _apply_grid(ax) -> None:
    ax.grid(
        True,
        color=Config.GRID_COLOR,
        alpha=Config.GRID_ALPHA,
        linewidth=Config.GRID_LINEWIDTH,
        linestyle=Config.GRID_LINESTYLE,
    )
    ax.set_axisbelow(True)


def _make_axes() -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(
        Config.N_ROWS,
        Config.N_COLS,
        figsize=Config.FIG_SIZE,
        sharex=False,
        sharey=False,
    )
    return fig, axes.ravel()


def plot_timeseries_grid(samples: list[dict], title_prefix: str, unpacker: UNPACK, cache: dict[str, np.ndarray]) -> plt.Figure:
    fig, axes = _make_axes()
    for i, sample in enumerate(samples):
        ax = axes[i]
        in_data, out_data = load_sample_pair(sample, unpacker, cache)
        in_plot = _trim(in_data)
        out_plot = _trim(out_data)
        time_axis = np.arange(len(in_plot)) / Config.FS + Config.TRIM_START_SECOND

        ax.plot(time_axis, in_plot, color=Config.INPLANE_COLOR, linewidth=Config.LINEWIDTH, label="面内" if i == 0 else None)
        ax.plot(time_axis, out_plot, color=Config.OUTPLANE_COLOR, linewidth=Config.LINEWIDTH, alpha=0.85, label="面外" if i == 0 else None)
        ax.set_title(_format_title(sample, i), fontproperties=ENG_FONT, fontsize=FONT_SIZE - 12, pad=3)
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 12)
        _apply_grid(ax)

    for ax in axes[len(samples):]:
        ax.set_visible(False)
    for ax in axes[-Config.N_COLS:]:
        ax.set_xlabel("时间 (s)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 8)
    for ax in axes[::Config.N_COLS]:
        ax.set_ylabel(r"加速度 ($m/s^2$)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, prop=CN_FONT, fontsize=FONT_SIZE - 8)
    fig.suptitle(title_prefix, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.07, top=0.88, hspace=0.45, wspace=0.25)
    return fig


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


def plot_spectrum_grid(samples: list[dict], title_prefix: str, unpacker: UNPACK, cache: dict[str, np.ndarray]) -> plt.Figure:
    fig, axes = _make_axes()
    for i, sample in enumerate(samples):
        ax = axes[i]
        in_data, out_data = load_sample_pair(sample, unpacker, cache)
        f_in, p_in = _welch(in_data)
        f_out, p_out = _welch(out_data)

        ax.plot(f_in, p_in, color=Config.INPLANE_COLOR, linewidth=Config.LINEWIDTH, label="面内" if i == 0 else None)
        ax.plot(f_out, p_out, color=Config.OUTPLANE_COLOR, linewidth=Config.LINEWIDTH, alpha=0.85, label="面外" if i == 0 else None)
        # ax.axvline(Config.FREQ_THRESHOLD, color="#404040", linewidth=0.8, linestyle="--", alpha=0.8)
        ax.set_xlim(0, Config.FREQ_LIMIT)
        ax.set_title(_format_title(sample, i), fontproperties=ENG_FONT, fontsize=FONT_SIZE - 12, pad=3)
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 12)
        _apply_grid(ax)

    for ax in axes[len(samples):]:
        ax.set_visible(False)
    for ax in axes[-Config.N_COLS:]:
        ax.set_xlabel("频率 (Hz)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 8)
    for ax in axes[::Config.N_COLS]:
        ax.set_ylabel("PSD", fontproperties=ENG_FONT, fontsize=FONT_SIZE - 8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, prop=CN_FONT, fontsize=FONT_SIZE - 8)
    fig.suptitle(title_prefix, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.07, top=0.88, hspace=0.45, wspace=0.25)
    return fig


def plot_trajectory_grid(samples: list[dict], title_prefix: str, unpacker: UNPACK, cache: dict[str, np.ndarray]) -> plt.Figure:
    fig, axes = _make_axes()
    for i, sample in enumerate(samples):
        ax = axes[i]
        in_data, out_data = load_sample_pair(sample, unpacker, cache)
        ax.scatter(out_data, in_data, s=Config.SCATTER_SIZE, color=Config.TRAJ_COLOR, alpha=Config.SCATTER_ALPHA, linewidths=0)

        all_vals = np.concatenate([in_data, out_data])
        v_min = float(all_vals.min())
        v_max = float(all_vals.max())
        margin = max((v_max - v_min) * 0.05, 1e-6)
        ax.set_xlim(v_min - margin, v_max + margin)
        ax.set_ylim(v_min - margin, v_max + margin)
        ax.set_title(_format_title(sample, i), fontproperties=ENG_FONT, fontsize=FONT_SIZE - 12, pad=3)
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 12)
        _apply_grid(ax)

    for ax in axes[len(samples):]:
        ax.set_visible(False)
    for ax in axes[-Config.N_COLS:]:
        ax.set_xlabel(r"面外加速度 ($m/s^2$)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 8)
    for ax in axes[::Config.N_COLS]:
        ax.set_ylabel(r"面内加速度 ($m/s^2$)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 8)

    fig.suptitle(title_prefix, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.07, top=0.88, hspace=0.45, wspace=0.30)
    return fig


def main() -> None:
    print("=" * 80)
    print(f"图4-x 低频 VIV 样本探索图（主导模态 < {Config.FREQ_THRESHOLD:g} Hz）")
    print("=" * 80)

    print("\n[步骤1] 加载并筛选低频样本...")
    groups, side_predictions = load_low_freq_groups()
    sampled_groups = {group: random_sample(samples, group) for group, samples in groups.items()}
    _write_selection_snapshot(side_predictions, groups, sampled_groups)

    print("\n[步骤2] 绘制探索图...")
    unpacker = UNPACK(init_path=False)
    cache: dict[str, np.ndarray] = {}
    slot = 0
    for group, samples in sampled_groups.items():
        if not samples:
            print(f"  跳过 {GROUPS[group]}：无样本")
            continue

        group_title = GROUPS[group]
        figures = [
            (plot_timeseries_grid(samples, f"{group_title} - 时程图", unpacker, cache), "时程图"),
            (plot_spectrum_grid(samples, f"{group_title} - 频谱图", unpacker, cache), "频谱图"),
            (plot_trajectory_grid(samples, f"{group_title} - 轨迹图", unpacker, cache), "轨迹图"),
        ]
        for fig, fig_type in figures:
            web_push(
                fig,
                page=Config.WEB_PAGE,
                slot=slot,
                title=f"{group_title} {fig_type}",
                page_cols=3 if slot == 0 else None,
            )
            plt.close(fig)
            slot += 1
        print(f"  [OK] {group_title} 已推送 3 张图")

    print(f"[OK] 已推送到 WebUI：{Config.WEB_PAGE}，共 {slot} 张图")


if __name__ == "__main__":
    main()
