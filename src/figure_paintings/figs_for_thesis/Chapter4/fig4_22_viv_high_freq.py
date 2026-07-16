import json
import socket
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import orjson

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chapter4_characteristics.feature_analysis.entry import ensure_enriched_for_figures
from src.data_processer.io_unpacker import UNPACK
from src.figure_paintings.figs_for_thesis.Chapter4._data_loader import get_enriched_class_dir, iter_enriched_json_files
from src.figure_paintings.figs_for_thesis.Chapter4._viv_pipeline import _WINDOW_SIZE, psd_ranked_energy_cumsum
from src.figure_paintings.figs_for_thesis.config import (
    ABOVE_THRESHOLD_COLOR,
    CN_FONT,
    SQUARE_FIG_SIZE,
    SQUARE_FONT_SIZE,
    VIV_INPLANE_COLOR,
    VIV_OUTPLANE_COLOR,
)
from src.visualize_tools.web_dashboard import push as web_push


class Config:
    VIV_CLASS_ID = 1
    FEATURE_BATCH_SIZE = 512
    FREQ_THRESHOLD = 7.0

    N_MODES = 50
    NFFT = 128
    N_BINS = 80
    RMS_X_PERCENTILE = 99.0
    SCATTER_AXIS_PERCENTILE = 99.5
    SCATTER_AXIS_PAD = 1.08

    FIG_SIZE = SQUARE_FIG_SIZE
    LABEL_FONT_SIZE = SQUARE_FONT_SIZE
    TICK_FONT_SIZE = SQUARE_FONT_SIZE - 4
    LEGEND_FONT_SIZE = SQUARE_FONT_SIZE - 4

    GRID_COLOR = "gray"
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = "--"
    SHADE_ALPHA = 0.18
    LINE_WIDTH = 2.2

    INPLANE_COLOR = VIV_INPLANE_COLOR
    OUTPLANE_COLOR = VIV_OUTPLANE_COLOR
    BOTH_COLOR = ABOVE_THRESHOLD_COLOR
    BAR_ALPHA = 0.72
    SCATTER_SIZE = 8
    SCATTER_ALPHA = 0.35

    ENRICHED_STATS_DIR = get_enriched_class_dir(1)
    INFERENCE_RESULT_PATH = project_root / "results" / "chapter4_characteristics" / "inference" / "inference.json"
    SNAPSHOT_PATH = (
        project_root
        / "results"
        / "chapter4_characteristics"
        / "figure_snapshots"
        / "fig4_22_viv_high_freq_selection.json"
    )
    WEB_PAGE = "fig4_22 VIV高频主导样本"
    WEB_DASHBOARD_PORT = 15678


def _dominant_frequency(freqs: list[float] | None, powers: list[float] | None) -> float | None:
    if not freqs or not powers:
        return None
    dom_idx = int(np.argmax(powers))
    if dom_idx >= len(freqs):
        return None
    return float(freqs[dom_idx])


def _is_high_freq(freqs: list[float] | None, powers: list[float] | None) -> bool:
    dom_freq = _dominant_frequency(freqs, powers)
    return dom_freq is not None and dom_freq >= Config.FREQ_THRESHOLD


def _snapshot_config() -> dict:
    return {
        "inference_result_path": str(Config.INFERENCE_RESULT_PATH),
        "inference_result_mtime": Config.INFERENCE_RESULT_PATH.stat().st_mtime,
        "enriched_stats_dir": str(Config.ENRICHED_STATS_DIR),
        "freq_threshold": Config.FREQ_THRESHOLD,
    }


def _iter_record_bytes(path: Path):
    key = b'"records"'
    buffer = b""
    found_records = False
    in_object = False
    in_string = False
    escape = False
    depth = 0
    obj = bytearray()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024 * 8)
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


def _read_side_prediction_snapshot(candidate_ids: set[int]) -> dict[int, tuple[int, int]] | None:
    if not Config.SNAPSHOT_PATH.exists():
        return None
    with open(Config.SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("config") != _snapshot_config():
        return None
    side_predictions = payload.get("side_predictions") or {}
    if not candidate_ids.issubset({int(k) for k in side_predictions}):
        return None
    print(f"  读取高频分向快照：{Config.SNAPSHOT_PATH}")
    return {
        int(idx): (int(preds["inplane_prediction"]), int(preds["outplane_prediction"]))
        for idx, preds in side_predictions.items()
        if int(idx) in candidate_ids
    }


def _load_side_predictions(candidate_ids: set[int]) -> dict[int, tuple[int, int]]:
    cached = _read_side_prediction_snapshot(candidate_ids)
    if cached is not None:
        return cached
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
    return side_predictions


def _write_selection_snapshot(side_predictions: dict[int, tuple[int, int]], groups: dict[str, list[int]]) -> None:
    Config.SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": _snapshot_config(),
        "side_predictions": {
            str(idx): {"inplane_prediction": int(preds[0]), "outplane_prediction": int(preds[1])}
            for idx, preds in sorted(side_predictions.items())
        },
        "group_sample_ids": groups,
    }
    with open(Config.SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  写出高频分向快照：{Config.SNAPSHOT_PATH}")


def _append_if_high_freq(
    rms_list: list[float],
    job_list: list[tuple[str, int]],
    rms: float | None,
    freqs: list[float] | None,
    powers: list[float] | None,
    file_path: str | None,
    window_idx: int | None,
) -> bool:
    if rms is None or file_path is None or window_idx is None:
        return False
    if not _is_high_freq(freqs, powers):
        return False
    rms_list.append(float(rms))
    job_list.append((file_path, int(window_idx)))
    return True


def _group_name(sample_idx: int, high_in: bool, high_out: bool, side_predictions: dict[int, tuple[int, int]]) -> str | None:
    pred_in, pred_out = side_predictions.get(sample_idx, (0, 0))
    is_in = pred_in == Config.VIV_CLASS_ID
    is_out = pred_out == Config.VIV_CLASS_ID
    if is_in and not is_out and high_in:
        return "inplane_only"
    if is_out and not is_in and high_out:
        return "outplane_only"
    if is_in and is_out and (high_in or high_out):
        return "both"
    return None


def load_high_freq_data() -> dict:
    ensure_enriched_for_figures(class_id=Config.VIV_CLASS_ID, batch_size=Config.FEATURE_BATCH_SIZE)
    json_files = iter_enriched_json_files(Config.ENRICHED_STATS_DIR)
    if not json_files:
        raise FileNotFoundError(f"目录下无 JSON 文件：{Config.ENRICHED_STATS_DIR}")

    samples: list[dict] = []
    candidate_ids: set[int] = set()
    for json_file in json_files:
        print(f"  加载：{json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sample in data["samples"]:
            sample_idx = sample.get("sample_idx")
            if sample_idx is None:
                continue
            psd_in = sample.get("psd_inplane") or {}
            psd_out = sample.get("psd_outplane") or {}
            high_in = _is_high_freq(psd_in.get("frequencies"), psd_in.get("powers"))
            high_out = _is_high_freq(psd_out.get("frequencies"), psd_out.get("powers"))
            if high_in or high_out:
                samples.append(sample)
                candidate_ids.add(int(sample_idx))

    print(f"  enriched 高频候选：{len(samples)} 个")
    side_predictions = _load_side_predictions(candidate_ids)

    inplane_rms: list[float] = []
    outplane_rms: list[float] = []
    paired_inplane_rms: list[float] = []
    paired_outplane_rms: list[float] = []
    jobs_in: list[tuple[str, int]] = []
    jobs_out: list[tuple[str, int]] = []
    groups = {"inplane_only": [], "outplane_only": [], "both": []}

    for sample in samples:
        sample_idx = int(sample["sample_idx"])
        ts_in = sample.get("time_stats_inplane") or {}
        ts_out = sample.get("time_stats_outplane") or {}
        psd_in = sample.get("psd_inplane") or {}
        psd_out = sample.get("psd_outplane") or {}
        freqs_in = psd_in.get("frequencies")
        powers_in = psd_in.get("powers")
        freqs_out = psd_out.get("frequencies")
        powers_out = psd_out.get("powers")
        high_in = _is_high_freq(freqs_in, powers_in)
        high_out = _is_high_freq(freqs_out, powers_out)
        rms_in = ts_in.get("rms")
        rms_out = ts_out.get("rms")
        window_idx = sample.get("window_idx")

        _append_if_high_freq(inplane_rms, jobs_in, rms_in, freqs_in, powers_in, sample.get("inplane_file_path"), window_idx)
        _append_if_high_freq(outplane_rms, jobs_out, rms_out, freqs_out, powers_out, sample.get("outplane_file_path"), window_idx)

        if rms_in is not None and rms_out is not None and (high_in or high_out):
            paired_inplane_rms.append(float(rms_in))
            paired_outplane_rms.append(float(rms_out))

        group = _group_name(sample_idx, high_in, high_out, side_predictions)
        if group is not None:
            groups[group].append(sample_idx)

    _write_selection_snapshot(side_predictions, groups)
    return {
        "inplane_rms": np.asarray(inplane_rms, dtype=np.float64),
        "outplane_rms": np.asarray(outplane_rms, dtype=np.float64),
        "paired_inplane_rms": np.asarray(paired_inplane_rms, dtype=np.float64),
        "paired_outplane_rms": np.asarray(paired_outplane_rms, dtype=np.float64),
        "jobs_in": jobs_in,
        "jobs_out": jobs_out,
        "category_counts": {group: len(ids) for group, ids in groups.items()},
    }


def _slice_window(raw: np.ndarray, window_idx: int) -> np.ndarray | None:
    start = window_idx * _WINDOW_SIZE
    end = start + _WINDOW_SIZE
    if end > len(raw):
        return None
    return raw[start:end]


def compute_full_psd_cumsum(jobs: list[tuple[str, int]], side_label: str) -> list[np.ndarray]:
    by_path: dict[str, list[int]] = {}
    for file_path, window_idx in jobs:
        by_path.setdefault(file_path, []).append(window_idx)

    unpacker = UNPACK(init_path=False)
    curves: list[np.ndarray] = []
    total_files = len(by_path)
    print(f"  [{side_label}] 高频窗口：{len(jobs)}，唯一 VIC 文件：{total_files}")
    for file_i, (file_path, window_indices) in enumerate(by_path.items(), start=1):
        if not Path(file_path).exists():
            continue
        raw = np.asarray(unpacker.VIC_DATA_Unpack(str(file_path)), dtype=np.float64)
        for window_idx in window_indices:
            sig = _slice_window(raw, window_idx)
            if sig is None:
                continue
            curve = psd_ranked_energy_cumsum(sig, n_modes=Config.N_MODES, nfft=Config.NFFT)
            if curve is not None:
                curves.append(curve)
        if file_i % 200 == 0 or file_i == total_files:
            print(f"    [{side_label}] 已解包文件 {file_i}/{total_files}")
    return curves


def _apply_grid(ax) -> None:
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA, linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)


def _add_legend(ax) -> None:
    leg = ax.legend(fontsize=Config.LEGEND_FONT_SIZE, framealpha=0.9, prop=CN_FONT)
    for text in leg.get_texts():
        text.set_fontproperties(CN_FONT)


def _is_web_dashboard_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _aggregate(curves: list[np.ndarray], n_modes: int) -> dict:
    mat = np.full((len(curves), n_modes), np.nan)
    for i, curve in enumerate(curves):
        length = min(len(curve), n_modes)
        mat[i, :length] = curve[:length]
        if 0 < length < n_modes:
            mat[i, length:] = curve[length - 1]
    return {"mean": np.nanmean(mat, axis=0), "std": np.nanstd(mat, axis=0), "n": len(curves)}


def plot_category_ratio(counts: dict[str, int]) -> plt.Figure:
    raw_items = [
        ("仅面内", counts["inplane_only"], Config.INPLANE_COLOR),
        ("仅面外", counts["outplane_only"], Config.OUTPLANE_COLOR),
        ("面内外同时", counts["both"], Config.BOTH_COLOR),
    ]
    total = sum(count for _, count, _ in raw_items)
    non_zero_items = [(label, count, color) for label, count, color in raw_items if count > 0]
    if not non_zero_items:
        raise ValueError("高频类别样本数为 0，无法绘制饼图")
    values = np.asarray([count for _, count, _ in non_zero_items], dtype=np.float64)
    colors = [color for _, _, color in non_zero_items]
    total = float(total)

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    ax.pie(
        values,
        colors=colors,
        startangle=90,
        counterclock=False,
        autopct=lambda pct: f"{pct:.1f}%",
        pctdistance=0.68,
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
        textprops={"fontproperties": CN_FONT, "fontsize": Config.TICK_FONT_SIZE, "color": "black"},
    )
    legend_handles = [Patch(facecolor=color, edgecolor="white") for _, _, color in raw_items]
    legend_labels = [f"{label}  {count:,}（{count / total * 100:.1f}%）" for label, count, _ in raw_items]
    legend = fig.legend(
        legend_handles,
        legend_labels,
        loc="center right",
        bbox_to_anchor=(0.98, 0.5),
        ncol=1,
        frameon=True,
        fancybox=True,
        framealpha=0.96,
        edgecolor="#d0d0d0",
        handlelength=1.0,
        handletextpad=0.7,
        borderpad=0.5,
        prop=CN_FONT,
        fontsize=Config.LEGEND_FONT_SIZE - 3,
    )
    for text in legend.get_texts():
        text.set_fontproperties(CN_FONT)
    ax.set(aspect="equal")
    fig.subplots_adjust(left=0.03, right=0.66, bottom=0.05, top=0.96)
    return fig


def plot_rms_histogram(data: dict) -> plt.Figure:
    in_rms = data["inplane_rms"]
    out_rms = data["outplane_rms"]
    combined = np.concatenate([in_rms, out_rms])
    x_max = max(float(np.percentile(combined, Config.RMS_X_PERCENTILE)), 1e-6)
    bins = np.linspace(0, x_max, Config.N_BINS + 1)
    width = bins[1] - bins[0]
    centers = (bins[:-1] + bins[1:]) / 2
    counts_in, _ = np.histogram(in_rms[in_rms <= x_max], bins=bins)
    counts_out, _ = np.histogram(out_rms[out_rms <= x_max], bins=bins)

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    half = width * 0.46
    ax.bar(centers - half / 2, counts_in, width=half, color=Config.INPLANE_COLOR, alpha=Config.BAR_ALPHA, label=f"面内（n={len(in_rms)}）")
    ax.bar(centers + half / 2, counts_out, width=half, color=Config.OUTPLANE_COLOR, alpha=Config.BAR_ALPHA, label=f"面外（n={len(out_rms)}）")
    ax.set_xlim(0, x_max)
    ax.set_xlabel(r"RMS ($m/s^2$)", labelpad=10, fontproperties=CN_FONT, fontsize=Config.LABEL_FONT_SIZE)
    ax.set_ylabel("样本数（个）", labelpad=10, fontproperties=CN_FONT, fontsize=Config.LABEL_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=Config.TICK_FONT_SIZE)
    _add_legend(ax)
    _apply_grid(ax)
    fig.tight_layout()
    return fig


def plot_rms_scatter(data: dict) -> plt.Figure:
    in_rms = data["paired_inplane_rms"]
    out_rms = data["paired_outplane_rms"]
    combined = np.concatenate([in_rms, out_rms])
    xy_max = max(float(np.percentile(combined, Config.SCATTER_AXIS_PERCENTILE)) * Config.SCATTER_AXIS_PAD, 1e-6)

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    ax.scatter(out_rms, in_rms, s=Config.SCATTER_SIZE, color=Config.INPLANE_COLOR, alpha=Config.SCATTER_ALPHA, linewidths=0, label=f"高频样本（n={len(in_rms)}）")
    ax.set_xlim(0, xy_max)
    ax.set_ylim(0, xy_max)
    ax.set_xlabel(r"面外 RMS ($m/s^2$)", labelpad=10, fontproperties=CN_FONT, fontsize=Config.LABEL_FONT_SIZE)
    ax.set_ylabel(r"面内 RMS ($m/s^2$)", labelpad=10, fontproperties=CN_FONT, fontsize=Config.LABEL_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=Config.TICK_FONT_SIZE)
    _add_legend(ax)
    _apply_grid(ax)
    fig.tight_layout()
    return fig


def plot_energy_cumsum(data: dict) -> plt.Figure:
    stats_in = _aggregate(data["cumsum_in"], Config.N_MODES)
    stats_out = _aggregate(data["cumsum_out"], Config.N_MODES)
    x = np.arange(1, Config.N_MODES + 1)
    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    for stats, color, marker, label in [
        (stats_in, Config.INPLANE_COLOR, "o", "面内"),
        (stats_out, Config.OUTPLANE_COLOR, "s", "面外"),
    ]:
        mean = stats["mean"]
        std = stats["std"]
        ax.plot(x, mean, color=color, linewidth=Config.LINE_WIDTH, marker=marker, markersize=4, label=f'{label}（n={stats["n"]}）')
        ax.fill_between(x, np.clip(mean - std, 0, None), mean + std, color=color, alpha=Config.SHADE_ALPHA)
    ax.axhline(y=1.0, color="gray", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlim(0.5, Config.N_MODES + 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0, Config.N_MODES + 1, 5))
    ax.set_xlabel("主频阶序", labelpad=10, fontproperties=CN_FONT, fontsize=Config.LABEL_FONT_SIZE)
    ax.set_ylabel("累积能量占比", labelpad=10, fontproperties=CN_FONT, fontsize=Config.LABEL_FONT_SIZE)
    ax.tick_params(axis="both", which="major", labelsize=Config.TICK_FONT_SIZE)
    _add_legend(ax)
    _apply_grid(ax)
    fig.tight_layout()
    return fig


def push_figures(figures: list[tuple[plt.Figure, str]]) -> None:
    if not _is_web_dashboard_available(Config.WEB_DASHBOARD_PORT):
        print("  未检测到 VibDash 服务，跳过 WebUI 推送；如需预览请先运行：python -m src.visualize_tools.web_dashboard")
        return
    for slot, (fig, title) in enumerate(figures):
        web_push(fig, page=Config.WEB_PAGE, slot=slot, title=title, page_cols=2 if slot == 0 else None)
    print(f"[OK] 已推送到 WebUI：{Config.WEB_PAGE}")


def main() -> None:
    print("=" * 80)
    print(f"图4-22 高频主导 VIV 样本（主导模态 >= {Config.FREQ_THRESHOLD:g} Hz）")
    print("=" * 80)
    print("\n[步骤1] 加载并筛选 enriched 数据...")
    print(f"  数据目录：{Config.ENRICHED_STATS_DIR}")
    data = load_high_freq_data()
    print(f"[OK] 面内高频样本：{len(data['inplane_rms'])}，面外高频样本：{len(data['outplane_rms'])}")
    print(f"  面内 RMS median={float(np.median(data['inplane_rms'])):.4f}")
    print(f"  面外 RMS median={float(np.median(data['outplane_rms'])):.4f}")
    print(f"  配对 RMS 散点样本：{len(data['paired_inplane_rms'])}")

    print(f"\n[步骤2] 从原始窗口重算完整 PSD 累积能量（nfft={Config.NFFT}）...")
    data["cumsum_in"] = compute_full_psd_cumsum(data["jobs_in"], "面内")
    data["cumsum_out"] = compute_full_psd_cumsum(data["jobs_out"], "面外")
    print(f"[OK] 面内累积曲线：{len(data['cumsum_in'])}，面外累积曲线：{len(data['cumsum_out'])}")

    category_counts = data["category_counts"]
    total_categories = sum(category_counts.values())
    print(
        "\n[步骤3] 高频分向类别："
        f"仅面内={category_counts['inplane_only']}，"
        f"仅面外={category_counts['outplane_only']}，"
        f"同时={category_counts['both']}，"
        f"合计={total_categories}"
    )

    print("\n[步骤4] 绘制四张正方形图像...")
    fig_category = plot_category_ratio(category_counts)
    fig_rms = plot_rms_histogram(data)
    fig_scatter = plot_rms_scatter(data)
    fig_energy = plot_energy_cumsum(data)
    print("[OK] 已生成类别比例、RMS 分布、RMS 散点与累积能量分布")

    push_figures([
        (fig_category, "高频样本类别比例"),
        (fig_rms, "高频主导样本 RMS 分布"),
        (fig_scatter, "高频主导样本 RMS 散点"),
        (fig_energy, "高频主导样本累积能量分布"),
    ])
    plt.close(fig_category)
    plt.close(fig_rms)
    plt.close(fig_scatter)
    plt.close(fig_energy)


if __name__ == "__main__":
    main()
