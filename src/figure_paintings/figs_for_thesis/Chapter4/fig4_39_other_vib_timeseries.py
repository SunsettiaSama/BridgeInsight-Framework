"""图4-39：其他振动时域波形 20 样本总览。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chapter4_characteristics._bootstrap import ensure_paths
from src.data_processer.io_unpacker import UNPACK
from src.data_processer.signals.wavelets import denoise
from src.figure_paintings.figs_for_thesis.Chapter4 import data_config
from src.figure_paintings.figs_for_thesis.Chapter4._viv_pipeline import load_dl_result
from src.figure_paintings.figs_for_thesis.config import CN_FONT, ENG_FONT, FONT_SIZE
from src.visualize_tools.web_dashboard import push as web_push

ensure_paths()


class Config:
    FS = 50.0
    WINDOW_SIZE = 3000
    TRIM_START_SECOND = 0
    TRIM_END_SECOND = 20

    NUM_SAMPLES_TO_PLOT = 20
    RANDOM_SEED = 42

    FIG_SIZE = (18, 11)
    N_ROWS = 4
    N_COLS = 5
    INPLANE_COLOR = "#7895C1"
    OUTPLANE_COLOR = "#E3625D"
    LINEWIDTH = 0.7
    GRID_COLOR = "gray"
    GRID_ALPHA = 0.25
    GRID_LINEWIDTH = 0.5
    GRID_LINESTYLE = "--"

    OTHER_CLASS_ID = 3
    WAVELET_TYPE = "db4"
    WAVELET_LEVEL = 3
    THRESHOLD_TYPE = "soft"
    THRESHOLD_METHOD = "sqtwolog"

    ROUND9_INFERENCE_PATH = (
        project_root / "results" / "augment" / "final" / "rounds" / "round_09" / "inference.json"
    )
    ANNOTATION_PATH = project_root / "results" / "augment" / "annotations" / "merged_for_training.json"
    SAMPLE_COPY_PATH = (
        project_root
        / "results"
        / "chapter4_characteristics"
        / "figure_snapshots"
        / "other_round9_annotation_2023_pairs.json"
    )
    IDENTICAL_SAMPLE_COPY_PATH = (
        project_root
        / "results"
        / "chapter4_characteristics"
        / "figure_snapshots"
        / "other_round9_annotation_2023_near_identical_pairs.json"
    )
    MATCH_NRMSE_MAX = 0.20
    MATCH_CORR_MIN = 0.98
    WEB_PAGE = "fig4_39 其他振动时程"


def _sibling_path(file_path: str) -> Path | None:
    path = Path(file_path)
    name = path.name
    if "-01_" in name:
        return path.with_name(name.replace("-01_", "-02_", 1))
    if "-02_" in name:
        return path.with_name(name.replace("-02_", "-01_", 1))
    return None


def _parse_timestamp(time_str: str, metadata: dict) -> list:
    month = metadata.get("month")
    day = metadata.get("day")
    hour = metadata.get("hour")
    if month is not None and day is not None and hour is not None:
        return [int(month), int(day), int(hour)]
    if isinstance(time_str, str) and "/" in time_str:
        date_part, _, clock_part = time_str.partition(" ")
        m_s, _, d_s = date_part.partition("/")
        h_s = clock_part.split(":")[0] if clock_part else "0"
        return [int(m_s), int(d_s), int(h_s)]
    return []


def _is_excluded_sensor(sensor_id: str) -> bool:
    return sensor_id in data_config.EXCLUDED_SENSOR_IDS


def _is_excluded_pair(in_id: str, out_id: str) -> bool:
    return _is_excluded_sensor(in_id) or _is_excluded_sensor(out_id)


def _sample_dedupe_key(sample: dict) -> tuple[str, str, int]:
    return (
        os.path.normcase(str(sample["inplane_file_path"])),
        os.path.normcase(str(sample["outplane_file_path"])),
        int(sample["window_idx"]),
    )


def _annotation_entry_to_pair(entry: dict) -> dict | None:
    file_path = entry.get("file_path")
    window_idx = entry.get("window_index")
    sensor_id = entry.get("sensor_id", "")
    if file_path is None or window_idx is None or not sensor_id:
        return None
    if _is_excluded_sensor(sensor_id):
        return None

    sibling = _sibling_path(file_path)
    if sibling is None or not sibling.exists():
        return None

    if sensor_id.endswith("-01"):
        in_id, out_id = sensor_id, sensor_id[:-3] + "-02"
        in_path, out_path = str(Path(file_path)), str(sibling)
    elif sensor_id.endswith("-02"):
        in_id, out_id = sensor_id[:-3] + "-01", sensor_id
        in_path, out_path = str(sibling), str(Path(file_path))
    else:
        return None

    if _is_excluded_pair(in_id, out_id):
        return None

    meta = entry.get("metadata") or {}
    return {
        "idx": f"annotation_{Path(in_path).stem}_{int(window_idx)}",
        "window_idx": int(window_idx),
        "inplane_sensor_id": in_id,
        "outplane_sensor_id": out_id,
        "inplane_file_path": in_path,
        "outplane_file_path": out_path,
        "timestamp": _parse_timestamp(entry.get("time", ""), meta),
        "source": "annotation",
        "time_str": entry.get("time", ""),
    }


def load_annotation_other_samples() -> tuple[list[dict], dict[tuple[str, str, int], int]]:
    if not Config.ANNOTATION_PATH.exists():
        raise FileNotFoundError(f"找不到标注结果：{Config.ANNOTATION_PATH}")
    with open(Config.ANNOTATION_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)

    label_by_key: dict[tuple[str, str, int], int] = {}
    other_samples: list[dict] = []
    for entry in entries:
        pair = _annotation_entry_to_pair(entry)
        if pair is None:
            continue
        label = int(entry.get("annotation", -1))
        key = _sample_dedupe_key(pair)
        label_by_key[key] = label
        if label == Config.OTHER_CLASS_ID:
            other_samples.append(pair)
    print(f"  标注结果其他振动配对样本：{len(other_samples)} 个")
    return other_samples, label_by_key


def _record_to_pair(record: dict, source: str) -> dict | None:
    in_path = record.get("inplane_file_path")
    out_path = record.get("outplane_file_path")
    window_idx = record.get("window_index", record.get("window_idx"))
    in_id = record.get("inplane_sensor_id", "")
    out_id = record.get("outplane_sensor_id", "")
    if not in_path or not out_path or window_idx is None:
        return None
    if _is_excluded_pair(str(in_id), str(out_id)):
        return None
    return {
        "idx": f"{source}_{record.get('sample_idx', '')}_{int(window_idx)}",
        "window_idx": int(window_idx),
        "inplane_sensor_id": str(in_id),
        "outplane_sensor_id": str(out_id),
        "inplane_file_path": str(in_path),
        "outplane_file_path": str(out_path),
        "timestamp": record.get("timestamp", []),
        "source": source,
        "time_str": "",
    }


def _annotation_allows_sample(
    sample: dict,
    annotation_labels: dict[tuple[str, str, int], int],
) -> bool:
    label = annotation_labels.get(_sample_dedupe_key(sample))
    return label is None or label == Config.OTHER_CLASS_ID


def load_round9_other_samples(annotation_labels: dict[tuple[str, str, int], int]) -> list[dict]:
    if not Config.ROUND9_INFERENCE_PATH.exists():
        raise FileNotFoundError(f"找不到 round9 全量识别结果：{Config.ROUND9_INFERENCE_PATH}")
    with open(Config.ROUND9_INFERENCE_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    samples: list[dict] = []
    for record in payload.get("records", []):
        if int(record.get("prediction", -1)) != Config.OTHER_CLASS_ID:
            continue
        sample = _record_to_pair(record, "round9_inference")
        if sample is None or not _annotation_allows_sample(sample, annotation_labels):
            continue
        samples.append(sample)
    print(f"  round9 全量识别其他振动配对样本：{len(samples)} 个")
    return samples


def load_2023_other_samples(annotation_labels: dict[tuple[str, str, int], int]) -> list[dict]:
    result = load_dl_result()
    predictions = {int(k): int(v) for k, v in result["predictions"].items()}
    sample_metadata = result.get("sample_metadata", {})

    samples: list[dict] = []
    for idx, pred_label in predictions.items():
        if pred_label != Config.OTHER_CLASS_ID:
            continue
        meta = sample_metadata.get(str(idx))
        if meta is None:
            continue
        sample = _record_to_pair(
            {
                "sample_idx": idx,
                "prediction": pred_label,
                "inplane_file_path": meta.get("inplane_file_path"),
                "outplane_file_path": meta.get("outplane_file_path"),
                "window_idx": meta.get("window_idx"),
                "inplane_sensor_id": meta.get("inplane_sensor_id", ""),
                "outplane_sensor_id": meta.get("outplane_sensor_id", ""),
                "timestamp": meta.get("timestamp", []),
            },
            "dl_2023",
        )
        if sample is None or not _annotation_allows_sample(sample, annotation_labels):
            continue
        samples.append(sample)
    print(f"  2023 全年识别其他振动配对样本：{len(samples)} 个")
    return samples


def merge_sample_sources(source_samples: list[list[dict]]) -> list[dict]:
    merged: dict[tuple[str, str, int], dict] = {}
    for samples in source_samples:
        for sample in samples:
            key = _sample_dedupe_key(sample)
            if key in merged:
                prev = merged[key]
                src_a = str(prev.get("source", ""))
                src_b = str(sample.get("source", ""))
                if src_b and src_b not in src_a:
                    prev["source"] = f"{src_a}+{src_b}" if src_a else src_b
                continue
            merged[key] = dict(sample)
    samples = list(merged.values())
    print(f"  合并去重后其他振动配对样本：{len(samples)} 个")
    return samples


def load_other_samples_for_figure(refresh_sample_copy: bool) -> list[dict]:
    if Config.SAMPLE_COPY_PATH.exists() and not refresh_sample_copy:
        with open(Config.SAMPLE_COPY_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        samples = payload.get("samples", [])
        print(f"  读取样本池副本：{Config.SAMPLE_COPY_PATH}  n={len(samples)}")
        return samples

    annotation_samples, annotation_labels = load_annotation_other_samples()
    round9_samples = load_round9_other_samples(annotation_labels)
    dl_2023_samples = load_2023_other_samples(annotation_labels)
    samples = merge_sample_sources([annotation_samples, round9_samples, dl_2023_samples])

    payload = {
        "version": "other_round9_annotation_2023_pairs_v1",
        "round9_inference": str(Config.ROUND9_INFERENCE_PATH),
        "annotation_path": str(Config.ANNOTATION_PATH),
        "n_samples": len(samples),
        "samples": samples,
    }
    Config.SAMPLE_COPY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(Config.SAMPLE_COPY_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  写出样本池副本：{Config.SAMPLE_COPY_PATH}")
    return samples


def _source_bucket(sample: dict) -> str:
    source = str(sample.get("source", ""))
    if "annotation" in source:
        return "annotation"
    if "round9_inference" in source:
        return "round9"
    if "dl_2023" in source:
        return "dl_2023"
    return "other"


def _choose_samples(pool: list[dict], k: int, rng: np.random.Generator) -> list[dict]:
    if k <= 0:
        return []
    chosen_indices = rng.choice(len(pool), size=k, replace=False)
    return [pool[int(i)] for i in chosen_indices.tolist()]


def _signal_match_metrics(in_win: np.ndarray, out_win: np.ndarray) -> tuple[float, float]:
    in_plot = _prepare_plot_window(in_win)
    out_plot = _prepare_plot_window(out_win)
    diff = in_plot - out_plot
    rmse = float(np.sqrt(np.mean(diff * diff)))
    scale = float(
        max(
            np.ptp(in_plot),
            np.ptp(out_plot),
            np.max(np.abs(in_plot)),
            np.max(np.abs(out_plot)),
            1e-12,
        )
    )
    nrmse = rmse / scale
    if float(np.std(in_plot)) <= 1e-12 or float(np.std(out_plot)) <= 1e-12:
        corr = 1.0 if nrmse <= Config.MATCH_NRMSE_MAX else 0.0
    else:
        corr = float(np.corrcoef(in_plot, out_plot)[0, 1])
    return nrmse, corr


def _signals_are_nearly_identical(nrmse: float, corr: float) -> bool:
    return nrmse <= Config.MATCH_NRMSE_MAX or (
        corr >= Config.MATCH_CORR_MIN and nrmse <= 0.35
    )


def filter_identical_signal_samples(samples: list[dict], refresh_cache: bool) -> list[dict]:
    if Config.IDENTICAL_SAMPLE_COPY_PATH.exists() and not refresh_cache:
        with open(Config.IDENTICAL_SAMPLE_COPY_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
        identical = payload.get("samples", [])
        print(
            f"  读取面内外近似一致样本缓存：{Config.IDENTICAL_SAMPLE_COPY_PATH}  "
            f"n={len(identical)}"
        )
        return identical

    print(f"  扫描面内外近似一致样本：候选 {len(samples)} 个")
    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for sample in samples:
        by_pair[(str(sample["inplane_file_path"]), str(sample["outplane_file_path"]))].append(sample)

    unpacker = UNPACK(init_path=False)
    identical: list[dict] = []
    n_pairs = len(by_pair)
    for pair_i, ((in_path, out_path), pair_samples) in enumerate(by_pair.items(), start=1):
        if not Path(in_path).exists() or not Path(out_path).exists():
            continue
        raw_in = np.asarray(unpacker.VIC_DATA_Unpack(in_path), dtype=np.float64)
        raw_out = np.asarray(unpacker.VIC_DATA_Unpack(out_path), dtype=np.float64)
        for sample in pair_samples:
            start = int(sample["window_idx"]) * Config.WINDOW_SIZE
            end = start + Config.WINDOW_SIZE
            if end > len(raw_in) or end > len(raw_out):
                continue
            nrmse, corr = _signal_match_metrics(raw_in[start:end], raw_out[start:end])
            if _signals_are_nearly_identical(nrmse, corr):
                item = dict(sample)
                item["near_identical_in_out"] = True
                item["match_nrmse"] = float(nrmse)
                item["match_corr"] = float(corr)
                identical.append(item)
        if pair_i % 50 == 0 or pair_i == n_pairs:
            print(f"    已扫描文件对 {pair_i}/{n_pairs}，近似一致样本 {len(identical)}")

    payload = {
        "version": "other_near_identical_in_out_pairs_v1",
        "criterion": (
            "displayed denoised 0-20s windows are near-identical: "
            f"nRMSE<={Config.MATCH_NRMSE_MAX} or "
            f"corr>={Config.MATCH_CORR_MIN} with nRMSE<=0.35"
        ),
        "n_candidates": len(samples),
        "n_samples": len(identical),
        "samples": identical,
    }
    Config.IDENTICAL_SAMPLE_COPY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(Config.IDENTICAL_SAMPLE_COPY_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  写出面内外近似一致样本缓存：{Config.IDENTICAL_SAMPLE_COPY_PATH}  n={len(identical)}")
    return identical


def select_most_similar_samples(samples: list[dict]) -> list[dict]:
    if not samples:
        raise ValueError("无面内外近似一致样本可供抽取")
    ranked = sorted(
        samples,
        key=lambda sample: (
            float(sample.get("match_nrmse", float("inf"))),
            -float(sample.get("match_corr", 0.0)),
        ),
    )
    selected = ranked[: Config.NUM_SAMPLES_TO_PLOT]
    print(
        f"  取相似度最高 {len(selected)} 个："
        f"nRMSE {float(selected[0].get('match_nrmse', 0.0)):.4f}"
        f"–{float(selected[-1].get('match_nrmse', 0.0)):.4f}"
    )
    return selected


def filter_by_cable(samples: list[dict], cable_id: str) -> list[dict]:
    needle = cable_id.strip()
    if not needle:
        return samples
    filtered = [
        sample
        for sample in samples
        if needle in str(sample.get("inplane_sensor_id", ""))
        or needle in str(sample.get("outplane_sensor_id", ""))
    ]
    print(f"  测点过滤 {needle}：{len(filtered)}/{len(samples)} 个")
    return filtered


def random_sample(samples: list[dict]) -> list[dict]:
    n = len(samples)
    if n == 0:
        raise ValueError("无其他振动样本可供抽取")
    k = min(Config.NUM_SAMPLES_TO_PLOT, n)
    rng = np.random.default_rng(Config.RANDOM_SEED)

    groups: dict[str, list[dict]] = {"annotation": [], "round9": [], "dl_2023": [], "other": []}
    for sample in samples:
        groups[_source_bucket(sample)].append(sample)
    active = [name for name in ("annotation", "round9", "dl_2023", "other") if groups[name]]

    quota = {name: k // len(active) for name in active}
    for name in active[: k % len(active)]:
        quota[name] += 1

    remaining = 0
    for name in active:
        if quota[name] > len(groups[name]):
            remaining += quota[name] - len(groups[name])
            quota[name] = len(groups[name])
    for name in active:
        if remaining <= 0:
            break
        capacity = len(groups[name]) - quota[name]
        add = min(capacity, remaining)
        quota[name] += add
        remaining -= add

    chosen: list[dict] = []
    for name in active:
        chosen.extend(_choose_samples(groups[name], quota[name], rng))
    rng.shuffle(chosen)
    print(
        "  分层抽样："
        + "  ".join(f"{name}={quota[name]}/{len(groups[name])}" for name in active)
        + f"  seed={Config.RANDOM_SEED}"
    )
    return chosen


def _wavelet_denoise(data: np.ndarray) -> np.ndarray:
    denoised, _ = denoise(
        signal=data,
        wavelet=Config.WAVELET_TYPE,
        level=Config.WAVELET_LEVEL,
        threshold_type=Config.THRESHOLD_TYPE,
        threshold_method=Config.THRESHOLD_METHOD,
    )
    return denoised


def _load_window(file_path: str, window_idx: int, unpacker: UNPACK) -> np.ndarray:
    raw = np.asarray(unpacker.VIC_DATA_Unpack(file_path), dtype=np.float64)
    start = window_idx * Config.WINDOW_SIZE
    end = start + Config.WINDOW_SIZE
    if end > len(raw):
        raise ValueError(f"窗口越界：window_idx={window_idx}, data_len={len(raw)}")
    return raw[start:end]


def _prepare_plot_window(data: np.ndarray) -> np.ndarray:
    denoised = _wavelet_denoise(data)
    trim_start = int(Config.TRIM_START_SECOND * Config.FS)
    trim_end = int(Config.TRIM_END_SECOND * Config.FS)
    return denoised[trim_start:trim_end]


def _format_title(sample: dict, sample_idx: int) -> str:
    timestamp = sample.get("timestamp", [])
    time_str = "-".join(str(t) for t in timestamp) if timestamp else ""
    sensor_id = sample.get("inplane_sensor_id", "")
    cable_id = sensor_id.replace("ST-VIC-", "").rsplit("-", 1)[0]
    peak_amp = sample.get("peak_amp")
    peak_text = f"  |A|max={float(peak_amp):.2f}" if peak_amp is not None else ""
    match_text = ""
    if sample.get("match_nrmse") is not None and sample.get("match_corr") is not None:
        match_text = (
            f"  e={float(sample['match_nrmse']):.2f}"
            f" r={float(sample['match_corr']):.2f}"
        )
    return (
        f"{sample_idx + 1}. {cable_id}  {time_str}  "
        f"win={sample['window_idx']}{peak_text}{match_text}"
    )


def plot_other_vib_timeseries_grid(samples: list[dict], unpacker: UNPACK) -> plt.Figure:
    fig, axes = plt.subplots(
        Config.N_ROWS,
        Config.N_COLS,
        figsize=Config.FIG_SIZE,
        sharex=True,
    )
    axes_flat = axes.ravel()

    for i, sample in enumerate(samples):
        ax = axes_flat[i]
        print(
            f"  [样本 {i + 1}] sensor="
            f"{sample['inplane_sensor_id']} / {sample['outplane_sensor_id']} "
            f"source={sample.get('source', '?')}"
        )

        inplane_data = _load_window(
            sample["inplane_file_path"], sample["window_idx"], unpacker
        )
        outplane_data = _load_window(
            sample["outplane_file_path"], sample["window_idx"], unpacker
        )
        inplane_plot = _prepare_plot_window(inplane_data)
        outplane_plot = _prepare_plot_window(outplane_data)
        time_axis = np.arange(len(inplane_plot)) / Config.FS + Config.TRIM_START_SECOND

        ax.plot(
            time_axis,
            inplane_plot,
            color=Config.INPLANE_COLOR,
            linewidth=Config.LINEWIDTH,
            label="面内" if i == 0 else None,
        )
        ax.plot(
            time_axis,
            outplane_plot,
            color=Config.OUTPLANE_COLOR,
            linewidth=Config.LINEWIDTH,
            alpha=0.85,
            label="面外" if i == 0 else None,
        )

        ax.set_title(
            _format_title(sample, i),
            fontproperties=ENG_FONT,
            fontsize=FONT_SIZE - 12,
            pad=3,
        )
        ax.grid(
            True,
            color=Config.GRID_COLOR,
            alpha=Config.GRID_ALPHA,
            linewidth=Config.GRID_LINEWIDTH,
            linestyle=Config.GRID_LINESTYLE,
        )
        ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 12)

    for ax in axes_flat[len(samples) :]:
        ax.set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("时间 (s)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 8)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"加速度 ($m/s^2$)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 8)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        prop=CN_FONT,
        fontsize=FONT_SIZE - 8,
    )
    fig.text(
        0.99,
        0.01,
        f"窗口 {Config.WINDOW_SIZE / Config.FS:.0f} s；"
        f"展示 {Config.TRIM_START_SECOND:g}-{Config.TRIM_END_SECOND:g} s；"
        f"seed={Config.RANDOM_SEED}",
        ha="right",
        va="bottom",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 12,
        color="#404040",
    )
    fig.subplots_adjust(
        left=0.06,
        right=0.985,
        bottom=0.07,
        top=0.91,
        hspace=0.42,
        wspace=0.22,
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="图4-39 其他振动时程")
    parser.add_argument("--refresh-sample-copy", action="store_true")
    parser.add_argument("--refresh-identical-cache", action="store_true")
    parser.add_argument("--only-cable", type=str, default=None)
    parser.add_argument("--slot", type=int, default=None)
    args = parser.parse_args()

    print("=" * 80)
    print("图4-39 其他振动时域波形 20 样本总览（面内 & 面外）")
    print(f"  round9={Config.ROUND9_INFERENCE_PATH}")
    print(f"  annotation={Config.ANNOTATION_PATH}")
    print("=" * 80)

    print("\n[步骤1] 加载其他振动样本池...")
    all_samples = load_other_samples_for_figure(args.refresh_sample_copy)
    print(f"其他振动配对样本：{len(all_samples)} 个")

    print("\n[步骤2] 筛选面内外近似一致样本...")
    identical_samples = filter_identical_signal_samples(
        all_samples,
        refresh_cache=args.refresh_identical_cache,
    )
    print(f"面内外近似一致样本：{len(identical_samples)} 个")
    if args.only_cable:
        identical_samples = filter_by_cable(identical_samples, args.only_cable)

    print("\n[步骤3] 选取最相似样本...")
    samples = select_most_similar_samples(identical_samples)

    print(f"\n[步骤4] 加载数据并绘制时程总图（{len(samples)} 个样本）...")
    unpacker = UNPACK(init_path=False)
    figure = plot_other_vib_timeseries_grid(samples, unpacker)
    slot = args.slot if args.slot is not None else 0
    title = "其他振动 面内外近似一致时程 20 样本"
    if args.only_cable:
        title = f"其他振动 {args.only_cable} 面内外近似一致时程"

    web_push(
        figure,
        page=Config.WEB_PAGE,
        slot=slot,
        title=title,
        page_cols=1,
    )
    plt.close(figure)
    print(f"已推送到 WebUI：{Config.WEB_PAGE} / slot={slot}")


if __name__ == "__main__":
    main()
