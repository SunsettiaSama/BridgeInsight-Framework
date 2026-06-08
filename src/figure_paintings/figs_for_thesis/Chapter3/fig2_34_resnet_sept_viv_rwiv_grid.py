"""
ResCNN 2024年9月 VIV / RWIV 时程九宫格

流程：
1. 对 2024年9月推理池跑 ResCNN 全量识别（含面内/面外独立结果），缓存到
   results/chapter1/augment/1-fullly_recognize/
2. 按面内/面外识别标签随机各抽 9 个样本，绘制对应通道原始时程九宫格（图1、图2）
3. 从黄金标注集随机各抽 9 个 VIV / RWIV 样本，绘制九宫格（图3、图4）
4. 通过 web_dashboard.push() 推送到已运行的 VibDash（脚本内不启动 WebUI）
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.chapter1_identifier.augment._bootstrap import ensure_paths
from src.chapter1_identifier.augment.infer.dataset_loader import load_staycable_dataset
from src.data_processer.io_unpacker import UNPACK
from src.figure_paintings.figs_for_thesis.config import CN_FONT, FONT_SIZE
from src.visualize_tools.web_dashboard import push as web_push

ensure_paths()

from src.identifier.dl.identifier import DLVibrationIdentifier
from src.identifier.dl.runner import FullDatasetRunner


RECOGNIZE_CACHE_DIR = Path(project_root) / "results" / "chapter1" / "augment" / "1-fullly_recognize"
LATEST_RESULT_PATH = RECOGNIZE_CACHE_DIR / "latest.json"
DATASET_CONFIG = Path(project_root) / "config" / "datasets" / "total_staycable_vib_202409.yaml"
CHECKPOINT_PATH = (
    Path(project_root)
    / "results" / "training_result" / "deep_learning_module"
    / "res_cnn" / "checkpoints" / "ResCNN_20260402_111429" / "best_checkpoint.pth"
)
MODEL_CONFIG = Path(project_root) / "config" / "train" / "models" / "res_cnn.yaml"
GOLD_ANNOTATION_PATH = (
    Path(project_root) / "results" / "dataset_annotation" / "annotation_results.json"
)

# 设为 True 可强制重跑全量识别
FORCE_RERUN = False

WEBUI_PORT = 5678
WEBUI_PAGE = "fig2_34 ResCNN九月VIV/RWIV时程"


class Config:
    FS = 50.0
    WINDOW_SIZE = 3000

    TRIM_START_SECOND = 0
    TRIM_END_SECOND = 20

    NUM_SAMPLES = 9
    GRID_ROWS = 3
    GRID_COLS = 3
    RANDOM_SEED = 7
    GOLD_RANDOM_SEED = 17

    VIV_CLASS_ID = 1
    RWIV_CLASS_ID = 2

    FIG_SIZE = (18, 14)
    LINEWIDTH = 0.9
    GRID_COLOR = "gray"
    GRID_ALPHA = 0.35
    GRID_LINEWIDTH = 0.4
    GRID_LINESTYLE = "--"

    WAVEFORM_COLOR = "#333333"

    BATCH_SIZE = 256
    NUM_WORKERS = 4


def _load_result(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_sept2024_recognition() -> Path:
    if not DATASET_CONFIG.exists():
        raise FileNotFoundError(f"数据集配置不存在：{DATASET_CONFIG}")
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{CHECKPOINT_PATH}")

    RECOGNIZE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RECOGNIZE_CACHE_DIR / f"res_cnn_sept2024_{timestamp}.json"

    print(f"  加载 202409 数据集：{DATASET_CONFIG.name}")
    print("  （首次加载需读取元数据/索引缓存，约 1~5 分钟，并非死锁）")
    dataset = load_staycable_dataset(str(DATASET_CONFIG))
    print(f"  数据集就绪，样本数：{len(dataset)}")

    print(f"  加载 ResCNN：{CHECKPOINT_PATH.name}")
    identifier = DLVibrationIdentifier.from_checkpoint(
        checkpoint_path=str(CHECKPOINT_PATH),
        model_type="res_cnn",
        model_config_path=str(MODEL_CONFIG),
        num_classes=4,
    )

    runner = FullDatasetRunner(
        identifier=identifier,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )
    merged, pred_in, pred_out = runner.run(dataset)

    model_info = f"ResCNN | checkpoint={CHECKPOINT_PATH.parent.name} | dataset=202409"
    FullDatasetRunner.save_predictions(
        path=str(output_path),
        predictions=merged,
        dataset=dataset,
        model_info=model_info,
        inplane_predictions=pred_in,
        outplane_predictions=pred_out,
    )

    with open(LATEST_RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump({"result_path": str(output_path), "generated_at": timestamp}, f, ensure_ascii=False, indent=2)

    print(f"  识别结果已保存：{output_path}")
    print(f"  面内={len(pred_in)} | 面外={len(pred_out)} | 合并={len(merged)}")
    return output_path


def ensure_recognition_result() -> dict:
    if not FORCE_RERUN and LATEST_RESULT_PATH.exists():
        pointer = _load_result(LATEST_RESULT_PATH)
        cached = Path(pointer["result_path"])
        if cached.exists():
            print(f"  使用缓存识别结果：{cached.name}")
            return _load_result(cached)

    print("  缓存不存在或 FORCE_RERUN=True，开始 202409 全量识别...")
    result_path = run_sept2024_recognition()
    return _load_result(result_path)


def _plot_channel_for_class(in_pred: int, out_pred: int, class_id: int) -> str | None:
    """与 FullDatasetRunner._merge_predictions 一致：面内优先。"""
    if in_pred == class_id:
        return "inplane"
    if out_pred == class_id:
        return "outplane"
    return None


def filter_class_samples(result: dict, class_id: int) -> list:
    pred_in = {int(k): int(v) for k, v in result.get("predictions_inplane", {}).items()}
    pred_out = {int(k): int(v) for k, v in result.get("predictions_outplane", {}).items()}
    if not pred_in or not pred_out:
        raise ValueError(
            "识别结果缺少 predictions_inplane / predictions_outplane，"
            "请删除缓存后重新运行全量识别"
        )

    sample_metadata = result.get("sample_metadata", {})
    samples = []
    for idx_str, meta in sample_metadata.items():
        idx = int(idx_str)
        in_pred = pred_in.get(idx, 0)
        out_pred = pred_out.get(idx, 0)
        channel = _plot_channel_for_class(in_pred, out_pred, class_id)
        if channel is None:
            continue

        file_path = meta.get(f"{channel}_file_path")
        sensor_id = meta.get(f"{channel}_sensor_id", "")
        if not file_path or not Path(file_path).exists():
            continue

        samples.append({
            "idx": idx,
            "window_idx": meta["window_idx"],
            "channel": channel,
            "file_path": file_path,
            "sensor_id": sensor_id,
            "in_pred": in_pred,
            "out_pred": out_pred,
            "timestamp": meta.get("timestamp", []),
        })
    return samples


def random_sample(samples: list, class_name: str, seed: int | None = None) -> list:
    n = len(samples)
    k = min(Config.NUM_SAMPLES, n)
    if k == 0:
        raise ValueError(f"{class_name} 无可用样本（面内/面外独立识别后为空）")

    rng = np.random.default_rng(Config.RANDOM_SEED if seed is None else seed)
    chosen = sorted(rng.choice(n, size=k, replace=False).tolist())
    print(f"  {class_name}：从 {n} 个样本中随机抽取 {k} 个，索引={chosen}")
    return [samples[i] for i in chosen]


def _sensor_channel(sensor_id: str) -> str:
    if sensor_id.endswith("-02"):
        return "outplane"
    return "inplane"


def _timestamp_from_gold_record(record: dict) -> list:
    meta = record.get("metadata", {})
    month = meta.get("month")
    day = meta.get("day")
    hour = meta.get("hour")
    if month is not None and day is not None and hour is not None:
        return [int(month), int(day), int(hour)]
    return []


def _is_september_gold_record(record: dict) -> bool:
    meta = record.get("metadata", {})
    if meta.get("month") == "09":
        return True
    file_path = record.get("file_path", "")
    return "2024September" in file_path or "\\09\\" in file_path.replace("/", "\\")


def load_gold_annotations() -> list:
    if not GOLD_ANNOTATION_PATH.exists():
        raise FileNotFoundError(f"黄金标注文件不存在：{GOLD_ANNOTATION_PATH}")
    with open(GOLD_ANNOTATION_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"黄金标注格式异常：期望 list，实际 {type(data)}")
    print(f"  读取黄金标注：{len(data)} 条")
    return data


def filter_gold_class_samples(annotation_data: list, class_id: int) -> list:
    label = str(class_id)
    samples = []
    for i, record in enumerate(annotation_data):
        if record.get("annotation") != label:
            continue
        if not _is_september_gold_record(record):
            continue

        file_path = record.get("file_path") or record.get("metadata", {}).get("file_path")
        sensor_id = record.get("sensor_id", "")
        window_idx = record.get("window_index")
        if not file_path or window_idx is None or not Path(file_path).exists():
            continue

        channel = _sensor_channel(sensor_id)
        samples.append({
            "idx": i,
            "window_idx": int(window_idx),
            "channel": channel,
            "file_path": file_path,
            "sensor_id": sensor_id,
            "timestamp": _timestamp_from_gold_record(record),
        })
    return samples


def _load_window(sample: dict, unpacker: UNPACK) -> np.ndarray:
    raw = np.array(unpacker.VIC_DATA_Unpack(sample["file_path"]))
    start = sample["window_idx"] * Config.WINDOW_SIZE
    end = start + Config.WINDOW_SIZE
    if end > len(raw):
        raise ValueError(
            f"窗口越界：idx={sample['idx']} window={sample['window_idx']} "
            f"len={len(raw)} file={sample['file_path']}"
        )
    return raw[start:end]


def _format_timestamp(timestamp: list) -> str:
    if not timestamp:
        return ""
    if len(timestamp) >= 3:
        month, day, hour = timestamp[:3]
        return f"{month:02d}/{day:02d} {hour:02d}:00"
    return "-".join(str(t) for t in timestamp)


def _channel_label(channel: str) -> str:
    return "面内" if channel == "inplane" else "面外"


def plot_nine_grid(
    samples: list,
    unpacker: UNPACK,
    waveform_color: str,
    suptitle: str,
) -> plt.Figure:
    fig, axes = plt.subplots(
        Config.GRID_ROWS,
        Config.GRID_COLS,
        figsize=Config.FIG_SIZE,
        sharex=True,
        sharey=True,
    )

    trim_start = int(Config.TRIM_START_SECOND * Config.FS)
    trim_end = int(Config.TRIM_END_SECOND * Config.FS)

    for ax, sample in zip(axes.flat, samples):
        data = _load_window(sample, unpacker)
        data_plot = data[trim_start:trim_end]
        time_axis = np.arange(len(data_plot)) / Config.FS + Config.TRIM_START_SECOND

        ax.plot(time_axis, data_plot, color=waveform_color, linewidth=Config.LINEWIDTH)
        ax.set_title(
            f"{_channel_label(sample['channel'])} {sample['sensor_id']} @ "
            f"{_format_timestamp(sample['timestamp'])}",
            fontproperties=CN_FONT,
            fontsize=FONT_SIZE - 6,
            pad=4,
        )
        ax.grid(
            True,
            color=Config.GRID_COLOR,
            alpha=Config.GRID_ALPHA,
            linewidth=Config.GRID_LINEWIDTH,
            linestyle=Config.GRID_LINESTYLE,
        )
        ax.tick_params(axis="both", labelsize=FONT_SIZE - 8)

    for ax in axes[-1, :]:
        ax.set_xlabel("时间 (s)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 4)
    for ax in axes[:, 0]:
        ax.set_ylabel("加速度 (m/s²)", fontproperties=CN_FONT, fontsize=FONT_SIZE - 4)

    fig.suptitle(suptitle, fontproperties=CN_FONT, fontsize=FONT_SIZE, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def push_figures(figures: list[tuple[plt.Figure, str]]) -> None:
    for slot, (fig, title) in enumerate(figures):
        web_push(
            fig,
            page=WEBUI_PAGE,
            slot=slot,
            title=title,
            port=WEBUI_PORT,
            page_cols=1 if slot == 0 else None,
        )
        plt.close(fig)


def main():
    print("=" * 80)
    print("ResCNN 2024年9月 VIV / RWIV 时程九宫格")
    print("=" * 80)

    print("\n[步骤1] 202409 全量识别（或加载缓存）...")
    result = ensure_recognition_result()

    print("\n[步骤2] 按面内/面外独立识别结果筛选样本...")
    viv_samples = filter_class_samples(result, Config.VIV_CLASS_ID)
    rwiv_samples = filter_class_samples(result, Config.RWIV_CLASS_ID)
    viv_in = sum(1 for s in viv_samples if s["channel"] == "inplane")
    viv_out = len(viv_samples) - viv_in
    rwiv_in = sum(1 for s in rwiv_samples if s["channel"] == "inplane")
    rwiv_out = len(rwiv_samples) - rwiv_in
    print(f"  VIV  样本：{len(viv_samples)}（面内 {viv_in} / 面外 {viv_out}）")
    print(f"  RWIV 样本：{len(rwiv_samples)}（面内 {rwiv_in} / 面外 {rwiv_out}）")

    print("\n[步骤3] 随机抽样...")
    viv_plot = random_sample(viv_samples, "VIV")
    rwiv_plot = random_sample(rwiv_samples, "RWIV")

    print("\n[步骤4] 绘制 ResCNN 识别九宫格（原始时程，无去噪）...")
    unpacker = UNPACK(init_path=False)
    fig_viv = plot_nine_grid(
        viv_plot,
        unpacker,
        Config.WAVEFORM_COLOR,
        "ResCNN 识别涡激共振 (VIV) 时程 — 2024年9月",
    )
    fig_rwiv = plot_nine_grid(
        rwiv_plot,
        unpacker,
        Config.WAVEFORM_COLOR,
        "ResCNN 识别风雨振 (RWIV) 时程 — 2024年9月",
    )
    print("  ResCNN 九宫格 2 张完成")

    print("\n[步骤5] 黄金标注集抽样...")
    gold_data = load_gold_annotations()
    gold_viv_samples = filter_gold_class_samples(gold_data, Config.VIV_CLASS_ID)
    gold_rwiv_samples = filter_gold_class_samples(gold_data, Config.RWIV_CLASS_ID)
    print(f"  黄金 VIV  样本：{len(gold_viv_samples)}")
    print(f"  黄金 RWIV 样本：{len(gold_rwiv_samples)}")
    gold_viv_plot = random_sample(gold_viv_samples, "黄金 VIV", seed=Config.GOLD_RANDOM_SEED)
    gold_rwiv_plot = random_sample(gold_rwiv_samples, "黄金 RWIV", seed=Config.GOLD_RANDOM_SEED + 1)

    print("\n[步骤6] 绘制黄金标注九宫格...")
    fig_gold_viv = plot_nine_grid(
        gold_viv_plot,
        unpacker,
        Config.WAVEFORM_COLOR,
        "黄金标注涡激共振 (VIV) 时程 — 2024年9月",
    )
    fig_gold_rwiv = plot_nine_grid(
        gold_rwiv_plot,
        unpacker,
        Config.WAVEFORM_COLOR,
        "黄金标注风雨振 (RWIV) 时程 — 2024年9月",
    )
    print("  黄金标注九宫格 2 张完成")

    print(f"\n[步骤7] 推送到 VibDash（port={WEBUI_PORT}，不启动 WebUI）...")
    push_figures([
        (fig_viv, "ResCNN 识别 VIV 九宫格"),
        (fig_rwiv, "ResCNN 识别 RWIV 九宫格"),
        (fig_gold_viv, "黄金标注 VIV 九宫格"),
        (fig_gold_rwiv, "黄金标注 RWIV 九宫格"),
    ])

    print("\n" + "=" * 80)
    print(f"图像已 push 至 {WEBUI_PAGE}，请在已运行的 VibDash 中查看")
    print("=" * 80)


if __name__ == "__main__":
    main()
