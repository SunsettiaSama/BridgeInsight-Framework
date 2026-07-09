import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_processer.io_unpacker import UNPACK
from src.data_processer.signals.wavelets import denoise
from src.visualize_tools.web_dashboard import push as web_push
from src.chapter4_characteristics._bootstrap import ensure_paths

ensure_paths()
from src.figure_paintings.figs_for_thesis.Chapter4 import data_config
from src.figure_paintings.figs_for_thesis.config import ENG_FONT, CN_FONT, FONT_SIZE


# ==================== 常量配置 ====================
class Config:
    FS = 50.0
    WINDOW_SIZE = 3000          # 60s @ 50Hz

    TRIM_START_SECOND = 0
    TRIM_END_SECOND = 20

    NUM_SAMPLES_TO_PLOT = 20
    RANDOM_SEED = 42            # 随机种子，None 表示不固定

    FIG_SIZE = (18, 11)
    N_ROWS = 4
    N_COLS = 5
    INPLANE_COLOR = '#7895C1'
    OUTPLANE_COLOR = '#E3625D'
    LINEWIDTH = 0.7
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.25
    GRID_LINEWIDTH = 0.5
    GRID_LINESTYLE = '--'

    WAVELET_TYPE = 'db4'
    WAVELET_LEVEL = 3
    THRESHOLD_TYPE = 'soft'
    THRESHOLD_METHOD = 'sqtwolog'

    NORMAL_VIB_CLASS_ID = 0     # 随机振动对应的类别编号


# ==================== 数据获取 ====================
def load_filtered_dl_result() -> dict:
    result_path = data_config.PROJECT_ROOT / data_config.CHAPTER4["predictions_enriched"]
    if not result_path.exists():
        raise FileNotFoundError(f"已剔除版本识别结果不存在：{result_path}")
    print(f"  加载识别结果（已剔除 201/202/301）：{result_path.name}")
    with open(result_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_normal_vib_samples(result: dict) -> list:
    predictions = {int(k): int(v) for k, v in result["predictions"].items()}
    sample_metadata = result.get("sample_metadata", {})

    normal_samples = []
    for idx, pred_label in predictions.items():
        if pred_label != Config.NORMAL_VIB_CLASS_ID:
            continue
        meta = sample_metadata.get(str(idx))
        if meta is None:
            continue
        inplane_path = meta.get("inplane_file_path")
        outplane_path = meta.get("outplane_file_path")
        if not inplane_path or not outplane_path:
            continue
        normal_samples.append({
            "idx": idx,
            "window_idx": meta["window_idx"],
            "inplane_sensor_id": meta.get("inplane_sensor_id", ""),
            "outplane_sensor_id": meta.get("outplane_sensor_id", ""),
            "inplane_file_path": inplane_path,
            "outplane_file_path": outplane_path,
            "timestamp": meta.get("timestamp", []),
        })

    return normal_samples


def random_sample(samples: list) -> list:
    n = len(samples)
    k = min(Config.NUM_SAMPLES_TO_PLOT, n)
    rng = np.random.default_rng(Config.RANDOM_SEED)
    chosen_indices = rng.choice(n, size=k, replace=False)
    chosen_indices_sorted = sorted(chosen_indices.tolist())
    print(f"  随机抽取索引：{chosen_indices_sorted}（共 {n} 个样本中选 {k} 个，seed={Config.RANDOM_SEED}）")
    return [samples[i] for i in chosen_indices_sorted]


# ==================== 数据预处理 ====================
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
    raw = unpacker.VIC_DATA_Unpack(file_path)
    raw = np.array(raw)
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


# ==================== 绘图函数 ====================
def plot_normal_vib_timeseries_grid(samples: list, unpacker: UNPACK) -> plt.Figure:
    fig, axes = plt.subplots(
        Config.N_ROWS,
        Config.N_COLS,
        figsize=Config.FIG_SIZE,
        sharex=True,
    )
    axes_flat = axes.ravel()

    for i, sample in enumerate(samples):
        ax = axes_flat[i]
        print(f"  [样本 {i + 1}] sensor={sample['inplane_sensor_id']} / {sample['outplane_sensor_id']}")

        inplane_data = _load_window(sample["inplane_file_path"], sample["window_idx"], unpacker)
        outplane_data = _load_window(sample["outplane_file_path"], sample["window_idx"], unpacker)
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

        time_str = "-".join(str(t) for t in sample["timestamp"]) if sample["timestamp"] else ""
        cable_id = sample["inplane_sensor_id"].replace("ST-VIC-", "").rsplit("-", 1)[0]
        ax.set_title(
            f"{i + 1}. {cable_id}  {time_str}  win={sample['window_idx']}",
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

    for ax in axes_flat[len(samples):]:
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
        f"窗口 {Config.WINDOW_SIZE / Config.FS:.0f} s；展示 {Config.TRIM_START_SECOND:g}-{Config.TRIM_END_SECOND:g} s；seed={Config.RANDOM_SEED}",
        ha="right",
        va="bottom",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE - 12,
        color="#404040",
    )
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.07, top=0.91, hspace=0.42, wspace=0.22)
    return fig


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("第三章 随机振动时域波形绘制（面内 & 面外）")
    print("=" * 80)

    print("\n[步骤1] 加载识别结果（已剔除版本）...")
    result = load_filtered_dl_result()

    print("\n[步骤2] 筛选随机振动（class 0）样本...")
    samples = get_normal_vib_samples(result)
    print(f"✓ 共筛选到 {len(samples)} 个随机振动样本")

    print("\n[步骤3] 随机抽取样本...")
    samples = random_sample(samples)

    print("\n[步骤4] 加载原始数据并绘制 20 面板总图...")
    unpacker = UNPACK(init_path=False)
    figure = plot_normal_vib_timeseries_grid(samples, unpacker)

    print("\n" + "=" * 80)
    print(f"✓ 总图：1 张  |  子图：{len(samples)} 个")
    print("=" * 80)

    page = "fig4_8 随机振动时程"
    web_push(
        figure,
        page=page,
        slot=0,
        title="随机振动时域波形 20 样本总览",
        page_cols=1,
    )
    plt.close(figure)
    print(f"✓ 已推送到 WebUI：{page}")


if __name__ == "__main__":
    main()
