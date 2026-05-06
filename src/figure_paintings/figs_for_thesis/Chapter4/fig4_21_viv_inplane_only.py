import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_processer.io_unpacker import UNPACK
from src.visualize_tools.web_dashboard import push as web_push
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT, FONT_SIZE, SQUARE_FIG_SIZE,
    VIV_VIB_COLOR, VIV_INPLANE_COLOR,
)

_chapter4_dir = str(Path(__file__).parent)
if _chapter4_dir not in sys.path:
    sys.path.insert(0, _chapter4_dir)
from _viv_pipeline import load_latest_result


# ==================== 常量配置 ====================
_VIV_CLASS = 1

class Config:
    FS = 50.0
    WINDOW_SIZE = 3000

    TRIM_START_SECOND = 0
    TRIM_END_SECOND = 20        # 展示前 20s，VIV 周期性更明显

    NUM_SAMPLES = 20
    RANDOM_SEED = 42

    FIG_SIZE      = SQUARE_FIG_SIZE
    WAVEFORM_COLOR = VIV_VIB_COLOR
    SCATTER_COLOR  = VIV_INPLANE_COLOR
    LINEWIDTH     = 1.0
    SCATTER_SIZE  = 10
    SCATTER_ALPHA = 0.35
    GRID_COLOR    = 'gray'
    GRID_ALPHA    = 0.4
    GRID_LINEWIDTH = 0.5
    GRID_LINESTYLE = '--'

    APPLY_DENOISE = False

    DL_RESULT_GLOB = project_root / "results" / "identification_result" / "res_cnn_full_dataset_*.json"


# ==================== 数据筛选 ====================
def get_inplane_only_samples(result: dict) -> list:
    pred_in  = {int(k): int(v) for k, v in result.get("predictions_inplane",  {}).items()}
    pred_out = {int(k): int(v) for k, v in result.get("predictions_outplane", {}).items()}
    metadata = result.get("sample_metadata", {})

    samples = []
    for idx, label_in in pred_in.items():
        if label_in != _VIV_CLASS:
            continue
        if pred_out.get(idx, 0) == _VIV_CLASS:
            continue                             # 同时振动，排除

        meta = metadata.get(str(idx))
        if meta is None:
            continue
        in_path  = meta.get("inplane_file_path")
        out_path = meta.get("outplane_file_path")
        if not in_path or not out_path:
            continue

        samples.append({
            "idx":                idx,
            "window_idx":         meta["window_idx"],
            "inplane_sensor_id":  meta.get("inplane_sensor_id",  ""),
            "outplane_sensor_id": meta.get("outplane_sensor_id", ""),
            "inplane_file_path":  in_path,
            "outplane_file_path": out_path,
            "timestamp":          meta.get("timestamp", []),
        })

    return samples


def random_sample(samples: list) -> list:
    n   = len(samples)
    k   = min(Config.NUM_SAMPLES, n)
    rng = np.random.default_rng(Config.RANDOM_SEED)
    chosen = sorted(rng.choice(n, size=k, replace=False).tolist())
    print(f"  随机抽取索引：{chosen}（共 {n} 个仅面内 VIV 样本中选 {k} 个，seed={Config.RANDOM_SEED}）")
    return [samples[i] for i in chosen]


# ==================== 数据加载 ====================
def _load_window(file_path: str, window_idx: int, unpacker: UNPACK) -> np.ndarray:
    raw   = np.array(unpacker.VIC_DATA_Unpack(file_path))
    start = window_idx * Config.WINDOW_SIZE
    end   = start + Config.WINDOW_SIZE
    if end > len(raw):
        raise ValueError(f"窗口越界：window_idx={window_idx}, data_len={len(raw)}")
    return raw[start:end]


def _trim(data: np.ndarray) -> np.ndarray:
    s = int(Config.TRIM_START_SECOND * Config.FS)
    e = int(Config.TRIM_END_SECOND   * Config.FS)
    return data[s:e]


# ==================== 绘图函数 ====================
def _grid(ax):
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA,
            linewidth=Config.GRID_LINEWIDTH, linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)


def _plot_timeseries(data: np.ndarray, sensor_id: str, direction: str,
                     timestamp: list) -> plt.Figure:
    data_plot   = _trim(data)
    time_axis   = np.arange(len(data_plot)) / Config.FS + Config.TRIM_START_SECOND
    time_str    = "-".join(str(t) for t in timestamp) if timestamp else ""

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    ax.plot(time_axis, data_plot, color=Config.WAVEFORM_COLOR, linewidth=Config.LINEWIDTH)
    ax.set_title(f"{sensor_id} @ {time_str}\n{direction}时程",
                 fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.set_xlabel('时间 (s)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel(r'加速度 ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)
    _grid(ax)
    fig.tight_layout()
    return fig


def _plot_trajectory(in_data: np.ndarray, out_data: np.ndarray,
                     sample: dict) -> plt.Figure:
    ts  = sample.get("timestamp", [])
    in_id  = sample.get("inplane_sensor_id",  "未知")
    out_id = sample.get("outplane_sensor_id", "未知")
    time_str = (f"{int(ts[0]):02d}月{int(ts[1]):02d}日  {int(ts[2]):02d}时"
                if len(ts) >= 3 else
                f"{int(ts[0]):02d}月{int(ts[1]):02d}日" if len(ts) >= 2 else "")
    title = f"面内: {in_id}  |  面外: {out_id}\n{time_str}"

    all_vals   = np.concatenate([in_data, out_data])
    g_min, g_max = float(all_vals.min()), float(all_vals.max())
    margin = (g_max - g_min) * 0.05

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    ax.scatter(out_data, in_data,
               s=Config.SCATTER_SIZE, color=Config.SCATTER_COLOR,
               alpha=Config.SCATTER_ALPHA, linewidths=0)
    ax.set_xlim(g_min - margin, g_max + margin)
    ax.set_ylim(g_min - margin, g_max + margin)
    ax.set_xlabel(r'面外加速度 ($m/s^2$)', labelpad=10,
                  fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel(r'面内加速度 ($m/s^2$)', labelpad=10,
                  fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title(title, fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)
    _grid(ax)
    fig.tight_layout()
    return fig


# ==================== 绘图与推送 ====================
def plot_and_push(samples: list, page_name: str):
    unpacker = UNPACK(init_path=False)
    total    = len(samples)
    print(f"  共 {total} 个样本，每样本 3 张图（面内时程 / 面外时程 / 加速度云图）")

    for i, sample in enumerate(samples):
        print(f"  [样本 {i + 1}/{total}] sensor={sample['inplane_sensor_id']}")

        in_data  = _load_window(sample["inplane_file_path"],  sample["window_idx"], unpacker)
        out_data = _load_window(sample["outplane_file_path"], sample["window_idx"], unpacker)

        fig_in   = _plot_timeseries(in_data,  sample["inplane_sensor_id"],  "面内", sample["timestamp"])
        fig_out  = _plot_timeseries(out_data, sample["outplane_sensor_id"], "面外", sample["timestamp"])
        fig_traj = _plot_trajectory(in_data, out_data, sample)

        base = i * 3
        web_push(fig_in,   page=page_name, slot=base,
                 title=f'样本{i + 1} 面内时程',
                 page_cols=3 if i == 0 else None)
        web_push(fig_out,  page=page_name, slot=base + 1,
                 title=f'样本{i + 1} 面外时程')
        web_push(fig_traj, page=page_name, slot=base + 2,
                 title=f'样本{i + 1} 加速度云图')

        plt.close('all')
        print(f"    ✓ 已推送")

    print(f"  ✓ 全部 {total} 组推送完成")


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("仅面内涡激共振样本特征（面内时程 / 面外时程 / 加速度云图）")
    print("=" * 80)

    print("\n[步骤1] 加载 DL 识别结果...")
    dl_result = load_latest_result(Config.DL_RESULT_GLOB)

    print("\n[步骤2] 筛选仅面内 VIV 样本...")
    inplane_only = get_inplane_only_samples(dl_result)
    print(f"✓ 仅面内 VIV 样本：{len(inplane_only)} 个")

    print("\n[步骤3] 随机抽取样本...")
    plot_samples = random_sample(inplane_only)

    print("\n[步骤4] 绘图并推送到 WebUI...")
    plot_and_push(plot_samples, "fig4_19b 仅面内VIV")

    print("\n" + "=" * 80)
    print("完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
