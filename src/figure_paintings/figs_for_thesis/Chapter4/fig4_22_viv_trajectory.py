import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_processer.io_unpacker import UNPACK
from src.data_processer.signals.wavelets import denoise
from src.identifier.deeplearning_methods import FullDatasetRunner
from src.visualize_tools.web_dashboard import push as web_push
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT, FONT_SIZE, SQUARE_FIG_SIZE,
    VIV_VIB_COLOR, VIV_INPLANE_COLOR, VIV_OUTPLANE_COLOR,
)

_chapter4_dir = str(Path(__file__).parent)
if _chapter4_dir not in sys.path:
    sys.path.insert(0, _chapter4_dir)
from _viv_pipeline import load_latest_result, get_viv_samples as _pipeline_get_viv_samples


# ==================== 常量配置 ====================
class Config:
    FS = 50.0
    WINDOW_SIZE = 3000

    WAVELET_TYPE = 'db4'
    WAVELET_LEVEL = 3
    THRESHOLD_TYPE = 'soft'
    THRESHOLD_METHOD = 'sqtwolog'
    APPLY_DENOISE = False   # True: 先去噪再绘图；False: 直接用原始信号

    NUM_SAMPLES_TO_PLOT = 40
    RANDOM_SEED = 7

    FIG_SIZE = SQUARE_FIG_SIZE
    SCATTER_SIZE = 10
    SCATTER_ALPHA = 0.35
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = '--'
    LINEWIDTH = 1.0

    WAVEFORM_COLOR = VIV_VIB_COLOR

    NFFT = 2048
    FREQ_LIMIT = 25.0

    SAMPLE_COLORS: list = [VIV_INPLANE_COLOR, VIV_OUTPLANE_COLOR]

    DL_RESULT_GLOB   = project_root / "results" / "identification_result"        / "res_cnn_full_dataset_*.json"
    MECC_RESULT_GLOB = project_root / "results" / "identification_result_mecc_viv" / "mecc_viv_only_*.json"


# ==================== 数据加载 ====================
def _load_window(file_path: str, window_idx: int, unpacker: UNPACK) -> np.ndarray:
    raw = unpacker.VIC_DATA_Unpack(file_path)
    raw = np.array(raw)
    start = window_idx * Config.WINDOW_SIZE
    end   = start + Config.WINDOW_SIZE
    if end > len(raw):
        raise ValueError(f"窗口越界：window_idx={window_idx}, data_len={len(raw)}")
    return raw[start:end]


def _wavelet_denoise(data: np.ndarray) -> np.ndarray:
    denoised, _ = denoise(
        signal=data,
        wavelet=Config.WAVELET_TYPE,
        level=Config.WAVELET_LEVEL,
        threshold_type=Config.THRESHOLD_TYPE,
        threshold_method=Config.THRESHOLD_METHOD,
    )
    return denoised


def random_sample(samples: list) -> list:
    n = len(samples)
    k = min(Config.NUM_SAMPLES_TO_PLOT, n)
    rng = np.random.default_rng(Config.RANDOM_SEED)
    chosen = sorted(rng.choice(n, size=k, replace=False).tolist())
    print(f"  随机抽取索引：{chosen}（共 {n} 个样本中选 {k} 个，seed={Config.RANDOM_SEED}）")
    return [samples[i] for i in chosen]


def load_sample_pair(sample: dict, unpacker: UNPACK):
    in_raw  = _load_window(sample["inplane_file_path"],  sample["window_idx"], unpacker)
    out_raw = _load_window(sample["outplane_file_path"], sample["window_idx"], unpacker)
    return (
        _wavelet_denoise(in_raw)  if Config.APPLY_DENOISE else in_raw,
        _wavelet_denoise(out_raw) if Config.APPLY_DENOISE else out_raw,
    )


# ==================== 绘图函数 ====================
def _apply_grid(ax):
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA,
            linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)


def _make_timeseries_fig(data: np.ndarray, direction: str, title_prefix: str) -> plt.Figure:
    t = np.arange(len(data)) / Config.FS
    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    ax.plot(t, data, color=Config.WAVEFORM_COLOR, linewidth=Config.LINEWIDTH)
    ax.set_title(f"{title_prefix}\n{direction}时程", fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.set_xlabel('时间 (s)', fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel(r'加速度 ($m/s^2$)', fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.tick_params(labelsize=FONT_SIZE - 2)
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA, linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def _make_spectrum_fig(data: np.ndarray, direction: str, title_prefix: str) -> plt.Figure:
    f, psd = signal.welch(
        data,
        fs=Config.FS,
        nperseg=int(Config.NFFT / 2),
        noverlap=int(Config.NFFT / 4),
        nfft=Config.NFFT,
    )
    mask = f <= Config.FREQ_LIMIT
    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    ax.plot(f[mask], psd[mask], color=Config.WAVEFORM_COLOR, linewidth=Config.LINEWIDTH)
    ax.set_title(f"{title_prefix}\n{direction}频谱", fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.set_xlabel('频率 (Hz)', fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel(r'PSD $(m/s^2)^2$/Hz', fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_xlim(0, Config.FREQ_LIMIT)
    ax.tick_params(labelsize=FONT_SIZE - 2)
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA, linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def _plot_aux_timeseries(in_data: np.ndarray, out_data: np.ndarray, title_prefix: str = "") -> tuple:
    fig_in  = _make_timeseries_fig(in_data,  "面内", title_prefix)
    fig_out = _make_timeseries_fig(out_data, "面外", title_prefix)
    return fig_in, fig_out


def _plot_aux_spectra(in_data: np.ndarray, out_data: np.ndarray, title_prefix: str = "") -> tuple:
    fig_in  = _make_spectrum_fig(in_data,  "面内", title_prefix)
    fig_out = _make_spectrum_fig(out_data, "面外", title_prefix)
    return fig_in, fig_out


def _plot_single_trajectory(sample: dict, color, sample_idx: int, total: int, unpacker: UNPACK) -> tuple:
    in_data, out_data = load_sample_pair(sample, unpacker)

    ts = sample.get("timestamp", [])
    in_id  = sample.get("inplane_sensor_id",  "未知")
    out_id = sample.get("outplane_sensor_id", "未知")
    sensor_line = f"面内: {in_id}  |  面外: {out_id}"
    if ts and len(ts) >= 3:
        time_line = f"{int(ts[0]):02d}月{int(ts[1]):02d}日  {int(ts[2]):02d}时"
    elif ts and len(ts) >= 2:
        time_line = f"{int(ts[0]):02d}月{int(ts[1]):02d}日"
    else:
        time_line = f"样本{sample_idx + 1}"
    title = f"{sensor_line}\n{time_line}"

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)
    ax.scatter(
        out_data, in_data,
        s=Config.SCATTER_SIZE,
        color=color,
        alpha=Config.SCATTER_ALPHA,
        linewidths=0,
    )

    all_vals = np.concatenate([in_data, out_data])
    global_min, global_max = float(all_vals.min()), float(all_vals.max())
    margin = (global_max - global_min) * 0.05
    ax.set_xlim(global_min - margin, global_max + margin)
    ax.set_ylim(global_min - margin, global_max + margin)

    ax.set_xlabel(r'面外加速度 ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel(r'面内加速度 ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title(title, fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)

    _apply_grid(ax)
    plt.tight_layout()
    print(f"  ✓ 样本 {sample_idx + 1}/{total} 已绘制  sensor={sample['inplane_sensor_id']}")

    fig_in_ts,  fig_out_ts  = _plot_aux_timeseries(in_data, out_data, title_prefix=title)
    fig_in_sp,  fig_out_sp  = _plot_aux_spectra(in_data, out_data, title_prefix=title)
    return fig_in_ts, fig_out_ts, fig_in_sp, fig_out_sp, fig


def plot_trajectory_cloud(samples: list, unpacker: UNPACK) -> list:
    figs = []
    for i, sample in enumerate(samples):
        color = Config.SAMPLE_COLORS[i % len(Config.SAMPLE_COLORS)]
        fig_in_ts, fig_out_ts, fig_in_sp, fig_out_sp, fig_traj = _plot_single_trajectory(
            sample, color, i, len(samples), unpacker
        )
        figs.append((fig_in_ts, fig_out_ts, fig_in_sp, fig_out_sp, fig_traj))
    return figs


# ==================== 共享绘图入口 ====================
def _draw_and_push(samples: list, page_name: str):
    unpacker = UNPACK(init_path=False)
    figs     = plot_trajectory_cloud(samples, unpacker)

    print(f"  推送 {len(figs)} 组（每组5张）图到 WebUI 页面「{page_name}」...")
    for sample_idx, (fig_in_ts, fig_out_ts, fig_in_sp, fig_out_sp, fig_traj) in enumerate(figs):
        base_slot = sample_idx * 5
        web_push(fig_in_ts,  page=page_name, slot=base_slot,
                 title=f'样本{sample_idx + 1} 面内时程',
                 page_cols=5 if sample_idx == 0 else None)
        web_push(fig_out_ts, page=page_name, slot=base_slot + 1,
                 title=f'样本{sample_idx + 1} 面外时程')
        web_push(fig_in_sp,  page=page_name, slot=base_slot + 2,
                 title=f'样本{sample_idx + 1} 面内频谱')
        web_push(fig_out_sp, page=page_name, slot=base_slot + 3,
                 title=f'样本{sample_idx + 1} 面外频谱')
        web_push(fig_traj,   page=page_name, slot=base_slot + 4,
                 title=f'样本{sample_idx + 1} 轨迹')
    print(f"  ✓ 推送完成")


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("涡激共振振动轨迹云图（DL vs MECC，面内 vs 面外）")
    print("=" * 80)

    print("\n[步骤1] 加载 DL 识别结果...")
    dl_result   = load_latest_result(Config.DL_RESULT_GLOB)
    dl_samples  = _pipeline_get_viv_samples(dl_result)
    print(f"✓ DL VIV 样本：{len(dl_samples)} 个")
    dl_plot = random_sample(dl_samples)

    print("\n[步骤2] 加载 MECC 识别结果...")
    mecc_result  = load_latest_result(Config.MECC_RESULT_GLOB)
    mecc_samples = _pipeline_get_viv_samples(mecc_result)
    print(f"✓ MECC VIV 样本：{len(mecc_samples)} 个")
    mecc_plot = random_sample(mecc_samples)

    print("\n[步骤3] 绘制 DL 样本轨迹并推送到 WebUI...")
    _draw_and_push(dl_plot, "fig4_22 VIV轨迹 DL")

    print("\n[步骤4] 绘制 MECC 样本轨迹并推送到 WebUI...")
    _draw_and_push(mecc_plot, "fig4_22 VIV轨迹 MECC")

    print("\n" + "=" * 80)
    print("全部轨迹图已推送完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
