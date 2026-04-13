import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_processer.io_unpacker import UNPACK
from src.data_processer.signals.wavelets import denoise
from src.visualize_tools.utils import PlotLib
from src.figure_paintings.figs_for_thesis.Chapter3.fig3_2_all_data_display import load_identification_result
from src.figure_paintings.figs_for_thesis.config import (
    ENG_FONT, CN_FONT, SQUARE_FIG_SIZE, SQUARE_FONT_SIZE,
    NORMAL_VIB_COLOR, NORMAL_EDGE_COLOR, DEFAULT_COLOR,
)

FONT_SIZE = SQUARE_FONT_SIZE


# ==================== 常量配置 ====================
class Config:
    FS            = 50.0
    WINDOW_SIZE   = 3000          # 60 s @ 50 Hz

    TRIM_START_SECOND = 0
    TRIM_END_SECOND   = 10        # 展示前 20 s

    NUM_SAMPLES   = 3             # 绘制样本组数
    RANDOM_SEED   = 42

    TOP_N_FREQS   = 10            # 用于重建的前 N 阶主频分量

    # 颜色：复用 config.py 配色
    COLOR_TIME  = NORMAL_EDGE_COLOR   # 原始时域 — 深灰
    COLOR_RECON = DEFAULT_COLOR       # 重建时域 — 蓝紫
    COLOR_FREQ  = NORMAL_EDGE_COLOR   # 频域幅值谱 — 深灰

    LINEWIDTH      = 3
    GRID_COLOR     = 'gray'
    GRID_ALPHA     = 0.4
    GRID_LINEWIDTH = 0.5
    GRID_LINESTYLE = '--'

    WAVELET_TYPE    = 'db4'
    WAVELET_LEVEL   = 3
    THRESHOLD_TYPE  = 'soft'
    THRESHOLD_METHOD = 'sqtwolog'

    NORMAL_VIB_CLASS_ID = 0

    FIG_SIZE = SQUARE_FIG_SIZE    # (10, 8) 正方形


# ==================== 数据获取 ====================
def get_normal_vib_samples(result: dict) -> list:
    predictions     = {int(k): int(v) for k, v in result["predictions"].items()}
    sample_metadata = result.get("sample_metadata", {})

    samples = []
    for idx, pred_label in predictions.items():
        if pred_label != Config.NORMAL_VIB_CLASS_ID:
            continue
        meta = sample_metadata.get(str(idx))
        if meta is None:
            continue
        if not meta.get("inplane_file_path"):
            continue
        samples.append({
            "idx":               idx,
            "window_idx":        meta["window_idx"],
            "inplane_sensor_id": meta.get("inplane_sensor_id", ""),
            "inplane_file_path": meta["inplane_file_path"],
            "timestamp":         meta.get("timestamp", []),
        })
    return samples


def random_sample(samples: list) -> list:
    n   = len(samples)
    k   = min(Config.NUM_SAMPLES, n)
    rng = np.random.default_rng(Config.RANDOM_SEED)
    chosen = sorted(rng.choice(n, size=k, replace=False).tolist())
    print(f"  随机抽取：{chosen}（共 {n} 个样本中选 {k} 个，seed={Config.RANDOM_SEED}）")
    return [samples[i] for i in chosen]


# ==================== 信号处理 ====================
def load_window(file_path: str, window_idx: int, unpacker: UNPACK) -> np.ndarray:
    raw   = np.array(unpacker.VIC_DATA_Unpack(file_path))
    start = window_idx * Config.WINDOW_SIZE
    end   = start + Config.WINDOW_SIZE
    if end > len(raw):
        raise ValueError(f"窗口越界：window_idx={window_idx}, len={len(raw)}")
    return raw[start:end]


def wavelet_denoise(data: np.ndarray) -> np.ndarray:
    denoised, _ = denoise(
        signal=data,
        wavelet=Config.WAVELET_TYPE,
        level=Config.WAVELET_LEVEL,
        threshold_type=Config.THRESHOLD_TYPE,
        threshold_method=Config.THRESHOLD_METHOD,
    )
    return denoised


def compute_spectrum(data: np.ndarray):
    """返回 (freqs_hz, amplitude)，仅保留正频率部分。"""
    n      = len(data)
    fft_c  = np.fft.rfft(data)
    amp    = np.abs(fft_c) / n * 2      # 单边幅值（DC 项不需 ×2，但为对称显示统一处理）
    amp[0] /= 2                          # 修正 DC
    freqs  = np.fft.rfftfreq(n, d=1.0 / Config.FS)
    return freqs, amp, fft_c


def reconstruct_top_n(fft_c: np.ndarray, n: int) -> np.ndarray:
    """保留幅值最大的前 n 个频率分量，其余置零后 IFFT。"""
    amp      = np.abs(fft_c)
    top_idx  = np.argsort(amp)[::-1][:n]
    mask     = np.zeros_like(fft_c)
    mask[top_idx] = fft_c[top_idx]
    return np.fft.irfft(mask, n=Config.WINDOW_SIZE)


# ==================== 绘图 ====================
def _ax_grid(ax):
    ax.grid(
        True,
        color=Config.GRID_COLOR,
        alpha=Config.GRID_ALPHA,
        linewidth=Config.GRID_LINEWIDTH,
        linestyle=Config.GRID_LINESTYLE,
    )
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)


def _prepare_data(data_full: np.ndarray):
    """公共预处理：去噪 → 截取展示段 → 计算频谱与重建信号。"""
    data      = wavelet_denoise(data_full)
    t_start   = int(Config.TRIM_START_SECOND * Config.FS)
    t_end     = int(Config.TRIM_END_SECOND   * Config.FS)
    data_show = data[t_start:t_end]
    time_axis = np.arange(len(data_show)) / Config.FS + Config.TRIM_START_SECOND
    freqs, amp, fft_c = compute_spectrum(data)
    recon_show = reconstruct_top_n(fft_c, Config.TOP_N_FREQS)[t_start:t_end]
    return time_axis, data_show, recon_show, freqs, amp


def plot_timeseries(data_full: np.ndarray, sensor_id: str, timestamp: list) -> plt.Figure:
    """独立正方形图 A：原始时域 + 前 N 阶重建叠加对比。"""
    time_axis, data_show, recon_show, _, _ = _prepare_data(data_full)

    fig, ax = plt.subplots(1, 1, figsize=Config.FIG_SIZE)

    ax.plot(time_axis, data_show,  color=Config.COLOR_TIME,  linewidth=Config.LINEWIDTH,
            label='原始时域信号')
    ax.plot(time_axis, recon_show, color=Config.COLOR_RECON, linewidth=Config.LINEWIDTH,
            linestyle='--', label=f'前 {Config.TOP_N_FREQS} 阶重建')

    ax.set_xlabel('时间 (s)', labelpad=8, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel(r'加速度 ($\mathrm{m/s^2}$)', labelpad=8, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)

    legend = ax.legend(framealpha=0.8)
    for text in legend.get_texts():
        text.set_fontproperties(CN_FONT)
        text.set_fontsize(FONT_SIZE - 4)

    _ax_grid(ax)
    fig.tight_layout()
    return fig


def plot_spectrum(data_full: np.ndarray, sensor_id: str, timestamp: list) -> plt.Figure:
    """独立正方形图 B：频域幅值谱（0 ~ 10 Hz）。"""
    _, _, _, freqs, amp = _prepare_data(data_full)

    freq_max = 10.0
    mask_f   = freqs <= freq_max

    fig, ax = plt.subplots(1, 1, figsize=Config.FIG_SIZE)

    ax.plot(freqs[mask_f], amp[mask_f], color=Config.COLOR_FREQ, linewidth=Config.LINEWIDTH)

    ax.set_xlabel('频率 (Hz)', labelpad=8, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel('幅值', labelpad=8, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)

    _ax_grid(ax)
    fig.tight_layout()
    return fig


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("Fig 3-4  时域 → 频域 → 重建时域（随机振动）")
    print("=" * 80)

    result_dir   = project_root / "results" / "identification_result"
    result_files = sorted(result_dir.glob("res_cnn_full_dataset_*.json"))
    if not result_files:
        raise FileNotFoundError("未找到识别结果文件 res_cnn_full_dataset_*.json")

    result_path = result_files[-1]
    print(f"\n[步骤1] 加载识别结果：{result_path.name}")
    result = load_identification_result(str(result_path))

    print("\n[步骤2] 筛选随机振动（class 0）样本...")
    samples = get_normal_vib_samples(result)
    print(f"  共 {len(samples)} 个随机振动样本")

    print("\n[步骤3] 随机抽取样本...")
    samples = random_sample(samples)

    print("\n[步骤4] 加载数据并绘图...")
    unpacker = UNPACK(init_path=False)
    figs     = []

    for i, sample in enumerate(samples, 1):
        sid = sample["inplane_sensor_id"]
        ts  = sample["timestamp"]
        print(f"  [样本 {i}] {sid}  window={sample['window_idx']}")
        data = load_window(sample["inplane_file_path"], sample["window_idx"], unpacker)
        figs.append(plot_timeseries(data, sid, ts))
        figs.append(plot_spectrum(data, sid, ts))
        print(f"    ✓ 时域图 + 频域图已生成")

    print(f"\n共生成 {len(figs)} 张独立图像（每样本 2 张）")
    print("=" * 80)

    ploter = PlotLib()
    for fig in figs:
        ploter.figs.append(fig)
    ploter.show()


if __name__ == "__main__":
    main()
