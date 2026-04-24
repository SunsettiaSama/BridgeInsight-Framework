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
from src.visualize_tools.utils import PlotLib
from src.identifier.deeplearning_methods import FullDatasetRunner
from src.figure_paintings.figs_for_thesis.config import (
    ENG_FONT, CN_FONT, FONT_SIZE, SQUARE_FIG_SIZE, VIV_VIB_COLOR,
    get_viridis_color_map,
)


# ==================== 常量配置 ====================
class Config:
    FS = 50.0
    WINDOW_SIZE = 3000          # 60s @ 50Hz

    TRIM_START_SECOND = 0
    TRIM_END_SECOND = 20        # 展示前20s，涡激共振周期性更明显

    NUM_SAMPLES_TO_PLOT = 20
    RANDOM_SEED = 7

    FIG_SIZE = SQUARE_FIG_SIZE
    WAVEFORM_COLOR = VIV_VIB_COLOR
    LINEWIDTH = 1.0
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.4
    GRID_LINEWIDTH = 0.5
    GRID_LINESTYLE = '--'

    APPLY_DENOISE = False            # True: 先去噪再绘图；False: 直接用原始信号

    WAVELET_TYPE = 'db4'
    WAVELET_LEVEL = 3
    THRESHOLD_TYPE = 'soft'
    THRESHOLD_METHOD = 'sqtwolog'

    NFFT = 2048
    FREQ_LIMIT = 25.0           # Nyquist @ 50 Hz
    SPECTROGRAM_SEGMENT_S = 2   # 每段时长（秒）用于时频谱

    VIV_CLASS_ID = 1            # 涡激共振对应的类别编号


# ==================== 数据获取 ====================
def get_viv_samples(result: dict) -> list:
    predictions = {int(k): int(v) for k, v in result["predictions"].items()}
    sample_metadata = result.get("sample_metadata", {})

    viv_samples = []
    for idx, pred_label in predictions.items():
        if pred_label != Config.VIV_CLASS_ID:
            continue
        meta = sample_metadata.get(str(idx))
        if meta is None:
            continue
        inplane_path = meta.get("inplane_file_path")
        outplane_path = meta.get("outplane_file_path")
        if not inplane_path or not outplane_path:
            continue
        viv_samples.append({
            "idx": idx,
            "window_idx": meta["window_idx"],
            "inplane_sensor_id": meta.get("inplane_sensor_id", ""),
            "outplane_sensor_id": meta.get("outplane_sensor_id", ""),
            "inplane_file_path": inplane_path,
            "outplane_file_path": outplane_path,
            "timestamp": meta.get("timestamp", []),
        })

    return viv_samples


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


# ==================== 绘图函数 ====================
def _plot_single(data: np.ndarray, sensor_id: str, timestamp: list) -> plt.Figure:
    data_plot_src = _wavelet_denoise(data) if Config.APPLY_DENOISE else data

    trim_start = int(Config.TRIM_START_SECOND * Config.FS)
    trim_end = int(Config.TRIM_END_SECOND * Config.FS)
    data_plot = data_plot_src[trim_start:trim_end]

    time_axis = np.arange(len(data_plot)) / Config.FS + Config.TRIM_START_SECOND

    fig = plt.figure(figsize=Config.FIG_SIZE)
    ax = fig.add_subplot(111)

    ax.plot(time_axis, data_plot, color=Config.WAVEFORM_COLOR, linewidth=Config.LINEWIDTH)

    time_str = "-".join(str(t) for t in timestamp) if timestamp else ""
    ax.set_title(f"{sensor_id} @ {time_str}", fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_xlabel('时间 (s)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel(r'加速度 ($m/s^2$)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)

    ax.grid(
        True,
        color=Config.GRID_COLOR,
        alpha=Config.GRID_ALPHA,
        linewidth=Config.GRID_LINEWIDTH,
        linestyle=Config.GRID_LINESTYLE,
    )
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)

    return fig


def _plot_spectrum(data: np.ndarray, sensor_id: str, timestamp: list) -> plt.Figure:
    data_src = _wavelet_denoise(data) if Config.APPLY_DENOISE else data

    f, psd = signal.welch(
        data_src,
        fs=Config.FS,
        nperseg=int(Config.NFFT / 2),
        noverlap=int(Config.NFFT / 4),
        nfft=Config.NFFT,
    )
    mask = f <= Config.FREQ_LIMIT

    fig = plt.figure(figsize=Config.FIG_SIZE)
    ax = fig.add_subplot(111)

    ax.plot(f[mask], psd[mask], color=Config.WAVEFORM_COLOR, linewidth=Config.LINEWIDTH)

    time_str = "-".join(str(t) for t in timestamp) if timestamp else ""
    ax.set_title(f"{sensor_id} @ {time_str}", fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_xlabel('频率 (Hz)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel(r'PSD $(m/s^2)^2$/Hz', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_xlim(0, Config.FREQ_LIMIT)

    ax.grid(
        True,
        color=Config.GRID_COLOR,
        alpha=Config.GRID_ALPHA,
        linewidth=Config.GRID_LINEWIDTH,
        linestyle=Config.GRID_LINESTYLE,
    )
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)

    return fig


def _plot_spectrogram(data: np.ndarray, sensor_id: str, timestamp: list) -> plt.Figure:
    data_src = _wavelet_denoise(data) if Config.APPLY_DENOISE else data

    seg_len = int(Config.SPECTROGRAM_SEGMENT_S * Config.FS)
    n_segments = len(data_src) // seg_len

    psd_list = []
    f_ref = None
    for i in range(n_segments):
        seg = data_src[i * seg_len: (i + 1) * seg_len]
        f, psd = signal.welch(
            seg,
            fs=Config.FS,
            nperseg=min(int(Config.FS * 0.8), len(seg)),
            noverlap=int(Config.FS * 0.4),
            nfft=Config.NFFT,
        )
        if f_ref is None:
            f_ref = f
        mask = f_ref <= Config.FREQ_LIMIT
        psd_list.append(psd[mask])

    if not psd_list:
        raise ValueError("无法生成时频谱：分段数为零")

    spec_array = np.array(psd_list)
    cmap = get_viridis_color_map(start_gray=0.2)

    total_seconds = n_segments * Config.SPECTROGRAM_SEGMENT_S

    fig = plt.figure(figsize=Config.FIG_SIZE)
    ax = fig.add_subplot(111)

    im = ax.imshow(
        spec_array,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        extent=[0, Config.FREQ_LIMIT, 0, total_seconds],
        interpolation='bilinear',
    )

    time_str = "-".join(str(t) for t in timestamp) if timestamp else ""
    ax.set_title(f"{sensor_id} @ {time_str}", fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_xlabel('频率 (Hz)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel('时间 (s)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'PSD $(m/s^2)^2$/Hz', fontproperties=CN_FONT, fontsize=FONT_SIZE)
    cbar.ax.tick_params(labelsize=FONT_SIZE)

    return fig


def plot_viv_timeseries(samples: list, unpacker: UNPACK):
    inplane_figs = []
    outplane_figs = []

    for i, sample in enumerate(samples, 1):
        if len(inplane_figs) >= Config.NUM_SAMPLES_TO_PLOT and len(outplane_figs) >= Config.NUM_SAMPLES_TO_PLOT:
            break

        print(f"  [样本 {i}] sensor={sample['inplane_sensor_id']} / {sample['outplane_sensor_id']}")

        if len(inplane_figs) < Config.NUM_SAMPLES_TO_PLOT:
            inplane_data = _load_window(sample["inplane_file_path"], sample["window_idx"], unpacker)
            fig_in = _plot_single(inplane_data, sample["inplane_sensor_id"], sample["timestamp"])
            inplane_figs.append(fig_in)
            print(f"    ✓ 面内图已生成（共 {len(inplane_figs)}）")

        if len(outplane_figs) < Config.NUM_SAMPLES_TO_PLOT:
            outplane_data = _load_window(sample["outplane_file_path"], sample["window_idx"], unpacker)
            fig_out = _plot_single(outplane_data, sample["outplane_sensor_id"], sample["timestamp"])
            outplane_figs.append(fig_out)
            print(f"    ✓ 面外图已生成（共 {len(outplane_figs)}）")

    return inplane_figs, outplane_figs


def plot_viv_spectra(samples: list, unpacker: UNPACK):
    inplane_figs = []
    outplane_figs = []

    for i, sample in enumerate(samples, 1):
        if len(inplane_figs) >= Config.NUM_SAMPLES_TO_PLOT and len(outplane_figs) >= Config.NUM_SAMPLES_TO_PLOT:
            break

        print(f"  [样本 {i}] sensor={sample['inplane_sensor_id']} / {sample['outplane_sensor_id']}")

        if len(inplane_figs) < Config.NUM_SAMPLES_TO_PLOT:
            inplane_data = _load_window(sample["inplane_file_path"], sample["window_idx"], unpacker)
            fig_in = _plot_spectrum(inplane_data, sample["inplane_sensor_id"], sample["timestamp"])
            inplane_figs.append(fig_in)
            print(f"    ✓ 面内频谱图已生成（共 {len(inplane_figs)}）")

        if len(outplane_figs) < Config.NUM_SAMPLES_TO_PLOT:
            outplane_data = _load_window(sample["outplane_file_path"], sample["window_idx"], unpacker)
            fig_out = _plot_spectrum(outplane_data, sample["outplane_sensor_id"], sample["timestamp"])
            outplane_figs.append(fig_out)
            print(f"    ✓ 面外频谱图已生成（共 {len(outplane_figs)}）")

    return inplane_figs, outplane_figs


def plot_viv_spectrograms(samples: list, unpacker: UNPACK):
    inplane_figs = []
    outplane_figs = []

    for i, sample in enumerate(samples, 1):
        if len(inplane_figs) >= Config.NUM_SAMPLES_TO_PLOT and len(outplane_figs) >= Config.NUM_SAMPLES_TO_PLOT:
            break

        print(f"  [样本 {i}] sensor={sample['inplane_sensor_id']} / {sample['outplane_sensor_id']}")

        if len(inplane_figs) < Config.NUM_SAMPLES_TO_PLOT:
            inplane_data = _load_window(sample["inplane_file_path"], sample["window_idx"], unpacker)
            fig_in = _plot_spectrogram(inplane_data, sample["inplane_sensor_id"], sample["timestamp"])
            inplane_figs.append(fig_in)
            print(f"    ✓ 面内时频谱图已生成（共 {len(inplane_figs)}）")

        if len(outplane_figs) < Config.NUM_SAMPLES_TO_PLOT:
            outplane_data = _load_window(sample["outplane_file_path"], sample["window_idx"], unpacker)
            fig_out = _plot_spectrogram(outplane_data, sample["outplane_sensor_id"], sample["timestamp"])
            outplane_figs.append(fig_out)
            print(f"    ✓ 面外时频谱图已生成（共 {len(outplane_figs)}）")

    return inplane_figs, outplane_figs


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("第三章 涡激共振时域波形绘制（面内 & 面外）")
    print("=" * 80)

    result_dir = project_root / "results" / "identification_result"
    if not result_dir.exists():
        raise FileNotFoundError(f"识别结果目录不存在：{result_dir}")

    result_files = sorted(result_dir.glob("res_cnn_full_dataset_*.json"))
    if not result_files:
        raise FileNotFoundError("未找到识别结果文件 res_cnn_full_dataset_*.json")

    result_path = result_files[-1]
    print(f"\n[步骤1] 加载识别结果：{result_path.name}")
    result = FullDatasetRunner.load_result(str(result_path))

    print("\n[步骤2] 筛选涡激共振（class 1）样本...")
    samples = get_viv_samples(result)
    print(f"✓ 共筛选到 {len(samples)} 个涡激共振样本")

    print("\n[步骤3] 随机抽取样本...")
    samples = random_sample(samples)

    print("\n[步骤4] 加载原始数据并绘图...")
    unpacker = UNPACK(init_path=False)

    print("\n  -- 时域波形 --")
    inplane_ts, outplane_ts = plot_viv_timeseries(samples, unpacker)

    print("\n  -- 功率谱密度（PSD）--")
    inplane_sp, outplane_sp = plot_viv_spectra(samples, unpacker)

    print("\n  -- 时频谱 --")
    inplane_sg, outplane_sg = plot_viv_spectrograms(samples, unpacker)

    print("\n" + "=" * 80)
    print(f"✓ 时域  面内：{len(inplane_ts)} 张  面外：{len(outplane_ts)} 张")
    print(f"✓ 频谱  面内：{len(inplane_sp)} 张  面外：{len(outplane_sp)} 张")
    print(f"✓ 时频  面内：{len(inplane_sg)} 张  面外：{len(outplane_sg)} 张")
    print("=" * 80)

    all_figs = (
        inplane_ts + outplane_ts
        + inplane_sp + outplane_sp
        + inplane_sg + outplane_sg
    )
    ploter = PlotLib()
    for fig in all_figs:
        ploter.figs.append(fig)
    ploter.show()


if __name__ == "__main__":
    main()
