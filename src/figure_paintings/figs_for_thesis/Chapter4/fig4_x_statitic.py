import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal as scipy_signal

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_processer.io_unpacker import UNPACK
from src.identifier.deeplearning_methods import FullDatasetRunner
from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT, FONT_SIZE, REC_FIG_SIZE,
    VIV_INPLANE_COLOR, VIV_OUTPLANE_COLOR,
)


# ==================== 常量配置 ====================
class Config:
    FS          = 50.0
    WINDOW_SIZE = 3000
    NFFT        = 2048
    FREQ_LIMIT  = 25.0
    DOM_ENERGY_HALF_WIDTH = 3   # Welch bin 宽度，主频两侧各 ±3 个 bin 计为主频能量

    VIV_CLASS_ID = 1
    MAX_SAMPLES  = 3000         # 每个数据源最多处理样本数
    RANDOM_SEED  = 42

    DL_RESULT_DIR   = project_root / "results" / "identification_result"        / "res_cnn_full_dataset_*.json"
    MECC_RESULT_DIR = project_root / "results" / "identification_result_mecc_viv" / "mecc_viv_only_*.json"

    # DL 数据源颜色（与其他 VIV 图一致）
    DL_COLOR   = VIV_INPLANE_COLOR    # '#8074C8' 深紫
    # MECC 数据源颜色（高对比度区分）
    MECC_COLOR = '#2CA02C'            # 草绿色

    # 面内/面外方向颜色（用于能量直方图子图）
    DL_INPLANE_COLOR    = VIV_INPLANE_COLOR    # '#8074C8' 深紫
    DL_OUTPLANE_COLOR   = VIV_OUTPLANE_COLOR   # '#E3625D' 珊瑚红
    MECC_INPLANE_COLOR  = '#2CA02C'             # 草绿色
    MECC_OUTPLANE_COLOR = '#FF7F0E'             # 橙色

    FIG_SIZE      = REC_FIG_SIZE
    GRID_COLOR    = 'gray'
    GRID_ALPHA    = 0.3
    GRID_LINESTYLE = '--'
    SCATTER_SIZE  = 6
    SCATTER_ALPHA = 0.30
    BAR_ALPHA     = 0.62
    N_BINS        = 60
    ENERGY_PERCENTILE = 98.0   # 直方图 x 轴上界取该分位数


# ==================== 数据加载 ====================
def _load_latest_result(full_glob: Path) -> dict:
    parent  = full_glob.parent
    pattern = full_glob.name
    if not parent.exists():
        raise FileNotFoundError(f"结果目录不存在：{parent}")
    files = sorted(parent.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"在 {parent} 中未找到匹配 {pattern!r} 的文件"
        )
    latest = files[-1]
    print(f"    文件：{latest.name}")
    return FullDatasetRunner.load_result(str(latest))


def _get_viv_samples(result: dict) -> list:
    predictions     = {int(k): int(v) for k, v in result["predictions"].items()}
    sample_metadata = result.get("sample_metadata", {})
    viv_samples = []
    for idx, pred_label in predictions.items():
        if pred_label != Config.VIV_CLASS_ID:
            continue
        meta = sample_metadata.get(str(idx))
        if meta is None:
            continue
        inplane_path  = meta.get("inplane_file_path")
        outplane_path = meta.get("outplane_file_path")
        if not inplane_path or not outplane_path:
            continue
        viv_samples.append({
            "idx":               idx,
            "window_idx":        meta["window_idx"],
            "inplane_file_path":  inplane_path,
            "outplane_file_path": outplane_path,
        })
    return viv_samples


def _subsample(samples: list) -> list:
    n   = len(samples)
    k   = Config.MAX_SAMPLES
    if n <= k:
        return samples
    rng     = np.random.default_rng(Config.RANDOM_SEED)
    chosen  = rng.choice(n, size=k, replace=False)
    return [samples[i] for i in sorted(chosen.tolist())]


# ==================== 信号处理 ====================
def _load_window(file_path: str, window_idx: int, unpacker: UNPACK):
    path = Path(file_path)
    if not path.exists():
        return None
    raw   = np.array(unpacker.VIC_DATA_Unpack(file_path))
    start = window_idx * Config.WINDOW_SIZE
    end   = start + Config.WINDOW_SIZE
    if end > len(raw):
        return None
    return raw[start:end]


def _dominant_freq_and_energy(sig: np.ndarray):
    f, psd = scipy_signal.welch(
        sig,
        fs=Config.FS,
        nperseg=int(Config.NFFT / 2),
        noverlap=int(Config.NFFT / 4),
        nfft=Config.NFFT,
    )
    mask    = f <= Config.FREQ_LIMIT
    f_lim   = f[mask]
    psd_lim = psd[mask]
    if len(psd_lim) == 0 or psd_lim.sum() == 0:
        return None, None
    dom_idx  = int(np.argmax(psd_lim))
    dom_freq = float(f_lim[dom_idx])
    half_w   = Config.DOM_ENERGY_HALF_WIDTH
    lo       = max(0, dom_idx - half_w)
    hi       = min(len(psd_lim) - 1, dom_idx + half_w)
    dom_energy = float(psd_lim[lo:hi + 1].sum() / psd_lim.sum())
    return dom_freq, dom_energy


def _compute_stats(samples: list, source_name: str) -> dict:
    unpacker = UNPACK(init_path=False)

    rms_pairs      = []          # (rms_in, rms_out) — 成对
    inplane_freq   = []
    outplane_freq  = []
    inplane_energy = []
    outplane_energy = []

    total = len(samples)
    for i, s in enumerate(samples):
        if (i + 1) % 200 == 0:
            print(f"    [{source_name}] 处理 {i + 1}/{total}...")

        sig_in  = _load_window(s["inplane_file_path"],  s["window_idx"], unpacker)
        sig_out = _load_window(s["outplane_file_path"], s["window_idx"], unpacker)

        rms_in = rms_out = None

        if sig_in is not None:
            rms_in = float(np.sqrt(np.mean(sig_in ** 2)))
            freq_in, energy_in = _dominant_freq_and_energy(sig_in)
            if freq_in is not None:
                inplane_freq.append(freq_in)
                inplane_energy.append(energy_in)

        if sig_out is not None:
            rms_out = float(np.sqrt(np.mean(sig_out ** 2)))
            freq_out, energy_out = _dominant_freq_and_energy(sig_out)
            if freq_out is not None:
                outplane_freq.append(freq_out)
                outplane_energy.append(energy_out)

        if rms_in is not None and rms_out is not None:
            rms_pairs.append((rms_in, rms_out))

    rms_arr = np.array(rms_pairs, dtype=np.float64) if rms_pairs else np.empty((0, 2))
    return {
        "inplane_rms":    rms_arr[:, 0] if rms_arr.shape[0] > 0 else np.array([]),
        "outplane_rms":   rms_arr[:, 1] if rms_arr.shape[0] > 0 else np.array([]),
        "inplane_freq":   np.array(inplane_freq,    dtype=np.float64),
        "outplane_freq":  np.array(outplane_freq,   dtype=np.float64),
        "inplane_energy": np.array(inplane_energy,  dtype=np.float64),
        "outplane_energy": np.array(outplane_energy, dtype=np.float64),
    }


# ==================== 工具函数 ====================
def _apply_grid(ax):
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA,
            linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)


def _add_legend(ax):
    leg = ax.legend(fontsize=FONT_SIZE - 2, framealpha=0.9)
    for t in leg.get_texts():
        t.set_fontproperties(CN_FONT)


def _downsample_xy(x, y, seed, max_pts=50_000):
    n = len(x)
    if n <= max_pts:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_pts, replace=False)
    return x[idx], y[idx]


def _downsample_1d(arr, seed, max_pts=50_000):
    n = len(arr)
    if n <= max_pts:
        return arr
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_pts, replace=False)
    return arr[idx]


# ==================== 绘图函数 ====================
def plot_rms_comparison(dl_stats: dict, mecc_stats: dict) -> plt.Figure:
    dl_out_s,   dl_in_s   = _downsample_xy(
        dl_stats["outplane_rms"],   dl_stats["inplane_rms"],
        Config.RANDOM_SEED
    )
    mecc_out_s, mecc_in_s = _downsample_xy(
        mecc_stats["outplane_rms"], mecc_stats["inplane_rms"],
        Config.RANDOM_SEED + 1
    )

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)

    ax.scatter(
        dl_out_s, dl_in_s,
        s=Config.SCATTER_SIZE, color=Config.DL_COLOR,
        alpha=Config.SCATTER_ALPHA, linewidths=0,
        label=f'深度学习 (VIV, n={len(dl_in_s)})',
        zorder=1,
    )
    ax.scatter(
        mecc_out_s, mecc_in_s,
        s=Config.SCATTER_SIZE, color=Config.MECC_COLOR,
        alpha=Config.SCATTER_ALPHA, linewidths=0,
        label=f'MECC (VIV, n={len(mecc_in_s)})',
        zorder=2,
    )

    ax.set_xlabel('面外 RMS (m/s²)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel('面内 RMS (m/s²)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title('VIV 样本 RMS 分布对比（深度学习 vs MECC）',
                 fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)
    _add_legend(ax)
    _apply_grid(ax)
    plt.tight_layout()
    return fig


def plot_freq_energy_comparison(dl_stats: dict, mecc_stats: dict) -> plt.Figure:
    # 面内：实心圆 o；面外：向上三角 ^
    # DL：深紫；MECC：草绿
    dl_fi,   dl_ei   = _downsample_xy(
        dl_stats["inplane_freq"],   dl_stats["inplane_energy"],  Config.RANDOM_SEED
    )
    dl_fo,   dl_eo   = _downsample_xy(
        dl_stats["outplane_freq"],  dl_stats["outplane_energy"], Config.RANDOM_SEED + 1
    )
    mecc_fi, mecc_ei = _downsample_xy(
        mecc_stats["inplane_freq"],  mecc_stats["inplane_energy"],  Config.RANDOM_SEED + 2
    )
    mecc_fo, mecc_eo = _downsample_xy(
        mecc_stats["outplane_freq"], mecc_stats["outplane_energy"], Config.RANDOM_SEED + 3
    )

    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)

    ax.scatter(dl_fi,   dl_ei,   s=Config.SCATTER_SIZE, marker='o',
               color=Config.DL_COLOR,   alpha=Config.SCATTER_ALPHA, linewidths=0,
               label='DL 面内')
    ax.scatter(dl_fo,   dl_eo,   s=Config.SCATTER_SIZE, marker='^',
               color=Config.DL_COLOR,   alpha=Config.SCATTER_ALPHA, linewidths=0,
               label='DL 面外')
    ax.scatter(mecc_fi, mecc_ei, s=Config.SCATTER_SIZE, marker='o',
               color=Config.MECC_COLOR, alpha=Config.SCATTER_ALPHA, linewidths=0,
               label='MECC 面内')
    ax.scatter(mecc_fo, mecc_eo, s=Config.SCATTER_SIZE, marker='^',
               color=Config.MECC_COLOR, alpha=Config.SCATTER_ALPHA, linewidths=0,
               label='MECC 面外')

    ax.set_xlabel('主频 (Hz)', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel('主频能量占比', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title('VIV 主频与能量占比散点对比（深度学习 vs MECC）',
                 fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=14)
    ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)
    _add_legend(ax)
    _apply_grid(ax)
    plt.tight_layout()
    return fig


def plot_energy_hist_comparison(dl_stats: dict, mecc_stats: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=Config.FIG_SIZE, sharey=False)

    pairs = [
        (axes[0], '面内',
         dl_stats["inplane_energy"],   Config.DL_INPLANE_COLOR,
         mecc_stats["inplane_energy"], Config.MECC_INPLANE_COLOR),
        (axes[1], '面外',
         dl_stats["outplane_energy"],  Config.DL_OUTPLANE_COLOR,
         mecc_stats["outplane_energy"], Config.MECC_OUTPLANE_COLOR),
    ]

    for ax, direction, dl_e, dl_c, mecc_e, mecc_c in pairs:
        combined = np.concatenate([dl_e, mecc_e]) if (len(dl_e) and len(mecc_e)) else (
            dl_e if len(dl_e) else mecc_e
        )
        if len(combined) == 0:
            ax.set_title(direction, fontproperties=CN_FONT, fontsize=FONT_SIZE)
            continue

        x_max = min(float(np.percentile(combined, Config.ENERGY_PERCENTILE)), 1.0)
        bins  = np.linspace(0, x_max, Config.N_BINS + 1)

        def _hist(arr, bins):
            clipped = arr[arr <= x_max]
            counts, _ = np.histogram(clipped, bins=bins)
            return counts

        dl_counts   = _hist(dl_e,   bins)
        mecc_counts = _hist(mecc_e, bins)

        centers = 0.5 * (bins[:-1] + bins[1:])
        bar_w   = (bins[1] - bins[0]) * 0.42

        ax.bar(centers - bar_w / 2, dl_counts,   width=bar_w, color=dl_c,
               alpha=Config.BAR_ALPHA, label='深度学习 (VIV)')
        ax.bar(centers + bar_w / 2, mecc_counts, width=bar_w, color=mecc_c,
               alpha=Config.BAR_ALPHA, label='MECC (VIV)')

        ax.set_xlabel('主频能量占比', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
        ax.set_ylabel('样本数（个）', labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
        ax.set_title(f'VIV {direction}能量占比分布对比',
                     fontproperties=CN_FONT, fontsize=FONT_SIZE, pad=12)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE - 2)
        _add_legend(ax)
        _apply_grid(ax)

    plt.tight_layout()
    return fig


# ==================== 主函数 ====================
def main():
    print("=" * 80)
    print("VIV 统计特征对比（深度学习 vs MECC）")
    print("=" * 80)

    print("\n[步骤1] 加载识别结果...")
    print("  深度学习结果：")
    dl_result   = _load_latest_result(Config.DL_RESULT_DIR)
    print("  MECC 结果：")
    mecc_result = _load_latest_result(Config.MECC_RESULT_DIR)

    print("\n[步骤2] 筛选 VIV 样本...")
    dl_samples   = _get_viv_samples(dl_result)
    mecc_samples = _get_viv_samples(mecc_result)
    print(f"  深度学习 VIV 样本数：{len(dl_samples)}")
    print(f"  MECC VIV 样本数：{len(mecc_samples)}")

    print("\n[步骤3] 随机抽样（上限 MAX_SAMPLES={})...".format(Config.MAX_SAMPLES))
    dl_samples   = _subsample(dl_samples)
    mecc_samples = _subsample(mecc_samples)
    print(f"  深度学习抽样后：{len(dl_samples)}  MECC 抽样后：{len(mecc_samples)}")

    print("\n[步骤4] 计算统计量（加载原始信号 → RMS / 主频 / 能量占比）...")
    print("  [深度学习]")
    dl_stats   = _compute_stats(dl_samples,   "DL")
    print(f"    RMS 成对：{len(dl_stats['inplane_rms'])}  "
          f"主频：面内 {len(dl_stats['inplane_freq'])} / 面外 {len(dl_stats['outplane_freq'])}")

    print("  [MECC]")
    mecc_stats = _compute_stats(mecc_samples, "MECC")
    print(f"    RMS 成对：{len(mecc_stats['inplane_rms'])}  "
          f"主频：面内 {len(mecc_stats['inplane_freq'])} / 面外 {len(mecc_stats['outplane_freq'])}")

    print("\n[步骤5] 绘制对比图...")
    fig_rms    = plot_rms_comparison(dl_stats, mecc_stats)
    print("  ✓ RMS 散点对比图")
    fig_freq   = plot_freq_energy_comparison(dl_stats, mecc_stats)
    print("  ✓ 主频-能量散点对比图")
    fig_energy = plot_energy_hist_comparison(dl_stats, mecc_stats)
    print("  ✓ 能量占比直方图对比")

    print("\n[步骤6] 注入 WebUI 浏览器（HTTP POST → 已运行的 web_dashboard 服务）...")
    from src.visualize_tools.web_dashboard import push as web_push

    PAGE   = 'fig4_x VIV统计对比 DL vs MECC'
    TITLES = ['RMS 分布对比', '主频-能量散点对比', '能量占比分布对比']
    for slot, (fig, title) in enumerate(zip([fig_rms, fig_freq, fig_energy], TITLES)):
        web_push(fig, page=PAGE, slot=slot, title=title,
                 page_cols=3 if slot == 0 else None)
        print(f"  ✓ 推送 slot {slot}：{title}")
    print("=" * 80)


if __name__ == "__main__":
    main()
