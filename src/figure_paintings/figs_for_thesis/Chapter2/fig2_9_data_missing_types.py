import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from pathlib import Path
from scipy import signal

_project_root = Path(__file__).parent.parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.figure_paintings.figs_for_thesis.config import ENG_FONT, CN_FONT, FONT_SIZE


# ==================== 常量配置 ====================
class Config:
    FS = 50.0
    DURATION = 120.0
    RANDOM_SEED = 42

    # 模拟拉索振动信号组成：基频 + 谐波 + 低频漂移 + 噪声
    COMPONENTS = [
        (1.0, 1.2),    # 幅值, 频率(Hz) - 拉索基频
        (0.40, 2.4),   # 二阶谐波
        (0.20, 3.6),   # 三阶谐波
        (0.15, 0.4),   # 低频漂移分量
    ]
    NOISE_LEVEL = 0.06

    # 突发块状缺失参数
    SUDDEN_START_S = 35.0
    SUDDEN_END_S = 72.0

    # 周期性块状缺失参数
    PERIODIC_PERIOD_S = 20.0
    PERIODIC_MISSING_S = 5.0

    # 随机单点缺失参数
    RANDOM_MISSING_RATE = 0.008

    # 频谱计算参数
    NFFT = 4096
    FREQ_MAX_PLOT = 6.0

    # 绘图颜色
    COLOR_SIGNAL = '#333333'
    COLOR_ORIGINAL_PSD = '#8074C8'
    COLOR_MISSING_PSD = '#E3625D'
    COLOR_MISSING_SPAN = '#E3625D'
    COLOR_RANDOM_MARKER = '#E3625D'
    ALPHA_MISSING_SPAN = 0.18

    LINEWIDTH_SIGNAL = 0.9
    LINEWIDTH_PSD = 1.2
    GRID_COLOR = 'gray'
    GRID_ALPHA = 0.3
    GRID_LINEWIDTH = 0.5
    GRID_LINESTYLE = '--'

    TS_FONTSIZE = FONT_SIZE - 6    # 轴标签字号
    TITLE_FONTSIZE = FONT_SIZE - 4  # 子图标题字号

    FIG_SIZE = (16, 14)


# ==================== 信号生成 ====================
def generate_signal():
    rng = np.random.default_rng(Config.RANDOM_SEED)
    n = int(Config.FS * Config.DURATION)
    t = np.arange(n) / Config.FS

    sig = sum(
        amp * np.sin(2 * np.pi * freq * t)
        for amp, freq in Config.COMPONENTS
    )
    sig += Config.NOISE_LEVEL * rng.standard_normal(n)
    return t, sig


# ==================== 缺失模拟 ====================
def apply_sudden_missing(sig):
    masked = sig.copy().astype(float)
    i0 = int(Config.SUDDEN_START_S * Config.FS)
    i1 = int(Config.SUDDEN_END_S * Config.FS)
    masked[i0:i1] = np.nan
    return masked


def apply_periodic_missing(sig):
    masked = sig.copy().astype(float)
    n = len(sig)
    period = int(Config.PERIODIC_PERIOD_S * Config.FS)
    miss_len = int(Config.PERIODIC_MISSING_S * Config.FS)
    for start in range(0, n, period):
        masked[start:min(start + miss_len, n)] = np.nan
    return masked


def apply_random_missing(sig):
    rng = np.random.default_rng(Config.RANDOM_SEED + 7)
    masked = sig.copy().astype(float)
    n = len(sig)
    miss_idx = rng.choice(n, size=int(n * Config.RANDOM_MISSING_RATE), replace=False)
    masked[miss_idx] = np.nan
    return masked, miss_idx


# ==================== 频谱计算 ====================
def compute_psd(sig_with_nan):
    filled = np.where(np.isnan(sig_with_nan), 0.0, sig_with_nan)
    freqs, psd = signal.welch(
        filled, fs=Config.FS,
        nperseg=Config.NFFT // 2,
        nfft=Config.NFFT,
        window='hann',
    )
    return freqs, psd


# ==================== 辅助绘图 ====================
def _shade_missing_spans(ax, t, missing_mask):
    in_block = False
    t_start = None
    for i, m in enumerate(missing_mask):
        if m and not in_block:
            in_block = True
            t_start = t[i]
        elif not m and in_block:
            in_block = False
            ax.axvspan(t_start, t[i], alpha=Config.ALPHA_MISSING_SPAN,
                       color=Config.COLOR_MISSING_SPAN, zorder=1, linewidth=0)
    if in_block:
        ax.axvspan(t_start, t[-1], alpha=Config.ALPHA_MISSING_SPAN,
                   color=Config.COLOR_MISSING_SPAN, zorder=1, linewidth=0)


def _style_ax(ax, xlabel=None, ylabel=None, xlim=None, ylim=None, hide_xticklabels=False):
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA,
            linewidth=Config.GRID_LINEWIDTH, linestyle=Config.GRID_LINESTYLE)
    ax.tick_params(axis='both', which='major', labelsize=Config.TS_FONTSIZE)
    if xlabel:
        ax.set_xlabel(xlabel, fontproperties=CN_FONT, fontsize=Config.TS_FONTSIZE, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, fontproperties=CN_FONT, fontsize=Config.TS_FONTSIZE, labelpad=6)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if hide_xticklabels:
        ax.set_xticklabels([])
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune='both'))


def _plot_psd_pair(ax, freqs, psd_orig, psd_miss, show_legend=True, show_xlabel=False):
    mask = freqs <= Config.FREQ_MAX_PLOT
    ax.semilogy(freqs[mask], psd_orig[mask],
                color=Config.COLOR_ORIGINAL_PSD, linewidth=Config.LINEWIDTH_PSD,
                label='原始', zorder=3)
    ax.semilogy(freqs[mask], psd_miss[mask],
                color=Config.COLOR_MISSING_PSD, linewidth=Config.LINEWIDTH_PSD,
                linestyle='--', label='缺失后', alpha=0.9, zorder=4)
    ax.set_xlim([0, Config.FREQ_MAX_PLOT])
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA,
            linewidth=Config.GRID_LINEWIDTH, linestyle=Config.GRID_LINESTYLE)
    ax.tick_params(axis='both', which='major', labelsize=Config.TS_FONTSIZE)
    ax.tick_params(axis='y', which='minor', labelsize=Config.TS_FONTSIZE - 2)
    if show_legend:
        ax.legend(prop=CN_FONT, fontsize=Config.TS_FONTSIZE,
                  loc='upper right', framealpha=0.8)
    if show_xlabel:
        ax.set_xlabel('频率 (Hz)', fontproperties=CN_FONT,
                      fontsize=Config.TS_FONTSIZE, labelpad=6)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel('PSD', fontproperties=ENG_FONT, fontsize=Config.TS_FONTSIZE, labelpad=6)

    # 标记已知频率分量
    for _, freq in Config.COMPONENTS:
        if freq <= Config.FREQ_MAX_PLOT:
            ax.axvline(freq, color='#A0A0A0', linewidth=0.7, linestyle=':', zorder=2)


# ==================== 数据准备 ====================
_FIG_SIZE_SPLIT = (16, 7)
_SUBPLOTS_ADJUST = dict(hspace=0.48, wspace=0.32,
                        left=0.08, right=0.97, top=0.93, bottom=0.12)


def _prepare_data() -> dict:
    t, sig_orig = generate_signal()
    sig_sudden  = apply_sudden_missing(sig_orig)
    sig_periodic = apply_periodic_missing(sig_orig)
    sig_random, random_miss_idx = apply_random_missing(sig_orig)

    freqs, psd_orig   = compute_psd(sig_orig)
    _, psd_sudden     = compute_psd(sig_sudden)
    _, psd_periodic   = compute_psd(sig_periodic)
    _, psd_random     = compute_psd(sig_random)

    y_lim = (sig_orig.min() * 1.3, sig_orig.max() * 1.3)
    return dict(
        t=t, sig_orig=sig_orig,
        sig_sudden=sig_sudden, sig_periodic=sig_periodic,
        sig_random=sig_random, random_miss_idx=random_miss_idx,
        freqs=freqs, psd_orig=psd_orig,
        psd_sudden=psd_sudden, psd_periodic=psd_periodic, psd_random=psd_random,
        y_lim=y_lim,
    )


# ==================== 分图绘制 ====================
def plot_fig1(data: dict) -> plt.Figure:
    """图一：原始完整信号 + 突发块状缺失（2×2 子图）"""
    t         = data['t']
    sig_orig  = data['sig_orig']
    sig_sudden = data['sig_sudden']
    freqs, psd_orig, psd_sudden = data['freqs'], data['psd_orig'], data['psd_sudden']
    y_lim     = data['y_lim']

    fig, axes = plt.subplots(2, 2, figsize=_FIG_SIZE_SPLIT)
    fig.subplots_adjust(**_SUBPLOTS_ADJUST)

    # ── 第0行：原始信号 ──
    ax = axes[0, 0]
    ax.plot(t, sig_orig, color=Config.COLOR_SIGNAL, linewidth=Config.LINEWIDTH_SIGNAL)
    _style_ax(ax, ylabel=r'加速度 ($\rm{m/s^2}$)', xlim=[0, Config.DURATION],
              ylim=y_lim, hide_xticklabels=True)
    ax.set_title('原始完整信号',
                 fontproperties=CN_FONT, fontsize=Config.TITLE_FONTSIZE, loc='left', pad=4)

    ax = axes[0, 1]
    mask = freqs <= Config.FREQ_MAX_PLOT
    ax.semilogy(freqs[mask], psd_orig[mask],
                color=Config.COLOR_ORIGINAL_PSD, linewidth=Config.LINEWIDTH_PSD)
    for _, freq in Config.COMPONENTS:
        if freq <= Config.FREQ_MAX_PLOT:
            ax.axvline(freq, color='#A0A0A0', linewidth=0.7, linestyle=':', zorder=2)
    ax.set_xlim([0, Config.FREQ_MAX_PLOT])
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA,
            linewidth=Config.GRID_LINEWIDTH, linestyle=Config.GRID_LINESTYLE)
    ax.tick_params(axis='both', which='major', labelsize=Config.TS_FONTSIZE)
    ax.set_xticklabels([])
    ax.set_ylabel('PSD', fontproperties=ENG_FONT, fontsize=Config.TS_FONTSIZE, labelpad=6)
    ax.set_title('功率谱密度',
                 fontproperties=CN_FONT, fontsize=Config.TITLE_FONTSIZE, loc='left', pad=4)

    # ── 第1行：突发块状缺失 ──
    ax = axes[1, 0]
    missing_mask_sudden = np.isnan(sig_sudden)
    _shade_missing_spans(ax, t, missing_mask_sudden)
    ax.plot(t, np.where(missing_mask_sudden, np.nan, sig_sudden),
            color=Config.COLOR_SIGNAL, linewidth=Config.LINEWIDTH_SIGNAL, zorder=3)
    _style_ax(ax, xlabel='时间 (s)', ylabel=r'加速度 ($\rm{m/s^2}$)',
              xlim=[0, Config.DURATION], ylim=y_lim)
    missing_rate_sudden = missing_mask_sudden.sum() / len(missing_mask_sudden) * 100
    ax.set_title(f'突发块状缺失  (缺失率 {missing_rate_sudden:.1f}%)',
                 fontproperties=CN_FONT, fontsize=Config.TITLE_FONTSIZE, loc='left', pad=4)
    ax.annotate('', xy=(Config.SUDDEN_END_S, y_lim[1] * 0.85),
                xytext=(Config.SUDDEN_START_S, y_lim[1] * 0.85),
                arrowprops=dict(arrowstyle='<->', color=Config.COLOR_MISSING_PSD, lw=1.5))
    ax.text((Config.SUDDEN_START_S + Config.SUDDEN_END_S) / 2, y_lim[1] * 0.95,
            f'{Config.SUDDEN_END_S - Config.SUDDEN_START_S:.0f} s',
            ha='center', va='top', color=Config.COLOR_MISSING_PSD,
            fontproperties=ENG_FONT, fontsize=Config.TS_FONTSIZE)

    ax = axes[1, 1]
    _plot_psd_pair(ax, freqs, psd_orig, psd_sudden, show_legend=True, show_xlabel=True)
    ax.set_title('功率谱密度对比',
                 fontproperties=CN_FONT, fontsize=Config.TITLE_FONTSIZE, loc='left', pad=4)

    return fig


def plot_fig2(data: dict) -> plt.Figure:
    """图二：周期性块状缺失 + 随机单点缺失（2×2 子图）"""
    t         = data['t']
    sig_orig  = data['sig_orig']
    sig_periodic = data['sig_periodic']
    sig_random, random_miss_idx = data['sig_random'], data['random_miss_idx']
    freqs     = data['freqs']
    psd_orig, psd_periodic, psd_random = data['psd_orig'], data['psd_periodic'], data['psd_random']
    y_lim     = data['y_lim']

    fig, axes = plt.subplots(2, 2, figsize=_FIG_SIZE_SPLIT)
    fig.subplots_adjust(**_SUBPLOTS_ADJUST)

    # ── 第0行：周期性块状缺失 ──
    ax = axes[0, 0]
    missing_mask_periodic = np.isnan(sig_periodic)
    _shade_missing_spans(ax, t, missing_mask_periodic)
    ax.plot(t, np.where(missing_mask_periodic, np.nan, sig_periodic),
            color=Config.COLOR_SIGNAL, linewidth=Config.LINEWIDTH_SIGNAL, zorder=3)
    _style_ax(ax, ylabel=r'加速度 ($\rm{m/s^2}$)', xlim=[0, Config.DURATION],
              ylim=y_lim, hide_xticklabels=True)
    missing_rate_periodic = missing_mask_periodic.sum() / len(missing_mask_periodic) * 100
    ax.set_title(f'周期性块状缺失  (缺失率 {missing_rate_periodic:.1f}%)',
                 fontproperties=CN_FONT, fontsize=Config.TITLE_FONTSIZE, loc='left', pad=4)
    period_arrow_y = y_lim[1] * 0.88
    ax.annotate('', xy=(Config.PERIODIC_PERIOD_S, period_arrow_y),
                xytext=(0, period_arrow_y),
                arrowprops=dict(arrowstyle='<->', color='#606060', lw=1.2))
    ax.text(Config.PERIODIC_PERIOD_S / 2, y_lim[1] * 0.97,
            f'T={Config.PERIODIC_PERIOD_S:.0f} s',
            ha='center', va='top', color='#404040',
            fontproperties=ENG_FONT, fontsize=Config.TS_FONTSIZE)

    ax = axes[0, 1]
    _plot_psd_pair(ax, freqs, psd_orig, psd_periodic, show_legend=True, show_xlabel=False)
    ax.set_title('功率谱密度对比',
                 fontproperties=CN_FONT, fontsize=Config.TITLE_FONTSIZE, loc='left', pad=4)

    # ── 第1行：随机单点缺失 ──
    ax = axes[1, 0]
    missing_mask_random = np.isnan(sig_random)
    ax.plot(t, sig_orig, color=Config.COLOR_SIGNAL, linewidth=Config.LINEWIDTH_SIGNAL, zorder=2)
    ax.vlines(t[random_miss_idx], y_lim[0] * 0.9, y_lim[1] * 0.9,
              color=Config.COLOR_RANDOM_MARKER, linewidth=1.2, alpha=0.7, zorder=3)
    _style_ax(ax, xlabel='时间 (s)', ylabel=r'加速度 ($\rm{m/s^2}$)',
              xlim=[0, Config.DURATION], ylim=y_lim)
    missing_rate_random = missing_mask_random.sum() / len(missing_mask_random) * 100
    ax.set_title(f'随机单点缺失  (缺失率 {missing_rate_random:.2f}%)',
                 fontproperties=CN_FONT, fontsize=Config.TITLE_FONTSIZE, loc='left', pad=4)
    patch_miss = mpatches.Patch(color=Config.COLOR_RANDOM_MARKER, alpha=0.7, label='缺失点位置')
    ax.legend(handles=[patch_miss], prop=CN_FONT, fontsize=Config.TS_FONTSIZE,
              loc='upper right', framealpha=0.8)

    ax = axes[1, 1]
    _plot_psd_pair(ax, freqs, psd_orig, psd_random, show_legend=True, show_xlabel=True)
    ax.set_title('功率谱密度对比',
                 fontproperties=CN_FONT, fontsize=Config.TITLE_FONTSIZE, loc='left', pad=4)

    return fig


# ==================== 主函数 ====================
def main():
    data = _prepare_data()
    fig1 = plot_fig1(data)
    fig2 = plot_fig2(data)

    from src.visualize_tools.utils import PlotLib
    ploter = PlotLib()
    ploter.figs.extend([fig1, fig2])
    ploter.show_web(
        page='fig2_9 数据缺失类型',
        cols=2,
        titles=['原始信号 + 突发缺失', '周期缺失 + 随机缺失'],
    )


if __name__ == '__main__':
    main()
