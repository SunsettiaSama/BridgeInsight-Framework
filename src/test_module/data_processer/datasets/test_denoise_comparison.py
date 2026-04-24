"""
去噪前后对比测试

对 Annotation 数据集中第2、3、4类别（class 1/2/3：VIV / RWIV / Transition）
各随机抽取10个样本，展示小波去噪前后的时域波形对比。
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import signal as scipy_signal

current_dir  = Path(__file__).parent
project_root = current_dir.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_processer.preprocess.get_data_vib import VICWindowExtractor
from src.visualize_tools.utils import PlotLib

# ==================== 配置 ====================
ANNOTATION_JSON = project_root / "results" / "dataset_annotation" / "annotation_results.json"
PROJECT_ROOT_MARKER = "Vibration Characteristics In Cable Vibration"

WINDOW_SIZE   = 3000
FS            = 50.0
N_SAMPLES     = 10
RANDOM_SEED   = 42

TARGET_CLASSES = {
    1: "VIV",
    2: "RWIV",
    3: "Transition",
}

RAW_COLOR      = "#606060"
DENOISED_COLOR = "#C03030"
LINEWIDTH      = 0.8

# PSD 参数（Welch，窗口 512 点，限 0–25 Hz）
NFFT       = 1024
FREQ_LIMIT = 25.0   # Nyquist @ 50 Hz

# 屏幕高度约束
SCREEN_DPI        = 100          # matplotlib 默认 DPI
MAX_HEIGHT_PX     = 1080
MAX_HEIGHT_IN     = MAX_HEIGHT_PX / SCREEN_DPI   # 10.8 英寸
PREFERRED_ROW_H   = 2.0          # 每行希望高度（英寸），行数多时会被压缩

plt.rcParams["font.sans-serif"] = ["Times New Roman", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False
FONT_SIZE = 9     # 行高压缩后字体稍小


# ==================== 工具函数 ====================
def _resolve_path(raw_path: str) -> Path:
    """将可能包含旧机器绝对路径的字符串解析为当前机器的绝对路径。"""
    p = Path(raw_path)
    if p.exists():
        return p
    # 尝试从 marker 后面的部分重建
    raw_str = raw_path.replace("\\", "/")
    marker = PROJECT_ROOT_MARKER
    idx = raw_str.find(marker)
    if idx != -1:
        rel = raw_str[idx + len(marker):].lstrip("/")
        resolved = project_root / rel
        if resolved.exists():
            return resolved
    raise FileNotFoundError(f"无法解析路径：{raw_path}")


def _load_annotation_json() -> list:
    if not ANNOTATION_JSON.exists():
        raise FileNotFoundError(f"标注文件不存在：{ANNOTATION_JSON}")
    with open(ANNOTATION_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _class_id_from_item(item: dict) -> int:
    annotation = str(item.get("annotation", "")).strip()
    if not annotation:
        return -1
    return int(annotation)


def _sample_by_class(records: list, target_classes: dict, n: int, seed: int) -> dict:
    """按类别分组后各随机抽 n 条，返回 {class_id: [record, ...]}。"""
    buckets: dict = {cid: [] for cid in target_classes}
    for item in records:
        cid = _class_id_from_item(item)
        if cid in buckets:
            buckets[cid].append(item)

    rng = np.random.default_rng(seed)
    sampled: dict = {}
    for cid, items in buckets.items():
        k = min(n, len(items))
        if k == 0:
            print(f"  [警告] 类别 {cid} ({TARGET_CLASSES[cid]}) 无样本，跳过")
            continue
        chosen = rng.choice(len(items), size=k, replace=False).tolist()
        sampled[cid] = [items[i] for i in sorted(chosen)]
        print(f"  类别 {cid} ({TARGET_CLASSES[cid]})：共 {len(items)} 条，抽取 {k} 条")
    return sampled


# ==================== 数据加载与去噪 ====================
def _load_pair(item: dict,
               extractor_raw: VICWindowExtractor,
               extractor_den: VICWindowExtractor) -> tuple:
    """
    对同一条标注记录分别用「不去噪」和「分层去噪」提取窗口。

    分层去噪逻辑（由 VICWindowExtractor 内部实现）：
      - 若窗口主频 > 全量数据主频的 95% 分位数 → 跳过去噪，返回原始窗口
      - 否则 → 应用小波去噪（db4, soft, sqtwolog, layer-wise threshold）
    主频来源优先级：metadata["dominant_freq_per_window"][window_index] → 实时 FFT fallback
    """
    file_path  = _resolve_path(item["file_path"])
    window_idx = int(item["window_index"])
    metadata   = item.get("metadata")          # 含 dominant_freq_per_window 等预处理信息

    vic_data = extractor_raw.load_file(str(file_path))

    raw = extractor_raw.extract_window_from_data(
        vic_data, window_idx, WINDOW_SIZE, metadata=metadata, file_path=str(file_path)
    )
    den = extractor_den.extract_window_from_data(
        vic_data, window_idx, WINDOW_SIZE, metadata=metadata, file_path=str(file_path)
    )
    # extract_window_from_data 可能返回 (N, 1) 的二维数组，统一压成一维
    return np.asarray(raw, dtype=float).ravel(), np.asarray(den, dtype=float).ravel()


# ==================== 频域辅助 ====================
def _compute_psd(data: np.ndarray):
    """返回 (f_masked, psd_masked)，限制在 0–FREQ_LIMIT Hz。"""
    nperseg = min(NFFT // 2, len(data))
    f, psd = scipy_signal.welch(
        data, fs=FS, nperseg=nperseg, noverlap=nperseg // 2, nfft=NFFT
    )
    mask = f <= FREQ_LIMIT
    return f[mask], psd[mask]


def _ax_style(ax, fontsize: int, last_row: bool,
              xlabel: str = "", ylabel: str = ""):
    ax.grid(True, color="gray", alpha=0.3, linewidth=0.35, linestyle="--")
    ax.tick_params(axis="both", labelsize=fontsize - 1)
    if last_row and xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize)


# ==================== 绘图 ====================
def _plot_class_comparison(class_id: int, class_name: str, items: list) -> plt.Figure:
    """
    布局：N 行 × 4 列
      Col 0  时域·原始        Col 1  时域·去噪后
      Col 2  PSD·原始         Col 3  PSD·去噪后

    图像竖向高度被约束在 MAX_HEIGHT_IN 英寸（≈ 1080 px @ 100 DPI）以内。
    """
    n = len(items)

    # ---- 尺寸计算 ----
    fig_h = min(PREFERRED_ROW_H * n, MAX_HEIGHT_IN)
    row_h = fig_h / n           # 实际每行高度（英寸）
    font  = max(6, int(row_h * 4.5))   # 行越小字越小，保底 6pt

    fig = plt.figure(figsize=(20, fig_h), dpi=SCREEN_DPI)
    fig.suptitle(
        f"Class {class_id} – {class_name}  |  Before / After Denoising  (n={n}, dpi={SCREEN_DPI})",
        fontsize=font + 1,
        fontweight="bold",
        y=1.002,
    )

    # 列标题行（第 0 行 ax 的上方写一次）
    col_labels = ["Time · Raw", "Time · Denoised", "PSD · Raw", "PSD · Denoised"]

    gs = gridspec.GridSpec(
        n, 4, figure=fig,
        hspace=0.55, wspace=0.32,
        left=0.05, right=0.98, top=0.96, bottom=0.04,
    )

    extractor_raw = VICWindowExtractor(enable_denoise=False)
    extractor_den = VICWindowExtractor(enable_denoise=True)
    time_axis     = np.arange(WINDOW_SIZE) / FS

    for row, item in enumerate(items):
        sensor_id  = item.get("sensor_id", "")
        time_str   = item.get("time", "")
        window_idx = int(item["window_index"])
        is_last    = row == n - 1

        raw, den = _load_pair(item, extractor_raw, extractor_den)

        diff_rms     = float(np.sqrt(np.mean((raw - den) ** 2)))
        was_denoised = diff_rms > 1e-10
        den_tag      = f"ΔRMS={diff_rms:.4f}" if was_denoised else "Skipped·high-freq"
        den_color    = "black" if was_denoised else "dimgray"

        f_raw,  p_raw  = _compute_psd(raw)
        f_den,  p_den  = _compute_psd(den)

        sample_label = f"[{row+1}] {sensor_id} win={window_idx}"

        # --- 时域 ---
        ax_t0 = fig.add_subplot(gs[row, 0])
        ax_t1 = fig.add_subplot(gs[row, 1])

        ax_t0.plot(time_axis, raw, color=RAW_COLOR,      linewidth=LINEWIDTH)
        ax_t1.plot(time_axis, den, color=DENOISED_COLOR, linewidth=LINEWIDTH)

        ax_t0.set_title(f"{sample_label}", fontsize=font, loc="left")
        ax_t1.set_title(den_tag, fontsize=font, color=den_color, loc="left")

        # 仅第一行在顶部注释列标题
        if row == 0:
            for ax_hdr, lbl in zip([ax_t0, ax_t1], col_labels[:2]):
                ax_hdr.set_title(lbl, fontsize=font, fontweight="bold", loc="center")

        _ax_style(ax_t0, font, is_last, xlabel="Time (s)", ylabel=r"$m/s^2$")
        _ax_style(ax_t1, font, is_last, xlabel="Time (s)")
        ax_t0.set_xlim(0, WINDOW_SIZE / FS)
        ax_t1.set_xlim(0, WINDOW_SIZE / FS)

        # --- 频域 ---
        ax_f0 = fig.add_subplot(gs[row, 2])
        ax_f1 = fig.add_subplot(gs[row, 3])

        ax_f0.plot(f_raw, p_raw, color=RAW_COLOR,      linewidth=LINEWIDTH)
        ax_f1.plot(f_den, p_den, color=DENOISED_COLOR, linewidth=LINEWIDTH)

        if row == 0:
            for ax_hdr, lbl in zip([ax_f0, ax_f1], col_labels[2:]):
                ax_hdr.set_title(lbl, fontsize=font, fontweight="bold", loc="center")

        _ax_style(ax_f0, font, is_last, xlabel="Freq (Hz)", ylabel=r"$(m/s^2)^2$/Hz")
        _ax_style(ax_f1, font, is_last, xlabel="Freq (Hz)")
        ax_f0.set_xlim(0, FREQ_LIMIT)
        ax_f1.set_xlim(0, FREQ_LIMIT)

    return fig


# ==================== 主函数 ====================
def main():
    print("=" * 70)
    print("去噪前后对比测试（Class 1/2/3 各随机 10 个样本）")
    print("=" * 70)

    print(f"\n[步骤1] 加载标注文件：{ANNOTATION_JSON.name}")
    records = _load_annotation_json()
    print(f"  共 {len(records)} 条标注记录")

    print("\n[步骤2] 按类别随机抽样...")
    sampled = _sample_by_class(records, TARGET_CLASSES, N_SAMPLES, RANDOM_SEED)

    print("\n[步骤3] 逐类别生成对比图...")
    ploter = PlotLib()
    for cid in sorted(sampled.keys()):
        class_name = TARGET_CLASSES[cid]
        items      = sampled[cid]
        print(f"  绘制 Class {cid} ({class_name})，{len(items)} 个样本...")
        fig = _plot_class_comparison(cid, class_name, items)
        ploter.figs.append(fig)
        print(f"    ✓ 完成")

    print(f"\n[步骤4] 展示（共 {len(ploter.figs)} 张图）...")
    ploter.show()


if __name__ == "__main__":
    main()
