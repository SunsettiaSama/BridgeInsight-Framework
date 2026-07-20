"""图4-30：风雨振前50阶主频累积能量收敛。

样式对齐 fig4_12 / fig4_20。样本池与 fig4_25 共用（合并副本或仅 DL）；
能量定义与 VIV 一致：全谱 PSD 按幅值排序后的累积占比。
y=85% 水平线；三类收敛阶数以图例标注（涡激 / 风雨振 / 随机）。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_processer.io_unpacker import UNPACK
from src.chapter4_characteristics._bootstrap import ensure_paths

ensure_paths()
from src.figure_paintings.figs_for_thesis.Chapter4._rwiv_pipeline import (
    RWIV_SAMPLE_COPY_PATH,
    USE_MERGED_DATASET,
    add_dataset_switch_args,
    load_rwiv_samples_for_figures,
    resolve_use_merged,
)
from src.figure_paintings.figs_for_thesis.Chapter4.fig4_25_rwiv_timeseries import (
    Config as SharedConfig,
)
from src.figure_paintings.figs_for_thesis.Chapter4._viv_pipeline import (
    psd_ranked_energy_cumsum,
)
from src.figure_paintings.figs_for_thesis.config import (
    ANNOTATION_COLOR,
    CN_FONT,
    FONT_SIZE,
    REC_FIG_SIZE,
    VIV_INPLANE_COLOR,
    VIV_OUTPLANE_COLOR,
)
from src.visualize_tools.web_dashboard import push as web_push


class Config:
    FS = SharedConfig.FS
    WINDOW_SIZE = SharedConfig.WINDOW_SIZE

    N_MODES = 50
    NFFT_LIST = (128,)
    ENERGY_LEVEL = 0.85  # 达到该累积能量占比的阶序

    FIG_SIZE = REC_FIG_SIZE
    GRID_COLOR = "gray"
    GRID_ALPHA = 0.3
    GRID_LINESTYLE = "--"
    SHADE_ALPHA = 0.18
    LINE_WIDTH = 2.2

    INPLANE_COLOR = VIV_INPLANE_COLOR
    OUTPLANE_COLOR = VIV_OUTPLANE_COLOR

    # 三类 85% 收敛点：涡激 / 风雨振 / 随机
    STAIR_VIV_COLOR = "#4A4A4A"
    STAIR_RWIV_COLOR = VIV_INPLANE_COLOR
    STAIR_NORMAL_COLOR = "#B8B8B8"
    STAIR_LEVEL_COLOR = ANNOTATION_COLOR
    STAIR_MARKER_SIZE = 70
    STAIR_MARKERS = {"viv": "^", "rwiv": "o", "normal": "D"}

    SNAPSHOT_DIR = (
        project_root
        / "results"
        / "chapter4_characteristics"
        / "figure_snapshots"
    )
    # 与 fig4_12 / fig4_20 对应的累积能量快照（文件名沿用生成时编号）
    NORMAL_CUMSUM_SNAPSHOT = SNAPSHOT_DIR / "fig4_11_normal_vib_energy_cumsum_nfft128.npz"
    VIV_CUMSUM_SNAPSHOT = SNAPSHOT_DIR / "fig4_22_viv_energy_cumsum_nfft128.npz"
    WEB_PAGE = "fig4_30 风雨振累积能量"


def _snapshot_path(nfft: int) -> Path:
    return Config.SNAPSHOT_DIR / f"fig4_30_rwiv_energy_cumsum_nfft{nfft}.npz"


def _snapshot_config(nfft: int, use_merged: bool) -> dict:
    return {
        "figure": "fig4_30_rwiv_energy_cumsum",
        "nfft": int(nfft),
        "n_modes": Config.N_MODES,
        "window_size": int(Config.WINDOW_SIZE),
        "fs": float(Config.FS),
        "use_merged": bool(use_merged),
        "energy_definition": "ranked_psd_bin_linear_sum",
        "sample_copy": str(RWIV_SAMPLE_COPY_PATH) if use_merged else "dl_only",
    }


def load_snapshot(nfft: int, use_merged: bool, force_refresh: bool) -> dict | None:
    path = _snapshot_path(nfft)
    if force_refresh or not path.exists():
        return None

    payload = np.load(path, allow_pickle=True)
    saved_cfg = json.loads(str(payload["config_json"]))
    if saved_cfg != _snapshot_config(nfft, use_merged):
        print(f"  快照参数不匹配，将重新计算：{path}")
        return None

    cumsum_in = np.asarray(payload["cumsum_in"], dtype=np.float64)
    cumsum_out = np.asarray(payload["cumsum_out"], dtype=np.float64)
    print(f"  读取结果快照：{path}")
    print(
        f"    created_at={payload['created_at']}  "
        f"n_in={cumsum_in.shape[0]}  n_out={cumsum_out.shape[0]}"
    )
    return {
        "cumsum_in": [row for row in cumsum_in],
        "cumsum_out": [row for row in cumsum_out],
        "nfft": int(nfft),
    }


def save_snapshot(data: dict, use_merged: bool) -> None:
    nfft = int(data["nfft"])
    path = _snapshot_path(nfft)
    path.parent.mkdir(parents=True, exist_ok=True)

    mat_in = (
        np.stack(data["cumsum_in"]).astype(np.float64)
        if data["cumsum_in"]
        else np.empty((0, Config.N_MODES))
    )
    mat_out = (
        np.stack(data["cumsum_out"]).astype(np.float64)
        if data["cumsum_out"]
        else np.empty((0, Config.N_MODES))
    )

    np.savez_compressed(
        path,
        created_at=np.asarray(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        config_json=np.asarray(
            json.dumps(_snapshot_config(nfft, use_merged), ensure_ascii=False)
        ),
        nfft=np.asarray(nfft, dtype=np.int32),
        cumsum_in=mat_in,
        cumsum_out=mat_out,
    )
    print(f"  写出结果快照：{path}")


def _slice_window(raw: np.ndarray, window_idx: int) -> np.ndarray | None:
    start = window_idx * Config.WINDOW_SIZE
    end = start + Config.WINDOW_SIZE
    if end > len(raw):
        return None
    return raw[start:end]


def compute_psd_modes(samples: list[dict], nfft: int) -> dict:
    jobs = [
        (s["inplane_file_path"], s["outplane_file_path"], int(s["window_idx"]))
        for s in samples
    ]
    n = len(jobs)
    curves_in: list[np.ndarray | None] = [None] * n
    curves_out: list[np.ndarray | None] = [None] * n

    by_path: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for i, (in_path, out_path, window_idx) in enumerate(jobs):
        by_path[in_path].append((window_idx, i, "in"))
        by_path[out_path].append((window_idx, i, "out"))

    unpacker = UNPACK(init_path=False)
    n_files = len(by_path)
    print(f"  [nfft={nfft}] 待处理样本：{n}，唯一 VIC 文件：{n_files}")

    for file_i, (path, items) in enumerate(by_path.items(), start=1):
        if not Path(path).exists():
            raise FileNotFoundError(f"VIC 文件不存在：{path}")
        raw = np.asarray(unpacker.VIC_DATA_Unpack(str(path)), dtype=np.float64)
        for window_idx, sample_i, side in items:
            sig = _slice_window(raw, window_idx)
            if sig is None:
                raise ValueError(f"窗口越界：path={path} win={window_idx} len={len(raw)}")
            curve = psd_ranked_energy_cumsum(
                sig,
                n_modes=Config.N_MODES,
                fs=Config.FS,
                nfft=nfft,
            )
            if curve is None:
                raise ValueError(f"累积能量计算失败：path={path} win={window_idx}")
            if side == "in":
                curves_in[sample_i] = curve
            else:
                curves_out[sample_i] = curve

        if file_i % 50 == 0 or file_i == n_files:
            print(f"    [nfft={nfft}] 已解包文件 {file_i}/{n_files}")

    return {
        "cumsum_in": [c for c in curves_in if c is not None],
        "cumsum_out": [c for c in curves_out if c is not None],
        "nfft": nfft,
    }


def load_or_compute(
    nfft: int,
    use_merged: bool,
    force_refresh: bool,
    refresh_sample_copy: bool,
) -> dict:
    cached = load_snapshot(nfft, use_merged=use_merged, force_refresh=force_refresh)
    if cached is not None:
        return cached

    print(f"  未命中快照，开始重算 nfft={nfft} ...")
    if use_merged:
        print(f"  副本路径：{RWIV_SAMPLE_COPY_PATH}")
    samples = load_rwiv_samples_for_figures(
        use_merged=use_merged,
        force_refresh=refresh_sample_copy,
    )
    print(f"  配对样本：{len(samples)}")
    data = compute_psd_modes(samples, nfft=nfft)
    save_snapshot(data, use_merged=use_merged)
    return data


def _aggregate(curves: list[np.ndarray], n_modes: int) -> dict:
    mat = np.full((len(curves), n_modes), np.nan)
    for i, c in enumerate(curves):
        length = min(len(c), n_modes)
        mat[i, :length] = c[:length]

    mean = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    return {"mean": mean, "std": std, "n": len(curves)}


def _mean_curve_from_snapshot(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"找不到参考累积能量快照：{path}")
    payload = np.load(path, allow_pickle=True)
    mean_in = np.nanmean(np.asarray(payload["cumsum_in"], dtype=np.float64), axis=0)
    mean_out = np.nanmean(np.asarray(payload["cumsum_out"], dtype=np.float64), axis=0)
    return 0.5 * (mean_in + mean_out)


def _first_rank_at_level(mean_curve: np.ndarray, level: float) -> tuple[int, float]:
    idx = np.where(mean_curve >= level)[0]
    if len(idx) == 0:
        raise ValueError(
            f"累积能量曲线未达到 {level:.0%}（末值={float(mean_curve[-1]):.4f}）"
        )
    k = int(idx[0]) + 1
    return k, float(mean_curve[idx[0]])


def build_convergence_stairs(nfft: int, rwiv_data: dict) -> list[dict]:
    """组装三类达到 ENERGY_LEVEL 的阶序：涡激 / 风雨振 / 随机。"""
    if int(nfft) != 128:
        raise ValueError(f"当前仅配置了 nfft=128 的参考快照，收到 nfft={nfft}")

    stats_in = _aggregate(rwiv_data["cumsum_in"], Config.N_MODES)
    stats_out = _aggregate(rwiv_data["cumsum_out"], Config.N_MODES)
    rwiv_mean = 0.5 * (stats_in["mean"] + stats_out["mean"])
    rwiv_rank, rwiv_value = _first_rank_at_level(rwiv_mean, Config.ENERGY_LEVEL)

    specs = [
        {
            "key": "viv",
            "label": "涡激共振",
            "color": Config.STAIR_VIV_COLOR,
            "curve": _mean_curve_from_snapshot(Config.VIV_CUMSUM_SNAPSHOT),
            "source": Config.VIV_CUMSUM_SNAPSHOT.name,
        },
        {
            "key": "rwiv",
            "label": "风雨振",
            "color": Config.STAIR_RWIV_COLOR,
            "curve": rwiv_mean,
            "source": "本图风雨振面内外均值",
        },
        {
            "key": "normal",
            "label": "随机振动",
            "color": Config.STAIR_NORMAL_COLOR,
            "curve": _mean_curve_from_snapshot(Config.NORMAL_CUMSUM_SNAPSHOT),
            "source": Config.NORMAL_CUMSUM_SNAPSHOT.name,
        },
    ]

    stairs: list[dict] = []
    for spec in specs:
        if spec["key"] == "rwiv":
            rank, value = rwiv_rank, rwiv_value
        else:
            rank, value = _first_rank_at_level(spec["curve"], Config.ENERGY_LEVEL)
        item = {
            "key": spec["key"],
            "label": spec["label"],
            "color": spec["color"],
            "rank": rank,
            "value": value,
        }
        stairs.append(item)
        print(
            f"  [{item['label']}] {spec['source']}："
            f"首次 ≥{Config.ENERGY_LEVEL:.0%} → 第 {rank} 阶（实际={value:.4f}）"
        )

    stairs.sort(key=lambda d: d["rank"])
    return stairs


def _apply_grid(ax) -> None:
    ax.grid(True, color=Config.GRID_COLOR, alpha=Config.GRID_ALPHA, linestyle=Config.GRID_LINESTYLE)
    ax.set_axisbelow(True)


def _add_legend(ax) -> None:
    leg = ax.legend(fontsize=FONT_SIZE - 2, framealpha=0.9, prop=CN_FONT)
    for text in leg.get_texts():
        text.set_fontproperties(CN_FONT)


def _auto_ylim(
    mean_in: np.ndarray,
    std_in: np.ndarray,
    mean_out: np.ndarray,
    std_out: np.ndarray,
) -> float:
    upper = max(
        float(np.nanmax(mean_in + std_in)),
        float(np.nanmax(mean_out + std_out)),
    )
    return min(1.05, max(0.35, upper * 1.08))


def _draw_convergence_markers(ax, stairs: list[dict]) -> None:
    """y=ENERGY_LEVEL 水平线 + 三点；收敛阶数写入图例。"""
    level = Config.ENERGY_LEVEL
    ax.axhline(
        y=level,
        color=Config.STAIR_LEVEL_COLOR,
        linewidth=1.2,
        linestyle="-.",
        alpha=0.75,
        zorder=3,
        label=f"{level:.0%} 能量水平",
    )
    for item in stairs:
        key = item["key"]
        ax.scatter(
            [int(item["rank"])],
            [level],
            s=Config.STAIR_MARKER_SIZE,
            marker=Config.STAIR_MARKERS[key],
            color=item["color"],
            edgecolors="white",
            linewidths=0.8,
            zorder=5,
            label=f'{item["label"]}（第{int(item["rank"])}阶）',
        )


def plot_energy_cumsum(data: dict, stairs: list[dict]) -> plt.Figure:
    n = Config.N_MODES
    nfft = int(data["nfft"])
    stats_in = _aggregate(data["cumsum_in"], n)
    stats_out = _aggregate(data["cumsum_out"], n)

    x = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=Config.FIG_SIZE)

    mean_in = stats_in["mean"]
    std_in = stats_in["std"]
    ax.plot(
        x,
        mean_in,
        color=Config.INPLANE_COLOR,
        linewidth=Config.LINE_WIDTH,
        marker="o",
        markersize=4,
        label=f'面内（n={stats_in["n"]:,}）',
    )
    ax.fill_between(
        x,
        np.clip(mean_in - std_in, 0, None),
        mean_in + std_in,
        color=Config.INPLANE_COLOR,
        alpha=Config.SHADE_ALPHA,
    )

    mean_out = stats_out["mean"]
    std_out = stats_out["std"]
    ax.plot(
        x,
        mean_out,
        color=Config.OUTPLANE_COLOR,
        linewidth=Config.LINE_WIDTH,
        marker="s",
        markersize=4,
        label=f'面外（n={stats_out["n"]:,}）',
    )
    ax.fill_between(
        x,
        np.clip(mean_out - std_out, 0, None),
        mean_out + std_out,
        color=Config.OUTPLANE_COLOR,
        alpha=Config.SHADE_ALPHA,
    )

    _draw_convergence_markers(ax, stairs)

    ax.set_xlim(0.5, n + 0.5)
    ax.set_ylim(0, _auto_ylim(mean_in, std_in, mean_out, std_out))
    ax.set_xticks(np.arange(0, n + 1, 5))
    ax.set_xlabel("主频阶序", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_ylabel("累积能量占比", labelpad=10, fontproperties=CN_FONT, fontsize=FONT_SIZE)
    ax.set_title(
        f"风雨振前{n}阶主频累积能量分布（nfft={nfft}）",
        fontproperties=CN_FONT,
        fontsize=FONT_SIZE,
        pad=14,
    )
    ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE - 2)
    _add_legend(ax)
    _apply_grid(ax)
    plt.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="图4-30 风雨振前50阶累积能量（支持快照）")
    add_dataset_switch_args(parser)
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="忽略已有快照，强制从原始 VIC 重算并覆盖快照",
    )
    args = parser.parse_args()
    use_merged = resolve_use_merged(args.use_merged)

    print("=" * 80)
    print("图4-30 风雨振前50阶主频累积能量分布（nfft=128）")
    print(f"  默认开关 USE_MERGED_DATASET={USE_MERGED_DATASET}  → 本次 use_merged={use_merged}")
    print(f"  分辨率：{Config.NFFT_LIST}")
    print(f"  快照目录：{Config.SNAPSHOT_DIR}")
    if args.refresh_cache:
        print("  模式：--refresh-cache（强制重算）")
    else:
        print("  模式：优先读快照，缺失才重算")
    print("=" * 80)

    for slot, nfft in enumerate(Config.NFFT_LIST):
        print("\n" + "-" * 80)
        print(f"[分辨率 nfft={nfft}]")
        print("-" * 80)
        print(f"\n[步骤1] 加载/计算 PSD 累积能量（全谱 PSD 线性求和，nfft={nfft}）...")
        data = load_or_compute(
            nfft=nfft,
            use_merged=use_merged,
            force_refresh=args.refresh_cache,
            refresh_sample_copy=args.refresh_sample_copy,
        )

        n_in = len(data["cumsum_in"])
        n_out = len(data["cumsum_out"])
        print(f"[OK] 面内有效样本：{n_in}，面外有效样本：{n_out}")

        stats_in = _aggregate(data["cumsum_in"], Config.N_MODES)
        stats_out = _aggregate(data["cumsum_out"], Config.N_MODES)
        print(
            f"  面内  第1阶占比均值：{stats_in['mean'][0]:.4f}  "
            f"前50阶累积均值：{stats_in['mean'][-1]:.4f}"
        )
        print(
            f"  面外  第1阶占比均值：{stats_out['mean'][0]:.4f}  "
            f"前50阶累积均值：{stats_out['mean'][-1]:.4f}"
        )

        print(f"\n[步骤2] 组装三类 {Config.ENERGY_LEVEL:.0%} 能量收敛阶梯...")
        stairs = build_convergence_stairs(nfft=nfft, rwiv_data=data)

        print(f"\n[步骤3] 绘制图像（nfft={nfft}）...")
        fig = plot_energy_cumsum(data, stairs)
        web_push(
            fig,
            page=Config.WEB_PAGE,
            slot=slot,
            title=f"风雨振前50阶累积能量 nfft={nfft}",
            page_cols=1,
        )
        plt.close(fig)
        print(f"[OK] 已推送到 WebUI：{Config.WEB_PAGE} / slot={slot}")

    print("=" * 80)
    print("完成")


if __name__ == "__main__":
    main()
