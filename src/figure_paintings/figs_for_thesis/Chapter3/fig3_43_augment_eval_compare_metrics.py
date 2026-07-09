from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.figure_paintings.figs_for_thesis.config import (
    CN_FONT,
    ENG_FONT,
    SQUARE_FIG_SIZE,
    SQUARE_FONT_SIZE,
    get_full_color_map,
)
from src.visualize_tools.web_dashboard import DEFAULT_PORT as DASHBOARD_DEFAULT_PORT
from src.visualize_tools.web_dashboard import push as web_push

DEFAULT_COMPARE_REPORT = (
    PROJECT_ROOT / "results" / "augment_eval_compare" / "rounds" / "round_09" / "compare_report.json"
)
OUTPUT_DIR = PROJECT_ROOT / "results" / "figure_paintings" / "figs_for_thesis" / "Chapter3"
FIGURE_NAME = "fig3_43_augment_eval_compare_metrics.png"
WEBUI_PAGE = "fig3_43 解耦vs联合模型指标对比"

CLASS_NAMES = ("Normal", "VIV", "RWIV", "Others")
X_LABELS = (*CLASS_NAMES, "双面命中")
MODEL_LABELS = ("ResNet", "四特征通道ResNet")
DIRECTION_SPECS = (
    ("inplane", "面内"),
    ("outplane", "面外"),
)
_PALETTE = get_full_color_map(style="discrete").colors
DECOUPLED_COLOR = _PALETTE[7]
JOINT_COLOR = _PALETTE[0]


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _round_idx_from_checkpoint(path_text: str | None) -> int | None:
    if not path_text:
        return None
    for part in Path(path_text).parts:
        if part.startswith("round_"):
            suffix = part.removeprefix("round_")
            if suffix.isdigit():
                return int(suffix)
    return None


def validate_report_scope(report: dict[str, Any]) -> None:
    report_round = int(report["round_idx"])
    mismatches: list[str] = []
    for block_name in ("decoupled", "joint"):
        ckpt_round = _round_idx_from_checkpoint(report.get(block_name, {}).get("checkpoint"))
        if ckpt_round is not None and ckpt_round != report_round:
            mismatches.append(f"{block_name}=round_{ckpt_round:02d}")
    if mismatches:
        raise ValueError(
            f"compare_report 的 checkpoint round 与报告 round_{report_round:02d} 不一致："
            + ", ".join(mismatches)
            + "。请先重跑同一 round / 同一评估语义下的 augment_eval_compare。"
        )


def _metric(dm: dict[str, float], direction: str, cls: str, suffix: str) -> float:
    return float(dm[f"{direction}_{cls}{suffix}"])


def extract_direction_series(model_block: dict[str, Any], direction: str) -> dict[str, np.ndarray]:
    dm = model_block["direction_metrics"]
    f1 = [_metric(dm, direction, cls, "_f1") for cls in CLASS_NAMES]
    pre = [_metric(dm, direction, cls, "_precision") for cls in CLASS_NAMES]
    rec = [_metric(dm, direction, cls, "_recall") for cls in CLASS_NAMES]
    pair_hit = float(model_block["pair_joint_accuracy"])
    return {
        "f1": np.asarray(f1 + [pair_hit], dtype=float),
        "precision": np.asarray(pre + [np.nan], dtype=float),
        "recall": np.asarray(rec + [np.nan], dtype=float),
    }


def setup_axis(ax: plt.Axes) -> None:
    ax.set_ylabel("分数", fontproperties=CN_FONT, fontsize=SQUARE_FONT_SIZE - 8)
    ax.set_ylim(0.0, 1.08)
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=SQUARE_FONT_SIZE - 11)


def _annotate_f1_bar(ax: plt.Axes, x: float, f1: float) -> None:
    if not np.isfinite(f1):
        return
    ax.text(
        x,
        min(f1 + 0.014, 1.03),
        f"{f1:.2f}",
        ha="center",
        va="bottom",
        fontsize=SQUARE_FONT_SIZE - 12,
        fontproperties=ENG_FONT,
        color="#303030",
    )


def plot_direction_panel(
    ax: plt.Axes,
    decoupled: dict[str, np.ndarray],
    joint: dict[str, np.ndarray],
) -> None:
    n_groups = len(X_LABELS)
    x = np.arange(n_groups, dtype=float)
    bar_width = 0.34

    dec_f1 = decoupled["f1"]
    joint_f1 = joint["f1"]
    ax.bar(x - bar_width / 2, dec_f1, width=bar_width, color=DECOUPLED_COLOR, label=MODEL_LABELS[0], zorder=3)
    ax.bar(x + bar_width / 2, joint_f1, width=bar_width, color=JOINT_COLOR, label=MODEL_LABELS[1], zorder=3)

    for idx in range(len(CLASS_NAMES)):
        _annotate_f1_bar(ax, x[idx] - bar_width / 2, dec_f1[idx])
        _annotate_f1_bar(ax, x[idx] + bar_width / 2, joint_f1[idx])

    acc_idx = len(CLASS_NAMES)
    for bar_x, value in (
        (x[acc_idx] - bar_width / 2, dec_f1[acc_idx]),
        (x[acc_idx] + bar_width / 2, joint_f1[acc_idx]),
    ):
        ax.text(
            bar_x,
            min(value + 0.012, 1.02),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=SQUARE_FONT_SIZE - 13,
            fontproperties=ENG_FONT,
            color="#303030",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(X_LABELS, fontproperties=CN_FONT, fontsize=SQUARE_FONT_SIZE - 11)
    setup_axis(ax)


def build_figures(report: dict[str, Any]) -> list[tuple[str, plt.Figure]]:
    dec = report["decoupled"]
    joint = report["joint"]
    figures: list[tuple[str, plt.Figure]] = []

    for direction, title_suffix in DIRECTION_SPECS:
        fig, ax = plt.subplots(figsize=SQUARE_FIG_SIZE)
        dec_series = extract_direction_series(dec, direction)
        joint_series = extract_direction_series(joint, direction)
        plot_direction_panel(ax, dec_series, joint_series)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=2,
            frameon=False,
            prop=CN_FONT,
            fontsize=SQUARE_FONT_SIZE - 10,
        )
        fig.tight_layout()
        figures.append((title_suffix, fig))
    return figures


def save_figures(figures: list[tuple[str, plt.Figure]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for title, fig in figures:
        suffix = "inplane" if title == "面内" else "outplane"
        path = output_dir / FIGURE_NAME.replace(".png", f"_{suffix}.png")
        fig.savefig(path, dpi=240, bbox_inches="tight")
        print(f"saved: {path}")
        saved_paths.append(path)
    return saved_paths


def save_plot_payload(report: dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fig3_43_augment_eval_compare_metrics.json"
    payload = {
        "round_idx": report.get("round_idx"),
        "eval_split": report.get("eval_split"),
        "eval_pair_count": report.get("eval_pair_count"),
        "decoupled": {
            direction: extract_direction_series(report["decoupled"], direction)
            for direction, _ in DIRECTION_SPECS
        },
        "joint": {
            direction: extract_direction_series(report["joint"], direction)
            for direction, _ in DIRECTION_SPECS
        },
        "delta": report.get("delta", {}),
    }
    cleaned = json.loads(json.dumps(payload, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else v))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    print(f"saved: {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制 augment_eval_compare 解耦 vs 联合模型指标对比图")
    parser.add_argument("--report", type=Path, default=DEFAULT_COMPARE_REPORT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--port", type=int, default=DASHBOARD_DEFAULT_PORT)
    parser.add_argument("--no-web", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = load_json(args.report)
    validate_report_scope(report)
    figures = build_figures(report)
    output_paths = save_figures(figures, args.output_dir)
    save_plot_payload(report, args.output_dir)

    if not args.no_web:
        for slot, (title, fig) in enumerate(figures):
            web_push(
                fig,
                page=WEBUI_PAGE,
                slot=slot,
                title=f"fig3_43 {title} F1 对比",
                port=args.port,
                page_cols=2 if slot == 0 else None,
            )
        print(f"pushed to VibDash page: {WEBUI_PAGE}")
    for _, fig in figures:
        plt.close(fig)
    print("figures:")
    for output_path in output_paths:
        print(f"  {output_path}")


if __name__ == "__main__":
    main()
