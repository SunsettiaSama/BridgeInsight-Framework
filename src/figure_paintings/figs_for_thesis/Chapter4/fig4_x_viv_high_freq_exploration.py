import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent.parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.chapter4_characteristics.feature_analysis.entry import ensure_enriched_for_figures
from src.data_processer.io_unpacker import UNPACK
from src.figure_paintings.figs_for_thesis.Chapter4._data_loader import get_enriched_class_dir, iter_enriched_json_files
from src.figure_paintings.figs_for_thesis.Chapter4.fig4_x_viv_low_freq_exploration import (
    GROUPS,
    plot_spectrum_grid,
    plot_timeseries_grid,
    plot_trajectory_grid,
)
from src.visualize_tools.web_dashboard import push as web_push


class Config:
    VIV_CLASS_ID = 1
    FEATURE_BATCH_SIZE = 512
    FREQ_THRESHOLD = 7.0
    NUM_SAMPLES_PER_GROUP = 20
    RANDOM_SEED = 52

    ENRICHED_STATS_DIR = get_enriched_class_dir(1)
    SELECTION_SNAPSHOT_PATH = (
        project_root
        / "results"
        / "chapter4_characteristics"
        / "figure_snapshots"
        / "fig4_22_viv_high_freq_selection.json"
    )
    WEB_PAGE = "fig4_x VIV高频样本探索"


def _load_sample_lookup() -> dict[int, dict]:
    ensure_enriched_for_figures(class_id=Config.VIV_CLASS_ID, batch_size=Config.FEATURE_BATCH_SIZE)
    json_files = iter_enriched_json_files(Config.ENRICHED_STATS_DIR)
    if not json_files:
        raise FileNotFoundError(f"目录下无 JSON 文件：{Config.ENRICHED_STATS_DIR}")

    samples_by_id: dict[int, dict] = {}
    for json_file in json_files:
        print(f"  加载：{json_file.name}")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        for sample in data["samples"]:
            sample_idx = sample.get("sample_idx")
            if sample_idx is not None:
                samples_by_id[int(sample_idx)] = sample
    return samples_by_id


def load_high_freq_groups() -> dict[str, list[dict]]:
    if not Config.SELECTION_SNAPSHOT_PATH.exists():
        raise FileNotFoundError(
            f"高频筛选快照不存在：{Config.SELECTION_SNAPSHOT_PATH}\n"
            "请先运行：python -m src.figure_paintings.figs_for_thesis.Chapter4.fig4_22_viv_high_freq"
        )

    with open(Config.SELECTION_SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    group_ids = payload.get("group_sample_ids") or {}
    samples_by_id = _load_sample_lookup()

    groups: dict[str, list[dict]] = {name: [] for name in GROUPS}
    for group in GROUPS:
        for sample_idx in group_ids.get(group, []):
            sample = samples_by_id.get(int(sample_idx))
            if sample is not None:
                groups[group].append(sample)

    for group, samples in groups.items():
        print(f"  {GROUPS[group]}：{len(samples)} 个高频样本")
    return groups


def random_sample(samples: list[dict], group: str) -> list[dict]:
    n = len(samples)
    k = min(Config.NUM_SAMPLES_PER_GROUP, n)
    if k == 0:
        return []
    rng = np.random.default_rng(Config.RANDOM_SEED + list(GROUPS).index(group))
    chosen = sorted(rng.choice(n, size=k, replace=False).tolist())
    print(f"  {GROUPS[group]} 抽样索引：{chosen}（共 {n} 个中选 {k} 个）")
    return [samples[i] for i in chosen]


def main() -> None:
    print("=" * 80)
    print(f"图4-x 高频 VIV 样本探索图（主导模态 >= {Config.FREQ_THRESHOLD:g} Hz）")
    print("=" * 80)

    print("\n[步骤1] 读取高频筛选快照并加载 enriched 样本...")
    groups = load_high_freq_groups()
    sampled_groups = {group: random_sample(samples, group) for group, samples in groups.items()}

    print("\n[步骤2] 绘制探索图...")
    unpacker = UNPACK(init_path=False)
    cache: dict[str, np.ndarray] = {}
    slot = 0
    for group, samples in sampled_groups.items():
        if not samples:
            print(f"  跳过 {GROUPS[group]}：无样本")
            continue

        group_title = GROUPS[group]
        figures = [
            (plot_timeseries_grid(samples, f"{group_title} - 高频时程图", unpacker, cache), "时程图"),
            (plot_spectrum_grid(samples, f"{group_title} - 高频频谱图", unpacker, cache), "频谱图"),
            (plot_trajectory_grid(samples, f"{group_title} - 高频轨迹图", unpacker, cache), "轨迹图"),
        ]
        for fig, fig_type in figures:
            web_push(
                fig,
                page=Config.WEB_PAGE,
                slot=slot,
                title=f"{group_title} {fig_type}",
                page_cols=3 if slot == 0 else None,
            )
            plt.close(fig)
            slot += 1
        print(f"  [OK] {group_title} 已推送 3 张图")

    print(f"[OK] 已推送到 WebUI：{Config.WEB_PAGE}，共 {slot} 张图")


if __name__ == "__main__":
    main()
