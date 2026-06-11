from __future__ import annotations

import json
import sys
from pathlib import Path


def _prepend_project_root() -> None:
    root = Path(__file__).resolve().parents[3]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def run_smoke_tests() -> None:
    _prepend_project_root()
    from src.chapter3_identifier.augment._bootstrap import ensure_paths
    from src.chapter3_identifier.augment.annotation.gold_index import annotation_key
    from src.chapter3_identifier.augment.annotation.store import (
        annotation_store_for_round,
        load_cumulative_manual_edits,
    )
    from src.chapter3_identifier.augment.features.spectrum import welch_psd
    from src.chapter3_identifier.augment.settings import (
        get_round_manual_edits_path,
        get_round_merged_training_path,
        load_config,
    )
    from src.chapter3_identifier.augment.smoke_fixtures.bootstrap import ensure_smoke_fixtures, smoke_config_path
    from src.chapter3_identifier.augment.train.run import resolve_gold_only
    from src.chapter3_identifier.augment.figures.render.context import plot_long_spectrogram

    import numpy as np

    ensure_paths()
    ensure_smoke_fixtures(force=True)
    cfg_path = str(smoke_config_path())
    cfg = load_config(cfg_path)

    passed = 0

    short = np.random.default_rng(0).standard_normal(100)
    f, psd = welch_psd(short, fs=50.0, nfft=2048, freq_max_hz=25.0)
    assert len(f) == len(psd) and len(f) > 0
    passed += 1
    print("[ok] welch_psd 短片段")

    long_signal = np.random.default_rng(1).standard_normal(7000)
    png = plot_long_spectrogram(long_signal, 50.0, 2.0, 10.0, 70.0, "smoke")
    assert len(png) > 100
    passed += 1
    print("[ok] plot_long_spectrogram 短分段")

    round1_store = annotation_store_for_round(cfg, 1)
    inference_path = Path(cfg["rounds_output_dir"]) / "round_01" / "inference.json"
    record = json.loads(inference_path.read_text(encoding="utf-8"))["records"][0]
    in_fp = record["inplane_file_path"]
    wi = int(record["window_index"])

    row = round1_store.upsert_manual(
        file_path=in_fp,
        window_index=wi,
        inplane_annotation=2,
        outplane_annotation=1,
        outplane_file_path=record.get("outplane_file_path"),
        round_idx=1,
    )
    assert row["inplane_annotation"] == 2
    assert row["outplane_annotation"] == 1
    manual_path = get_round_manual_edits_path(cfg, 1)
    assert manual_path.exists()
    manual_rows = json.loads(manual_path.read_text(encoding="utf-8"))
    assert len(manual_rows) == 1
    passed += 1
    print("[ok] round_01/manual_edits.json 写入")

    merged = round1_store.merge(gold_only=False, prior_manual=[])
    merged_path = get_round_merged_training_path(cfg, 1)
    assert merged_path.exists()
    assert len(merged) >= 2
    key = annotation_key(in_fp, wi)
    merged_map = {
        annotation_key(e["file_path"], e.get("window_index", 0)): e for e in merged
    }
    assert merged_map[key]["annotation"] == 2
    out_key = annotation_key(record["outplane_file_path"], wi)
    assert merged_map[out_key]["annotation"] == 1
    passed += 1
    print("[ok] round_01/merged_training.json 含人工标注")

    cumulative = load_cumulative_manual_edits(cfg, 1)
    assert len(cumulative) == 1
    passed += 1
    print("[ok] cumulative manual round 1")

    assert resolve_gold_only(1, cfg, None) is True
    assert resolve_gold_only(2, cfg, None) is False
    passed += 1
    print("[ok] round2+ 训练策略为金标+人工")

    round2_store = annotation_store_for_round(cfg, 2)
    round2_merged = round2_store.merge(
        gold_only=False,
        prior_manual=load_cumulative_manual_edits(cfg, 1),
    )
    assert len(round2_merged) >= 2
    assert merged_map[key]["annotation"] == 2
    passed += 1
    print("[ok] round2 混合训练集含 round1 人工标注")

    print(f"\nAugment smoke: {passed}/{passed} passed")


if __name__ == "__main__":
    run_smoke_tests()
