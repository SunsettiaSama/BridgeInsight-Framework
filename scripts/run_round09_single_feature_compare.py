from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chapter3_identifier.augment.train.run import run_training
from src.chapter3_identifier.augment_eval_compare.dataset.eval_pairs import build_eval_dataset
from src.chapter3_identifier.augment_eval_compare.infer.run_compare import run_compare
from src.chapter3_identifier.augment_eval_compare.report.build_report import build_compare_report
from src.chapter3_identifier.augment_eval_compare.settings import load_compare_config


SOURCE_ROUNDS_DIR = ROOT / "results" / "augment" / "rounds"
BASELINE_ROOT = ROOT / "results" / "augment_eval_compare" / "single_feature_baseline"
BASELINE_ROUNDS_DIR = BASELINE_ROOT / "rounds"
GENERATED_DIR = BASELINE_ROOT / "generated_config"
ROUND_IDX = 9


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def prepare_baseline_artifacts() -> tuple[Path, Path, Path]:
    augment_default = ROOT / "src" / "chapter3_identifier" / "augment" / "config" / "default.yaml"
    compare_default = ROOT / "src" / "chapter3_identifier" / "augment_eval_compare" / "config" / "default.yaml"
    source_round = SOURCE_ROUNDS_DIR / f"round_{ROUND_IDX:02d}"
    baseline_round = BASELINE_ROUNDS_DIR / f"round_{ROUND_IDX:02d}"

    cfg = _load_yaml(augment_default)
    cfg["rounds_output_dir"] = str(BASELINE_ROUNDS_DIR)
    cfg["job_state_path"] = str(BASELINE_ROOT / "job_state.json")
    cfg["workflow_config_path"] = str(BASELINE_ROOT / "workflow_config.json")
    cfg["enable_preload_cache"] = False
    cfg["preload_num_workers"] = 0
    cfg["show_preload_progress"] = False
    cfg["dataloader_num_workers"] = 0
    generated_augment = GENERATED_DIR / "augment_round09_single_feature.yaml"
    _write_yaml(generated_augment, cfg)

    # Preserve the exact round09 eval-pair source and annotation state in the isolated baseline tree.
    for round_num in range(1, ROUND_IDX + 1):
        src_dir = SOURCE_ROUNDS_DIR / f"round_{round_num:02d}"
        dst_dir = BASELINE_ROUNDS_DIR / f"round_{round_num:02d}"
        for name in ("manual_edits.json", "manual_edits_delta.jsonl", "manual_edits_history.jsonl"):
            _copy_if_exists(src_dir / name, dst_dir / name)

    for name in ("merged_training_pair_key.json", "pair_key_migration_report.json"):
        _copy_if_exists(source_round / name, baseline_round / name)

    # The single-head trainer expects a previous-round checkpoint for round09.
    warm_start = SOURCE_ROUNDS_DIR / "round_01" / "best_checkpoint.pth"
    baseline_prev = BASELINE_ROUNDS_DIR / "round_08" / "best_checkpoint.pth"
    _copy_if_exists(warm_start, baseline_prev)

    source_profile = json.load((source_round / "train_profile.json").open("r", encoding="utf-8"))
    profile = {
        **source_profile,
        "schema_version": 2,
        "model_type": "dual_stream_single_head",
        "context_mode": "short_only",
        "enable_long_context": False,
        "enable_legacy_regularization": False,
        "enable_same_structure_regularization": False,
        "require_bidirectional_labels": False,
        "enable_gold_fill": False,
        "prediction_fill_mode": "off",
        "batch_size": min(int(source_profile.get("batch_size", 16)), 8),
    }
    generated_profile = GENERATED_DIR / "round09_single_feature_profile.json"
    _write_json(generated_profile, profile)

    compare_cfg = _load_yaml(compare_default)
    compare_cfg["augment_config"] = str(generated_augment)
    compare_cfg["decoupled_checkpoint"] = str(baseline_round / "best_checkpoint.pth")
    compare_cfg["decoupled_round_idx"] = ROUND_IDX
    compare_cfg["joint_checkpoint"] = str(source_round / "best_checkpoint.pth")
    compare_cfg["joint_round_idx"] = ROUND_IDX
    compare_cfg["allow_cross_round_baseline"] = False
    compare_cfg["eval_split"] = "all_pairs"
    compare_cfg["reuse_joint_predictions"] = True
    generated_compare = GENERATED_DIR / "compare_round09_single_vs_fusion.yaml"
    _write_yaml(generated_compare, compare_cfg)

    return generated_augment, generated_profile, generated_compare


def main() -> None:
    parser = argparse.ArgumentParser(description="Train round09 single-feature baseline and rerun compare.")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args()

    augment_cfg, profile_path, compare_cfg = prepare_baseline_artifacts()
    baseline_ckpt = BASELINE_ROUNDS_DIR / f"round_{ROUND_IDX:02d}" / "best_checkpoint.pth"
    train_result_path = BASELINE_ROUNDS_DIR / f"round_{ROUND_IDX:02d}" / "single_feature_train_result.json"

    if args.force_train or (not args.skip_train and not (baseline_ckpt.exists() and train_result_path.exists())):
        result = run_training(
            round_idx=ROUND_IDX,
            gold_only=False,
            config_path=str(augment_cfg),
            profile_path=str(profile_path),
        )
        _write_json(train_result_path, result)

    cfg = load_compare_config(str(compare_cfg))
    pair_key_path = BASELINE_ROUNDS_DIR / f"round_{ROUND_IDX:02d}" / "merged_training_pair_key.json"
    dataset = build_eval_dataset(pair_key_path, cfg["_augment_cfg"], eval_split=str(cfg.get("eval_split", "val")))
    run_compare(cfg, ROUND_IDX, dataset)
    report_path = build_compare_report(cfg, ROUND_IDX)
    print(f"compare_report: {report_path}")


if __name__ == "__main__":
    main()
