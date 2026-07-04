#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.chapter3_identifier.augment.infer.dataset_loader import load_staycable_dataset
from src.chapter3_identifier.augment.infer.identifier import DualStreamIdentifier
from src.chapter3_identifier.augment.infer.profile_hooks import StageTimer
from src.chapter3_identifier.augment.infer.runner import DualStreamInferenceRunner
from src.chapter3_identifier.augment.settings import get_round_checkpoint_path, load_config
from src.chapter3_identifier.augment.workflow_config import (
    apply_workflow_to_runtime_cfg,
    ensure_workflow_config,
    resolve_round_workflow,
)


def _limited_dataset(dataset, limit: int):
    if int(limit) <= 0:
        return dataset
    dataset._samples = dataset._samples[: int(limit)]
    return dataset


def _apply_overrides(cfg: dict, args: argparse.Namespace, scenario: dict) -> dict:
    out = copy.deepcopy(cfg)
    for key, value in scenario.items():
        out[key] = value
    if args.batch_size is not None:
        out["infer_batch_size"] = int(args.batch_size)
    if args.prefetch_files is not None:
        out["infer_prefetch_files"] = int(args.prefetch_files)
    if args.prefetch_batches is not None:
        out["infer_prefetch_batches"] = int(args.prefetch_batches)
    if args.prefetch_workers is not None:
        out["infer_prefetch_workers"] = int(args.prefetch_workers)
    if args.context_workers is not None:
        out["infer_context_workers"] = int(args.context_workers)
    if args.context_cache_entries is not None:
        out["infer_context_cache_entries"] = int(args.context_cache_entries)
    if args.context_batch_mode is not None:
        out["infer_context_batch_mode"] = str(args.context_batch_mode)
    if args.time_block_producer_workers is not None:
        out["infer_time_block_producer_workers"] = int(args.time_block_producer_workers)
    if args.cache_max_mb is not None:
        out["infer_cache_max_mb"] = float(args.cache_max_mb)
    return out


def _profile_once(base_cfg: dict, args: argparse.Namespace, scenario_name: str, scenario: dict) -> dict:
    cfg = _apply_overrides(base_cfg, args, scenario)
    timer = StageTimer()
    round_idx = int(args.round)

    with timer.stage("dataset"):
        dataset = load_staycable_dataset(cfg["inference_dataset_config"])
        dataset = _limited_dataset(dataset, int(args.limit))
        timer.timings.extra["sample_count"] = int(len(getattr(dataset, "_samples", [])))

    checkpoint_path = get_round_checkpoint_path(cfg, round_idx)
    with timer.stage("checkpoint"):
        identifier = DualStreamIdentifier.from_checkpoint(
            str(checkpoint_path),
            model_config_path=cfg.get("dual_stream_config"),
            fs=float(cfg["fs"]),
            nfft=int(cfg["nfft"]),
            freq_max_hz=float(cfg["freq_max_hz"]),
            wind_config=cfg,
        )

    runner = DualStreamInferenceRunner(
        identifier,
        batch_size=int(cfg.get("infer_batch_size", 256)),
        num_workers=int(cfg.get("infer_dataloader_workers", 4)),
        psd_workers=int(cfg.get("infer_psd_workers", 2)),
        fs=float(cfg["fs"]),
        nfft=int(cfg["nfft"]),
        freq_max_hz=float(cfg["freq_max_hz"]),
        wind_config=cfg,
        cache_max_mb=float(cfg.get("infer_cache_max_mb", 512)),
        prefetch_files=int(cfg.get("infer_prefetch_files", 4)),
        prefetch_batches=int(cfg.get("infer_prefetch_batches", 2)),
        prefetch_workers=int(cfg.get("infer_prefetch_workers", 2)),
        context_workers=int(cfg.get("infer_context_workers", 2)),
        context_cache_entries=int(cfg.get("infer_context_cache_entries", 0)),
        context_batch_mode=str(cfg.get("infer_context_batch_mode", "time_block")),
        time_block_producer_workers=int(cfg.get("infer_time_block_producer_workers", 1)),
        joint_queue_depth=int(cfg.get("infer_joint_queue_depth", 2)),
        profile_stats=timer.timings.prefetch,
    )

    with timer.stage("runner"):
        inplane_p, outplane_p = runner.run_with_proba(dataset)
    with timer.stage("merge"):
        merged = runner.merge_proba_predictions(
            inplane_p,
            outplane_p,
            projection_mode=str(cfg.get("prediction_projection_mode", "direction")),
            projection_direction=str(cfg.get("prediction_projection_direction", "outplane")),
        )

    timer.timings.extra.update(
        {
            "scenario": scenario_name,
            "round": round_idx,
            "merged_count": len(merged),
            "model_type": identifier.model_type,
            "context_mode": getattr(identifier, "context_mode", "short_only"),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "infer_batch_size": int(cfg.get("infer_batch_size", 256)),
            "infer_cache_max_mb": float(cfg.get("infer_cache_max_mb", 512)),
            "infer_prefetch_files": int(cfg.get("infer_prefetch_files", 4)),
            "infer_prefetch_batches": int(cfg.get("infer_prefetch_batches", 2)),
            "infer_prefetch_workers": int(cfg.get("infer_prefetch_workers", 2)),
            "infer_context_workers": int(cfg.get("infer_context_workers", 2)),
            "infer_context_cache_entries": int(cfg.get("infer_context_cache_entries", 0)),
            "infer_context_batch_mode": str(cfg.get("infer_context_batch_mode", "time_block")),
            "infer_time_block_producer_workers": int(cfg.get("infer_time_block_producer_workers", 1)),
            "infer_joint_queue_depth": int(cfg.get("infer_joint_queue_depth", 2)),
            "checkpoint": str(checkpoint_path),
            "inference_dataset_config": str(cfg["inference_dataset_config"]),
        }
    )
    return timer.timings.to_dict()


def _scenarios(args: argparse.Namespace) -> list[tuple[str, dict]]:
    if args.compare_context_batch_mode:
        return [
            (
                "legacy",
                {
                    "infer_context_batch_mode": "legacy",
                    "infer_context_workers": 2,
                    "infer_context_cache_entries": 0,
                },
            ),
            (
                "time_block_p1",
                {
                    "infer_context_batch_mode": "time_block",
                    "infer_time_block_producer_workers": 1,
                    "infer_context_workers": 2,
                    "infer_context_cache_entries": 0,
                },
            ),
            (
                "time_block_p2",
                {
                    "infer_context_batch_mode": "time_block",
                    "infer_time_block_producer_workers": 2,
                    "infer_context_workers": 2,
                    "infer_context_cache_entries": 0,
                },
            ),
        ]
    if args.compare_context_workers:
        return [
            ("context_workers_1", {"infer_context_workers": 1}),
            ("context_workers_2", {"infer_context_workers": 2}),
            ("context_workers_4", {"infer_context_workers": 4}),
        ]
    if not args.compare_context_cache:
        return [("current", {})]
    return [
        ("context_cache_0", {"infer_context_cache_entries": 0}),
        ("context_cache_16", {"infer_context_cache_entries": 16}),
        ("context_cache_64", {"infer_context_cache_entries": 64}),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment 全量识别小样本 profiling")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--round", type=int, default=9)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--prefetch-files", type=int, default=None)
    parser.add_argument("--prefetch-batches", type=int, default=None)
    parser.add_argument("--prefetch-workers", type=int, default=None)
    parser.add_argument("--context-workers", type=int, default=None)
    parser.add_argument("--context-cache-entries", type=int, default=None)
    parser.add_argument("--context-batch-mode", type=str, choices=["legacy", "time_block"], default=None)
    parser.add_argument("--time-block-producer-workers", type=int, default=None)
    parser.add_argument("--cache-max-mb", type=float, default=None)
    parser.add_argument("--compare-context-batch-mode", action="store_true")
    parser.add_argument("--compare-context-cache", action="store_true")
    parser.add_argument("--compare-context-workers", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_workflow_config(cfg)
    resolved = resolve_round_workflow(cfg, int(args.round))
    runtime_cfg = apply_workflow_to_runtime_cfg(cfg, resolved)

    results = {
        name: _profile_once(runtime_cfg, args, name, scenario)
        for name, scenario in _scenarios(args)
    }
    payload = {
        "generated_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "limit": int(args.limit),
        "round": int(args.round),
        "scenarios": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    output = Path(args.output) if args.output else ROOT / "results" / "augment" / "profiles" / (
        "augment_infer_profile_" + payload["generated_at"] + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"profile 已写入：{output}")


if __name__ == "__main__":
    main()
