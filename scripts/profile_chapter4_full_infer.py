#!/usr/bin/env python
"""Chapter4 全量识别性能 profiling：测量 I/O / PSD / GPU forward 的 gap。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chapter4_characteristics._bootstrap import ensure_paths, resolve_path

ensure_paths()

import torch

from src.chapter3_identifier.augment.infer.identifier import DualStreamIdentifier
from src.chapter3_identifier.augment.infer.profile_hooks import InferStageTimings, StageTimer
from src.chapter3_identifier.augment.infer.runner import DualStreamInferenceRunner
from src.chapter4_characteristics.infer.preflight import run_preflight
from src.chapter4_characteristics.infer.run import (
    _apply_limit,
    _assemble_records_parallel,
    _load_inference_dataset,
    _stream_write_inference_json,
)
from src.chapter4_characteristics.infer.result_adapter import records_to_enriched_json
from src.chapter4_characteristics.settings import (
    get_chapter4_root,
    get_identifier_checkpoint,
    get_inference_dir,
    get_inference_path,
    get_predictions_enriched_path,
    load_config,
)


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    out = dict(cfg)
    if args.batch_size is not None:
        out["infer_batch_size"] = int(args.batch_size)
    if args.workers is not None:
        out["infer_dataloader_workers"] = int(args.workers)
    if args.psd_workers is not None:
        out["infer_psd_workers"] = int(args.psd_workers)
    if args.prefetch_files is not None:
        out["infer_prefetch_files"] = int(args.prefetch_files)
    if args.prefetch_batches is not None:
        out["infer_prefetch_batches"] = int(args.prefetch_batches)
    if args.cache_max_mb is not None:
        out["infer_cache_max_mb"] = float(args.cache_max_mb)
    if args.prefetch_workers is not None:
        out["infer_prefetch_workers"] = int(args.prefetch_workers)
    return out


def run_profile(args: argparse.Namespace) -> dict:
    cfg = _apply_overrides(load_config(args.config), args)
    timer = StageTimer()

    if not args.skip_preflight:
        with timer.stage("preflight"):
            run_preflight(config_path=args.config)

    with timer.stage("dataset"):
        dataset = _load_inference_dataset(cfg["inference_dataset_config"])
        dataset = _apply_limit(dataset, args.limit)

    ckpt_path = get_identifier_checkpoint(cfg)
    with timer.stage("checkpoint"):
        identifier = DualStreamIdentifier.from_checkpoint(
            str(ckpt_path),
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
        profile_stats=timer.timings.prefetch,
    )

    with timer.stage("runner"):
        inplane_p, outplane_p = runner.run_with_proba(dataset)

    projection_mode = str(cfg.get("prediction_projection_mode", "direction"))
    projection_direction = str(cfg.get("prediction_projection_direction", "outplane"))
    with timer.stage("merge"):
        merged = runner.merge_proba_predictions(
            inplane_p,
            outplane_p,
            projection_mode=projection_mode,
            projection_direction=projection_direction,
        )
    projection_label = (
        f"direction:{projection_direction}"
        if projection_mode == "direction"
        else projection_mode
    )

    merged_indices = sorted(merged.keys())
    chunk_size = int(cfg.get("infer_record_chunk_size", 4096))
    record_workers = int(cfg.get("infer_record_workers", 8))

    with timer.stage("records"):
        records = _assemble_records_parallel(
            merged_indices=merged_indices,
            samples=dataset._samples,
            merged=merged,
            inplane_p=inplane_p,
            outplane_p=outplane_p,
            chunk_size=chunk_size,
            workers=record_workers,
            projection_mode=projection_label,
        )

    out_dir = get_inference_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    header = {
        "generated_at": ts,
        "checkpoint": str(ckpt_path),
        "projection_mode": projection_label,
    }

    if not args.skip_write:
        with timer.stage("inference_json"):
            _stream_write_inference_json(
                get_inference_path(cfg), header, records, chunk_size
            )
        with timer.stage("enriched_json"):
            enriched_payload = records_to_enriched_json(
                records,
                dataset,
                checkpoint_path=str(ckpt_path),
                dataset_config=str(cfg["inference_dataset_config"]),
            )
            with open(get_predictions_enriched_path(cfg), "w", encoding="utf-8") as f:
                json.dump(enriched_payload, f, ensure_ascii=False, indent=2)

    timer.timings.extra = {
        "sample_count": len(merged_indices),
        "record_count": len(records),
        "limit": args.limit,
        "model_type": identifier.model_type,
        "checkpoint": str(ckpt_path),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "infer_batch_size": int(cfg.get("infer_batch_size", 256)),
        "infer_dataloader_workers": int(cfg.get("infer_dataloader_workers", 4)),
        "infer_psd_workers": int(cfg.get("infer_psd_workers", 2)),
        "infer_cache_max_mb": float(cfg.get("infer_cache_max_mb", 512)),
        "infer_prefetch_files": int(cfg.get("infer_prefetch_files", 4)),
        "infer_prefetch_batches": int(cfg.get("infer_prefetch_batches", 2)),
        "infer_prefetch_workers": int(cfg.get("infer_prefetch_workers", 2)),
        "infer_context_workers": int(cfg.get("infer_context_workers", 2)),
        "skip_write": bool(args.skip_write),
    }
    return timer.timings.to_dict()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chapter4 全量识别性能 profiling")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--limit", type=int, default=512)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--psd-workers", type=int, default=None)
    parser.add_argument("--prefetch-files", type=int, default=None)
    parser.add_argument("--prefetch-batches", type=int, default=None)
    parser.add_argument("--prefetch-workers", type=int, default=None)
    parser.add_argument("--cache-max-mb", type=float, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--skip-write", action="store_true")
    parser.add_argument(
        "--compare-prefetch",
        action="store_true",
        help="依次运行无预取 / 保守预取 / 提高预取距离三组对比",
    )
    args = parser.parse_args(argv)
    if args.config is None:
        args.config = str(
            ROOT / "src" / "chapter4_characteristics" / "config" / "profile.yaml"
        )

    cfg = load_config(args.config)
    profile_dir = get_chapter4_root(cfg) / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_prefetch:
        scenarios = [
            ("no_prefetch", {"prefetch_files": 0, "prefetch_batches": 0, "prefetch_workers": 0}),
            ("conservative", {"prefetch_files": 2, "prefetch_batches": 1, "prefetch_workers": 1}),
            ("aggressive", {"prefetch_files": 6, "prefetch_batches": 3, "prefetch_workers": 2}),
        ]
        results = {}
        for name, overrides in scenarios:
            run_args = argparse.Namespace(**vars(args))
            run_args.prefetch_files = overrides["prefetch_files"]
            run_args.prefetch_batches = overrides["prefetch_batches"]
            run_args.prefetch_workers = overrides["prefetch_workers"]
            print(f"\n=== scenario: {name} ===")
            results[name] = run_profile(run_args)
            print(json.dumps(results[name], ensure_ascii=False, indent=2))
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = profile_dir / f"infer_profile_compare_{ts}.json"
        payload = {"generated_at": ts, "limit": args.limit, "scenarios": results}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\n对比结果已写入：{out_path}")
        return

    result = run_profile(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        out_path = resolve_path(args.output)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = profile_dir / f"infer_profile_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"profile 已写入：{out_path}")


if __name__ == "__main__":
    main()
