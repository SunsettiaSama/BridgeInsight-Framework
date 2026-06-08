from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.chapter1_identifier.augment._bootstrap import ensure_paths, resolve_path
from src.chapter1_identifier.augment.annotation.store import AnnotationStore
from src.chapter1_identifier.augment.datasets.dual_stream_dataset import build_dataloaders
from src.chapter1_identifier.augment.features.spectrum import psd_bin_count
from src.chapter1_identifier.augment.models.dual_stream_res_cnn import DualStreamResCNN
from src.chapter1_identifier.augment.settings import (
    DualStreamResCNNConfig,
    load_best_params,
    load_config,
    load_dual_stream_model_config,
)
from src.chapter1_identifier.augment.train.trainer import DualStreamTrainer
from src.chapter1_identifier.augment.train.warm_start import (
    create_frozen_teacher,
    load_dual_stream_checkpoint,
    load_time_branch_from_baseline,
    resolve_training_checkpoints,
)

ensure_paths()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def resolve_gold_only(round_idx: int, store: AnnotationStore, gold_only: bool | None) -> bool:
    if gold_only is not None:
        return gold_only
    manual_count = len(store.load_manual_edits())
    if round_idx == 1 and manual_count == 0:
        return True
    return False


def run_training(round_idx: int = 1, gold_only: bool | None = None, config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    store = AnnotationStore(
        gold_path=cfg["gold_annotation_path"],
        manual_edits_path=cfg["manual_edits_path"],
        merged_output_path=cfg["merged_training_path"],
    )
    use_gold_only = resolve_gold_only(round_idx, store, gold_only)
    entries = store.merge(gold_only=use_gold_only)
    logger.info(
        f"round {round_idx} 训练数据：{'仅金标' if use_gold_only else '金标+人工'}，共 {len(entries)} 条"
    )
    if not entries:
        raise ValueError("训练标注为空，请检查 gold/manual 标注路径")

    model_cfg_dict = load_dual_stream_model_config(cfg.get("dual_stream_config"))
    psd_bins = psd_bin_count(cfg["fs"], cfg["nfft"], cfg["freq_max_hz"])
    model_cfg_dict.setdefault("spec_branch", {})["input_size"] = psd_bins
    model_cfg = DualStreamResCNNConfig.from_dict(model_cfg_dict)

    best_params = load_best_params(cfg["best_params"])
    batch_size = int(best_params.get("batch_size", cfg.get("batch_size", 16)))

    train_loader, val_loader = build_dataloaders(
        entries=entries,
        split_path=cfg["split_indices_path"],
        batch_size=batch_size,
        train_val_ratio=float(cfg["train_val_ratio"]),
        random_seed=int(cfg["random_seed"]),
        window_size=int(cfg["window_size"]),
        fs=float(cfg["fs"]),
        nfft=int(cfg["nfft"]),
        freq_max_hz=float(cfg["freq_max_hz"]),
    )

    init_ckpt, teacher_ckpt = resolve_training_checkpoints(
        cfg["training_output_dir"],
        round_idx,
        cfg.get("baseline_checkpoint"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualStreamResCNN(model_cfg)

    if init_ckpt:
        load_dual_stream_checkpoint(model, init_ckpt, device)
        logger.info(f"round {round_idx} 从 checkpoint 初始化：{init_ckpt}")
    else:
        baseline = cfg.get("baseline_checkpoint")
        if baseline:
            load_time_branch_from_baseline(model, baseline, device)

    teacher = create_frozen_teacher(
        model_cfg,
        device,
        checkpoint_path=teacher_ckpt if round_idx > 1 else None,
        source_model=model if round_idx == 1 else None,
    )

    output_dir = resolve_path(cfg["training_output_dir"]) / f"round_{round_idx:02d}"
    trainer = DualStreamTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=str(output_dir),
        epochs=int(cfg["epochs"]),
        learning_rate=float(best_params.get("learning_rate", cfg.get("learning_rate", 1e-4))),
        weight_decay=float(best_params.get("weight_decay", cfg.get("weight_decay", 1e-5))),
        gradient_clip_norm=float(best_params.get("gradient_clip_norm", cfg.get("gradient_clip_norm", 0.5))),
        class_weights=list(cfg["class_weights"]),
        focal_gamma=float(cfg["focal_gamma"]),
        teacher_model=teacher,
        teacher_reg_lambda=float(cfg.get("teacher_reg_lambda", 0.5)),
        teacher_reg_temperature=float(cfg.get("teacher_reg_temperature", 2.0)),
        device=str(device),
    )
    return trainer.train(round_idx=round_idx)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Augment DualStream 训练")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--gold-only", action="store_true", help="强制仅使用金标训练")
    parser.add_argument("--with-manual", action="store_true", help="强制使用金标+人工标注")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args(argv)
    if args.gold_only and args.with_manual:
        raise ValueError("--gold-only 与 --with-manual 不能同时使用")
    gold_only = True if args.gold_only else (False if args.with_manual else None)
    result = run_training(round_idx=args.round, gold_only=gold_only, config_path=args.config)
    logger.info(f"训练完成：{result}")


if __name__ == "__main__":
    main()
