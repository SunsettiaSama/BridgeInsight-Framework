from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from src.chapter3_identifier.augment._bootstrap import ensure_paths, resolve_path
from src.chapter3_identifier.augment.annotation.store import (
    annotation_store_for_round,
    load_cumulative_manual_edits,
)
from src.chapter3_identifier.augment.datasets.dual_stream_dataset import build_dataloaders
from src.chapter3_identifier.augment.datasets.dual_stream_dataset import (
    build_round2_dataloaders,
    build_round2_pair_entries,
)
from src.chapter3_identifier.augment.features.spectrum import psd_bin_count
from src.chapter3_identifier.augment.models.dual_stream_res_cnn import DualStreamResCNN
from src.chapter3_identifier.augment.models.quad_stream_dual_head_res_cnn import (
    QuadStreamDualHeadResCNN,
)
from src.chapter3_identifier.augment.labels import get_label_names
from src.chapter3_identifier.augment.settings import (
    DualStreamResCNNConfig,
    get_round_dir,
    get_round_merged_training_path,
    load_best_params,
    load_config,
    load_dual_stream_model_config,
)
from src.chapter3_identifier.augment.train.trainer import (
    DualStreamTrainer,
    QuadStreamDualHeadTrainer,
)
from src.chapter3_identifier.augment.train.warm_start import (
    create_frozen_teacher,
    load_dual_stream_checkpoint,
    load_time_branch_from_baseline,
    resolve_training_checkpoints,
)

ensure_paths()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def resolve_gold_only(round_idx: int, cfg: dict, gold_only: bool | None) -> bool:
    if gold_only is not None:
        return gold_only
    if round_idx >= 2:
        return False
    prior_manual = load_cumulative_manual_edits(cfg, round_idx - 1)
    return len(prior_manual) == 0


def run_training(round_idx: int = 1, gold_only: bool | None = None, config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    store = annotation_store_for_round(cfg, round_idx)
    use_gold_only = resolve_gold_only(round_idx, cfg, gold_only)
    prior_manual = load_cumulative_manual_edits(cfg, round_idx - 1)
    entries = store.merge(
        gold_only=use_gold_only,
        prior_manual=prior_manual if not use_gold_only else None,
        shuffle_seed=int(cfg.get("merge_shuffle_seed", 42)) + round_idx,
    )
    logger.info(
        f"round {round_idx} 训练数据：{'仅金标' if use_gold_only else '金标+人工'}，"
        f"共 {len(entries)} 条（prior_manual={len(prior_manual)}）"
    )
    if not entries:
        raise ValueError("训练标注为空，请检查 gold/manual 标注路径")

    model_cfg_dict = load_dual_stream_model_config(cfg.get("dual_stream_config"))
    psd_bins = psd_bin_count(cfg["fs"], cfg["nfft"], cfg["freq_max_hz"])
    model_cfg_dict.setdefault("spec_branch", {})["input_size"] = psd_bins
    model_cfg = DualStreamResCNNConfig.from_dict(model_cfg_dict)

    best_params = load_best_params(cfg["best_params"])
    batch_size = int(best_params.get("batch_size", cfg.get("batch_size", 16)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = get_round_dir(cfg, round_idx)
    logger.info(f"训练集快照：{get_round_merged_training_path(cfg, round_idx)}")
    label_names = get_label_names(cfg)
    num_classes = int(cfg.get("num_classes", len(label_names)))
    if round_idx >= 2:
        pair_entries = build_round2_pair_entries(entries)
        train_loader, val_loader = build_round2_dataloaders(
            entries=pair_entries,
            split_path=cfg["split_indices_path"],
            batch_size=batch_size,
            train_val_ratio=float(cfg["train_val_ratio"]),
            random_seed=int(cfg["random_seed"]),
            window_size=int(cfg["window_size"]),
            fs=float(cfg["fs"]),
            nfft=int(cfg["nfft"]),
            freq_max_hz=float(cfg["freq_max_hz"]),
            enable_denoise=bool(cfg.get("enable_denoise", False)),
            enable_preload_cache=bool(cfg.get("enable_preload_cache", True)),
            preload_num_workers=int(cfg.get("preload_num_workers", 4)),
            show_preload_progress=bool(cfg.get("show_preload_progress", True)),
            num_workers=int(cfg.get("dataloader_num_workers", 0)),
        )
        model = QuadStreamDualHeadResCNN(
            time_branch_cfg=model_cfg.time_branch,
            spec_branch_cfg=model_cfg.spec_branch,
            num_classes=num_classes,
            fusion_hidden_dim=int(cfg.get("round2_fusion_hidden_dim", 128)),
            fusion_dropout=float(cfg.get("round2_fusion_dropout", 0.1)),
        )
        init_ckpt, _ = resolve_training_checkpoints(
            cfg["rounds_output_dir"],
            round_idx,
            cfg.get("baseline_checkpoint"),
        )
        if init_ckpt:
            prev = torch.load(init_ckpt, map_location=device)
            state = prev.get("model_state_dict", {})
            prev_cfg = prev.get("config", {})
            prev_type = str(prev_cfg.get("model_type", "dual_stream_single_head"))
            if prev_type == "quad_stream_dual_head":
                model.load_state_dict(state)
                logger.info(f"round {round_idx} 从上一轮联合模型 checkpoint 初始化：{init_ckpt}")
            else:
                in_time_state = model.in_time_encoder.backbone.state_dict()
                in_spec_state = model.in_spec_encoder.backbone.state_dict()
                out_time_state = model.out_time_encoder.backbone.state_dict()
                out_spec_state = model.out_spec_encoder.backbone.state_dict()
                for k, v in state.items():
                    if k.startswith("time_branch."):
                        kk = k.replace("time_branch.", "", 1)
                        if kk in in_time_state and in_time_state[kk].shape == v.shape:
                            in_time_state[kk] = v
                        if kk in out_time_state and out_time_state[kk].shape == v.shape:
                            out_time_state[kk] = v
                    if k.startswith("spec_branch."):
                        kk = k.replace("spec_branch.", "", 1)
                        if kk in in_spec_state and in_spec_state[kk].shape == v.shape:
                            in_spec_state[kk] = v
                        if kk in out_spec_state and out_spec_state[kk].shape == v.shape:
                            out_spec_state[kk] = v
                model.in_time_encoder.backbone.load_state_dict(in_time_state)
                model.out_time_encoder.backbone.load_state_dict(out_time_state)
                model.in_spec_encoder.backbone.load_state_dict(in_spec_state)
                model.out_spec_encoder.backbone.load_state_dict(out_spec_state)
                logger.info(f"round {round_idx} 从旧双流 checkpoint 复制四分支参数：{init_ckpt}")

        legacy_teacher = None
        legacy_reg_lambda = 0.0
        legacy_reg_temperature = float(cfg.get("round3_legacy_reg_temperature", 2.0))
        if round_idx >= 3:
            legacy_reg_lambda = float(cfg.get("round3_legacy_reg_lambda", 0.2))
            if legacy_reg_lambda > 0:
                legacy_ckpt_cfg = cfg.get("legacy_teacher_checkpoint")
                legacy_ckpt = resolve_path(legacy_ckpt_cfg) if legacy_ckpt_cfg else None
                if legacy_ckpt is None or not legacy_ckpt.exists():
                    round1_ckpt = get_round_dir(cfg, 1) / "best_checkpoint.pth"
                    if round1_ckpt.exists():
                        legacy_ckpt = round1_ckpt
                    else:
                        baseline = cfg.get("baseline_checkpoint")
                        legacy_ckpt = resolve_path(baseline) if baseline else None
                if legacy_ckpt is None or not legacy_ckpt.exists():
                    raise FileNotFoundError(
                        "round3 及之后启用新旧模型正则时，需要可用的旧模型 checkpoint。"
                    )
                legacy_teacher = DualStreamResCNN(model_cfg)
                load_dual_stream_checkpoint(legacy_teacher, str(legacy_ckpt), device)
                logger.info(f"round {round_idx} 旧模型正则 teacher：{legacy_ckpt}")
        trainer = QuadStreamDualHeadTrainer(
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
            inplane_loss_weight=float(cfg.get("round2_inplane_loss_weight", 1.0)),
            outplane_loss_weight=float(cfg.get("round2_outplane_loss_weight", 1.0)),
            legacy_teacher=legacy_teacher,
            legacy_reg_lambda=legacy_reg_lambda,
            legacy_reg_temperature=legacy_reg_temperature,
            device=str(device),
            label_names=label_names,
            num_classes=num_classes,
        )
        return trainer.train(round_idx=round_idx)

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
        enable_preload_cache=bool(cfg.get("enable_preload_cache", True)),
        preload_num_workers=int(cfg.get("preload_num_workers", 4)),
        show_preload_progress=bool(cfg.get("show_preload_progress", True)),
        num_workers=int(cfg.get("dataloader_num_workers", 0)),
    )

    init_ckpt, teacher_ckpt = resolve_training_checkpoints(
        cfg["rounds_output_dir"],
        round_idx,
        cfg.get("baseline_checkpoint"),
    )
    model = DualStreamResCNN(model_cfg)

    if init_ckpt:
        load_dual_stream_checkpoint(model, init_ckpt, device)
        logger.info(f"round {round_idx} 从 checkpoint 初始化：{init_ckpt}")
    else:
        baseline = cfg.get("baseline_checkpoint")
        if baseline:
            load_time_branch_from_baseline(model, baseline, device)

    teacher_reg_lambda = float(cfg.get("teacher_reg_lambda", 0.5))
    if round_idx == 1:
        teacher_reg_lambda = 0.0
        teacher = None
    else:
        teacher = create_frozen_teacher(
            model_cfg,
            device,
            checkpoint_path=teacher_ckpt,
        )

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
        teacher_reg_lambda=teacher_reg_lambda,
        teacher_reg_temperature=float(cfg.get("teacher_reg_temperature", 2.0)),
        device=str(device),
        label_names=label_names,
        num_classes=num_classes,
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
