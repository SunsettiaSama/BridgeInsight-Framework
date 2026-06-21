from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import torch

from src.chapter3_identifier.augment._bootstrap import ensure_paths, resolve_path
from src.chapter3_identifier.augment.annotation.gold_index import annotation_key, build_gold_index
from src.chapter3_identifier.augment.annotation.split import load_saved_split_key_sets
from src.chapter3_identifier.augment.annotation.store import (
    annotation_store_for_round,
    load_cumulative_manual_edits,
)
from src.chapter3_identifier.augment.datasets.dual_stream_dataset import build_dataloaders
from src.chapter3_identifier.augment.datasets.dual_stream_dataset import (
    DEFAULT_CONTEXT_ALLOW_CROSS_FILE,
    DEFAULT_CONTEXT_INPUT_SIZE,
    DEFAULT_CONTEXT_TOTAL_SECONDS,
    build_round2_dataloaders,
    build_round2_pair_entries,
)
from src.chapter3_identifier.augment.features.spectrum import psd_bin_count
from src.chapter3_identifier.augment.models.dual_stream_res_cnn import DualStreamResCNN
from src.chapter3_identifier.augment.models.quad_stream_dual_head_res_cnn import (
    QuadStreamDualHeadContextResCNN,
    QuadStreamDualHeadResCNN,
)
from src.chapter3_identifier.augment.labels import get_label_names
from src.chapter3_identifier.augment.settings import (
    BranchConfig,
    DualStreamResCNNConfig,
    get_round_dir,
    get_round_inference_path,
    get_round_merged_training_path,
    load_config,
    load_dual_stream_model_config,
)
from src.chapter3_identifier.augment.train.trainer import (
    DualStreamTrainer,
    QuadStreamDualHeadTrainer,
)
from src.chapter3_identifier.augment.train.profile import load_training_profile
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


def _load_pair_hints_from_inference(cfg: dict, round_idx: int) -> dict[tuple[str, int], dict]:
    infer_round = max(1, round_idx - 1)
    infer_path = get_round_inference_path(cfg, infer_round)
    if not infer_path.exists():
        logger.warning(
            f"round {round_idx} 未找到 round {infer_round} inference.json，无法用预测补全缺失方向标签：{infer_path}"
        )
        return {}
    with open(infer_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("records", [])
    hints: dict[tuple[str, int], dict] = {}
    for row in records:
        in_fp = row.get("inplane_file_path")
        out_fp = row.get("outplane_file_path")
        if not in_fp or not out_fp:
            continue
        wi = int(row.get("window_index", 0))
        key = annotation_key(in_fp, wi)
        if key in hints:
            continue
        hints[key] = {
            "inplane_file_path": in_fp,
            "outplane_file_path": out_fp,
            "window_index": wi,
            "inplane_prediction": row.get("inplane_prediction", row.get("prediction")),
            "outplane_prediction": row.get("outplane_prediction", row.get("prediction")),
        }
    return hints


def _update_split_dataset_info(dataset_info: dict, cfg: dict, gold_keys: set[tuple[str, int]]) -> None:
    _, split_val_keys = load_saved_split_key_sets(cfg["split_indices_path"])
    dataset_info["gold_val_total"] = int(len(gold_keys & split_val_keys))
    dataset_info["split_val_total"] = int(len(split_val_keys))


def _select_pair_entries(
    cfg: dict,
    entries: list[dict],
    round_idx: int,
    train_profile: dict,
    num_classes: int,
) -> tuple[list[dict], dict]:
    prediction_fill_mode = str(train_profile.get("prediction_fill_mode", "auto"))
    enable_gold_fill = bool(train_profile.get("enable_gold_fill", False))
    require_bidirectional_labels = bool(train_profile.get("require_bidirectional_labels", False))
    min_pairs = int(train_profile.get("min_pairs_without_prediction", 1000))
    base_pair_hints = _load_pair_hints_from_inference(cfg, round_idx) if enable_gold_fill else {}
    base_entries, base_stats = build_round2_pair_entries(
        entries,
        pair_hints=base_pair_hints,
        num_classes=num_classes,
        enable_prediction_fill=False,
        enable_gold_fill=enable_gold_fill,
        require_bidirectional_labels=require_bidirectional_labels,
        raise_on_empty=False,
    )
    use_prediction_fill = False
    reason = "disabled"
    if require_bidirectional_labels:
        reason = "require_bidirectional_labels"
    elif prediction_fill_mode == "always":
        use_prediction_fill = True
        reason = "always"
    elif prediction_fill_mode == "auto":
        use_prediction_fill = len(base_entries) < min_pairs
        reason = "auto_insufficient_pairs" if use_prediction_fill else "auto_enough_pairs"

    if not use_prediction_fill:
        if not base_entries:
            raise ValueError("round2 训练需要双向标注 pair；当前配置关闭了 prediction 补全且无可用 pair")
        stats = {
            **base_stats,
            "base_pair_total_without_prediction": int(len(base_entries)),
            "pair_hints_total": int(len(base_pair_hints)),
            "prediction_fill_mode": prediction_fill_mode,
            "prediction_fill_used": False,
            "prediction_fill_reason": reason,
            "min_pairs_without_prediction": int(min_pairs),
        }
        return base_entries, stats

    pair_hints = base_pair_hints if base_pair_hints else _load_pair_hints_from_inference(cfg, round_idx)
    pair_entries, pair_stats = build_round2_pair_entries(
        entries,
        pair_hints=pair_hints,
        num_classes=num_classes,
        enable_prediction_fill=True,
        enable_gold_fill=enable_gold_fill,
        require_bidirectional_labels=require_bidirectional_labels,
    )
    stats = {
        **pair_stats,
        "base_pair_total_without_prediction": int(len(base_entries)),
        "pair_hints_total": int(len(pair_hints)),
        "prediction_fill_mode": prediction_fill_mode,
        "prediction_fill_used": True,
        "prediction_fill_reason": reason,
        "min_pairs_without_prediction": int(min_pairs),
    }
    return pair_entries, stats


def _context_branch_config(time_branch: BranchConfig, context_input_size: int) -> BranchConfig:
    return BranchConfig(
        in_channels=int(time_branch.in_channels),
        input_size=int(context_input_size),
        res_channels=list(time_branch.res_channels),
        num_blocks=int(time_branch.num_blocks),
        kernel_size=int(time_branch.kernel_size),
        fc_hidden_dims=list(time_branch.fc_hidden_dims),
        dropout_prob=float(time_branch.dropout_prob),
        num_classes=int(time_branch.num_classes),
    )


def _load_matching_checkpoint_state(model: torch.nn.Module, state: dict, init_ckpt: Path | str, round_idx: int) -> None:
    target = model.state_dict()
    matched = {}
    skipped = []
    for key, value in state.items():
        if key in target and target[key].shape == value.shape:
            matched[key] = value
        elif key in target:
            skipped.append(key)
    target.update(matched)
    model.load_state_dict(target)
    logger.info(
        "round %s 结构变化，仅从 checkpoint 加载可匹配参数：%s（matched=%s skipped_shape=%s，同结构 teacher 正则关闭）",
        round_idx,
        init_ckpt,
        len(matched),
        len(skipped),
    )


def _state_dict_shapes_match(model: torch.nn.Module, state: dict) -> bool:
    target = model.state_dict()
    if set(target.keys()) != set(state.keys()):
        return False
    return all(target[key].shape == state[key].shape for key in target)


def run_training(
    round_idx: int = 1,
    gold_only: bool | None = None,
    config_path: str | None = None,
    profile_path: str | None = None,
) -> dict:
    cfg = load_config(config_path)
    profile_payload = load_training_profile(cfg, round_idx, profile_path)
    train_profile = profile_payload["profile"]
    profile_meta = profile_payload["metadata"]
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
    logger.info(
        "round %s 训练集配置：train_val_ratio=%.3f, require_bidirectional=%s, gold_fill=%s, prediction_fill_mode=%s, min_no_pred_pairs=%s",
        round_idx,
        float(train_profile["train_val_ratio"]),
        bool(train_profile["require_bidirectional_labels"]),
        bool(train_profile["enable_gold_fill"]),
        str(train_profile["prediction_fill_mode"]),
        int(train_profile["min_pairs_without_prediction"]),
    )
    if not entries:
        raise ValueError("训练标注为空，请检查 gold/manual 标注路径")

    model_cfg_dict = load_dual_stream_model_config(cfg.get("dual_stream_config"))
    psd_bins = psd_bin_count(cfg["fs"], cfg["nfft"], cfg["freq_max_hz"])
    model_cfg_dict.setdefault("spec_branch", {})["input_size"] = psd_bins
    model_cfg = DualStreamResCNNConfig.from_dict(model_cfg_dict)

    batch_size = int(train_profile["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = get_round_dir(cfg, round_idx)
    logger.info(f"训练集快照：{get_round_merged_training_path(cfg, round_idx)}")
    label_names = get_label_names(cfg)
    num_classes = int(cfg.get("num_classes", len(label_names)))
    dataset_info: dict = {
        "round_idx": int(round_idx),
        "merged_total": int(len(entries)),
        "training_profile": profile_meta["summary"],
        "training_profile_source": profile_meta["source"],
        "training_profile_path": profile_meta["path"],
    }
    gold_keys = set(build_gold_index(store.load_gold()).keys())
    dataset_info["gold_total"] = int(len(gold_keys))
    split_seed = int(cfg["random_seed"]) + int(round_idx)
    dataset_info["split_seed"] = int(split_seed)
    train_val_ratio = float(train_profile["train_val_ratio"])
    dataset_info["train_val_ratio"] = float(train_val_ratio)

    model_type = str(train_profile["model_type"])

    if model_type in {"quad_stream_dual_head", "quad_stream_dual_head_context"}:
        context_mode = str(train_profile.get("context_mode", "short_long"))
        enable_long_context = (
            model_type == "quad_stream_dual_head_context"
            and bool(train_profile["enable_long_context"])
            and context_mode == "short_long"
        )
        context_input_size = int(train_profile.get("context_input_size", DEFAULT_CONTEXT_INPUT_SIZE))
        context_total_seconds = float(train_profile.get("context_total_seconds", DEFAULT_CONTEXT_TOTAL_SECONDS))
        context_allow_cross_file = bool(train_profile.get("context_allow_cross_file", DEFAULT_CONTEXT_ALLOW_CROSS_FILE))
        dataset_info["enable_long_context"] = bool(enable_long_context)
        dataset_info["context_mode"] = context_mode if model_type == "quad_stream_dual_head_context" else "short_only"
        if model_type == "quad_stream_dual_head_context":
            dataset_info["context_input_size"] = int(context_input_size)
            dataset_info["context_total_seconds"] = float(context_total_seconds)
            dataset_info["context_allow_cross_file"] = bool(context_allow_cross_file)
        pair_entries, pair_stats = _select_pair_entries(
            cfg=cfg,
            entries=entries,
            round_idx=round_idx,
            train_profile=train_profile,
            num_classes=num_classes,
        )
        dataset_info.update(pair_stats)
        logger.info(
            "round %s 配对样本：pair_total=%s，base_no_pred=%s，hint_total=%s，prediction_fill=%s(%s)，"
            "in(source m/g/gf/p)=%s/%s/%s/%s，out(source m/g/gf/p)=%s/%s/%s/%s",
            round_idx,
            dataset_info.get("pair_total", 0),
            dataset_info.get("base_pair_total_without_prediction", 0),
            dataset_info.get("pair_hints_total", 0),
            dataset_info.get("prediction_fill_used", False),
            dataset_info.get("prediction_fill_reason", "-"),
            dataset_info.get("inplane_manual", 0),
            dataset_info.get("inplane_gold", 0),
            dataset_info.get("inplane_gold_fill", 0),
            dataset_info.get("inplane_prediction", 0),
            dataset_info.get("outplane_manual", 0),
            dataset_info.get("outplane_gold", 0),
            dataset_info.get("outplane_gold_fill", 0),
            dataset_info.get("outplane_prediction", 0),
        )
        train_loader, val_loader = build_round2_dataloaders(
            entries=pair_entries,
            split_path=cfg["split_indices_path"],
            round_idx=round_idx,
            batch_size=batch_size,
            train_val_ratio=train_val_ratio,
            random_seed=split_seed,
            window_size=int(cfg["window_size"]),
            fs=float(cfg["fs"]),
            nfft=int(cfg["nfft"]),
            freq_max_hz=float(cfg["freq_max_hz"]),
            enable_denoise=bool(cfg.get("enable_denoise", False)),
            enable_long_context=enable_long_context,
            context_input_size=context_input_size,
            context_total_seconds=context_total_seconds,
            context_allow_cross_file=context_allow_cross_file,
            enable_preload_cache=bool(cfg.get("enable_preload_cache", True)),
            preload_num_workers=int(cfg.get("preload_num_workers", 4)),
            show_preload_progress=bool(cfg.get("show_preload_progress", True)),
            num_workers=int(cfg.get("dataloader_num_workers", 0)),
        )
        _update_split_dataset_info(dataset_info, cfg, gold_keys)
        train_ds = train_loader.dataset
        val_ds = val_loader.dataset
        dataset_info["round2_bad_train_total"] = int(getattr(train_ds, "round2_bad_total", 0))
        dataset_info["round2_bad_train_rate"] = float(getattr(train_ds, "round2_bad_rate", 0.0))
        dataset_info["round2_bad_train_reasons"] = dict(getattr(train_ds, "round2_bad_reason_counts", {}))
        dataset_info["round2_bad_train_examples"] = list(getattr(train_ds, "round2_bad_examples", []))
        dataset_info["round2_bad_val_total"] = int(getattr(val_ds, "round2_bad_total", 0))
        dataset_info["round2_bad_val_rate"] = float(getattr(val_ds, "round2_bad_rate", 0.0))
        dataset_info["round2_bad_val_reasons"] = dict(getattr(val_ds, "round2_bad_reason_counts", {}))
        dataset_info["round2_bad_val_examples"] = list(getattr(val_ds, "round2_bad_examples", []))
        model_kwargs = {
            "time_branch_cfg": model_cfg.time_branch,
            "spec_branch_cfg": model_cfg.spec_branch,
            "num_classes": num_classes,
            "fusion_hidden_dim": int(train_profile["fusion_hidden_dim"]),
            "fusion_dropout": float(train_profile["fusion_dropout"]),
            "cross_attn_heads": int(train_profile["cross_attn_heads"]),
        }
        if model_type == "quad_stream_dual_head_context":
            context_branch_cfg = _context_branch_config(model_cfg.time_branch, context_input_size)
            model = QuadStreamDualHeadContextResCNN(
                context_branch_cfg=context_branch_cfg,
                **model_kwargs,
            )
        else:
            model = QuadStreamDualHeadResCNN(**model_kwargs)
        current_model_type = "quad_stream_dual_head_context" if model_type == "quad_stream_dual_head_context" else "quad_stream_dual_head"
        init_ckpt, _ = resolve_training_checkpoints(
            cfg["rounds_output_dir"],
            round_idx,
            cfg.get("baseline_checkpoint"),
        )
        same_structure_teacher = None
        same_structure_reg_lambda = 0.0
        same_structure_reg_temperature = float(train_profile["same_structure_reg_temperature"])
        learning_rate = float(train_profile["learning_rate"])
        if init_ckpt:
            prev = torch.load(init_ckpt, map_location=device)
            state = prev.get("model_state_dict", {})
            prev_cfg = prev.get("config", {})
            prev_type = str(prev_cfg.get("model_type", "dual_stream_single_head"))
            same_structure = prev_type == current_model_type and _state_dict_shapes_match(model, state)
            if same_structure:
                model.load_state_dict(state)
                logger.info("round %s 同结构连续微调初始化：%s", round_idx, init_ckpt)
                if bool(train_profile["enable_same_structure_regularization"]):
                    same_structure_reg_lambda = float(train_profile["same_structure_reg_lambda"])
                    if same_structure_reg_lambda > 0:
                        same_structure_teacher = copy.deepcopy(model)
                        learning_rate *= float(train_profile["same_structure_lr_scale"])
                        dataset_info["same_structure_finetune"] = True
                        dataset_info["same_structure_teacher_checkpoint"] = str(init_ckpt)
                        dataset_info["same_structure_lr_scale"] = float(train_profile["same_structure_lr_scale"])
                        logger.info(
                            "round %s 启用同结构 teacher 正则：λ=%s T=%s lr=%s",
                            round_idx,
                            same_structure_reg_lambda,
                            same_structure_reg_temperature,
                            learning_rate,
                        )
            elif prev_type in {"quad_stream_dual_head", "quad_stream_dual_head_context"}:
                _load_matching_checkpoint_state(model, state, init_ckpt, round_idx)
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
        legacy_reg_temperature = float(train_profile["legacy_reg_temperature"])
        if bool(train_profile["enable_legacy_regularization"]) and same_structure_teacher is not None:
            logger.info("round %s 已启用同结构 teacher 正则，跳过旧模型正则", round_idx)
        elif bool(train_profile["enable_legacy_regularization"]):
            legacy_reg_lambda = float(train_profile["legacy_reg_lambda"])
            if legacy_reg_lambda > 0:
                legacy_ckpt_cfg = train_profile.get("legacy_teacher_checkpoint") or cfg.get("legacy_teacher_checkpoint")
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
            epochs=int(train_profile["epochs"]),
            learning_rate=learning_rate,
            weight_decay=float(train_profile["weight_decay"]),
            gradient_clip_norm=float(train_profile["gradient_clip_norm"]),
            class_weights=list(cfg["class_weights"]),
            focal_gamma=float(train_profile["focal_gamma"]),
            inplane_loss_weight=float(train_profile["inplane_loss_weight"]),
            outplane_loss_weight=float(train_profile["outplane_loss_weight"]),
            legacy_teacher=legacy_teacher,
            legacy_reg_lambda=legacy_reg_lambda,
            legacy_reg_temperature=legacy_reg_temperature,
            same_structure_teacher=same_structure_teacher,
            same_structure_reg_lambda=same_structure_reg_lambda,
            same_structure_reg_temperature=same_structure_reg_temperature,
            device=str(device),
            label_names=label_names,
            num_classes=num_classes,
            dataset_info=dataset_info,
        )
        return trainer.train(round_idx=round_idx)

    train_loader, val_loader = build_dataloaders(
        entries=entries,
        split_path=cfg["split_indices_path"],
        batch_size=batch_size,
        train_val_ratio=train_val_ratio,
        random_seed=split_seed,
        window_size=int(cfg["window_size"]),
        fs=float(cfg["fs"]),
        nfft=int(cfg["nfft"]),
        freq_max_hz=float(cfg["freq_max_hz"]),
        enable_preload_cache=bool(cfg.get("enable_preload_cache", True)),
        preload_num_workers=int(cfg.get("preload_num_workers", 4)),
        show_preload_progress=bool(cfg.get("show_preload_progress", True)),
        num_workers=int(cfg.get("dataloader_num_workers", 0)),
    )
    _update_split_dataset_info(dataset_info, cfg, gold_keys)

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
        epochs=int(train_profile["epochs"]),
        learning_rate=float(train_profile["learning_rate"]),
        weight_decay=float(train_profile["weight_decay"]),
        gradient_clip_norm=float(train_profile["gradient_clip_norm"]),
        class_weights=list(cfg["class_weights"]),
        focal_gamma=float(train_profile["focal_gamma"]),
        teacher_model=teacher,
        teacher_reg_lambda=teacher_reg_lambda,
        teacher_reg_temperature=float(cfg.get("teacher_reg_temperature", 2.0)),
        device=str(device),
        label_names=label_names,
        num_classes=num_classes,
        dataset_info=dataset_info,
    )
    return trainer.train(round_idx=round_idx)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Augment DualStream 训练")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--gold-only", action="store_true", help="强制仅使用金标训练")
    parser.add_argument("--with-manual", action="store_true", help="强制使用金标+人工标注")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--profile", type=str, default=None)
    args = parser.parse_args(argv)
    if args.gold_only and args.with_manual:
        raise ValueError("--gold-only 与 --with-manual 不能同时使用")
    gold_only = True if args.gold_only else (False if args.with_manual else None)
    result = run_training(round_idx=args.round, gold_only=gold_only, config_path=args.config, profile_path=args.profile)
    logger.info(f"训练完成：{result}")


if __name__ == "__main__":
    main()
