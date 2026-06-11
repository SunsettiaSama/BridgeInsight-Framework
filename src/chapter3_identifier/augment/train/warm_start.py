from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from src.chapter3_identifier.augment._bootstrap import ensure_paths
from src.chapter3_identifier.augment.models.dual_stream_res_cnn import DualStreamResCNN

ensure_paths()
logger = logging.getLogger(__name__)


def load_time_branch_from_baseline(
    model: DualStreamResCNN,
    checkpoint_path: str,
    device: torch.device,
) -> int:
    path = Path(checkpoint_path)
    if not path.exists():
        logger.warning(f"baseline checkpoint 不存在，跳过 warm-start：{path}")
        return 0

    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    target = model.time_branch.state_dict()
    matched = {}
    for k, v in state.items():
        if k in target and target[k].shape == v.shape:
            matched[k] = v
    model.time_branch.load_state_dict({**target, **matched})
    logger.info(f"time_branch warm-start：加载 {len(matched)}/{len(target)} 个参数")
    return len(matched)


def load_dual_stream_checkpoint(
    model: DualStreamResCNN,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"DualStream 全量加载：{path}")


def create_frozen_teacher(
    model_cfg,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
    source_model: Optional[DualStreamResCNN] = None,
) -> DualStreamResCNN:
    teacher = DualStreamResCNN(model_cfg)
    if checkpoint_path and Path(checkpoint_path).exists():
        load_dual_stream_checkpoint(teacher, checkpoint_path, device)
    elif source_model is not None:
        teacher.load_state_dict(source_model.state_dict())
    else:
        raise ValueError("teacher 需要 checkpoint_path 或 source_model")
    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def resolve_training_checkpoints(
    rounds_output_dir: str,
    round_idx: int,
    baseline_checkpoint: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    training_dir = Path(rounds_output_dir)
    if not training_dir.is_absolute():
        from src.chapter3_identifier.augment._bootstrap import resolve_path
        training_dir = resolve_path(rounds_output_dir)
    if round_idx > 1:
        prev_ckpt = training_dir / f"round_{round_idx - 1:02d}" / "best_checkpoint.pth"
        if not prev_ckpt.exists():
            raise FileNotFoundError(
                f"round {round_idx} 训练需要上一 round checkpoint：{prev_ckpt}"
            )
        ckpt = str(prev_ckpt)
        return ckpt, ckpt

    init_ckpt = None
    teacher_ckpt = baseline_checkpoint if baseline_checkpoint and Path(baseline_checkpoint).exists() else None
    return init_ckpt, teacher_ckpt
