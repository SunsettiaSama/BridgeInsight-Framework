from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.chapter3_identifier.augment.models.dual_stream_res_cnn import DualStreamResCNN
from src.chapter3_identifier.augment.models.quad_stream_dual_head_res_cnn import (
    QuadStreamDualHeadContextResCNN,
    QuadStreamDualHeadResCNN,
    QuadStreamSerialContextDualHeadResCNN,
)
from src.chapter3_identifier.augment.train.visualize import TrainingVisualizer, compute_class_metrics

logger = logging.getLogger(__name__)

_LONG_CONTEXT_MODEL_CLASSES = (QuadStreamDualHeadContextResCNN, QuadStreamSerialContextDualHeadResCNN)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def teacher_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    temp = max(float(temperature), 1e-6)
    student_log_prob = F.log_softmax(student_logits / temp, dim=1)
    teacher_prob = F.softmax(teacher_logits / temp, dim=1)
    return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (temp ** 2)


class DualStreamTrainer:
    def __init__(
        self,
        model: DualStreamResCNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str,
        epochs: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        gradient_clip_norm: float = 0.5,
        class_weights: Optional[List[float]] = None,
        focal_gamma: float = 2.0,
        teacher_model: Optional[DualStreamResCNN] = None,
        teacher_reg_lambda: float = 0.0,
        teacher_reg_temperature: float = 2.0,
        device: Optional[str] = None,
        label_names: Optional[List[str]] = None,
        num_classes: int = 4,
        dataset_info: Optional[Dict] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.gradient_clip_norm = gradient_clip_norm
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.teacher_model = teacher_model
        self.teacher_reg_lambda = float(teacher_reg_lambda)
        self.teacher_reg_temperature = float(teacher_reg_temperature)
        if self.teacher_model is not None:
            self.teacher_model.to(self.device)
            self.teacher_model.eval()

        weight = None
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        self.criterion = FocalLoss(gamma=focal_gamma, weight=weight)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        self.label_names = label_names
        self.num_classes = num_classes
        self.visualizer = TrainingVisualizer(
            self.output_dir, label_names=label_names, num_classes=num_classes
        )
        self.best_metric = -1.0
        self.best_epoch = 0
        self.dataset_info = dict(dataset_info or {})
        self.dataset_info.setdefault("train_size", int(len(self.train_loader.dataset)))
        self.dataset_info.setdefault("val_size", int(len(self.val_loader.dataset)))

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor, time_x, psd_x) -> Tuple[torch.Tensor, float]:
        focal = self.criterion(logits, labels)
        reg = torch.tensor(0.0, device=self.device)
        if self.teacher_model is not None and self.teacher_reg_lambda > 0:
            with torch.no_grad():
                teacher_logits = self.teacher_model(time_x, psd_x)
            reg = teacher_kl_loss(logits, teacher_logits, self.teacher_reg_temperature)
        total = focal + self.teacher_reg_lambda * reg
        return total, float(reg.item())

    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, float, Dict, List[int], List[int]]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_reg = 0.0
        n_batches = 0
        y_true: List[int] = []
        y_pred: List[int] = []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for time_x, psd_x, labels in loader:
                time_x = time_x.to(self.device)
                psd_x = psd_x.to(self.device)
                labels = labels.to(self.device)

                if train:
                    self.optimizer.zero_grad()
                logits = self.model(time_x, psd_x)
                loss, reg_val = self._compute_loss(logits, labels, time_x, psd_x)
                if train:
                    loss.backward()
                    if self.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.optimizer.step()

                total_loss += float(loss.item())
                total_reg += reg_val
                n_batches += 1
                preds = logits.argmax(dim=1).detach().cpu().tolist()
                y_pred.extend(preds)
                y_true.extend(labels.detach().cpu().tolist())

        avg_loss = total_loss / max(n_batches, 1)
        avg_reg = total_reg / max(n_batches, 1)
        metrics = compute_class_metrics(
            y_true, y_pred, label_names=self.label_names, num_classes=self.num_classes
        )
        return avg_loss, avg_reg, metrics, y_true, y_pred

    def train(self, round_idx: int = 1) -> Dict:
        teacher_note = (
            f"teacher_reg λ={self.teacher_reg_lambda}, T={self.teacher_reg_temperature}"
            if self.teacher_model is not None and self.teacher_reg_lambda > 0
            else "teacher_reg=off"
        )
        logger.info(f"开始 DualStream 训练，epochs={self.epochs}, device={self.device}, {teacher_note}")
        last_val_metrics = {}

        for epoch in range(1, self.epochs + 1):
            train_loss, train_reg, train_metrics, _, _ = self._run_epoch(self.train_loader, train=True)
            val_loss, val_reg, val_metrics, y_true, y_pred = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step()

            self.visualizer.log_epoch(epoch, "train", train_metrics, train_loss)
            self.visualizer.log_epoch(epoch, "val", val_metrics, val_loss)
            self.visualizer.log_confusion(epoch, y_true, y_pred)
            self.visualizer.append_history(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_teacher_reg": train_reg,
                    "val_teacher_reg": val_reg,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "dataset_info": self.dataset_info,
                }
            )

            score = val_metrics["viv_rwiv_mean_f1"]
            if score > self.best_metric:
                self.best_metric = score
                self.best_epoch = epoch
                cfg = self.model.cfg
                ckpt = {
                    "model_state_dict": self.model.state_dict(),
                    "config": {
                        "time_branch": vars(cfg.time_branch),
                        "spec_branch": vars(cfg.spec_branch),
                        "fusion": {"type": cfg.fusion_type},
                        "num_classes": cfg.num_classes,
                        "dropout_prob": cfg.time_branch.dropout_prob,
                        "fc_hidden_dims": cfg.time_branch.fc_hidden_dims,
                        "training_profile": self.dataset_info.get("training_profile"),
                        "training_profile_source": self.dataset_info.get("training_profile_source"),
                    },
                    "round_idx": round_idx,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "best_metric": score,
                }
                torch.save(ckpt, self.output_dir / "best_checkpoint.pth")

            last_val_metrics = val_metrics
            logger.info(
                f"Epoch {epoch}/{self.epochs} | train_loss={train_loss:.4f} "
                f"train_reg={train_reg:.4f} val_loss={val_loss:.4f} viv_rwiv_f1={score:.4f}"
            )

        self.visualizer.close()
        return {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "val_metrics": last_val_metrics,
            "checkpoint": str(self.output_dir / "best_checkpoint.pth"),
        }


class QuadStreamDualHeadTrainer:
    def __init__(
        self,
        model: QuadStreamDualHeadResCNN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: str,
        epochs: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        gradient_clip_norm: float = 0.5,
        class_weights: Optional[List[float]] = None,
        focal_gamma: float = 2.0,
        inplane_loss_weight: float = 1.0,
        outplane_loss_weight: float = 1.0,
        legacy_teacher: Optional[DualStreamResCNN] = None,
        legacy_reg_lambda: float = 0.0,
        legacy_reg_temperature: float = 2.0,
        same_structure_teacher: Optional[nn.Module] = None,
        same_structure_reg_lambda: float = 0.0,
        same_structure_reg_temperature: float = 2.0,
        device: Optional[str] = None,
        label_names: Optional[List[str]] = None,
        num_classes: int = 4,
        dataset_info: Optional[Dict] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.gradient_clip_norm = gradient_clip_norm
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.inplane_loss_weight = float(inplane_loss_weight)
        self.outplane_loss_weight = float(outplane_loss_weight)
        self.legacy_teacher = legacy_teacher
        self.legacy_reg_lambda = float(legacy_reg_lambda)
        self.legacy_reg_temperature = float(legacy_reg_temperature)
        self.same_structure_teacher = same_structure_teacher
        self.same_structure_reg_lambda = float(same_structure_reg_lambda)
        self.same_structure_reg_temperature = float(same_structure_reg_temperature)
        if self.legacy_teacher is not None:
            self.legacy_teacher.to(self.device)
            self.legacy_teacher.eval()
            for param in self.legacy_teacher.parameters():
                param.requires_grad = False
        if self.same_structure_teacher is not None:
            self.same_structure_teacher.to(self.device)
            self.same_structure_teacher.eval()
            for param in self.same_structure_teacher.parameters():
                param.requires_grad = False

        weight = None
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        self.criterion = FocalLoss(gamma=focal_gamma, weight=weight)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)
        self.label_names = label_names
        self.num_classes = num_classes
        self.visualizer = TrainingVisualizer(
            self.output_dir, label_names=label_names, num_classes=num_classes
        )
        self.best_metric = -1.0
        self.best_epoch = 0
        self.dataset_info = dict(dataset_info or {})
        self.dataset_info.setdefault("train_size", int(len(self.train_loader.dataset)))
        self.dataset_info.setdefault("val_size", int(len(self.val_loader.dataset)))
        self.uses_long_context = isinstance(self.model, _LONG_CONTEXT_MODEL_CLASSES)
        self.use_context_input = self.uses_long_context and self.dataset_info.get("context_mode") != "short_only"
        self.use_wind_features = int(getattr(self.model, "wind_feature_dim", 0)) > 0

    def _run_epoch(self, loader: DataLoader, train: bool):
        if train:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_in_loss = 0.0
        total_out_loss = 0.0
        total_reg = 0.0
        n_batches = 0
        in_true: List[int] = []
        in_pred: List[int] = []
        out_true: List[int] = []
        out_pred: List[int] = []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                in_context_x = None
                out_context_x = None
                wind_features = None
                if self.use_context_input:
                    if self.use_wind_features:
                        (
                            in_time_x,
                            in_psd_x,
                            in_context_x,
                            out_time_x,
                            out_psd_x,
                            out_context_x,
                            wind_features,
                            in_labels,
                            out_labels,
                        ) = batch
                    else:
                        (
                            in_time_x,
                            in_psd_x,
                            in_context_x,
                            out_time_x,
                            out_psd_x,
                            out_context_x,
                            in_labels,
                            out_labels,
                        ) = batch
                    in_context_x = in_context_x.to(self.device)
                    out_context_x = out_context_x.to(self.device)
                else:
                    if self.use_wind_features:
                        in_time_x, in_psd_x, out_time_x, out_psd_x, wind_features, in_labels, out_labels = batch
                    else:
                        in_time_x, in_psd_x, out_time_x, out_psd_x, in_labels, out_labels = batch
                in_time_x = in_time_x.to(self.device)
                in_psd_x = in_psd_x.to(self.device)
                out_time_x = out_time_x.to(self.device)
                out_psd_x = out_psd_x.to(self.device)
                if wind_features is not None:
                    wind_features = wind_features.to(self.device)
                in_labels = in_labels.to(self.device)
                out_labels = out_labels.to(self.device)

                if train:
                    self.optimizer.zero_grad()
                if self.uses_long_context:
                    in_logits, out_logits = self.model(
                        in_time_x,
                        in_psd_x,
                        in_context_x,
                        out_time_x,
                        out_psd_x,
                        out_context_x,
                        wind_features=wind_features,
                        use_context=self.use_context_input,
                    )
                else:
                    in_logits, out_logits = self.model(
                        in_time_x,
                        in_psd_x,
                        out_time_x,
                        out_psd_x,
                        wind_features=wind_features,
                    )
                in_loss = self.criterion(in_logits, in_labels)
                out_loss = self.criterion(out_logits, out_labels)
                reg_loss = torch.tensor(0.0, device=self.device)
                if self.legacy_teacher is not None and self.legacy_reg_lambda > 0:
                    with torch.no_grad():
                        legacy_in_logits = self.legacy_teacher(in_time_x, in_psd_x)
                        legacy_out_logits = self.legacy_teacher(out_time_x, out_psd_x)
                    reg_in = teacher_kl_loss(
                        in_logits, legacy_in_logits, self.legacy_reg_temperature
                    )
                    reg_out = teacher_kl_loss(
                        out_logits, legacy_out_logits, self.legacy_reg_temperature
                    )
                    reg_loss = reg_loss + self.legacy_reg_lambda * 0.5 * (reg_in + reg_out)
                if self.same_structure_teacher is not None and self.same_structure_reg_lambda > 0:
                    with torch.no_grad():
                        if self.uses_long_context:
                            same_in_logits, same_out_logits = self.same_structure_teacher(
                                in_time_x,
                                in_psd_x,
                                in_context_x,
                                out_time_x,
                                out_psd_x,
                                out_context_x,
                                wind_features=wind_features,
                                use_context=self.use_context_input,
                            )
                        else:
                            same_in_logits, same_out_logits = self.same_structure_teacher(
                                in_time_x,
                                in_psd_x,
                                out_time_x,
                                out_psd_x,
                                wind_features=wind_features,
                            )
                    same_reg_in = teacher_kl_loss(
                        in_logits, same_in_logits, self.same_structure_reg_temperature
                    )
                    same_reg_out = teacher_kl_loss(
                        out_logits, same_out_logits, self.same_structure_reg_temperature
                    )
                    reg_loss = reg_loss + self.same_structure_reg_lambda * 0.5 * (same_reg_in + same_reg_out)
                base_loss = (
                    self.inplane_loss_weight * in_loss
                    + self.outplane_loss_weight * out_loss
                )
                loss = base_loss + reg_loss
                if train:
                    loss.backward()
                    if self.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.optimizer.step()

                total_loss += float(loss.item())
                total_in_loss += float(in_loss.item())
                total_out_loss += float(out_loss.item())
                total_reg += float(reg_loss.item())
                n_batches += 1

                in_pred.extend(in_logits.argmax(dim=1).detach().cpu().tolist())
                out_pred.extend(out_logits.argmax(dim=1).detach().cpu().tolist())
                in_true.extend(in_labels.detach().cpu().tolist())
                out_true.extend(out_labels.detach().cpu().tolist())

        avg_loss = total_loss / max(n_batches, 1)
        avg_in_loss = total_in_loss / max(n_batches, 1)
        avg_out_loss = total_out_loss / max(n_batches, 1)
        avg_reg = total_reg / max(n_batches, 1)
        in_metrics = compute_class_metrics(
            in_true, in_pred, label_names=self.label_names, num_classes=self.num_classes
        )
        out_metrics = compute_class_metrics(
            out_true, out_pred, label_names=self.label_names, num_classes=self.num_classes
        )
        merged_metrics = {}
        for key, value in in_metrics.items():
            merged_metrics[f"inplane_{key}"] = value
        for key, value in out_metrics.items():
            merged_metrics[f"outplane_{key}"] = value
        merged_metrics["joint_viv_rwiv_mean_f1"] = (
            in_metrics["viv_rwiv_mean_f1"] + out_metrics["viv_rwiv_mean_f1"]
        ) / 2.0
        return (
            avg_loss,
            avg_in_loss,
            avg_out_loss,
            avg_reg,
            merged_metrics,
            in_true,
            in_pred,
            out_true,
            out_pred,
        )

    def train(self, round_idx: int = 2) -> Dict:
        reg_note = (
            f"legacy_reg λ={self.legacy_reg_lambda}, T={self.legacy_reg_temperature}"
            if self.legacy_teacher is not None and self.legacy_reg_lambda > 0
            else "legacy_reg=off"
        )
        same_note = (
            f"same_structure_reg λ={self.same_structure_reg_lambda}, T={self.same_structure_reg_temperature}"
            if self.same_structure_teacher is not None and self.same_structure_reg_lambda > 0
            else "same_structure_reg=off"
        )
        logger.info(
            f"开始 round {round_idx} 配对双头联合训练，epochs={self.epochs}, device={self.device}, "
            f"loss_w(in/out)=({self.inplane_loss_weight:.2f}/{self.outplane_loss_weight:.2f}), {reg_note}, {same_note}"
        )
        last_val_metrics = {}
        for epoch in range(1, self.epochs + 1):
            (
                train_loss,
                train_in_loss,
                train_out_loss,
                train_reg,
                train_metrics,
                _,
                _,
                _,
                _,
            ) = self._run_epoch(self.train_loader, train=True)
            (
                val_loss,
                val_in_loss,
                val_out_loss,
                val_reg,
                val_metrics,
                in_true,
                in_pred,
                out_true,
                out_pred,
            ) = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step()

            self.visualizer.log_epoch(epoch, "train", train_metrics, train_loss)
            self.visualizer.log_epoch(epoch, "val", val_metrics, val_loss)
            self.visualizer.log_confusion(epoch, in_true + out_true, in_pred + out_pred)
            self.visualizer.log_confusion_named(epoch, in_true, in_pred, "inplane")
            self.visualizer.log_confusion_named(epoch, out_true, out_pred, "outplane")
            self.visualizer.append_history(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_inplane_loss": train_in_loss,
                    "train_outplane_loss": train_out_loss,
                    "val_inplane_loss": val_in_loss,
                    "val_outplane_loss": val_out_loss,
                    "train_legacy_reg": train_reg,
                    "val_legacy_reg": val_reg,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "dataset_info": self.dataset_info,
                }
            )

            score = float(val_metrics["joint_viv_rwiv_mean_f1"])
            if score > self.best_metric:
                self.best_metric = score
                self.best_epoch = epoch
                if isinstance(self.model, QuadStreamSerialContextDualHeadResCNN):
                    model_type = "quad_stream_serial_context_dual_head"
                elif isinstance(self.model, QuadStreamDualHeadContextResCNN):
                    model_type = "quad_stream_dual_head_context"
                else:
                    model_type = "quad_stream_dual_head"
                config = {
                    "model_type": model_type,
                    "time_branch": vars(self.model.in_time_encoder.backbone.cfg),
                    "spec_branch": vars(self.model.in_spec_encoder.backbone.cfg),
                    "num_classes": self.model.num_classes,
                    "fusion_hidden_dim": self.model.fusion_hidden_dim,
                    "fusion_dropout": self.model.fusion_dropout,
                    "cross_attn_heads": self.model.cross_attn_heads,
                    "wind_feature_dim": int(getattr(self.model, "wind_feature_dim", 0)),
                    "enable_wind_features": bool(int(getattr(self.model, "wind_feature_dim", 0)) > 0),
                    "inplane_loss_weight": self.inplane_loss_weight,
                    "outplane_loss_weight": self.outplane_loss_weight,
                    "legacy_reg_lambda": self.legacy_reg_lambda,
                    "legacy_reg_temperature": self.legacy_reg_temperature,
                    "same_structure_reg_lambda": self.same_structure_reg_lambda,
                    "same_structure_reg_temperature": self.same_structure_reg_temperature,
                }
                if self.uses_long_context:
                    config["context_branch"] = vars(self.model.in_context_encoder.backbone.cfg)
                    config["context_mode"] = self.dataset_info.get("context_mode", "short_long")
                    for key in (
                        "context_input_size",
                        "context_total_seconds",
                        "context_allow_cross_file",
                    ):
                        if key in self.dataset_info:
                            config[key] = self.dataset_info[key]
                config["training_profile"] = self.dataset_info.get("training_profile")
                config["training_profile_source"] = self.dataset_info.get("training_profile_source")
                ckpt = {
                    "model_state_dict": self.model.state_dict(),
                    "config": config,
                    "round_idx": round_idx,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "best_metric": score,
                }
                torch.save(ckpt, self.output_dir / "best_checkpoint.pth")

            last_val_metrics = val_metrics
            logger.info(
                f"Epoch {epoch}/{self.epochs} | train_loss={train_loss:.4f} "
                f"(in={train_in_loss:.4f}, out={train_out_loss:.4f}, reg={train_reg:.4f}) "
                f"val_loss={val_loss:.4f} (in={val_in_loss:.4f}, out={val_out_loss:.4f}, reg={val_reg:.4f}) "
                f"joint_viv_rwiv_f1={score:.4f}"
            )

        self.visualizer.close()
        return {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "val_metrics": last_val_metrics,
            "checkpoint": str(self.output_dir / "best_checkpoint.pth"),
        }
