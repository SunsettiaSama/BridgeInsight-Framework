from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter


from src.chapter3_identifier.augment.labels import get_label_names

_DEFAULT_LABELS = get_label_names()


def compute_class_metrics(
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
) -> Dict:
    names = label_names or _DEFAULT_LABELS
    n_cls = num_classes or len(names)
    metrics = {}
    for cls in range(n_cls):
        yt = [1 if y == cls else 0 for y in y_true]
        yp = [1 if y == cls else 0 for y in y_pred]
        label = names[cls] if cls < len(names) else f"Class_{cls}"
        metrics[f"{label}_precision"] = float(precision_score(yt, yp, zero_division=0))
        metrics[f"{label}_recall"] = float(recall_score(yt, yp, zero_division=0))
        metrics[f"{label}_f1"] = float(f1_score(yt, yp, zero_division=0))
    viv_f1 = metrics[f"{names[1]}_f1"] if len(names) > 1 else 0.0
    rwiv_f1 = metrics[f"{names[2]}_f1"] if len(names) > 2 else 0.0
    metrics["viv_rwiv_mean_f1"] = (viv_f1 + rwiv_f1) / 2.0
    metrics["accuracy"] = float(np.mean(np.array(y_true) == np.array(y_pred)))
    return metrics


def save_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    output_path: Path,
    label_names: Optional[List[str]] = None,
    num_classes: Optional[int] = None,
) -> None:
    names = label_names or _DEFAULT_LABELS
    n_cls = num_classes or len(names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n_cls))
    ax.set_yticks(range(n_cls))
    ax.set_xticklabels(names[:n_cls])
    ax.set_yticklabels(names[:n_cls])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(n_cls):
        for j in range(n_cls):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


class TrainingVisualizer:
    def __init__(self, output_dir: Path, label_names: Optional[List[str]] = None, num_classes: int = 4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label_names = label_names or _DEFAULT_LABELS
        self.num_classes = num_classes or len(self.label_names)
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        self.history: List[Dict] = []

    def log_epoch(self, epoch: int, split: str, metrics: Dict[str, float], loss: float) -> None:
        self.writer.add_scalar(f"{split}/loss", loss, epoch)
        for k, v in metrics.items():
            self.writer.add_scalar(f"{split}/{k}", v, epoch)

    def log_confusion(self, epoch: int, y_true: List[int], y_pred: List[int]) -> None:
        path = self.output_dir / f"confusion_matrix_epoch_{epoch:03d}.png"
        save_confusion_matrix(
            y_true, y_pred, path, label_names=self.label_names, num_classes=self.num_classes
        )

    def log_confusion_named(self, epoch: int, y_true: List[int], y_pred: List[int], name: str) -> None:
        path = self.output_dir / f"confusion_matrix_{name}_epoch_{epoch:03d}.png"
        save_confusion_matrix(
            y_true, y_pred, path, label_names=self.label_names, num_classes=self.num_classes
        )

    def append_history(self, record: Dict) -> None:
        self.history.append(record)
        self.flush_history()

    def _write_json_atomic(self, path: Path, payload) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp.replace(path)

    def flush_history(self) -> None:
        self._write_json_atomic(self.output_dir / "metrics.json", self.history)
        if self.history:
            latest = dict(self.history[-1])
            latest["total_epochs"] = len(self.history)
            self._write_json_atomic(self.output_dir / "metrics_live.json", latest)

    def save_summary(self) -> None:
        self.flush_history()

    def close(self) -> None:
        self.save_summary()
        self.writer.close()
