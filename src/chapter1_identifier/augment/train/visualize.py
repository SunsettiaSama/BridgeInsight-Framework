from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.tensorboard import SummaryWriter


LABEL_NAMES = ["Normal", "VIV", "RWIV", "Transition"]
FOCUS_CLASSES = [1, 2]


def compute_class_metrics(y_true: List[int], y_pred: List[int], num_classes: int = 4) -> Dict:
    metrics = {}
    for cls in range(num_classes):
        yt = [1 if y == cls else 0 for y in y_true]
        yp = [1 if y == cls else 0 for y in y_pred]
        metrics[f"{LABEL_NAMES[cls]}_precision"] = float(precision_score(yt, yp, zero_division=0))
        metrics[f"{LABEL_NAMES[cls]}_recall"] = float(recall_score(yt, yp, zero_division=0))
        metrics[f"{LABEL_NAMES[cls]}_f1"] = float(f1_score(yt, yp, zero_division=0))
    viv_f1 = metrics["VIV_f1"]
    rwiv_f1 = metrics["RWIV_f1"]
    metrics["viv_rwiv_mean_f1"] = (viv_f1 + rwiv_f1) / 2.0
    metrics["accuracy"] = float(np.mean(np.array(y_true) == np.array(y_pred)))
    return metrics


def save_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    output_path: Path,
    num_classes: int = 4,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticklabels(LABEL_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


class TrainingVisualizer:
    def __init__(self, output_dir: Path, num_classes: int = 4):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_classes = num_classes
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        self.history: List[Dict] = []

    def log_epoch(self, epoch: int, split: str, metrics: Dict[str, float], loss: float) -> None:
        self.writer.add_scalar(f"{split}/loss", loss, epoch)
        for k, v in metrics.items():
            self.writer.add_scalar(f"{split}/{k}", v, epoch)

    def log_confusion(self, epoch: int, y_true: List[int], y_pred: List[int]) -> None:
        path = self.output_dir / f"confusion_matrix_epoch_{epoch:03d}.png"
        save_confusion_matrix(y_true, y_pred, path, self.num_classes)

    def append_history(self, record: Dict) -> None:
        self.history.append(record)
        self.flush_history()

    def flush_history(self) -> None:
        with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
        if self.history:
            latest = dict(self.history[-1])
            latest["total_epochs"] = len(self.history)
            with open(self.output_dir / "metrics_live.json", "w", encoding="utf-8") as f:
                json.dump(latest, f, ensure_ascii=False, indent=2)

    def save_summary(self) -> None:
        self.flush_history()

    def close(self) -> None:
        self.save_summary()
        self.writer.close()
