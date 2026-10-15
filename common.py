"""Shared model, data, metrics and config for the MNIST security lab.

Used by: train.py, evaluate.py, attacks/*, defense/*.
"""
from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
for _d in (DATA_DIR, MODEL_DIR, RESULT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    seed: int = 1337
    batch_size: int = 128
    test_batch_size: int = 1000
    epochs: int = 5
    lr: float = 1e-3
    num_classes: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class SimpleCNN(nn.Module):
    """A small CNN: 2 conv blocks + 2 FC layers."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    logits = model(x)
    return logits.argmax(dim=1).cpu().numpy()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str, criterion=None) -> dict:
    """Return loss, accuracy, per-sample inference time and raw predictions."""
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    y_true, y_pred = [], []
    t0 = time.perf_counter()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        n += y.size(0)
        y_true.append(y.cpu().numpy())
        y_pred.append(preds.cpu().numpy())
    elapsed = time.perf_counter() - t0
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return {
        "loss": total_loss / n,
        "accuracy": correct / n,
        "num_samples": n,
        "inference_time_total_s": elapsed,
        "inference_time_per_sample_ms": 1e3 * elapsed / n,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def confusion_and_report(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> dict:
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    report = classification_report(
        y_true, y_pred, labels=list(range(num_classes)), output_dict=True, zero_division=0
    )
    return {"confusion_matrix": cm.tolist(), "classification_report": report}


def save_metrics(metrics: dict, path: Path) -> None:
    """Persist metrics to JSON, dropping bulky arrays."""
    serializable = {
        k: v for k, v in metrics.items() if k not in ("y_true", "y_pred")
    }
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def save_confusion_png(cm, path: Path, title: str = "Confusion matrix") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# --------------------------------------------------------------------------- #
# Model persistence
# --------------------------------------------------------------------------- #
def save_model(model: nn.Module, path: Path) -> None:
    torch.save(model.state_dict(), path)


def load_model(path: Path, device: str) -> SimpleCNN:
    model = SimpleCNN().to(device)
    # weights_only=True prevents arbitrary code execution via pickle payloads
    # (SAST finding B614 / OWASP ML06 supply-chain mitigation).
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model
