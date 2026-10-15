"""Shared helpers for the red-team attack scripts (ART wrappers, data I/O)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import DATA_DIR, SimpleCNN, load_model  # noqa: E402
from training import MNIST_MEAN, MNIST_STD  # noqa: E402

ATK_DIR = Path(__file__).resolve().parent
ADV_DIR = ATK_DIR.parent / "data"
ADV_DIR.mkdir(parents=True, exist_ok=True)


class NormalizedModel(nn.Module):
    """Wrap the CNN so it accepts raw [0,1] pixels (as ART's clip_values expects)."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base((x - MNIST_MEAN) / MNIST_STD)


def build_classifier(model_path: str, device: str = "cpu", *, nb_classes: int = 10):
    from art.estimators.classification import PyTorchClassifier

    base = load_model(Path(model_path), device)
    wrapped = NormalizedModel(base).to(device)
    classifier = PyTorchClassifier(
        model=wrapped,
        loss=nn.CrossEntropyLoss(),
        input_shape=(1, 28, 28),
        nb_classes=nb_classes,
        clip_values=(0.0, 1.0),
    )
    return classifier, base


def _mnist_split(train: bool) -> tuple[np.ndarray, np.ndarray]:
    ds = datasets.MNIST(str(DATA_DIR), train=train, download=True, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)
    x, y = next(iter(loader))
    return x.numpy().astype(np.float32), y.numpy().astype(np.int64)


def clean_test() -> tuple[np.ndarray, np.ndarray]:
    return _mnist_split(train=False)


def train_subset(n: int, seed: int = 1337) -> tuple[np.ndarray, np.ndarray]:
    x, y = _mnist_split(train=True)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))[:n]
    return x[idx], y[idx]


def save_adv_npz(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    """x: float [0,1] NCHW -> stored as uint8 for compact files."""
    xi = np.clip(x, 0.0, 1.0)
    xi = (xi * 255.0).round().astype(np.uint8)
    np.savez_compressed(path, x=xi, y=y.astype(np.int64))
    print(f"[save] {path}  x={xi.shape} y={y.shape}")


def save_examples_png(clean: np.ndarray, adv: np.ndarray, y: np.ndarray, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = min(6, len(clean))
    fig, axes = plt.subplots(2, k, figsize=(2 * k, 4))
    for i in range(k):
        axes[0, i].imshow(clean[i, 0], cmap="gray")
        axes[0, i].set_title(f"clean {y[i]}", fontsize=9)
        axes[0, i].axis("off")
        axes[1, i].imshow(adv[i, 0], cmap="gray")
        axes[1, i].set_title("adv", fontsize=9)
        axes[1, i].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[save] {path}")
