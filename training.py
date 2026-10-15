"""Training helpers shared by baseline training, poisoning and defense."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common import Config

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def normalize(x: torch.Tensor) -> torch.Tensor:
    """x: float tensor in [0,1] -> standardised float tensor."""
    return (x - MNIST_MEAN) / MNIST_STD


def npz_to_loader(
    npz_path: str,
    batch_size: int,
    shuffle: bool,
    *,
    mean: float = MNIST_MEAN,
    std: float = MNIST_STD,
) -> DataLoader:
    """Build a DataLoader from an .npz file with `x` (uint8/float images) and `y` (labels).

    Images may be stored as uint8 [0,255], NCHW or NHWC, or float [0,1].
    """
    data = np.load(npz_path)
    x, y = data["x"], data["y"]
    x = np.asarray(x)
    if x.ndim == 3:  # (N,H,W) -> (N,1,H,W)
        x = x[:, None, :, :]
    elif x.ndim == 4 and x.shape[-1] == 1:  # (N,H,W,1) -> (N,1,H,W)
        x = np.transpose(x, (0, 3, 1, 2))
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    else:
        x = x.astype(np.float32)
        if x.max() > 1.0:
            x = x / 255.0
    xt = torch.from_numpy(x)
    xt = normalize(xt)
    yt = torch.from_numpy(y.astype(np.int64))
    loader = DataLoader(
        TensorDataset(xt, yt),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )
    return loader


def fit(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    verbose: bool = True,
) -> nn.Module:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        running, correct, n = 0.0, 0, 0
        t0 = time.perf_counter()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
        if verbose:
            print(
                f"  epoch {epoch + 1}/{epochs} "
                f"loss={running / n:.4f} acc={correct / n:.4f} "
                f"({time.perf_counter() - t0:.1f}s)"
            )
    return model
