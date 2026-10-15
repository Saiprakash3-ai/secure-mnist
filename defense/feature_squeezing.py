"""Blue team defense #2: feature squeezing (input sanitisation).

Median-filter denoising + bit-depth reduction of the input before classification.
Evaluated as a *post-hoc* defense: a clean-trained model's accuracy is recovered
on FGSM/PGD adversarial inputs without retraining.

    python defense/feature_squeezing.py --model models/baseline.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import RESULT_DIR, evaluate, confusion_and_report, load_model, save_metrics  # noqa: E402
from training import MNIST_MEAN, MNIST_STD  # noqa: E402
from train import mnist_loader  # noqa: E402


def median_squeeze(x: torch.Tensor, k: int = 3) -> torch.Tensor:
    pad = k // 2
    xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    patches = xp.unfold(2, k, 1).unfold(3, k, 1)  # (N, C, H, W, k, k)
    return patches.median(dim=-1).values.median(dim=-1).values


def bit_depth_reduce(x: torch.Tensor, bits: int = 4) -> torch.Tensor:
    levels = 2 ** bits
    return torch.round(x * (levels - 1)) / (levels - 1)


class SqueezedModel(nn.Module):
    """Apply feature squeezing in raw [0,1] space then the wrapped CNN."""

    def __init__(self, base: nn.Module, bits: int = 4, k: int = 3):
        super().__init__()
        self.base = base
        self.bits = bits
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Loader feeds normalised input; invert to [0,1] to squeeze, then renormalise.
        raw = x * MNIST_STD + MNIST_MEAN
        raw = bit_depth_reduce(raw, self.bits)
        if self.k > 1:
            raw = median_squeeze(raw, self.k)
        return self.base((raw - MNIST_MEAN) / MNIST_STD)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/baseline.pt")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    base = load_model(Path(args.model), args.device)
    squeezed = SqueezedModel(base, bits=args.bits, k=args.kernel).to(args.device).eval()

    # clean
    m = evaluate(squeezed, mnist_loader(train=False, batch_size=1000, shuffle=False), args.device)
    ex = confusion_and_report(m["y_true"], m["y_pred"])
    m.update({"tag": "squeezed_clean", **ex})
    save_metrics(m, RESULT_DIR / "squeezed_clean.json")
    print(f"[squeezed clean] acc={m['accuracy']:.4f}")

    # adversarial
    for name in ("adv_fgsm", "adv_pgd"):
        path = Path("data") / f"{name}.npz"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        from training import npz_to_loader
        m = evaluate(squeezed, npz_to_loader(str(path), 1000, shuffle=False), args.device)
        ex = confusion_and_report(m["y_true"], m["y_pred"])
        m.update({"tag": f"squeezed_{name}", **ex})
        save_metrics(m, RESULT_DIR / f"squeezed_{name}.json")
        print(f"[squeezed {name}] acc={m['accuracy']:.4f}")


if __name__ == "__main__":
    main()
