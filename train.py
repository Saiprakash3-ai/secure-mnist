"""Train the MNIST CNN, optionally on a custom (poisoned / adversarial-mixed) dataset.

Examples
--------
Baseline:
    python train.py --epochs 5 --out baseline.pt --tag baseline

Train on poisoned data:
    python train.py --train-npz data/poisoned_train.npz --out poisoned.pt --tag poisoned

Train on clean + adversarial (blue-team adversarial training):
    python train.py --train-npz data/adv_train_mix.npz --out adversarially_trained.pt --tag adv_trained
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common import (
    Config,
    DATA_DIR,
    MODEL_DIR,
    RESULT_DIR,
    SimpleCNN,
    evaluate,
    confusion_and_report,
    load_model,
    save_confusion_png,
    save_metrics,
    save_model,
    set_seed,
)
from training import MNIST_MEAN, MNIST_STD, fit


def mnist_loader(train: bool, batch_size: int, shuffle: bool) -> DataLoader:
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))])
    ds = datasets.MNIST(str(DATA_DIR), train=train, download=True, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def build_train_loader(cfg: Config, train_npz: str | None) -> DataLoader:
    if train_npz:
        from training import npz_to_loader
        print(f"[data] training on custom set: {train_npz}")
        return npz_to_loader(train_npz, cfg.batch_size, shuffle=True)
    print("[data] training on clean MNIST train split")
    return mnist_loader(train=True, batch_size=cfg.batch_size, shuffle=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=Config.epochs)
    ap.add_argument("--lr", type=float, default=Config.lr)
    ap.add_argument("--batch-size", type=int, default=Config.batch_size)
    ap.add_argument("--train-npz", type=str, default=None)
    ap.add_argument("--out", type=str, default="baseline.pt")
    ap.add_argument("--tag", type=str, default="baseline")
    args = ap.parse_args()

    cfg = Config(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)
    set_seed(cfg.seed)
    print(f"[env] device={cfg.device} seed={cfg.seed}")

    train_loader = build_train_loader(cfg, args.train_npz)
    test_loader = mnist_loader(train=False, batch_size=cfg.test_batch_size, shuffle=False)

    model = SimpleCNN().to(cfg.device)
    print(f"[train] tag={args.tag} epochs={cfg.epochs}")
    fit(model, train_loader, cfg.epochs, cfg.lr, cfg.device)

    model_path = MODEL_DIR / args.out
    save_model(model, model_path)
    print(f"[save] model -> {model_path}")

    print("[eval] clean MNIST test set")
    metrics = evaluate(model, test_loader, cfg.device)
    extra = confusion_and_report(metrics["y_true"], metrics["y_pred"])
    metrics.update(extra)
    metrics["tag"] = args.tag
    save_metrics(metrics, RESULT_DIR / f"{args.tag}_clean.json")
    save_confusion_png(
        np.array(extra["confusion_matrix"]),
        RESULT_DIR / f"{args.tag}_clean_confusion.png",
        title=f"{args.tag} - clean test confusion",
    )
    print(f"  accuracy={metrics['accuracy']:.4f}  loss={metrics['loss']:.4f}  "
          f"inference={metrics['inference_time_per_sample_ms']:.3f} ms/sample")


if __name__ == "__main__":
    main()
