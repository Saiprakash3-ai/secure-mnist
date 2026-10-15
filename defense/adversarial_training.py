"""Blue team: adversarial training + a second defense (feature squeezing).

Defense 1 - Adversarial training:
    Retrain the CNN on clean data mixed with FGSM/PGD adversarial examples
    (data/adv_train_mix.npz produced by attacks/fgsm_pgd.py).

Defense 2 - Feature squeezing (input preprocessing):
    Median-filter denoising + bit-depth reduction before classification,
    applied at inference to strip high-frequency adversarial perturbations.

Control mappings:
  * Robust model -> OWASP ML: mitigate ML01 Input Manipulation (evasion)
  * Feature squeezing -> input validation / sanitisation control
  * Retraining on mixed data -> partially mitigates ML02 Data Poisoning

    python defense/adversarial_training.py --mix data/adv_train_mix.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import (  # noqa: E402
    Config,
    MODEL_DIR,
    RESULT_DIR,
    SimpleCNN,
    confusion_and_report,
    evaluate,
    save_confusion_png,
    save_metrics,
    save_model,
    set_seed,
)
from training import fit, npz_to_loader  # noqa: E402
from train import mnist_loader  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mix", type=str, default="data/adv_train_mix.npz")
    ap.add_argument("--epochs", type=int, default=Config.epochs)
    ap.add_argument("--lr", type=float, default=Config.lr)
    ap.add_argument("--out", type=str, default="adversarially_trained.pt")
    args = ap.parse_args()

    cfg = Config(epochs=args.epochs, lr=args.lr)
    set_seed(cfg.seed)

    train_loader = npz_to_loader(args.mix, cfg.batch_size, shuffle=True)
    test_loader = mnist_loader(train=False, batch_size=cfg.test_batch_size, shuffle=False)

    model = SimpleCNN().to(cfg.device)
    print(f"[defense] adversarial training on {args.mix} for {cfg.epochs} epochs")
    fit(model, train_loader, cfg.epochs, cfg.lr, cfg.device)
    save_model(model, MODEL_DIR / args.out)
    print(f"[save] model -> {MODEL_DIR / args.out}")

    # clean metrics
    m = evaluate(model, test_loader, cfg.device)
    extra = confusion_and_report(m["y_true"], m["y_pred"])
    m.update({"tag": "adv_trained", **extra})
    save_metrics(m, RESULT_DIR / "adv_trained_clean.json")
    save_confusion_png(np.array(extra["confusion_matrix"]),
                       RESULT_DIR / "adv_trained_clean_confusion.png",
                       title="adversarially-trained - clean test")
    print(f"[clean] acc={m['accuracy']:.4f} loss={m['loss']:.4f}")

    # adversarial metrics
    for name in ("adv_fgsm", "adv_pgd"):
        path = Path("data") / f"{name}.npz"
        if not path.exists():
            print(f"[skip] {path} not found")
            continue
        loader = npz_to_loader(str(path), cfg.test_batch_size, shuffle=False)
        ma = evaluate(model, loader, cfg.device)
        ex = confusion_and_report(ma["y_true"], ma["y_pred"])
        ma.update({"tag": f"adv_trained_{name}", **ex})
        save_metrics(ma, RESULT_DIR / f"adv_trained_{name}.json")
        print(f"[{name}] acc={ma['accuracy']:.4f}")


if __name__ == "__main__":
    main()
