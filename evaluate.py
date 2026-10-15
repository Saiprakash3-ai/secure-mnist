"""Evaluate a saved model on clean MNIST or a custom .npz dataset (adversarial / poisoned).

Examples
--------
    python evaluate.py --model models/baseline.pt --data clean --tag baseline --metrics results/baseline_clean.json
    python evaluate.py --model models/baseline.pt --data data/adv_fgsm.npz --tag baseline_fgsm
"""
from __future__ import annotations

import argparse

import numpy as np

from common import (
    Config,
    RESULT_DIR,
    confusion_and_report,
    evaluate,
    load_model,
    save_confusion_png,
    save_metrics,
)
from train import mnist_loader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--data", type=str, default="clean", help="'clean' or path to .npz")
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--batch-size", type=int, default=Config.test_batch_size)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    model = load_model(__import__("pathlib").Path(args.model), cfg.device)

    if args.data == "clean":
        loader = mnist_loader(train=False, batch_size=args.batch_size, shuffle=False)
    else:
        from training import npz_to_loader
        loader = npz_to_loader(args.data, args.batch_size, shuffle=False)

    metrics = evaluate(model, loader, cfg.device)
    extra = confusion_and_report(metrics["y_true"], metrics["y_pred"])
    metrics.update(extra)
    metrics["tag"] = args.tag

    print(f"[{args.tag}] acc={metrics['accuracy']:.4f} loss={metrics['loss']:.4f} "
          f"n={metrics['num_samples']} infer={metrics['inference_time_per_sample_ms']:.3f} ms/sample")

    if not args.no_write:
        save_metrics(metrics, RESULT_DIR / f"{args.tag}.json")
        save_confusion_png(
            np.array(extra["confusion_matrix"]),
            RESULT_DIR / f"{args.tag}_confusion.png",
            title=f"{args.tag} - confusion",
        )
        print(f"[save] -> {RESULT_DIR / (args.tag + '.json')}")


if __name__ == "__main__":
    main()
