"""Red team: training-time data poisoning attacks.

Two variants:
  1. Label-flip poisoning  -> targeted misclassification after retraining.
  2. Backdoor / trojan     -> trigger patch forces a chosen target class.

OWASP ML Top 10 risks exercised:
  * ML02 - Data Poisoning Attack
  * ML07 - Transfer Learning Attack (concept of injected triggers leaking across models)

    python attacks/data_poisoning.py
Outputs (in data/):
    poisoned_label_flip.npz, poisoned_backdoor.npz, backdoor_test.npz,
    clean_test.npz, poison_examples.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _art_utils import ADV_DIR, _mnist_split, save_adv_npz  # noqa: E402


def add_trigger(x: np.ndarray, size: int = 3) -> np.ndarray:
    """Paint a white square in the bottom-right corner (in place on a copy)."""
    x = x.copy()
    x[:, :, -size:, -size:] = 1.0
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=int, default=1, help="source class to flip")
    ap.add_argument("--dst", type=int, default=7, help="target class")
    ap.add_argument("--flip-frac", type=float, default=1.0, help="fraction of source class to flip")
    ap.add_argument("--backdoor-frac", type=float, default=0.05, help="fraction of train to trojan")
    ap.add_argument("--backdoor-target", type=int, default=0)
    ap.add_argument("--trigger-size", type=int, default=3)
    args = ap.parse_args()

    x_tr, y_tr = _mnist_split(train=True)
    x_te, y_te = _mnist_split(train=False)

    # --- 1. Label-flip poisoning ----------------------------------------- #
    y_flip = y_tr.copy()
    src_idx = np.where(y_tr == args.src)[0]
    n_flip = int(len(src_idx) * args.flip_frac)
    chosen = np.random.default_rng(1).permutation(src_idx)[:n_flip]
    y_flip[chosen] = args.dst
    print(f"[label-flip] class {args.src}->{args.dst}: flipped {n_flip} labels")
    save_adv_npz(ADV_DIR / "poisoned_label_flip.npz", x_tr, y_flip)

    # --- 2. Backdoor / trojan -------------------------------------------- #
    y_bd = y_tr.copy()
    n_bd = int(len(x_tr) * args.backdoor_frac)
    bd_idx = np.random.default_rng(2).permutation(len(x_tr))[:n_bd]
    x_bd = x_tr.copy()
    x_bd[bd_idx] = add_trigger(x_bd[bd_idx], args.trigger_size)
    y_bd[bd_idx] = args.backdoor_target
    print(f"[backdoor] trojaned {n_bd} samples -> target class {args.backdoor_target}")
    save_adv_npz(ADV_DIR / "poisoned_backdoor.npz", x_bd, y_bd)

    # Triggered test set (labels unchanged) to measure backdoor success.
    x_te_trig = add_trigger(x_te, args.trigger_size)
    save_adv_npz(ADV_DIR / "backdoor_test.npz", x_te_trig, y_te)
    save_adv_npz(ADV_DIR / "clean_test.npz", x_te, y_te)

    # --- visualisation --------------------------------------------------- #
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 6, figsize=(12, 4))
    for i in range(6):
        axes[0, i].imshow(x_bd[bd_idx[i], 0], cmap="gray")
        axes[0, i].set_title(f"bd label={y_bd[bd_idx[i]]}", fontsize=9)
        axes[0, i].axis("off")
        axes[1, i].imshow(x_te_trig[i, 0], cmap="gray")
        axes[1, i].set_title(f"test trig={y_te[i]}", fontsize=9)
        axes[1, i].axis("off")
    fig.suptitle("Backdoor: trojaned train (top) and triggered test (bottom)")
    fig.tight_layout()
    fig.savefig(ADV_DIR / "poison_examples.png", dpi=120)
    plt.close(fig)
    print(f"[save] {ADV_DIR / 'poison_examples.png'}")


if __name__ == "__main__":
    main()
