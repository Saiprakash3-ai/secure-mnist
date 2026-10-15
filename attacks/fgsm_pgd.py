"""Red team: evasion attacks (FGSM and PGD) on the MNIST CNN via ART.

OWASP ML Top 10 risks exercised:
  * ML01 - Input Manipulation Attack (evasion)
  * ML02 - Data Poisoning (adversarially-mixed training data, used later for defense)

    python attacks/fgsm_pgd.py --model models/baseline.pt
Outputs (in data/):
    adv_fgsm.npz, adv_pgd.npz          -> adversarial test sets
    adv_train_mix.npz                  -> clean + adversarial train samples (for blue team)
    adv_examples.png                   -> qualitative grid
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _art_utils import (  # noqa: E402
    ADV_DIR,
    build_classifier,
    clean_test,
    save_adv_npz,
    save_examples_png,
    train_subset,
)


def main() -> None:
    from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="models/baseline.pt")
    ap.add_argument("--fgsm-eps", type=float, default=0.2)
    ap.add_argument("--pgd-eps", type=float, default=0.2)
    ap.add_argument("--pgd-steps", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=10000)
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    classifier, model = build_classifier(args.model, args.device)
    # The baseline model must be in eval mode for ART.
    model.eval()

    x_test, y_test = clean_test()
    x_test, y_test = x_test[: args.n_test], y_test[: args.n_test]
    x_tr, y_tr = train_subset(args.n_train)

    print(f"[fgsm] eps={args.fgsm_eps} on {len(x_test)} test samples")
    fgsm = FastGradientMethod(estimator=classifier, eps=args.fgsm_eps)
    x_fgsm = fgsm.generate(x=x_test)
    save_adv_npz(ADV_DIR / "adv_fgsm.npz", x_fgsm, y_test)

    print(f"[pgd] eps={args.pgd_eps} steps={args.pgd_steps} on {len(x_test)} test samples")
    pgd = ProjectedGradientDescent(
        estimator=classifier, eps=args.pgd_eps, eps_step=args.pgd_eps / 10,
        max_iter=args.pgd_steps, num_random_init=1,
    )
    x_pgd = pgd.generate(x=x_test)
    save_adv_npz(ADV_DIR / "adv_pgd.npz", x_pgd, y_test)

    save_examples_png(x_test, x_pgd, y_test, ADV_DIR / "adv_examples.png",
                      title="Clean vs PGD adversarial (eps=%.2f)" % args.pgd_eps)

    # Adversarial training set: clean train subset + adversarial versions of it.
    print(f"[train-adv] generating adversarial copies of {len(x_tr)} train samples")
    x_tr_fgsm = fgsm.generate(x=x_tr)
    x_tr_pgd = pgd.generate(x=x_tr)
    x_mix = np.concatenate([x_tr, x_tr_fgsm, x_tr_pgd], axis=0)
    y_mix = np.concatenate([y_tr, y_tr, y_tr], axis=0)
    perm = np.random.default_rng(0).permutation(len(x_mix))
    save_adv_npz(ADV_DIR / "adv_train_mix.npz", x_mix[perm], y_mix[perm])


if __name__ == "__main__":
    main()
