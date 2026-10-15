"""Run the full red/blue-team pipeline end to end and emit a summary table.

    python run_all.py --epochs 5

Steps: baseline train -> evasion attacks -> poisoning -> poisoned retrains ->
adversarial training -> feature squeezing -> results/summary.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=HERE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--skip-attacks", action="store_true")
    args = ap.parse_args()
    py = sys.executable

    # 1-2. Baseline model + clean metrics
    run([py, "train.py", "--epochs", str(args.epochs), "--out", "baseline.pt", "--tag", "baseline"])

    if not args.skip_attacks:
        # 4. Red team: evasion (FGSM / PGD) + build adversarial training set
        run([py, "attacks/fgsm_pgd.py", "--model", "models/baseline.pt"])
        # 4b. Red team: data poisoning (label flip + backdoor)
        run([py, "attacks/data_poisoning.py"])

    # 6. Retest baseline on adversarial data
    for name in ("adv_fgsm", "adv_pgd", "backdoor_test"):
        run([py, "evaluate.py", "--model", "models/baseline.pt",
             "--data", f"data/{name}.npz", "--tag", f"baseline_{name}"])

    # Poisoned retrains (shows poisoning impact)
    run([py, "train.py", "--epochs", str(args.epochs),
         "--train-npz", "data/poisoned_label_flip.npz",
         "--out", "poisoned_label_flip.pt", "--tag", "poisoned_label_flip"])
    run([py, "evaluate.py", "--model", "models/poisoned_label_flip.pt",
         "--data", "data/adv_fgsm.npz", "--tag", "poisoned_label_flip_adv_fgsm"])
    run([py, "train.py", "--epochs", str(args.epochs),
         "--train-npz", "data/poisoned_backdoor.npz",
         "--out", "poisoned_backdoor.pt", "--tag", "poisoned_backdoor"])
    run([py, "evaluate.py", "--model", "models/poisoned_backdoor.pt",
         "--data", "data/backdoor_test.npz", "--tag", "poisoned_backdoor_backdoor_test"])

    # 7. Blue team: adversarial training
    run([py, "defense/adversarial_training.py", "--epochs", str(args.epochs)])
    # 7b. Blue team: feature squeezing (post-hoc)
    run([py, "defense/feature_squeezing.py", "--model", "models/baseline.pt"])

    # summary
    run([py, "summary.py"])


if __name__ == "__main__":
    main()
