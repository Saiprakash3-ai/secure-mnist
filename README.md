# Secure AI Systems - Red / Blue Teaming an MNIST CNN

Course lab: build a CNN digit classifier, threat-model it, scan it with SAST,
attack it (evasion + poisoning), and defend it (adversarial training + feature
squeezing). Two-person team project.

## Layout

```
secure-mnist/
  common.py                    shared model, data, metrics, persistence
  training.py                  fit() + npz loader helpers
  train.py                     baseline / poisoned / mixed training
  evaluate.py                  evaluate a model on clean or .npz data
  summary.py                   collect results/*.json -> results/summary.csv
  run_all.py                   run the whole pipeline
  attacks/
    _art_utils.py              ART PyTorchClassifier wrapper + I/O
    fgsm_pgd.py                evasion: FGSM + PGD (ART)
    data_poisoning.py          label-flip + backdoor poisoning
  defense/
    adversarial_training.py    retrain on clean + adversarial data
    feature_squeezing.py       input sanitisation at inference
  sast/
    run_bandit.ps1             OWASP-listed SAST (Bandit) runner
  report/REPORT.md             full write-up (threat model, results, ...)
  data/ models/ results/       generated artifacts
```

## Install

```powershell
pip install -r requirements.txt
```

## Run

Everything (CPU, ~a few minutes per model):

```powershell
python run_all.py --epochs 5
```

Or step by step:

```powershell
python train.py --epochs 5 --out baseline.pt --tag baseline        # 1-2 baseline
python attacks/fgsm_pgd.py --model models/baseline.pt              # evasion attacks
python attacks/data_poisoning.py                                   # poisoning datasets
python evaluate.py --model models/baseline.pt --data data/adv_pgd.npz --tag baseline_adv_pgd
python defense/adversarial_training.py                             # blue team #1
python defense/feature_squeezing.py                                # blue team #2
python summary.py                                                  # results/summary.csv
```

## SAST

```powershell
powershell -File sast/run_bandit.ps1
```

## Outputs

- `models/*.pt` - trained weights
- `results/*.json` - per-run metrics (accuracy, loss, confusion matrix, inference time)
- `results/summary.csv` - one-row-per-run comparison table
- `results/*_confusion.png`, `data/adv_examples.png`, `data/poison_examples.png` - figures
