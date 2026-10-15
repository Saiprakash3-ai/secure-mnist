# Securing an MNIST CNN: A Red Team / Blue Team Exercise

**Course:** Secure AI Systems
**Team:** M. Saiprakash Nayak, D. Anvesh
**Repos:**
- CNN implementation: `https://github.com/Saiprakash3-ai/secure-mnist`
- Attacks & datasets: `https://github.com/Saiprakash3-ai/secure-mnist-attacks`

> Metrics in this report are produced by the scripts in this repository.
> Run `python run_all.py` and copy values from `results/summary.csv`.

---

## 1. Introduction

This project builds a convolutional neural network (CNN) that classifies
handwritten digits (MNIST), models its threats with STRIDE (ML interpretation) and
LINDDUN, scans the source with an OWASP-listed SAST tool (Bandit), attacks the
model with two families of OWASP ML Top-10 attacks (evasion and data poisoning),
and finally defends it with adversarial training plus input sanitisation
(feature squeezing).

## 2. System & Implementation

| Item | Value |
|------|-------|
| Framework | PyTorch (CPU) |
| Model | `SimpleCNN` - 2 conv blocks (32/64 filters, ReLU, 2x2 max-pool), FC 128, dropout 0.25, FC 10 |
| Input | 1x28x28 grayscale, normalized (mean 0.1307, std 0.3081) |
| Optimizer | Adam, lr 1e-3 |
| Loss | Cross-entropy |
| Epochs | 5 |
| Batch size | 128 (train), 1000 (test) |
| Seed | 1337 |

**Data flow / trust boundaries**

```
[data source] --(collect)--> [training store] --(fit)--> [model artifact] --(deploy)--> [inference API] <--(query)-- [user]
        TB1                         TB2                      TB3                         TB4
```
- **TB1** source handoff: is the dataset authentic/clean?
- **TB2** training store: can an attacker write to it? (poisoning surface)
- **TB3** model artifact: is it signed / integrity-checked? (supply chain)
- **TB4** inference boundary: untrusted user-supplied images.

## 3. Baseline Performance Metrics (clean test set)

| Metric | Value |
|--------|-------|
| Accuracy | **0.9902** |
| Loss | **0.0311** |
| Precision / Recall / F1 | per-class in `results/baseline_clean.json` |
| Inference time | **0.766 ms / sample** (CPU, batch 1000) |
| Confusion matrix | `results/baseline_clean_confusion.png` |

Training converged from 94.5% (epoch 1) to 99.3% train / 99.0% test. Clean accuracy
is high; this report shows how brittle it is under attack.

## 4. Threat Modeling

### 4.1 Attacker Model

| Aspect | Description |
|--------|-------------|
| White-box attacker | Full knowledge of architecture and weights (used for FGSM/PGD). |
| Black-box attacker | Query access only; can still transfer adversarial examples and run membership inference. |
| Data attacker | Can inject/relabel training samples (insider or scraping supply chain). |
| Goal | Cause misclassification (evasion), implant a backdoor, exfiltrate training data, or degrade service. |
| Capability assumed | Can add bounded perturbations (L-inf), modify a fraction of training data, and query the API. |

### 4.2 STRIDE (ML interpretation) -> OWASP ML Top 10 mapping

| STRIDE | ML interpretation in this system | OWASP ML risk | Likelihood | Impact |
|--------|----------------------------------|---------------|-----------|--------|
| **Spoofing** | Forged/fake model served in place of ours; spoofed identity of a queried user or of the data source | ML06 AI Supply Chain; ML05 Model Theft | Med | High |
| **Tampering** | Adversarial input at inference; poisoned/relabelled training data; tampered model weights | ML01 Input Manipulation; ML02 Data Poisoning; ML10 Model Poisoning | High | High |
| **Repudiation** | No audit log tying a prediction to a model version + input; cannot prove provenance | ML05 Model Theft (attribution); governance gap | Med | Med |
| **Information Disclosure** | Membership inference / model inversion recover training data; model extraction steals the function | ML04 Membership Inference; ML03 Model Inversion; ML05 Model Theft | Med | High |
| **Denial of Service** | Flooding inputs that maximize compute/misclassify; "sponge" examples; availability of the API | ML09 Output Integrity; availability | Med | Med |
| **Elevation of Privilege** | Adversarial input bypasses an access-control gate that trusts the model (e.g., digit/face auth); unsafe model deserialization -> code exec | ML06 AI Supply Chain; ML01 | Low | Critical |

### 4.3 OWASP ML Top 10 coverage of this lab

| Risk | Demonstrated? | Where |
|------|---------------|-------|
| ML01 Input Manipulation (evasion) | **Yes** | `attacks/fgsm_pgd.py` (FGSM, PGD) |
| ML02 Data Poisoning | **Yes** | `attacks/data_poisoning.py` (label flip, backdoor) |
| ML03 Model Inversion | Discussed | requires output-probability API; out of scope to run |
| ML04 Membership Inference | Discussed | shadow/confidence attack; out of scope to run |
| ML05 Model Theft | Discussed | model extraction via queries |
| ML06 AI Supply Chain | **Yes (static)** | B614 `torch.load` finding + fix (§5) |
| ML07 Transfer Learning Attack | Discussed | trigger generalization across retrains |
| ML08 Model Skewing | Discussed | feedback-loop drift |
| ML09 Output Integrity | **Partially** | adversarial label flips = integrity attack |
| ML10 Model Poisoning | Discussed | weight tampering |

### 4.4 LINDDUN privacy analysis (real-world deployment)

- **Linking** - a deployment that logs images + session IDs lets an operator link all
  digits written by the same person (handwriting is quasi-unique).
- **Identifying** - handwriting can be biometric; combined with other data it can
  re-identify a writer.
- **Non-repudiation** - predictions logged with user IDs create a permanent record a
  user cannot deny ("you wrote this digit").
- **Detecting** - an observer can infer *that* a user is interacting with the service
  from traffic/timing patterns.
- **Data Disclosure** - membership inference / model inversion can leak training
  digits; raw uploaded images may contain incidental sensitive content.
- **Unawareness** - users are typically not told their handwriting is stored or used
  to retrain models.
- **Non-compliance** - storing handwriting/biometrics triggers GDPR special-category
  and consent/retention obligations.

MNIST itself is not personal data, but the *architecture* above maps directly onto a
real handwriting/biometric service, so these threats become live.

## 5. SAST Findings and Fixes

Tool: **Bandit** (OWASP-listed), run via `sast/run_bandit.ps1`. Raw output in
`sast/bandit_report.txt` / `sast/bandit_report.json` (741 lines of code scanned).

| ID | Severity | Location | Finding | Status / Fix |
|----|----------|----------|---------|--------------|
| B614 | **Medium** | `common.py:load_model` | `torch.load` deserialization (CWE-502) can execute arbitrary code via a pickle payload when loading an untrusted model file | **FIXED** - now calls `torch.load(..., weights_only=True)`. Reproduced the finding in isolation on a bare `torch.load('x.pt')` call; with the fix Bandit no longer reports it. Maps to **OWASP ML06** (supply chain / unsafe deserialization). |
| B404 | Low | `run_all.py:11` | `import subprocess` (possible shell-injection surface) | Accepted - subprocess is used only to orchestrate our own scripts. |
| B603 | Low | `run_all.py:20` | `subprocess.run` without `shell=True` | Safe by design - args are a fixed list, `shell` is never enabled, and the interpreter is the absolute `sys.executable`. |

**Fix verification:** after adding `weights_only=True`, a re-scan reports **0 Medium and
0 High** issues; the two remaining Low findings are documented accepted risks (trusted,
list-based subprocess with no untrusted input). `numpy.load` uses the default
`allow_pickle=False`, so dataset loading is not a deserialization risk.

## 6. Red Team - Attacks

### 6.1 Evasion: FGSM and PGD (`attacks/fgsm_pgd.py`)

- **Tool:** Adversarial Robustness Toolbox (ART).
- **Method:** one-step Fast Gradient Sign Method (eps = 0.2) and Projected Gradient
  Descent (eps = 0.2, 40 steps, 1 random restart) in [0,1] pixel space.
- **OWASP risk exercised:** **ML01 Input Manipulation Attack** (evasion).
- **Why chosen:** evasion is the canonical, highest-likelihood ML risk for a deployed
  classifier and needs only query/inference access to be realistic. FGSM is cheap;
  PGD gives a strong upper bound on the attack surface.

### 6.2 Data Poisoning: label flip + backdoor (`attacks/data_poisoning.py`)

- **Method A - label flip:** relabel class 1 -> 7 in the training set (targeted).
- **Method B - backdoor/trojan:** stamp a 3x3 white corner patch on 5% of training
  samples and relabel them to 0. At test time the trigger forces class 0.
- **OWASP risk exercised:** **ML02 Data Poisoning Attack** (and **ML07 Transfer
  Learning** conceptually, **ML09 Output Integrity**).
- **Why chosen:** demonstrates a training-time integrity attack that survives normal
  training and is invisible on clean data - a distinct risk class from evasion.

Figures: `data/adv_examples.png` (evasion), `data/poison_examples.png` (backdoor).

## 7. Retest - Performance on Adversarial / Poisoned Data

| Scenario | Model | Data | Accuracy |
|----------|-------|------|----------|
| Baseline (reference) | `baseline.pt` | clean | **0.9902** |
| Evasion | `baseline.pt` | FGSM (eps 0.2) | **0.5088** |
| Evasion | `baseline.pt` | PGD (eps 0.2, 40 steps) | **0.0175** |
| Label-flip poison | `poisoned_label_flip.pt` | clean | **0.8773** |
| Backdoor | `poisoned_backdoor.pt` | clean | **0.9911** |
| Backdoor | `poisoned_backdoor.pt` | triggered test | **0.0980** (low = attack succeeded) |

**Observation:** a 20/255-bounded PGD perturbation drops accuracy from 99.0% to 1.8% -
the model is effectively blind to adversarial input. The label-flip model looks healthy
on its own training distribution but loses ~11 points on clean data because it was taught
the wrong 1->7 mapping. The backdoor model is the most dangerous: it keeps **99.1%** clean
accuracy yet fails to **9.8%** when the trigger is present - undetectable by clean-set
monitoring alone.

## 8. Blue Team - Defenses and Comparison

**Defense 1 - Adversarial training** (`defense/adversarial_training.py`):
retrain on clean data mixed with FGSM/PGD examples (`adv_train_mix.npz`).

**Defense 2 - Feature squeezing** (`defense/feature_squeezing.py`):
median-filter + 4-bit depth reduction of inputs before classification.

| Model | Clean acc | FGSM acc | PGD acc |
|-------|-----------|----------|---------|
| Baseline (no defense) | 0.9902 | 0.5088 | 0.0175 |
| **Adversarially trained** | **0.9772** | **0.9841** | **0.9827** |
| Baseline + feature squeezing | 0.9852 | 0.6085 | 0.1152 |

**Result:** adversarial training is the clear winner. It costs ~1.3 points of clean
accuracy (99.0% -> 97.7%) but raises robustness from 50.9% -> **98.4%** (FGSM) and from
1.8% -> **98.3%** (PGD). Feature squeezing, as a cheap post-hoc filter, recovers a little
FGSM accuracy (50.9% -> 60.9%) and PGD (1.8% -> 11.5%) while barely touching clean
accuracy - useful as defense-in-depth but not sufficient alone.

**Backdoor mitigation (note):** neither evasion defense removes a trigger. Detecting the
backdoor requires data-level controls - scanning the training set for anomalous label/pixel
correlations (e.g. the constant bottom-right patch), activation-clustering/STRIP-style
trigger detection, or provenance checks on the data pipeline. This is listed as future work.

## 9. Conclusion and Observations

- A high-accuracy MNIST CNN (**99.0%** clean) is trivially broken by evasion
  (**50.9%** on FGSM, **1.8%** on PGD) and silently compromised by poisoning
  (label flip: -11 points; backdoor: 99.1% clean but 9.8% under trigger),
  confirming accuracy alone is not security.
- Threat modeling with STRIDE + OWASP ML Top 10 surfaced supply-chain and privacy risks
  (via LINDDUN) that a bare training script ignores.
- SAST (Bandit) found a real deserialization risk (`torch.load`, B614, CWE-502) that maps
  to ML06; the one-line `weights_only=True` fix removes an arbitrary-code-execution vector
  and clears the finding.
- Adversarial training is the strongest single defense (**98.3%** under PGD for a 1.3-point
  clean-accuracy cost) but does not fix backdoors; defense-in-depth (sanitisation +
  trigger detection + provenance) is required.
- **Takeaway:** secure ML needs threat modeling, secure code, robust training, and
  runtime input controls together, not any single technique.

## Appendix - Reproduction

```powershell
pip install -r requirements.txt
python run_all.py --epochs 5
python summary.py
powershell -File sast/run_bandit.ps1
```
