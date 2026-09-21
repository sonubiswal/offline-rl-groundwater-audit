# Protocol-Dependent Verdicts in Offline Reinforcement Learning

Code and artifacts for the manuscript:

> **Protocol-Dependent Verdicts in Offline Reinforcement Learning: An Audit of
> Estimator, Baseline, and Distribution Sensitivity on a Groundwater Pumping Task**
> Sumit Kumar Biswal and Bhramara Bar Biswal
> Submitted to IEEE Transactions on Neural Networks and Learning Systems

## Contents

**RL environment and training**
- `src/rl/synthetic_reward.py` — synthetic groundwater pumping simulator
- `src/rl/generate_offline_dataset.py` — offline dataset generator
- `src/rl/bc_baseline.py` — behavioural cloning (BC) training
- `src/rl/train_cql.py` — conservative Q-learning (CQL) training

**Evaluation**
- `src/rl/fqe_eval.py` — fitted Q-evaluation
- `src/rl/rollout_eval.py` — direct simulator rollout
- `ood_return_test.py` — banded OOD diagnostic (QR critic)
- `ood_return_test_meanq.py` — banded OOD diagnostic (Mean-Q critic)
- `ood_state_distribution_test.py` — state-distance partitioning

**Sensitivity and validation**
- `a4a5_sensitivity.py` — A4/A5 parameter sweep
- `cgwb_behavioral_validation.py` — rainfall-groundwater sign check

**Audits and reproducibility**
- `check_leakage.py`, `verify_provenance.py`, `verify_ci.py`, `freeze_project.py`

**Figures**
- `generate_figures.py`

## Not in this repository

Large artifacts (`.h5` datasets, `.d3` checkpoints, per-episode return arrays)
are archived separately on Zenodo. DOI to be added.

## Setup

```bash
conda env create -f environment.yml
conda activate trishna-opal
pip install -r requirements.txt