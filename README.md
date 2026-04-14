# Privacy-Preserving Federated Learning Against Inference Attacks in Sensor Data

A complete, research-grade prototype demonstrating gradient-based identity inference attacks on federated learning systems and evaluating multiple privacy defense mechanisms on sensor activity data.

---

## Overview

This repository implements a full federated learning pipeline with an honest-but-curious server threat model. It includes:

- **Federated Learning** — FedAvg, FedMedian, and Ensemble FL across simulated clients
- **Sensor Dataset** — UCI HAR (561-dim activity recognition) with synthetic fallback
- **Inference Attack** — 5 gradient-based identity classifiers (random → MLP)
- **Defense Mechanisms** — Gaussian noise injection, gradient clipping, ensemble partitioning
- **Experiment Pipeline** — automated multi-condition runs with config overrides
- **Visualisation** — accuracy curves, attack success curves, privacy–utility tradeoff plots
- **Reproducibility** — global seed control, single-command reproduction script

---

## Repository Structure

```
PPFL-Sensors/
├── data/               Dataset loading and 5 partition strategies
├── models/             MLP and 1D-CNN architectures
├── client/             FL client (local training + defense application)
├── server/             FLServer, FedAvg/FedMedian, EnsembleServer
├── attack/             5 attack models + collect→train→eval pipeline
├── defense/            Gaussian noise + gradient clipping
├── experiments/        run_experiment.py + evaluate.py
├── utils/              seed, device, config, logger, metrics, gradient_processing, experiment_tracker
├── configs/            default.yaml, attack_only.yaml, defense_ablation.yaml
├── scripts/            reproduce_all.sh
├── docs/               methodology.md, experiment_design.md
├── tests/              pytest unit tests for all modules
├── main.py             CLI entry point
└── train.py            Standalone FL training (no attack)
```

---

## Quick Start

### 1. Install dependencies

```bash
git clone https://github.com/aliakarma/fl-privacy-project.git
cd fl-privacy-project
pip install -r requirements.txt
```

### 2. (Optional) Download UCI HAR Dataset

```bash
cd data/raw/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
unzip "UCI HAR Dataset.zip"
cd ../..
```

If the dataset is absent, the code automatically falls back to synthetic data.

### 3. Run a quick smoke test

```bash
python main.py --fast-dev
```

This runs 3 FL rounds with 3 clients and a tiny model — completes in under 60 seconds.

### 4. Run the full default experiment

```bash
python main.py --config configs/default.yaml
```

### 5. Reproduce all experiments

```bash
bash scripts/reproduce_all.sh
```

---

## CLI Reference

```bash
# Default experiment (baseline, no defense)
python main.py

# Specific config
python main.py --config configs/attack_only.yaml

# Fast dev mode
python main.py --fast-dev

# Runtime overrides (any config key, dot-notation)
python main.py --config configs/defense_ablation.yaml \
    --set defense.noise.enabled true \
    --set defense.noise.sigma 0.05

# Evaluate all completed runs → plots + summary table
python main.py --mode evaluate

# Validate config without training
python main.py --dry-run
```

---

## Partition Strategies

Set via `dataset.partition_strategy` in YAML:

| Strategy | Config value | Key parameter |
|---|---|---|
| IID | `iid` | — |
| Dirichlet label skew | `dirichlet` | `partition_params.alpha` |
| Pathological non-IID | `pathological` | `partition_params.classes_per_client` |
| Quantity skew | `quantity_skew` | `partition_params.beta` |
| Feature skew | `feature_skew` | `partition_params.noise_std_per_client` |

---

## Defense Configuration

```yaml
defense:
  noise:
    enabled: true
    sigma: 0.01          # noise std; multiplied by clip_norm if clipping enabled

  clipping:
    enabled: true
    max_norm: 1.0        # L2 clip threshold
```

```yaml
training:
  aggregation: ensemble  # enables EnsembleServer
  n_ensemble_groups: 3
  ensemble_strategy: round_robin
  ensemble_predict: average_logits
```

---

## Attack Configuration

```yaml
attack:
  enabled: true
  model: [random, majority, logistic, rf, mlp]
  collect_rounds: 8      # rounds 1–8 → attack training set
  eval_start_round: 9   # rounds 9–20 → attack test set (evaluated per round)
  grad_norm: l2
  gradient_store:
    storage_type: raw   # raw | topk
    topk_ratio: 0.1
```

---

## Fast Dev Mode

Adds `--fast-dev` flag or sets `fast_dev: true` in YAML to override:

| Parameter | Normal | fast_dev |
|---|---|---|
| `n_clients` | 10 | 3 |
| `rounds` | 20 | 3 |
| `local_epochs` | 5 | 1 |
| `hidden_dims` | [256, 128] | [64, 32] |
| `collect_rounds` | 8 | 2 |

---

## Experiment Outputs

Each run creates a timestamped directory:

```
results/logs/<run_id>/
    config.json            exact config used
    metrics.jsonl          per-round FL metrics
    attack_metrics.jsonl   per-round attack accuracy + privacy score
    artifacts/
        gradients/         stored gradient .pt files (collect phase)
    summary.json           final accuracy, attack accuracy, privacy score, wall time
```

After running evaluate.py:

```
results/plots/
    fl_accuracy_curves.png
    attack_privacy_curves.png
    privacy_utility_tradeoff.png
    summary_table.csv
```

---

## Key Metrics

| Metric | Description |
|---|---|
| `fl_accuracy` | Global model top-1 accuracy on held-out test set |
| `mean_attack_accuracy` | Mean accuracy across all enabled attack models |
| `privacy_score` | `1 − mean_attack_accuracy` (higher = more private) |
| `random_baseline` | `1 / n_clients` (lower bound for attack accuracy) |
| `attack_vs_random` | Net attacker advantage above random baseline |

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=. --cov-report=term-missing

# Single test file
pytest tests/test_defense.py -v
```

---

## Extending the Framework

### Add a new defense
1. Implement in `defense/` following `noise.py` / `clipping.py`
2. Call in `client.py → _apply_defense()`
3. Add config block in `configs/default.yaml`
4. Add tests in `tests/test_defense.py`

### Add a new attack model
1. Implement class with `.fit()`, `.predict()`, `.score()` in `attack/attack_model.py`
2. Register in `ATTACK_MODELS` dict
3. Add name to `attack.model` in YAML config
4. Add tests in `tests/test_attack.py`

### Add a new partition strategy
1. Implement `partition_<name>()` in `data/dataset_loader.py`
2. Add a case in `get_client_datasets()`
3. Document in `docs/experiment_design.md`

---

## References

- McMahan et al. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS.
- Melis et al. (2019). *Exploiting Unintended Feature Leakage in Collaborative Learning.* IEEE S&P.
- Yin et al. (2018). *Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates.* ICML.
- Abadi et al. (2016). *Deep Learning with Differential Privacy.* CCS.
- Zhao et al. (2018). *Federated Learning with Non-IID Data.* arXiv:1806.00582.

---

## License

MIT License. See `LICENSE` for details.
