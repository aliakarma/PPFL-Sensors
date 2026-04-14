# Experiment Design

## Overview

The experimental protocol is structured around four axes:

| Axis | Values |
|---|---|
| Defense condition | None, Noise, Clipping, Combined, Ensemble |
| Partition strategy | IID, Dirichlet (α=0.5), Pathological (K=2) |
| Number of clients | 10 (default), 5, 20 (ablation) |
| Attack model | Random, Majority, Logistic, RF, MLP |

Full factorial: 5 × 3 conditions reported; others available via config overrides.

---

## Experimental Conditions

| ID | Condition | Config File | Key Overrides |
|---|---|---|---|
| E1 | Baseline (no defense) | `attack_only.yaml` | — |
| E2 | Gaussian noise ablation | `defense_ablation.yaml` | `defense.noise.sigma` ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 0.5} |
| E3 | Gradient clipping ablation | `defense_ablation.yaml` | `defense.clipping.max_norm` ∈ {0.1, 0.5, 1.0, 2.0, 5.0} |
| E4 | Combined (noise + clipping) | `defense_ablation.yaml` | both enabled, σ=0.01, C=1.0 |
| E5 | Ensemble FL | `default.yaml` | `training.aggregation: ensemble` |
| E6 | Non-IID (Dirichlet) | `default.yaml` | `partition_strategy: dirichlet, alpha: 0.5` |
| E7 | Non-IID (Pathological) | `default.yaml` | `partition_strategy: pathological, classes_per_client: 2` |

---

## Metrics Definitions

### FL Task Metrics
- **`fl_accuracy`**: top-1 accuracy of the global model on the shared held-out test set, evaluated after each round.
- **`fl_loss`**: cross-entropy loss on the held-out test set.
- **`best_fl_accuracy`**: maximum `fl_accuracy` across all rounds.
- **`final_fl_accuracy`**: `fl_accuracy` at the last round.

### Attack Metrics (per round, eval phase only)
- **`attack_acc_{model}`**: accuracy of a specific attack model.
- **`mean_attack_accuracy`**: mean accuracy across all enabled attack models.
- **`privacy_score`**: `1 − mean_attack_accuracy`.
- **`random_baseline`**: `1 / n_clients` — theoretical floor.

### Summary Metrics (per run)
- **`final_privacy_score`**: privacy score at the last evaluated round.
- **`attack_vs_random`**: `mean_attack_accuracy − random_baseline` — net attacker advantage.
- **`wall_time_s`**: total experiment duration in seconds.

---

## Attack Training Protocol

```
Rounds:  1  2  3  ... N  |  N+1  N+2  ...  T
Phase:   ←── COLLECT ──→ |  ←── EVALUATE ─────→
                         ↑
                      TRAIN (once)
```

- **Collect phase** (rounds 1–N): every client gradient upload stored to disk.
- **Train** (after round N): all attack models trained once on collected gradients.
- **Eval phase** (rounds N+1–T): attack models evaluated on each new round's uploads.

Default: N=8, T=20. Configurable via `attack.collect_rounds` and `attack.eval_start_round`.

---

## Gradient Storage Policy

| Setting | Behaviour |
|---|---|
| `storage_type: raw` | Full 1-D gradient tensor saved as `.pt` file |
| `storage_type: topk` | Top-k% elements by magnitude; stored as (values, indices) pair |
| `topk_ratio: 0.1` | Keep 10% of gradient dimensions (reduces storage by 10×) |

Storage limited to rounds 1–`collect_rounds`. Later rounds are not stored.

Directory: `results/logs/<run_id>/artifacts/gradients/`

---

## Hypotheses

| Hypothesis | Expected Result |
|---|---|
| H1: No defense → high attack accuracy | MLP attack accuracy significantly above random baseline |
| H2: Noise defense → reduced attack accuracy | Attack accuracy decreases as σ increases |
| H3: Noise defense → reduced FL accuracy | FL accuracy decreases as σ increases (tradeoff) |
| H4: Clipping alone → modest privacy gain | Attack accuracy reduced but not to random level |
| H5: Combined defense → strongest privacy | Lowest attack accuracy; highest privacy score |
| H6: Ensemble FL → privacy gain without strong accuracy drop | Privacy improves; FL accuracy stable or slightly lower |
| H7: Non-IID makes attack easier | Higher attack accuracy under Dirichlet/Pathological (more distinctive gradients) |

---

## Reproducibility Protocol

### Fixed Seeds
All experiments use `seed: 42` (default). The seed controls:
- Data partition randomness
- Model weight initialisation
- DataLoader shuffling (via `worker_init_fn`)
- Attack model training (via `SeedContext`)

### How to reproduce exactly

```bash
# Option 1: full automated reproduction
bash scripts/reproduce_all.sh

# Option 2: single condition
python main.py --config configs/attack_only.yaml

# Option 3: fast smoke test (3 rounds, 3 clients)
bash scripts/reproduce_all.sh --fast-dev
```

### Expected runtimes (CPU, 10 clients, 20 rounds, synthetic data)

| Condition | Approx. time |
|---|---|
| Baseline | ~3–5 min |
| Noise/Clipping (per sigma) | ~3–5 min |
| Ensemble FL | ~5–8 min |
| Full reproduce_all.sh | ~45–90 min |

---

## Adding a New Defense

1. Implement defense function in `defense/` (follow `noise.py` / `clipping.py` pattern).
2. Add a config block under `defense:` in `configs/default.yaml`.
3. Import and call in `client.py → _apply_defense()`.
4. Add a new config YAML or override for the ablation.
5. Add unit tests in `tests/test_defense.py`.
6. Add a row to the Experimental Conditions table above.

## Adding a New Attack Model

1. Implement class with `.fit()`, `.predict()`, `.score()` in `attack/attack_model.py`.
2. Register in `ATTACK_MODELS` dict.
3. Add model name to `attack.model` list in YAML config.
4. Add unit tests in `tests/test_attack.py`.
