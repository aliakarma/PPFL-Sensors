# Privacy-Preserving Federated Learning Against Inference Attacks in Sensor Data

A complete, research-grade prototype demonstrating gradient-based identity inference attacks on federated learning systems, and evaluating multiple privacy defense mechanisms on sensor activity data.

---

## 📑 Table of Contents
- 📌 Overview
- 🧠 Methodology
- ⚙️ Installation
- 🚀 Quick Start
- 📊 Results
- 🔬 Evaluation
- ⚠️ Limitations
- 📂 Repository Structure
- 🛠️ Development & Testing
- 📖 Citation

---

## 📌 Overview

This repository implements a full federated learning (FL) pipeline designed to evaluate the privacy-utility tradeoff under an **honest-but-curious server** threat model. 

**Key Features:**
- **Federated Architectures:** Supports FedAvg, FedMedian, and Ensemble FL.
- **Sensor Dataset:** Native integration with the 561-dimensional UCI HAR dataset (human activity recognition).
- **Inference Attacks:** Includes 5 gradient-based identity classifiers routing from Random/Majority guessing up to trained MLPs.
- **Defenses:** Built-in controlled Gaussian noise injection and gradient clipping.
- **Automated Pipelines:** Config-driven multi-condition experiment runs with comprehensive logging and tradeoff plotting.

---

## 🧠 Methodology

### Threat Model: Honest-but-Curious Server

In this framework, the server faithfully executes the federated aggregation protocol but actively intercepts client gradient updates to train an attack model. The attacker's objective is to link intercepted gradients back to the specific participating client's identity.

```text
   [Client 1] --(ΔW₁)--> +-----------------------------+
   [Client 2] --(ΔW₂)--> | Honest-but-Curious Server   | --> [Global Model]
   [Client K] --(ΔWₖ)--> +-----------------------------+
                                  | (intercepts ΔW)
                                  v
                         +-----------------------------+
                         |     Inference Attack        |
                         | (Random → Logistic → MLP)   |
                         +-----------------------------+
                                  |
                                  v
                          [Client Identity (1 to K)]
```

---

## ⚙️ Installation

### 1. Clone & Install Dependencies
First, clone the repository and install the standard dependencies:

```bash
git clone https://github.com/aliakarma/PPFL-Sensors
cd PPFL-Sensors
pip install -r requirements.txt
```

### 2. Download the UCI HAR Dataset
The primary evaluations are conducted on real sensor data. 

**Linux / macOS:**
```bash
cd data/raw/
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
unzip "UCI HAR Dataset.zip"
cd ../..
```

**Windows (PowerShell):**
```powershell
cd data\raw\
Invoke-WebRequest -Uri "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip" -OutFile "UCI HAR Dataset.zip"
Expand-Archive -Path "UCI HAR Dataset.zip" -DestinationPath "."
cd ..\..
```
*(Note: If the dataset is absent or download fails, the code will automatically fall back to generating synthetic data.)*

---

## 🚀 Quick Start

Ensure your setup is working correctly using a minimal smoke test.

### Run a Smoke Test (`fast-dev`)
Executes 3 FL rounds with 3 clients using a minimal model architecture (completes in < 60 seconds).

```bash
python main.py --fast-dev
```

### Run Baseline Experiment
Executes the full default experiment (no defenses) to establish peak baseline utility and maximum vulnerability:

```bash
python main.py --config configs/default.yaml
```

---

## 📊 Results

### 1. Real HAR vs. Synthetic Data
- **Real (UCI HAR):** The network organically reaches ~87% Task Accuracy. Without defenses, Gradient Identity Inference achieves **100% Attack Accuracy**.
- **Synthetic Ablation:** Task utility drops to ~50%, but gradient inference remains 100%. Synthetic data generates highly separable gradients, acting purely as a stress test for identifiability boundaries, proving leakage is fundamentally decoupled from task utility.

### 2. Privacy-Utility Tradeoff (Gaussian Noise)
Pushing models through noise injection ($\sigma$) establishes an information-theoretic tradeoff explicitly analyzed using the `noise-sweep` pipeline:

- **The Tradeoff Region:** At $\sigma \approx 0.1-0.2$, the system yields a massive privacy gain (attack accuracy collapsing toward random guessing thresholds of ~44%) for a highly acceptable utility drop of merely 5–15%. This is the prime deployment operating region.
- **The Information Plateau:** For $\sigma \ge 0.2$, attack accuracy plateaus at ~0.44. Noise effectively destroys the identifiable latent signal. Injecting heavier noise ($\sigma = 1.0$) crashes global task accuracy to ~27% without providing meaningful additional privacy.

---

## 🔬 Evaluation

The repository relies on a robust `.yaml` configuration override system to run evaluation sweeps easily from the CLI.

### Generating Tradeoff Plots
You can sequence a fully automated noise sweep and multi-seed evaluation directly via the CLI:
```bash
# 1. Run the noise sweep payload
python main.py --experiment noise-sweep --n-seeds 3 --fast-dev

# 2. Evaluate all completed runs & generate plots
python main.py --mode evaluate
```

### CLI Overrides Example
Modify deep evaluation configurations on-the-fly using dot-notation:
```bash
python main.py --config defense_ablation.yaml \
    --set defense.noise.enabled true \
    --set defense.noise.sigma 0.15 \
    --set training.aggregation ensemble
```

---

## ⚠️ Limitations

- **Ensemble Limitations:** Ensemble partitioning limits honest-but-curious server perspectives theoretically. However, our results demonstrate that if an attacker intercepts *any* clean iteration of updates, identity is irrevocably exposed. Ensemble architectures alone do not reduce identity leakage.
- **Synthetic Constraints:** Synthetic data is only utilized for ablation stress-testing to highlight pure mathematical separability. It does not reflect realistic temporal or spatial sensor relations, therefore all main performance claims rely strictly on the UCI HAR set.
- **Observability:** Assumes the honest-but-curious server has bounded, episodic access to raw updates prior to central aggregation.

---

## 📂 Repository Structure

```text
PPFL-Sensors/
├── attack/             5 attack models + collect/train/eval pipeline
├── client/             FL client protocols + local defense application
├── configs/            YAML configs (default, attack_only, defense_ablation)
├── data/               Dataset loaders & 5 dynamic partition strategies
├── defense/            Cryptographic perturbation (Gaussian + Clipping)
├── docs/               Deep-dive methodology and experiment design
├── experiments/        Experiment runners and evaluation plotters
├── models/             MLP and 1D-CNN global architectures
├── scripts/            Single-command reproduction bash scripts
├── server/             FLServer, FedAvg/FedMedian, EnsembleServer
├── tests/              Pytest suite
├── utils/              Seed control, metrics, and artifact tracking
├── main.py             Aggregated CLI entry point
└── train.py            Standalone FL training (no attack)
```

---

## 🛠️ Development & Testing

### Fast-Dev Overrides
Using the `--fast-dev` flag (or setting `fast_dev: true` in YAML) scales down operations for immediate local verification:

| Parameter | Production | Fast Dev |
|-----------|------------|----------|
| `n_clients` | 10 | 3 |
| `rounds` | 20 | 3 |
| `local_epochs` | 5 | 1 |
| `hidden_dims` | [256, 128] | [64, 32] |
| `collect_rounds` | 8 | 2 |

### Testing
Run the comprehensive Pytest suite to validate module integrity (data partitioning, tensor shapes, gradient interception):
```bash
pip install pytest
pytest tests/
```

---

## 📖 Citation

If you use this codebase or its findings in your research computationally evaluating privacy, please cite:

```bibtex
@misc{ppfl_sensors_2026,
  author = {Your Name/Institution},
  title = {Privacy-Preserving Federated Learning Against Inference Attacks in Sensor Data},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/aliakarma/PPFL-Sensors}}
}
