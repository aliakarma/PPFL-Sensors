![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Reproducible](https://img.shields.io/badge/Reproducible-Yes-brightgreen)


# Privacy-Preserving Federated Learning Against Inference Attacks in Sensor Data

This repository contains the official implementation for evaluating privacy-utility tradeoffs in federated learning under honest-but-curious server threat models, specifically targeting human activity recognition (sensor data).

---

## 1. Overview

This codebase provides a complete federated learning (FL) pipeline to systematically evaluate the vulnerability of gradient updates to identity inference attacks. It highlights the inherent identifiability of clients in standard FL protocols (e.g., FedAvg, FedMedian, Ensembles) and quantifies the theoretical limitations of privacy preservation mechanisms such as controlled Gaussian noise injection.

## 2. Key Results

- **Inherent Identifiability:** Without explicit cryptographic defenses, highly separable gradient update distributions allow identity inference classifiers to achieve 100% attack accuracy. This strictly demonstrates an underlying information leakage bottleneck rather than a system flaw.
- **Privacy-Utility Tradeoff:** Noise-based defenses introduce a measurable, inevitable tradeoff. We identify a practical operating region ($\sigma \approx 0.1-0.2$) that achieves a significant privacy gain (reducing attack accuracy toward random guessing) with a manageable global utility degradation (5-15%).
- **Ensemble Limitations:** Partitioning training across bounded ensemble servers does not reduce fundamental identity leakage; even fractional exposure to clean gradients yields full identifiability.

## 3. Installation

**Hardware Requirements:**
- **CPU:** Supported natively
- **GPU:** Optional (CUDA-compatible recommended for full runs)
- **RAM:** $\ge$ 8GB

Clone the repository and install the standard dependencies:
```bash
git clone https://github.com/aliakarma/PPFL-Sensors
cd PPFL-Sensors
pip install -r requirements.txt
```

## 4. Dataset Setup

The primary evaluations are inherently dependent on real sensor data. **The UCI HAR dataset is meticulously required for all primary experiments.** The system will fail if the dataset is missing; synthetic data is explicitly reserved for isolated ablation stress-tests, not as a fallback.

**Linux / macOS:**
```bash
python scripts/download_har.py
```

**Windows (PowerShell):**
```powershell
python scripts/download_har.py
```

## 5. Running Experiments

The repository supports two distinct modes of execution depending on the research objective.

### 🔹 Fast Mode (Smoke Test)
**Purpose:** Quick pipeline validation, debugging, and environment checks.  
**Runtime:** 2-5 minutes.  
- Uses severely reduced rounds, clients, and dataset samples.  
- **Not used for paper results.**  
```bash
python main.py --fast-dev
```

### 🔹 Full Mode (Paper Reproduction)
**Purpose:** Generates the exact empirical artifacts, distributions, and tradeoff metrics reported in the study.  
**Runtime:** 15-40 minutes.  
- Uses the full UCI HAR dataset, complete training rounds, and multi-seed evaluation.  
- **Must be used to reproduce paper results.**  
```bash
python main.py --dataset har --n-seeds 5
```

## 6. Reproducing Paper Results

To reproduce the entire suite of artifacts (baseline evaluations, synthetic ablations, ensemble bounds, and noise tradeoffs), execute the Full Mode pipeline.

### Linux / macOS
```bash
bash scripts/reproduce_all.sh
```
*Or manually:*
```bash
python main.py --dataset har --n-seeds 5
python main.py --experiment har-vs-synthetic --n-seeds 3
python main.py --experiment ensemble-eval --n-seeds 3
python main.py --experiment noise-sweep --n-seeds 3
```

### Windows (PowerShell)
For exact reproducibility ensure deterministic environment variables are configured prior to execution:
```powershell
$env:PYTHONHASHSEED=42
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"

python main.py --dataset har --n-seeds 5
python main.py --experiment har-vs-synthetic --n-seeds 3
python main.py --experiment ensemble-eval --n-seeds 3
python main.py --experiment noise-sweep --n-seeds 3
```

## 7. Outputs

Executing the full pipeline will programmatically generate and aggregate all publication-ready artifacts directly into the `results/` directory.

After a full run, you should explicitly verify:
- `results/final_report.md`: Aggregated scientific analysis and tabular metrics.
- `results/noise_sweep.csv`: Raw tradeoff parameter tracking for defensive evaluations.
- `results/plots/privacy_utility_curve.png`: Output visualization demonstrating the explicit cost of privacy.

## 8. Evaluation

To manually evaluate cached logs, compile results, and generate the final output plots without retraining, invoke the standalone evaluate script:
```bash
python experiments/evaluate.py --results-dir results/logs/ --plot-dir results/plots/
```

## 9. Reproducibility

Scientific rigor necessitates strict deterministic reproducibility:
- **Deterministic Seeds:** All primary components (data partitioning, target model initialization, attack model sequences, and Gaussian sampling) are anchored to static cryptographic seeds (`PYTHONHASHSEED=42`).
- **Data Rigor:** No implicit data mixing or fallback behavior occurs.
- **Exact Commands:** The exact CLI arguments and configurations required for replication are cataloged explicitly above. Executing the sequences in Section 6 will flawlessly reconstruct the `results/final_report.md` metrics array.

## 10. Notes

- Ensure your Python environment is completely isolated (virtual environments are highly recommended) before batching the dependency installs.
- If `python scripts/download_har.py` encounters SSL restriction errors on custom networks, manually download and extract the raw UCI dataset directly into `data/UCI_HAR_Dataset/`.# Privacy-Preserving Federated Learning Against Inference Attacks in Sensor Data