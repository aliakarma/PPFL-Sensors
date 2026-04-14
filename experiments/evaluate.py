"""
experiments/evaluate.py
========================
Aggregate and visualise results across multiple experiment runs.

Reads all completed run directories via ExperimentRegistry, produces:
    • FL accuracy curves (per-round, per-condition)
    • Attack success curves (per-round, per-condition)
    • Privacy-utility tradeoff scatter plot
    • Final summary table (stdout + CSV)

Usage
-----
    python experiments/evaluate.py --results-dir results/logs/ --plot-dir results/plots/

    # Or from Python:
    from experiments.evaluate import evaluate_all
    evaluate_all("results/logs/", "results/plots/")
"""

import argparse
import json
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers: load per-run time-series
# ------------------------------------------------------------------

def _load_jsonl(path: str) -> List[dict]:
    """Load a .jsonl file into a list of dicts."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def load_run_metrics(run_dir: str) -> dict:
    """
    Load all metrics for a single run.

    Returns
    -------
    dict with keys: 'config', 'summary', 'fl_metrics', 'attack_metrics'
    """
    def _load_json(name):
        p = os.path.join(run_dir, name)
        if not os.path.exists(p):
            return {}
        with open(p) as f:
            return json.load(f)

    return {
        "run_id": os.path.basename(run_dir),
        "config": _load_json("config.json"),
        "summary": _load_json("summary.json"),
        "fl_metrics": _load_jsonl(os.path.join(run_dir, "metrics.jsonl")),
        "attack_metrics": _load_jsonl(os.path.join(run_dir, "attack_metrics.jsonl")),
    }


def _run_label(run_data: dict) -> str:
    """Generate a human-readable label for a run."""
    cfg = run_data.get("config", {})
    defense = cfg.get("defense", {})
    noise_on = defense.get("noise", {}).get("enabled", False)
    clip_on = defense.get("clipping", {}).get("enabled", False)
    agg = cfg.get("training", {}).get("aggregation", "fedavg")

    parts = [agg]
    if noise_on:
        sigma = defense["noise"].get("sigma", "?")
        parts.append(f"noise(σ={sigma})")
    if clip_on:
        norm = defense["clipping"].get("max_norm", "?")
        parts.append(f"clip(C={norm})")
    if not noise_on and not clip_on and agg != "ensemble":
        parts.append("no-defense")
    return " + ".join(parts)


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_accuracy_curves(runs: List[dict], plot_dir: str) -> None:
    """Plot FL task accuracy over rounds for each run condition."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plots.")
        return

    os.makedirs(plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    for run in runs:
        label = _run_label(run)
        fl = run["fl_metrics"]
        if not fl:
            continue
        rounds = [r["round"] for r in fl if "round" in r]
        accs = [r.get("fl_accuracy", None) for r in fl]
        accs_clean = [a for a in accs if a is not None]
        rounds_clean = rounds[: len(accs_clean)]
        if rounds_clean:
            ax.plot(rounds_clean, accs_clean, marker="o", markersize=3, label=label)

    ax.set_xlabel("FL Round")
    ax.set_ylabel("Global Test Accuracy")
    ax.set_title("FL Task Accuracy vs. Round")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(plot_dir, "fl_accuracy_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved: %s", out_path)


def plot_attack_success(runs: List[dict], plot_dir: str) -> None:
    """Plot attack accuracy and privacy score over rounds."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(plot_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for run in runs:
        label = _run_label(run)
        atk = run["attack_metrics"]
        if not atk:
            continue
        rounds = [r["round"] for r in atk]
        attack_accs = [r.get("mean_attack_accuracy", None) for r in atk]
        privacy_scores = [r.get("privacy_score", None) for r in atk]

        rounds_a = [rounds[i] for i, v in enumerate(attack_accs) if v is not None]
        accs_clean = [v for v in attack_accs if v is not None]
        rounds_p = [rounds[i] for i, v in enumerate(privacy_scores) if v is not None]
        priv_clean = [v for v in privacy_scores if v is not None]

        if rounds_a:
            ax1.plot(rounds_a, accs_clean, marker="s", markersize=3, label=label)
        if rounds_p:
            ax2.plot(rounds_p, priv_clean, marker="^", markersize=3, label=label)

    # Random baseline reference line
    all_configs = [r.get("config", {}) for r in runs]
    n_clients_list = [c.get("dataset", {}).get("n_clients", 10) for c in all_configs]
    n_clients = max(set(n_clients_list), key=n_clients_list.count) if n_clients_list else 10
    random_baseline = 1.0 / max(n_clients, 1)
    ax1.axhline(random_baseline, linestyle="--", color="gray",
                label=f"random baseline (1/{n_clients}={random_baseline:.2f})")

    ax1.set_xlabel("FL Round")
    ax1.set_ylabel("Attack Accuracy")
    ax1.set_title("Inference Attack Accuracy vs. Round")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("FL Round")
    ax2.set_ylabel("Privacy Score  (1 − attack_acc)")
    ax2.set_title("Privacy Score vs. Round")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(plot_dir, "attack_privacy_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved: %s", out_path)


def plot_privacy_utility_tradeoff(runs: List[dict], plot_dir: str) -> None:
    """Scatter: final FL accuracy (utility) vs final privacy score."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    os.makedirs(plot_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    for run in runs:
        summary = run.get("summary", {})
        fl_acc = summary.get("final_fl_accuracy")
        priv = summary.get("final_privacy_score")
        if fl_acc is None or priv is None:
            continue
        label = _run_label(run)
        ax.scatter(fl_acc, priv, s=80, zorder=5)
        ax.annotate(label, (fl_acc, priv), textcoords="offset points",
                    xytext=(6, 3), fontsize=7)

    ax.set_xlabel("Final FL Task Accuracy  (utility ↑)")
    ax.set_ylabel("Privacy Score  (1 − attack_acc)  (privacy ↑)")
    ax.set_title("Privacy–Utility Tradeoff")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(plot_dir, "privacy_utility_tradeoff.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved: %s", out_path)


# ------------------------------------------------------------------
# Summary table
# ------------------------------------------------------------------

def print_summary_table(runs: List[dict]) -> None:
    """Print a formatted comparison table to stdout."""
    headers = [
        "Run ID", "Condition",
        "Final FL Acc", "Mean Atk Acc", "NAA", "Privacy Score",
    ]
    rows = []
    for run in runs:
        s = run.get("summary", {})
        rows.append([
            run["run_id"][:8],
            _run_label(run),
            f"{s['final_fl_accuracy']:.4f}" if s.get("final_fl_accuracy") is not None else "N/A",
            f"{s['mean_attack_accuracy']:.4f}" if s.get("mean_attack_accuracy") is not None else "N/A",
            f"{s['mean_normalized_attacker_advantage']:.4f}" if s.get("mean_normalized_attacker_advantage") is not None else "N/A",
            f"{s['final_privacy_score']:.4f}" if s.get("final_privacy_score") is not None else "N/A",
        ])

    col_widths = [max(len(h), max((len(r[i]) for r in rows), default=0))
                  for i, h in enumerate(headers)]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  ".join("-" * w for w in col_widths)

    print("\n" + fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print()


def save_summary_csv(runs: List[dict], plot_dir: str) -> None:
    """Save a summary CSV for the notebook / external analysis."""
    try:
        import csv
    except ImportError:
        return

    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, "summary_table.csv")
    fieldnames = [
        "run_id", "condition", "final_fl_accuracy",
        "mean_attack_accuracy", "mean_normalized_attacker_advantage", "final_privacy_score",
        "n_clients", "partition", "defense_noise", "defense_clipping",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            s = run.get("summary", {})
            cfg = run.get("config", {})
            writer.writerow({
                "run_id": run["run_id"],
                "condition": _run_label(run),
                "final_fl_accuracy": s.get("final_fl_accuracy", ""),
                "mean_attack_accuracy": s.get("mean_attack_accuracy", ""),
                "final_privacy_score": s.get("final_privacy_score", ""),
                "n_clients": cfg.get("dataset", {}).get("n_clients", ""),
                "partition": cfg.get("dataset", {}).get("partition_strategy", ""),
                "defense_noise": cfg.get("defense", {}).get("noise", {}).get("enabled", False),
                "defense_clipping": cfg.get("defense", {}).get("clipping", {}).get("enabled", False),
            })
    logger.info("Summary CSV saved: %s", out_path)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def evaluate_all(
    results_dir: str = "results/logs/",
    plot_dir: str = "results/plots/",
    run_ids: Optional[List[str]] = None,
) -> None:
    """
    Load all completed runs and generate comparison artefacts.

    Parameters
    ----------
    results_dir : base log directory
    plot_dir    : where to save PNG plots and CSV
    run_ids     : optional list of specific run IDs to include
    """
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return

    all_run_dirs = [
        os.path.join(results_dir, d)
        for d in sorted(os.listdir(results_dir))
        if os.path.isdir(os.path.join(results_dir, d))
    ]

    if run_ids:
        all_run_dirs = [d for d in all_run_dirs if os.path.basename(d) in run_ids]

    runs = [load_run_metrics(d) for d in all_run_dirs]
    runs = [r for r in runs if r["summary"]]  # skip incomplete runs

    if not runs:
        print("No completed runs found.")
        return

    print(f"\nFound {len(runs)} completed run(s).")

    plot_accuracy_curves(runs, plot_dir)
    plot_attack_success(runs, plot_dir)
    plot_privacy_utility_tradeoff(runs, plot_dir)
    print_summary_table(runs)
    save_summary_csv(runs, plot_dir)

    print(f"Plots saved to: {plot_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate FL experiment results")
    parser.add_argument(
        "--results-dir", default="results/logs/",
        help="Base directory containing run subdirectories"
    )
    parser.add_argument(
        "--plot-dir", default="results/plots/",
        help="Output directory for plots and CSV"
    )
    parser.add_argument(
        "--run-ids", nargs="*", default=None,
        help="Specific run IDs to include (default: all)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    evaluate_all(
        results_dir=args.results_dir,
        plot_dir=args.plot_dir,
        run_ids=args.run_ids,
    )
