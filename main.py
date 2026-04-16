"""
main.py
=======
CLI entry point for fl-privacy-project.

Usage
-----
    python main.py                          # default config
    python main.py --config configs/attack_only.yaml
    python main.py --fast-dev               # 3 rounds, 3 clients
    python main.py --n-seeds 5              # run 5 seeds, report mean±std
    python main.py --set defense.noise.enabled true --set defense.noise.sigma 0.05
    python main.py --mode evaluate          # aggregate past results
    python main.py --dry-run               # validate config only
"""

import argparse
import logging
import sys
import numpy as np


def _parse_set_args(set_args: list) -> dict:
    if not set_args:
        return {}

    def _cast(v):
        if v.lower() == "true": return True
        if v.lower() == "false": return False
        try: return int(v)
        except ValueError: pass
        try: return float(v)
        except ValueError: pass
        return v

    if len(set_args) % 2 != 0:
        raise ValueError("--set requires pairs: KEY VALUE …")
    result = {}
    for i in range(0, len(set_args), 2):
        keys = set_args[i].split(".")
        val = _cast(set_args[i + 1])
        d = result
        for part in keys[:-1]:
            d = d.setdefault(part, {})
        d[keys[-1]] = val
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Privacy-Preserving FL — experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train")
    parser.add_argument("--fast-dev", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Number of seeds to run (report mean±std for N>1)")
    parser.add_argument("--experiment", type=str, choices=["none", "har-vs-synthetic", "ensemble-eval", "noise-sweep"], default="none",
                        help="Run a specific publication experiment")
    parser.add_argument("--set", nargs="+", metavar="KEY VALUE", dest="set_args")
    parser.add_argument("--results-dir", default="results/logs/")
    parser.add_argument("--plot-dir", default="results/plots/")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    if args.mode == "evaluate":
        from experiments.evaluate import evaluate_all
        evaluate_all(results_dir=args.results_dir, plot_dir=args.plot_dir)
        return 0

    if args.dry_run:
        from utils.config import load_config
        override = _parse_set_args(args.set_args or [])
        cfg = load_config(args.config, override=override, fast_dev=args.fast_dev)
        print(f"✓  Config loaded: {args.config}")
        print(f"   fast_dev          : {getattr(cfg, 'fast_dev', False)}")
        print(f"   n_clients         : {cfg.dataset.n_clients}")
        print(f"   partition_strategy: {cfg.dataset.partition_strategy}")
        print(f"   rounds            : {cfg.training.rounds}")
        print(f"   model             : {cfg.model.arch}")
        print(f"   defense.noise     : {cfg.defense.noise.enabled}")
        print(f"   defense.clipping  : {cfg.defense.clipping.enabled}")
        print(f"   attack.enabled    : {cfg.attack.enabled}")
        print(f"   attack.pca        : {getattr(cfg.attack, 'pca_components', 0)}")
        print("\n✓  Dry-run complete.")
        return 0

    from experiments.run_experiment import run
    override = _parse_set_args(args.set_args or [])

    # ── Publication Experiments ─────────────────────────────────────
    if args.experiment == "har-vs-synthetic":
        print("\n=== Running Experiment: HAR vs Synthetic ===\n")
        from utils.config import load_config
        
        # Run HAR
        print("--- HAR Dataset ---")
        har_override = _parse_set_args(args.set_args or [])
        har_override["dataset"] = {"name": "har"}
        har_summaries = _run_multi_seed(args.config, har_override, args.fast_dev, args.n_seeds)

        # Run Synthetic
        print("--- Synthetic Dataset ---")
        syn_override = _parse_set_args(args.set_args or [])
        syn_override["dataset"] = {"name": "synthetic", "synthetic": {"samples_per_client": 100, "n_features": 561, "n_classes": 6}}
        syn_summaries = _run_multi_seed(args.config, syn_override, args.fast_dev, args.n_seeds)
        return

    if args.experiment == "ensemble-eval":
        print("\n=== Running Experiment: Ensemble Eval ===\n")
        from utils.config import load_config
        
        for g in [2, 3, 5]:
            print(f"--- Ensemble groups: {g} ---")
            ens_override = _parse_set_args(args.set_args or [])
            ens_override["training"] = {"aggregation": "ensemble", "n_ensemble_groups": g}
            _run_multi_seed(args.config, ens_override, args.fast_dev, args.n_seeds)
        return

    if args.experiment == "noise-sweep":
        print("\n=== Running Experiment: Noise Sweep (Privacy-Utility Tradeoff) ===\n")
        noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
        results_table = []
        
        for sigma in noise_levels:
            print(f"\n--- Noise Level: sigma={sigma} ---")
            override_ns = _parse_set_args(args.set_args or [])
            override_ns["dataset"] = {"name": "har"}
            override_ns["defense"] = {"noise": {"enabled": True, "sigma": sigma}}
            
            summaries = _run_multi_seed(args.config, override_ns, args.fast_dev, args.n_seeds)
            
            fl_accs = [s["final_fl_accuracy"] for s in summaries if s.get("final_fl_accuracy") is not None]
            atk_accs = [s["mean_best_attack_accuracy"] for s in summaries if s.get("mean_best_attack_accuracy") is not None]
            privs = [s["final_privacy_score"] for s in summaries if s.get("final_privacy_score") is not None]
            
            if fl_accs and atk_accs:
                fl_mean, fl_std = np.mean(fl_accs), np.std(fl_accs)
                atk_mean, atk_std = np.mean(atk_accs), np.std(atk_accs)
                priv_mean = np.mean(privs)
                results_table.append({
                    "sigma": sigma, "fl_mean": fl_mean, "fl_std": fl_std,
                    "atk_mean": atk_mean, "atk_std": atk_std, "priv": priv_mean
                })
        
        # Save CSV
        import csv
        import os
        os.makedirs("results", exist_ok=True)
        csv_path = "results/noise_sweep.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sigma", "fl_acc_mean", "fl_acc_std", "attack_acc_mean", "attack_acc_std", "privacy_score"])
            for r in results_table:
                writer.writerow([r["sigma"], r["fl_mean"], r["fl_std"], r["atk_mean"], r["atk_std"], r["priv"]])
        
        # Generate plot
        try:
            import matplotlib.pyplot as plt
            os.makedirs("results/plots", exist_ok=True)
            sigmas = [r["sigma"] for r in results_table]
            fl_means = [r["fl_mean"] for r in results_table]
            atk_means = [r["atk_mean"] for r in results_table]
            
            plt.figure(figsize=(8, 5))
            plt.plot(sigmas, fl_means, marker='o', label='FL Accuracy (Utility)')
            plt.plot(sigmas, atk_means, marker='s', label='Attack Accuracy (Vulnerability)')
            plt.xlabel('Noise Sigma')
            plt.ylabel('Accuracy')
            plt.title('Privacy-Utility Tradeoff (Noise Sweep)')
            plt.legend()
            plt.grid(True)
            plt.savefig("results/plots/privacy_utility_curve.png")
        except Exception as e:
            print("Plotting failed:", e)

        print(f"\n=== Noise Sweep Results Saved to {csv_path} ===")
        return 0

    summaries = _run_multi_seed(args.config, override, args.fast_dev, args.n_seeds)

    if summaries:
        _print_multi_seed_summary(summaries)
    return 0

def _run_multi_seed(config_path, override, fast_dev, n_seeds):
    from experiments.run_experiment import run
    from utils.config import load_config
    import json
    import os
    
    base_cfg = load_config(config_path, override=override, fast_dev=fast_dev)
    base_seed = base_cfg.logging.seed

    summaries = []
    for i in range(n_seeds):
        seed = base_seed + i
        print(f"\n{'='*50}")
        print(f"  Seed {i+1}/{n_seeds}: seed={seed}")
        print(f"{'='*50}")
        try:
            s = run(config_path, override=override, fast_dev=fast_dev,
                    seed_override=seed)
            summaries.append(s)
        except Exception as exc:
            logging.exception("Seed %d failed: %s", seed, exc)
            
    if summaries:
        # Check standard deviation to prevent invalid identical runs showing up
        keys = ["final_fl_accuracy", "mean_attack_accuracy", "final_privacy_score", "mean_normalized_attacker_advantage"]
        for k in keys:
            vals = [s[k] for s in summaries if s.get(k) is not None]
            if len(vals) >= 2:
                import collections
                if len(set(vals)) == 1:
                    raise AssertionError("Invalid results: identical across seeds")

        # Save structured multi-seed results
        results_dir = summaries[0].get("run_dir", "results/logs/").rsplit('/', 1)[0]
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "per_seed_results.json"), "w") as f:
            json.dump(summaries, f, indent=2, default=str)
            
    return summaries

def _print_summary(summary: dict) -> None:
    print("\n── Run Summary ──────────────────────────────")
    for k, v in summary.items():
        if v is not None:
            print(f"  {k:<40} {v}")
    print("─────────────────────────────────────────────\n")


def _print_multi_seed_summary(summaries: list) -> None:
    keys = [
        "final_fl_accuracy", "mean_attack_accuracy",
        "final_privacy_score", "mean_normalized_attacker_advantage",
    ]
    print(f"\n── Multi-Seed Summary ({len(summaries)} seeds) ─────────────")
    for k in keys:
        vals = [s[k] for s in summaries if s.get(k) is not None]
        if vals:
            mean = np.mean(vals)
            std = np.std(vals)
            print(f"  {k:<40} {mean:.4f} ± {std:.4f}")
    print("─────────────────────────────────────────────\n")


if __name__ == "__main__":
    sys.exit(main())
