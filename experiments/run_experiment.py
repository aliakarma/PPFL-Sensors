"""
experiments/run_experiment.py
==============================
Orchestrates a single end-to-end experiment run.

Fixes applied
-------------
- Global model checkpointed every checkpoint_every rounds
- evaluate_all_attack_models() called at end of experiment
- mean_attack_accuracy + normalized_attacker_advantage tracked per round
- Multi-seed support via run() returning summary (called N times by main.py)
"""

import logging
import os

import torch

from attack.inference_attack import GradientInferenceAttack
from client.client import FLClient
from data.dataset_loader import get_client_datasets
from models import get_model
from server.server import get_server
from utils.config import load_config, config_to_dict
from utils.device import init_device, device_info
from utils.experiment_tracker import ExperimentTracker
from utils.logger import get_logger, configure_root_logger
from utils.metrics import MetricsTracker
from utils.seed import set_global_seed

logger = logging.getLogger(__name__)


def run(config_path: str, override: dict = None, fast_dev: bool = False,
        seed_override: int = None) -> dict:
    """
    Execute one complete experiment run.

    Parameters
    ----------
    config_path   : YAML config file
    override      : key-value overrides merged on top of config
    fast_dev      : reduced-scale preset
    seed_override : if set, overrides config.logging.seed (used for multi-seed)

    Returns
    -------
    summary dict (written to summary.json)
    """
    cfg = load_config(config_path, override=override, fast_dev=fast_dev)

    if seed_override is not None:
        cfg.logging.seed = seed_override

    configure_root_logger()
    base_log_dir = getattr(cfg.logging, "log_dir", "results/logs/")
    tracker = ExperimentTracker(base_log_dir=base_log_dir)
    run_id = tracker.start(cfg)
    run_log = get_logger("experiment", log_dir=tracker.run_dir)
    run_log.info("Run ID: %s  seed=%d", run_id, cfg.logging.seed)

    set_global_seed(cfg.logging.seed)
    init_device(getattr(cfg.training, "device", "auto"))
    run_log.info("Device: %s", device_info())

    # Data
    run_log.info("Loading dataset (partition=%s)…",
                 getattr(cfg.dataset, "partition_strategy", "iid"))
    client_datasets = get_client_datasets(cfg)
    test_dataset = client_datasets[0]

    # Model
    model_cfg = cfg.model
    global_model = get_model(
        name=model_cfg.arch,
        input_dim=model_cfg.input_dim,
        num_classes=model_cfg.num_classes,
        hidden_dims=getattr(model_cfg, "hidden_dims", [256, 128]),
        dropout=getattr(model_cfg, "dropout", 0.3),
    )
    run_log.info("Model: %s", global_model)

    # Clients + Server
    clients = [FLClient(ds.client_id, ds, cfg) for ds in client_datasets]
    server = get_server(cfg, global_model, clients, test_dataset)

    # Attack pipeline
    attack_enabled = getattr(cfg.attack, "enabled", True)
    attack = GradientInferenceAttack(cfg, tracker) if attack_enabled else None

    if attack is not None:
        collect_rounds = attack._collect_rounds
        eval_start_round = attack._eval_start
        assert eval_start_round > collect_rounds, (
            f"Invalid config: eval_start ({eval_start_round}) must be greater than "
            f"collect_rounds ({collect_rounds})"
        )
        assert collect_rounds < eval_start_round

    # Metrics tracker
    metrics = MetricsTracker(n_clients=cfg.dataset.n_clients)

    defense_active = (
        getattr(cfg.defense.noise, "enabled", False)
        or getattr(cfg.defense.clipping, "enabled", False)
    )

    checkpoint_every = getattr(cfg.training, "checkpoint_every", 5)
    n_rounds = cfg.training.rounds

    # ── FL Round loop ──────────────────────────────────────────────
    for round_idx in range(1, n_rounds + 1):

        updates, round_metrics = server.run_round(round_idx)

        metrics.update_fl(
            round_idx=round_idx,
            loss=round_metrics["fl_loss"],
            accuracy=round_metrics["fl_accuracy"],
            defense_active=defense_active,
        )
        tracker.log_round(round_idx, round_metrics)

        # Checkpoint global model
        if round_idx % checkpoint_every == 0 or round_idx == n_rounds:
            ckpt_path = os.path.join(
                tracker.artifact_dir, f"model_round_{round_idx:04d}.pt"
            )
            torch.save(server.model.state_dict(), ckpt_path)
            run_log.debug("Checkpoint saved: %s", ckpt_path)

        # Attack pipeline
        if attack is not None:
            collect_rounds = attack._collect_rounds
            train_round = collect_rounds + 1
            eval_start_round = attack._eval_start

            phase = 'collect' if round_idx <= collect_rounds else 'train' if round_idx == train_round else 'eval' if round_idx >= eval_start_round else 'gap'
            run_log.info(f"Round {round_idx}: phase={phase}")

            if round_idx <= collect_rounds:
                attack.collect(round_idx, updates)
                if round_idx == collect_rounds:
                    run_log.info("Finished collect phase")
            elif round_idx == train_round:
                attack.train()
                run_log.info("Finished train phase")
            elif round_idx >= eval_start_round:
                atk_results = attack.evaluate(round_idx, updates)
                if atk_results:
                    best_acc = atk_results.get("best_attack_accuracy", 0.0)
                    mean_acc = atk_results.get("mean_attack_accuracy", 0.0)
                    base_acc = atk_results.get("baseline_accuracy", 0.0)
                    metrics.update_attack(round_idx, best_acc, mean_acc, base_acc)

        run_log.info(
            "Round %d/%d  fl_acc=%.4f  fl_loss=%.4f",
            round_idx, n_rounds,
            round_metrics["fl_accuracy"], round_metrics["fl_loss"],
        )

    # ── Post-training: full attack model comparison ─────────────────
    if attack is not None and attack._is_trained:
        run_log.info("Running full 5-model attack comparison…")
        all_models_result = attack.evaluate_all_attack_models()
        if all_models_result:
            run_log.info("All-model comparison: %s", all_models_result)

    # ── Summary ─────────────────────────────────────────────────────
    summary = metrics.summary()
    summary["run_id"] = run_id
    summary["seed"] = cfg.logging.seed
    tracker.finish(summary)

    run_log.info("=== EXPERIMENT COMPLETE ===")
    run_log.info("Final FL accuracy : %.4f", summary.get("final_fl_accuracy") or 0)
    run_log.info("Mean attack acc   : %.4f", summary.get("mean_attack_accuracy") or 0)
    run_log.info("Privacy score     : %.4f", summary.get("final_privacy_score") or 0)
    run_log.info("Norm. adv. advantage: %.4f",
                 summary.get("mean_normalized_attacker_advantage") or 0)
    run_log.info("Results at        : %s", tracker.run_dir)

    return summary
