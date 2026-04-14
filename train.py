"""
train.py
========
Standalone FL training script — no attack, no defense overhead.
Useful for quickly verifying that the FL core (model + data + aggregation) works.

Usage
-----
    python train.py                          # uses configs/default.yaml
    python train.py --config configs/default.yaml --rounds 5
    python train.py --fast-dev               # 3 rounds, tiny model
"""

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(description="Standalone FL training (no attack)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override training.rounds from config")
    parser.add_argument("--fast-dev", action="store_true")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    log = logging.getLogger("train")

    from utils.config import load_config
    from utils.seed import set_global_seed
    from utils.device import init_device, device_info
    from utils.logger import configure_root_logger
    from data.dataset_loader import get_client_datasets
    from models import get_model
    from client.client import FLClient
    from server.server import get_server

    configure_root_logger()

    override = {}
    if args.rounds is not None:
        override = {"training": {"rounds": args.rounds}}

    # Disable attack in standalone training
    override.setdefault("attack", {})["enabled"] = False

    cfg = load_config(args.config, override=override, fast_dev=args.fast_dev)

    set_global_seed(cfg.logging.seed)
    init_device(getattr(cfg.training, "device", "auto"))
    log.info("Device: %s", device_info())

    log.info("Loading data…")
    client_datasets = get_client_datasets(cfg)
    test_dataset = client_datasets[0]

    model_cfg = cfg.model
    global_model = get_model(
        name=model_cfg.arch,
        input_dim=model_cfg.input_dim,
        num_classes=model_cfg.num_classes,
        hidden_dims=getattr(model_cfg, "hidden_dims", [256, 128]),
        dropout=getattr(model_cfg, "dropout", 0.3),
    )
    log.info("Model: %s  params=%d", global_model, global_model.count_parameters())

    clients = [
        FLClient(client_id=ds.client_id, dataset=ds, config=cfg)
        for ds in client_datasets
    ]

    server = get_server(
        config=cfg,
        model=global_model,
        clients=clients,
        test_dataset=test_dataset,
    )

    log.info("Starting FL training — %d rounds, %d clients",
             cfg.training.rounds, len(clients))

    best_acc = 0.0
    for round_idx in range(1, cfg.training.rounds + 1):
        _, metrics = server.run_round(round_idx)
        acc = metrics["fl_accuracy"]
        if acc > best_acc:
            best_acc = acc
        log.info(
            "Round %d/%d — acc=%.4f  loss=%.4f  (best=%.4f)",
            round_idx, cfg.training.rounds,
            acc, metrics["fl_loss"], best_acc,
        )

    log.info("Training complete.  Best accuracy: %.4f", best_acc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
