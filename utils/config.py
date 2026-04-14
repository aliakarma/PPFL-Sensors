"""
utils/config.py
===============
Load, validate, and merge YAML configuration files.

Key responsibilities
--------------------
* Load a YAML file into a nested ``SimpleNamespace`` (dot-accessible).
* Apply ``fast_dev`` preset overrides before any other key is processed.
* Deep-merge an override dict/namespace onto a base config.
* Validate required keys and value ranges.

Usage
-----
    from utils.config import load_config

    cfg = load_config("configs/default.yaml")
    print(cfg.training.rounds)          # 20
    print(cfg.dataset.n_clients)        # 10

    # fast_dev shrinks everything for quick smoke tests:
    cfg = load_config("configs/default.yaml", fast_dev=True)
    print(cfg.training.rounds)          # 3
"""

import copy
import logging
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional, Union

import yaml

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# fast_dev preset — applied BEFORE any user config key is processed
# ------------------------------------------------------------------
FAST_DEV_OVERRIDES: Dict[str, Any] = {
    "dataset": {
        "n_clients": 3,
    },
    "model": {
        "hidden_dims": [64, 32],
    },
    "training": {
        "rounds": 3,
        "local_epochs": 1,
        "batch_size": 16,
    },
    "attack": {
        "collect_rounds": 2,
        "eval_start_round": 3,
    },
}


def _dict_to_namespace(d: Dict) -> SimpleNamespace:
    """Recursively convert a dict to a SimpleNamespace."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns


def _namespace_to_dict(ns: Union[SimpleNamespace, Any]) -> Any:
    """Recursively convert a SimpleNamespace back to a dict."""
    if isinstance(ns, SimpleNamespace):
        return {k: _namespace_to_dict(v) for k, v in vars(ns).items()}
    return ns


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep-merge ``override`` into ``base``.  Values in ``override`` take
    precedence; dicts are merged recursively rather than replaced wholesale.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(
    path: str,
    override: Optional[Dict] = None,
    fast_dev: bool = False,
) -> SimpleNamespace:
    """
    Load a YAML config file and return a dot-accessible ``SimpleNamespace``.

    Parameters
    ----------
    path : str
        Path to the YAML config file.
    override : dict, optional
        Additional key-value overrides applied after the file is loaded and
        after ``fast_dev`` overrides (if any).  Supports nested dicts.
    fast_dev : bool
        If ``True`` (or if the config file contains ``fast_dev: true``),
        apply ``FAST_DEV_OVERRIDES`` to shrink the experiment for rapid
        iteration.

    Returns
    -------
    SimpleNamespace
        Dot-accessible config object.  E.g. ``cfg.training.rounds``.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw: Dict = yaml.safe_load(f) or {}

    # Step 1 — apply fast_dev preset (highest priority after explicit overrides)
    file_fast_dev = raw.get("fast_dev", False)
    if fast_dev or file_fast_dev:
        logger.info("fast_dev mode enabled — applying reduced-scale overrides")
        raw = _deep_merge(raw, FAST_DEV_OVERRIDES)
        raw["fast_dev"] = True  # keep the flag visible in the config object

    # Step 2 — apply caller-supplied overrides
    if override:
        raw = _deep_merge(raw, override)

    # Step 3 — validate
    _validate(raw)

    cfg = _dict_to_namespace(raw)
    logger.debug("Config loaded from %s", path)
    return cfg


def merge_configs(
    base: Union[SimpleNamespace, Dict],
    override: Union[SimpleNamespace, Dict],
) -> SimpleNamespace:
    """
    Merge two configs.  Both may be SimpleNamespace or plain dict.

    Returns a new SimpleNamespace (neither input is mutated).
    """
    if isinstance(base, SimpleNamespace):
        base = _namespace_to_dict(base)
    if isinstance(override, SimpleNamespace):
        override = _namespace_to_dict(override)
    merged = _deep_merge(base, override)
    return _dict_to_namespace(merged)


def config_to_dict(cfg: Union[SimpleNamespace, Dict]) -> Dict:
    """Convert a config (SimpleNamespace or dict) to a plain dict."""
    if isinstance(cfg, SimpleNamespace):
        return _namespace_to_dict(cfg)
    return copy.deepcopy(cfg)


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

_REQUIRED_TOP_LEVEL = {"dataset", "model", "training", "defense", "attack", "logging"}


def _validate(raw: Dict) -> None:
    """
    Validate required keys and sensible value ranges.
    Raises ValueError on invalid configuration.
    """
    missing = _REQUIRED_TOP_LEVEL - set(raw.keys())
    if missing:
        raise ValueError(f"Config is missing required top-level keys: {missing}")

    ds = raw.get("dataset", {})
    if ds.get("n_clients", 1) < 2:
        raise ValueError("dataset.n_clients must be >= 2")

    tr = raw.get("training", {})
    if tr.get("rounds", 0) < 1:
        raise ValueError("training.rounds must be >= 1")
    if tr.get("lr", 0) <= 0:
        raise ValueError("training.lr must be > 0")

    atk = raw.get("attack", {})
    collect = atk.get("collect_rounds", 0)
    total = tr.get("rounds", 0)
    eval_start = atk.get("eval_start_round", collect + 1)
    if collect >= total:
        raise ValueError(
            f"attack.collect_rounds ({collect}) must be < training.rounds ({total})"
        )
    if eval_start <= collect:
        raise ValueError(
            f"attack.eval_start_round ({eval_start}) must be > attack.collect_rounds ({collect})"
        )

    gs = atk.get("gradient_store", {})
    storage_type = gs.get("storage_type", "raw")
    if storage_type not in {"raw", "topk"}:
        raise ValueError(
            f"attack.gradient_store.storage_type must be 'raw' or 'topk', got '{storage_type}'"
        )
    topk_ratio = gs.get("topk_ratio", 0.1)
    if not (0.0 < topk_ratio <= 1.0):
        raise ValueError("attack.gradient_store.topk_ratio must be in (0, 1]")

    dev = raw.get("defense", {})
    sigma = dev.get("noise", {}).get("sigma", 0.0)
    if sigma < 0:
        raise ValueError("defense.noise.sigma must be >= 0")
    max_norm = dev.get("clipping", {}).get("max_norm", 1.0)
    if max_norm <= 0:
        raise ValueError("defense.clipping.max_norm must be > 0")
