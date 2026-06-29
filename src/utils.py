"""Utility helpers: seeding, logging, IO."""
from __future__ import annotations
import json
import logging
import os
import random
from pathlib import Path

import numpy as np

from . import config


def set_global_seed(seed: int = config.SEED) -> None:
    """Seed Python, NumPy, and (if available) TensorFlow for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


def get_logger(name: str = "glioma") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        config.LOGS.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(config.LOGS / "run.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)
