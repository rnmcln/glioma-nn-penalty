"""Helpers to persist and reload model predictions for downstream analysis."""
from __future__ import annotations
import pandas as pd

from . import config


def save_predictions(tag: str, y_true, preds: dict, sample=None) -> str:
    df = pd.DataFrame({"y_true": y_true})
    if sample is not None:
        df.insert(0, "sample", sample)
    for m, p in preds.items():
        df[f"prob__{m}"] = p
    path = config.PREDICTIONS / f"{tag}.csv"
    df.to_csv(path, index=False)
    return str(path)


def load_predictions(tag: str):
    df = pd.read_csv(config.PREDICTIONS / f"{tag}.csv")
    y = df["y_true"].values
    models = [c[len("prob__"):] for c in df.columns if c.startswith("prob__")]
    preds = {m: df[f"prob__{m}"].values for m in models}
    return y, preds, models
