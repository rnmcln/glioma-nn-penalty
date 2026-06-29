"""Leakage-safe preprocessing pipelines.

A single make_preprocessor() returns an unfitted sklearn ColumnTransformer.
It is ALWAYS fitted inside the training fold only (imputation, scaling, and
one-hot categories are learned from training data and applied to test data).
No step here is ever fitted on the full dataset before splitting.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_feature_types(df: pd.DataFrame, features: list[str]):
    present = [f for f in features if f in df.columns]
    numeric, categorical = [], []
    for f in present:
        if pd.api.types.is_numeric_dtype(df[f]):
            numeric.append(f)
        else:
            categorical.append(f)
    return numeric, categorical


def make_preprocessor(df: pd.DataFrame, features: list[str], scale: bool = True) -> ColumnTransformer:
    from . import config
    numeric, categorical = split_feature_types(df, features)

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    # Fixed categorical levels (prespecified) where known; keeps one-hot schema
    # constant across folds/cohorts. Falls back to learned categories otherwise.
    levels = getattr(config, "CATEGORICAL_LEVELS", {})
    if categorical and all(c in levels for c in categorical):
        cats = [levels[c] for c in categorical]
    else:
        cats = "auto"  # mixed/unknown -> learn from training fold
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(categories=cats, handle_unknown="ignore",
                                 sparse_output=False)),
    ])

    transformers = []
    if numeric:
        transformers.append(("num", num_pipe, numeric))
    if categorical:
        transformers.append(("cat", cat_pipe, categorical))
    return ColumnTransformer(transformers, remainder="drop")


def apply_missingness_rule(df: pd.DataFrame, features: list[str],
                           threshold: float, essential: set[str]):
    """Drop features with > threshold missingness unless essential. Returns (kept, report)."""
    report = {}
    kept = []
    for f in features:
        if f not in df.columns:
            report[f] = {"present": False, "missing_frac": 1.0, "kept": False}
            continue
        miss = float(df[f].isna().mean())
        keep = (miss <= threshold) or (f in essential)
        report[f] = {"present": True, "missing_frac": miss, "kept": keep}
        if keep:
            kept.append(f)
    return kept, report
