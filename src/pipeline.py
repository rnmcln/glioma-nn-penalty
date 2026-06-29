"""Orchestration: data assembly, internal cross-validation, external validation.

Leakage control: every model carries its own preprocessing, which is fitted only
on the training fold (internal CV) or on the full training cohort (external
validation). Expression-panel genes are z-scored WITHIN each cohort during
assembly, so external transfer between platforms (RNA-seq vs microarray) is on a
comparable scale; this per-cohort standardisation uses no outcome information.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from . import config
from .endpoints import make_binary_endpoint
from .models_baseline import build_baselines
from .models_tensorflow import TFMLPClassifier, tf_variants
from .preprocessing import apply_missingness_rule
from .utils import get_logger

log = get_logger()


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def load_cohort(name: str, with_expr: bool = True) -> pd.DataFrame:
    clin = pd.read_csv(config.PROCESSED / f"clinical_{name}.csv")
    if with_expr:
        ep = config.PROCESSED / f"expr_panel_{name}.csv"
        if ep.exists():
            expr = pd.read_csv(ep)
            genes = [c for c in expr.columns if c != "Sample"]
            # per-cohort z-score of expression (no outcome used)
            expr[genes] = (expr[genes] - expr[genes].mean()) / expr[genes].std(ddof=0)
            expr = expr.rename(columns={"Sample": "sample"})
            clin = clin.merge(expr, on="sample", how="left")
            clin.attrs["genes"] = genes
    return clin


def resolve_features(feature_set: str, df: pd.DataFrame, max_missing: float = 0.99) -> list[str]:
    """Return the columns for a feature set that are present and not (near-)all missing."""
    if feature_set == "C_expression":
        feats = df.attrs.get("genes", [c for c in df.columns if c in config.GENE_PANEL])
    elif feature_set.startswith("BC") or feature_set == "B+C":
        feats = config.FEATURE_SETS["B_clinical_molecular"] + df.attrs.get("genes", [])
    else:
        feats = config.FEATURE_SETS[feature_set]
    out = []
    for f in feats:
        if f in df.columns and df[f].isna().mean() < max_missing:
            out.append(f)
    return out


def model_factory(df_train: pd.DataFrame, features: list[str], include_tf: bool,
                  seed: int = config.SEED) -> dict:
    models = build_baselines(df_train, features, seed=seed)
    if include_tf:
        models.update(tf_variants(features, seed=seed))
    return models


# ---------------------------------------------------------------------------
# Internal repeated cross-validation (out-of-fold predictions averaged over repeats)
# ---------------------------------------------------------------------------
def internal_cv(df: pd.DataFrame, features: list[str], horizon: int,
                include_tf: bool = True, n_repeats: int = None, n_folds: int = None,
                seed: int = config.SEED):
    n_repeats = n_repeats or config.CV_REPEATS
    n_folds = n_folds or config.CV_FOLDS
    labelled, info = make_binary_endpoint(df, horizon)
    labelled = labelled.reset_index(drop=True)
    y = labelled["y"].values
    X = labelled

    names = list(model_factory(X, features, include_tf, seed).keys())
    # accumulate summed probability and counts per sample for averaging across repeats
    prob_sum = {m: np.zeros(len(X)) for m in names}
    prob_cnt = {m: np.zeros(len(X)) for m in names}

    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
    for fold, (tr, te) in enumerate(rskf.split(X, y)):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr = y[tr]
        models = model_factory(Xtr, features, include_tf, seed + fold)
        for m, est in models.items():
            est.fit(Xtr, ytr)
            p = est.predict_proba(Xte)[:, 1]
            prob_sum[m][te] += p
            prob_cnt[m][te] += 1

    preds = {m: prob_sum[m] / np.maximum(prob_cnt[m], 1) for m in names}
    return {"y_true": y, "preds": preds, "endpoint_info": info,
            "features": features, "n": len(X), "models": names,
            "sample": labelled["sample"].values}


# ---------------------------------------------------------------------------
# External validation: train on full source cohort, test on target cohort
# ---------------------------------------------------------------------------
def external_validate(train_df: pd.DataFrame, test_df: pd.DataFrame,
                      features: list[str], horizon: int, include_tf: bool = True,
                      seed: int = config.SEED):
    # restrict to features available (not all-missing) in BOTH cohorts
    shared = [f for f in features
              if f in train_df.columns and f in test_df.columns
              and train_df[f].notna().any() and test_df[f].notna().any()]
    tr_lab, tr_info = make_binary_endpoint(train_df, horizon)
    te_lab, te_info = make_binary_endpoint(test_df, horizon)
    ytr = tr_lab["y"].values
    yte = te_lab["y"].values

    models = model_factory(tr_lab, shared, include_tf, seed)
    preds = {}
    for m, est in models.items():
        est.fit(tr_lab, ytr)
        preds[m] = est.predict_proba(te_lab)[:, 1]
    return {"y_true": yte, "preds": preds, "shared_features": shared,
            "train_info": tr_info, "test_info": te_info,
            "n_train": len(tr_lab), "n_test": len(te_lab), "models": list(models.keys())}
