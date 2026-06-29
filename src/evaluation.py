"""Performance metrics, bootstrap confidence intervals, and paired comparison."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)

from . import config
from .calibration import calibration_intercept_slope


def point_metrics(y_true, y_prob) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-7, 1 - 1e-7)
    out = {"n": int(len(y_true)), "event_rate": float(y_true.mean())}
    if len(np.unique(y_true)) < 2:
        out.update({"auroc": np.nan, "auprc": np.nan})
    else:
        out["auroc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    out["brier"] = float(brier_score_loss(y_true, y_prob))
    ci, sl = calibration_intercept_slope(y_true, y_prob)
    out["calibration_intercept"] = ci
    out["calibration_slope"] = sl
    return out


def _fast_metric(y_true, y_prob, metric):
    """Cheap scalar metric for bootstrap (no GLM calibration refits)."""
    if metric == "auroc":
        return roc_auc_score(y_true, y_prob)
    if metric == "auprc":
        return average_precision_score(y_true, y_prob)
    if metric == "brier":
        yp = np.clip(y_prob, 1e-7, 1 - 1e-7)
        return brier_score_loss(y_true, yp)
    raise ValueError(metric)


def bootstrap_ci(y_true, y_prob, metric="auroc", n_boot=config.N_BOOTSTRAP,
                 seed=config.SEED):
    """Percentile bootstrap CI for a single model's metric."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if metric in ("auroc", "auprc") and len(np.unique(yt)) < 2:
            continue
        vals.append(_fast_metric(yt, yp, metric))
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"point": np.nan, "lo": np.nan, "hi": np.nan}
    return {"point": float(np.median(vals)),
            "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5))}


def paired_bootstrap(y_true, prob_a, prob_b, metric="auroc",
                     n_boot=config.N_BOOTSTRAP, seed=config.SEED):
    """Paired bootstrap of metric(A) - metric(B) over the SAME resampled patients.

    Positive difference favours model A. Returns delta with CI and a two-sided
    bootstrap p-value (proportion of replicates crossing zero, doubled).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    pa = np.asarray(prob_a, dtype=float)
    pb = np.asarray(prob_b, dtype=float)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if metric in ("auroc", "auprc") and len(np.unique(yt)) < 2:
            continue
        diffs.append(_fast_metric(yt, pa[idx], metric) - _fast_metric(yt, pb[idx], metric))
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) == 0:
        return {"delta": np.nan, "lo": np.nan, "hi": np.nan, "p_value": np.nan}
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"delta": float(np.median(diffs)),
            "lo": float(np.percentile(diffs, 2.5)),
            "hi": float(np.percentile(diffs, 97.5)),
            "p_value": float(min(1.0, p))}


def meaningful_improvement(delta_auroc: float, brier_nn: float, brier_ref: float,
                           slope_nn: float) -> dict:
    """Apply the prespecified clinically-meaningful rule (see config)."""
    better_disc = delta_auroc >= config.MEANINGFUL_AUROC_DELTA
    not_worse_brier = brier_nn <= brier_ref + 1e-6
    calib_ok = (slope_nn is not None) and (0.8 <= slope_nn <= 1.25)
    return {
        "delta_auroc": delta_auroc,
        "passes_auroc_margin": bool(better_disc),
        "not_worse_brier": bool(not_worse_brier),
        "calibration_acceptable": bool(calib_ok),
        "clinically_meaningful": bool(better_disc and not_worse_brier and calib_ok),
    }
