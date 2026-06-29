"""Calibration assessment: intercept, slope, and binned calibration curve.

Calibration intercept and slope follow the standard logistic-recalibration
framework (Cox). The slope is the coefficient of the linear predictor (logit of
predicted probability) in a logistic regression of outcome on that predictor;
the intercept is estimated with the slope fixed at 1. Slope < 1 indicates
overfitting (predictions too extreme); intercept != 0 indicates mis-calibration
in the mean (over- or under-prediction).
"""
from __future__ import annotations
import numpy as np


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def calibration_intercept_slope(y_true, y_prob):
    import statsmodels.api as sm
    y = np.asarray(y_true).astype(float)
    lp = _logit(y_prob)
    if len(np.unique(y)) < 2:
        return (np.nan, np.nan)
    # slope: outcome ~ intercept + slope * lp
    X = sm.add_constant(lp)
    try:
        slope_model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        slope = float(slope_model.params[1])
    except Exception:
        slope = np.nan
    # intercept (calibration-in-the-large): outcome ~ intercept + offset(lp)
    try:
        int_model = sm.GLM(y, np.ones((len(y), 1)),
                           family=sm.families.Binomial(), offset=lp).fit()
        intercept = float(int_model.params[0])
    except Exception:
        intercept = np.nan
    return (intercept, slope)


def calibration_curve_points(y_true, y_prob, n_bins=10):
    """Return (mean_pred, obs_freq, counts) per quantile bin for plotting."""
    y = np.asarray(y_true).astype(float)
    p = np.asarray(y_prob, dtype=float)
    order = np.argsort(p)
    p, y = p[order], y[order]
    bins = np.array_split(np.arange(len(p)), n_bins)
    mean_pred, obs_freq, counts = [], [], []
    for b in bins:
        if len(b) == 0:
            continue
        mean_pred.append(float(p[b].mean()))
        obs_freq.append(float(y[b].mean()))
        counts.append(int(len(b)))
    return np.array(mean_pred), np.array(obs_freq), np.array(counts)
