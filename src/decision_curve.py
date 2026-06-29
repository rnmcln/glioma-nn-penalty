"""Decision-curve analysis (net benefit).

Net benefit at threshold p_t:
    NB = TP/n - FP/n * (p_t / (1 - p_t))
where TP and FP are counts when classifying as positive if predicted prob >= p_t.
Reference strategies: treat-all and treat-none (NB = 0). The model is clinically
useful at p_t only if its net benefit exceeds both references.
"""
from __future__ import annotations
import numpy as np

from . import config


def net_benefit(y_true, y_prob, thresholds=None):
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    n = len(y)
    prev = y.mean()
    thresholds = np.asarray(thresholds if thresholds is not None else config.DCA_THRESHOLDS, dtype=float)

    nb_model, nb_all = [], []
    for pt in thresholds:
        w = pt / (1 - pt) if pt < 1 else np.inf
        pred_pos = p >= pt
        tp = np.sum((pred_pos) & (y == 1)) / n
        fp = np.sum((pred_pos) & (y == 0)) / n
        nb_model.append(tp - fp * w)
        nb_all.append(prev - (1 - prev) * w)
    return {
        "thresholds": thresholds,
        "net_benefit_model": np.asarray(nb_model),
        "net_benefit_all": np.asarray(nb_all),
        "net_benefit_none": np.zeros_like(thresholds),
    }


def net_benefit_table(y_true, models_probs: dict, report_thresholds=None):
    """Return per-threshold net benefit for several models + reference strategies."""
    report_thresholds = report_thresholds or config.REPORT_THRESHOLDS
    rows = []
    base = net_benefit(y_true, next(iter(models_probs.values())), report_thresholds)
    for i, pt in enumerate(base["thresholds"]):
        row = {"threshold": float(pt),
               "treat_all": float(base["net_benefit_all"][i]),
               "treat_none": 0.0}
        for name, probs in models_probs.items():
            nb = net_benefit(y_true, probs, [pt])
            row[name] = float(nb["net_benefit_model"][0])
        rows.append(row)
    return rows
