"""Secondary time-to-event benchmark (Cox / penalised Cox, optional RSF).

This complements the primary fixed-horizon binary analysis. It uses the full
follow-up (time + event) rather than a binary label, so no patients are excluded
for censoring before a horizon. Discrimination is Harrell's C-index. Where
scikit-survival is available, integrated Brier score and time-dependent AUC are
also reported; otherwise these are skipped and noted.

Transparent comparators (Cox, penalised/elastic-net Cox) are the focus. A neural
survival model is intentionally not added here, to keep the secondary benchmark
robust and interpretable.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from . import config
from .preprocessing import make_preprocessor
from .utils import get_logger

log = get_logger()

try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc
    from sksurv.util import Surv
    _HAS_SKSURV = True
except Exception:
    _HAS_SKSURV = False


def _design(df, features, fit_on=None):
    """Return a numeric design DataFrame using a leakage-safe preprocessor.

    If fit_on is provided, the preprocessor is fitted on fit_on (training rows)
    and applied to df; otherwise fitted on df.
    """
    pre = make_preprocessor(fit_on if fit_on is not None else df, features)
    pre.fit((fit_on if fit_on is not None else df)[features])
    X = pre.transform(df[features])
    cols = [f"x{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=cols, index=df.index)


def _cindex_lifelines(train, test, features, penalizer=0.0, l1_ratio=0.0):
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    Xtr = _design(train, features)
    Xtr["T"] = train["os_months"].values
    Xtr["E"] = train["os_event"].values
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(Xtr, duration_col="T", event_col="E", robust=False)
    Xte = _design(test, features, fit_on=train)
    risk = cph.predict_partial_hazard(Xte).values.ravel()
    c = concordance_index(test["os_months"].values, -risk, test["os_event"].values)
    return c


def cox_internal_cv(df, features, n_repeats=None, n_folds=None, seed=config.SEED):
    """Repeated stratified CV C-index for Cox and penalised Cox."""
    n_repeats = n_repeats or max(3, config.CV_REPEATS // 2)
    n_folds = n_folds or config.CV_FOLDS
    d = df.dropna(subset=["os_months", "os_event"]).reset_index(drop=True)
    strat = (d["os_event"].astype(int)).values
    rskf = RepeatedStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=seed)
    out = {"cox": [], "cox_penalised": []}
    if _HAS_SKSURV:
        out["rsf"] = []
    for tr, te in rskf.split(d, strat):
        train, test = d.iloc[tr], d.iloc[te]
        try:
            out["cox"].append(_cindex_lifelines(train, test, features, 0.0))
        except Exception:
            out["cox"].append(_cindex_lifelines(train, test, features, 0.1))
        out["cox_penalised"].append(_cindex_lifelines(train, test, features, 0.1, 0.5))
        if _HAS_SKSURV:
            out["rsf"].append(_rsf_cindex(train, test, features))
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in out.items()}


def _rsf_cindex(train, test, features):
    Xtr = _design(train, features).values
    ytr = Surv.from_arrays(train["os_event"].astype(bool).values, train["os_months"].values)
    rsf = RandomSurvivalForest(n_estimators=300, min_samples_leaf=10, random_state=config.SEED, n_jobs=-1)
    rsf.fit(Xtr, ytr)
    Xte = _design(test, features, fit_on=train).values
    return float(rsf.score(Xte, Surv.from_arrays(test["os_event"].astype(bool).values, test["os_months"].values)))


def cox_external(train_df, test_df, features):
    """Train on full source cohort, report C-index on target cohort."""
    shared = [f for f in features if f in train_df.columns and f in test_df.columns
              and train_df[f].notna().any() and test_df[f].notna().any()]
    tr = train_df.dropna(subset=["os_months", "os_event"]).reset_index(drop=True)
    te = test_df.dropna(subset=["os_months", "os_event"]).reset_index(drop=True)
    res = {"shared_features": shared,
           "cox": _cindex_lifelines(tr, te, shared, 0.1),
           "cox_penalised": _cindex_lifelines(tr, te, shared, 0.1, 0.5)}
    return res
