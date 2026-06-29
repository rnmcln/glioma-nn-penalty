"""Transparent and standard machine-learning baselines.

All models expose the sklearn estimator API (fit / predict_proba) so they can be
driven by a common evaluation loop. Preprocessing is attached as the first step
of each Pipeline, so fitting a model fits its preprocessing on the training fold
only.
"""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config
from .preprocessing import make_preprocessor

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


class AgeGradeLogistic(BaseEstimator, ClassifierMixin):
    """Transparent reference model: logistic regression on age + grade only."""

    def __init__(self, seed: int = config.SEED):
        self.seed = seed

    def fit(self, X, y):
        import pandas as pd
        cols = [c for c in ("age", "grade") if c in X.columns]
        self.cols_ = cols
        self.pre_ = make_preprocessor(X[cols], cols)
        Xt = self.pre_.fit_transform(X[cols])
        self.clf_ = LogisticRegression(max_iter=1000, random_state=self.seed)
        self.clf_.fit(Xt, y)
        self.classes_ = self.clf_.classes_
        return self

    def predict_proba(self, X):
        Xt = self.pre_.transform(X[self.cols_])
        return self.clf_.predict_proba(Xt)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _wrap(estimator, df, features, scale=True):
    return Pipeline([("pre", make_preprocessor(df, features, scale=scale)),
                     ("clf", estimator)])


def build_baselines(df, features: list[str], seed: int = config.SEED) -> dict:
    """Return dict name -> unfitted estimator (Pipeline or custom)."""
    models = {
        "base_rate": DummyClassifier(strategy="prior"),
        "logistic_age_grade": AgeGradeLogistic(seed=seed),
        "penalised_logistic": _wrap(
            LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5,
                               C=1.0, max_iter=5000, random_state=seed),
            df, features, scale=True),
        "random_forest": _wrap(
            RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                   random_state=seed, n_jobs=-1),
            df, features, scale=False),
        "hist_gboost": _wrap(
            HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                           max_iter=400, l2_regularization=1.0,
                                           random_state=seed),
            df, features, scale=False),
    }
    if _HAS_XGB:
        models["xgboost"] = _wrap(
            XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                          eval_metric="logloss", random_state=seed, n_jobs=-1),
            df, features, scale=False)
    return models


TRANSPARENT_MODELS = {"base_rate", "logistic_age_grade", "penalised_logistic"}
