"""Fast, resumable repeated cross-validation.

Speed: each TensorFlow architecture is built once per setting and its weights are
reset between folds, avoiding repeated graph construction. One-hot categories are
fixed a priori (config.CATEGORICAL_LEVELS), so the encoded dimension is constant
across folds and the same compiled model can be reused.

Resumability: progress is checkpointed per repeat to results/predictions/_ckpt,
so the runner can be invoked repeatedly under a wall-clock budget until a setting
is complete. Out-of-fold probabilities are averaged across repeats, matching the
semantics of the original internal_cv.

Rigour: numeric imputation and scaling are still fitted inside each training fold;
only the categorical level set (structural metadata, no outcome) is fixed.
"""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config
from .endpoints import make_binary_endpoint
from .models_baseline import build_baselines
from .models_tensorflow import ARCHITECTURES, build_mlp
from .preprocessing import make_preprocessor, split_feature_types
from .io_results import save_predictions
from .utils import get_logger, save_json

log = get_logger()
CKPT = config.PREDICTIONS / "_ckpt"
CKPT.mkdir(parents=True, exist_ok=True)
TF_VARIANTS = ["tf_compact", "tf_medium", "tf_regularised"]
ARCH_OF = {"tf_compact": "compact", "tf_medium": "medium", "tf_regularised": "regularised"}


def _encoded_dim(df, features):
    pre = make_preprocessor(df, features)
    return pre.fit_transform(df[features]).shape[1]


def _pad_to(X, y, n, rng):
    """Pad (by resampling) or truncate arrays to a fixed length n, so TensorFlow
    sees a constant input signature across folds and does not re-trace."""
    if len(X) == n:
        return X, y
    if len(X) > n:
        return X[:n], y[:n]
    idx = rng.integers(0, len(X), n - len(X))
    return np.vstack([X, X[idx]]), np.concatenate([y, y[idx]])


def _fit_tf(model, w0, Xtr, ytr, seed, target_a=None, target_v=None):
    import tensorflow as tf
    model.set_weights([w.copy() for w in w0])
    strat = ytr if len(np.unique(ytr)) > 1 else None
    Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.2, random_state=seed, stratify=strat)
    if target_a:
        rng = np.random.default_rng(seed)
        Xa, ya = _pad_to(Xa, ya, target_a, rng)
        Xv, yv = _pad_to(Xv, yv, target_v, rng)
    pos = float(ya.sum()); neg = float(len(ya) - pos)
    cw = {0: 1.0, 1: (neg / pos) if pos > 0 else 1.0}
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    model.fit(Xa, ya, validation_data=(Xv, yv), epochs=100, batch_size=64,
              class_weight=cw, callbacks=[es], verbose=0)


def run_setting(tag, df, features, horizon, n_repeats=None, n_folds=None,
                include_tf=True, budget_s=38, seed=config.SEED):
    """Process repeats for one setting within a wall-clock budget. Returns status."""
    import tensorflow as tf
    n_repeats = n_repeats or config.CV_REPEATS
    n_folds = n_folds or config.CV_FOLDS
    labelled, info = make_binary_endpoint(df, horizon)
    labelled = labelled.reset_index(drop=True)
    y = labelled["y"].values.astype(int)
    n = len(labelled)
    base_names = list(build_baselines(labelled, features, seed).keys())
    model_names = base_names + (TF_VARIANTS if include_tf else [])

    ck = CKPT / f"{tag}.npz"
    if ck.exists():
        d = np.load(ck, allow_pickle=True)
        prob_sum = {m: d[f"sum__{m}"] for m in model_names}
        prob_cnt = {m: d[f"cnt__{m}"] for m in model_names}
        done = set(d["done_repeats"].tolist())
    else:
        prob_sum = {m: np.zeros(n) for m in model_names}
        prob_cnt = {m: np.zeros(n) for m in model_names}
        done = set()

    # fixed array sizes (avoid TF retracing across folds)
    fold_test = n // n_folds + 1
    train_fold = n - n // n_folds
    target_a = int(0.8 * train_fold) + 2
    target_v = int(0.2 * train_fold) + 2
    target_te = fold_test + 2
    rng_pad = np.random.default_rng(seed)

    t0 = time.time()  # include build + TF init in the budget so calls stay bounded
    # Build TF models once (constant encoded dim)
    tf_models = {}
    if include_tf:
        dim = _encoded_dim(labelled, features)
        for v in TF_VARIANTS:
            spec = ARCHITECTURES[ARCH_OF[v]]
            m = build_mlp(dim, spec["layers"], spec["l2"], seed=seed)
            tf_models[v] = (m, [w.copy() for w in m.get_weights()])

    processed = 0
    for rep in range(n_repeats):
        if rep in done:
            continue
        if time.time() - t0 > budget_s and processed > 0:
            break
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + rep)
        for tr, te in skf.split(labelled, y):
            Xtr_df, Xte_df = labelled.iloc[tr], labelled.iloc[te]
            ytr = y[tr]
            # baselines
            for m in base_names:
                est = build_baselines(Xtr_df, features, seed + rep)[m]
                est.fit(Xtr_df, ytr)
                prob_sum[m][te] += est.predict_proba(Xte_df)[:, 1]
                prob_cnt[m][te] += 1
            # TF variants (shared fold preprocessing)
            if include_tf:
                pre = make_preprocessor(Xtr_df, features)
                Xtr = pre.fit_transform(Xtr_df[features]).astype("float32")
                Xte = pre.transform(Xte_df[features]).astype("float32")
                Xte_pad, _ = _pad_to(Xte, np.zeros(len(Xte)), target_te, rng_pad)
                m_te = len(te)
                for v in TF_VARIANTS:
                    model, w0 = tf_models[v]
                    _fit_tf(model, w0, Xtr, ytr, seed + rep, target_a, target_v)
                    pred = np.asarray(model(Xte_pad, training=False)).ravel()[:m_te]
                    prob_sum[v][te] += pred
                    prob_cnt[v][te] += 1
        done.add(rep)
        processed += 1
        # checkpoint after every repeat so an overrun never loses completed work
        np.savez(ck, done_repeats=np.array(sorted(done)),
                 **{f"sum__{m}": prob_sum[m] for m in model_names},
                 **{f"cnt__{m}": prob_cnt[m] for m in model_names})

    if len(done) >= n_repeats:
        preds = {m: prob_sum[m] / np.maximum(prob_cnt[m], 1) for m in model_names}
        save_predictions(tag, y, preds, sample=labelled["sample"].values)
        save_json({"setting": tag, "features": features, "n": n,
                   "endpoint_info": info, "models": model_names,
                   "cv": f"{n_repeats}x{n_folds}"}, config.METRICS / f"{tag}_meta.json")
        ck.unlink(missing_ok=True)
        return {"tag": tag, "status": "complete", "repeats": len(done), "n": n}
    return {"tag": tag, "status": "partial", "repeats": len(done), "target": n_repeats}
