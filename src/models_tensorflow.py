"""TensorFlow multilayer perceptrons with a scikit-learn-compatible wrapper.

Three prespecified architectures are provided. None is tuned outside
cross-validation; all share identical preprocessing, folds, endpoints, and
training settings. Preprocessing is fitted on the training fold only.

  compact     : Dense 32 -> Dropout -> Dense 16 -> Dropout -> sigmoid
  medium      : Dense 128 -> BN -> Dropout -> Dense 64 -> Dropout -> Dense 32 -> Dropout -> sigmoid, L2
  regularised : Dense 64 -> Dropout 0.4 -> Dense 32 -> Dropout 0.4 -> sigmoid, L2

All use Adam, binary cross-entropy, class weighting, and early stopping on a
validation split carved from the training fold.
"""
from __future__ import annotations
import os
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from . import config
from .preprocessing import make_preprocessor

# Prespecified architecture catalogue. Each layer: (units, dropout, batchnorm).
ARCHITECTURES = {
    "compact": {"layers": [(32, 0.2, False), (16, 0.2, False)], "l2": 0.0},
    "medium": {"layers": [(128, 0.3, True), (64, 0.3, False), (32, 0.3, False)], "l2": 1e-4},
    "regularised": {"layers": [(64, 0.4, False), (32, 0.4, False)], "l2": 1e-3},
}


def build_mlp(input_dim, layers, l2=0.0, lr=1e-3, seed=config.SEED):
    import tensorflow as tf
    from tensorflow.keras import regularizers
    tf.keras.utils.set_random_seed(seed)
    reg = regularizers.l2(l2) if l2 and l2 > 0 else None
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    for units, dropout, bn in layers:
        model.add(tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=reg))
        if bn:
            model.add(tf.keras.layers.BatchNormalization())
        if dropout and dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    # compile without an extra streaming metric to reduce per-epoch overhead in CV
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy")
    return model


class TFMLPClassifier(BaseEstimator, ClassifierMixin):
    """sklearn-style wrapper around a prespecified Keras MLP architecture."""

    def __init__(self, features=None, arch="compact", lr=1e-3, batch_size=64,
                 max_epochs=200, patience=20, val_frac=0.2, scale=True,
                 seed=config.SEED, verbose=0):
        self.features = features
        self.arch = arch
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_frac = val_frac
        self.scale = scale
        self.seed = seed
        self.verbose = verbose

    def fit(self, X, y):
        import tensorflow as tf
        spec = ARCHITECTURES[self.arch]
        y = np.asarray(y).astype("float32")
        feats = self.features or list(X.columns)
        self.features_ = [f for f in feats if f in X.columns]
        self.pre_ = make_preprocessor(X[self.features_], self.features_, scale=self.scale)
        Xt = self.pre_.fit_transform(X[self.features_]).astype("float32")

        strat = y if len(np.unique(y)) > 1 else None
        Xtr, Xval, ytr, yval = train_test_split(
            Xt, y, test_size=self.val_frac, random_state=self.seed, stratify=strat)
        pos = float(ytr.sum()); neg = float(len(ytr) - pos)
        cw = {0: 1.0, 1: (neg / pos) if pos > 0 else 1.0}

        self.model_ = build_mlp(Xt.shape[1], spec["layers"], spec["l2"], self.lr, self.seed)
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True)
        self.model_.fit(Xtr, ytr, validation_data=(Xval, yval),
                        epochs=self.max_epochs, batch_size=self.batch_size,
                        class_weight=cw, callbacks=[es], verbose=self.verbose)
        self.classes_ = np.array([0, 1])
        self.n_input_ = Xt.shape[1]
        return self

    def _predict_raw(self, X):
        import tensorflow as tf
        Xt = self.pre_.transform(X[self.features_]).astype("float32")
        # direct call avoids predict-function retracing overhead in CV loops
        return np.asarray(self.model_(Xt, training=False)).ravel()

    def predict_proba(self, X):
        p = self._predict_raw(X)
        return np.column_stack([1.0 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# Convenience constructors for the three prespecified variants
def tf_variants(features, seed=config.SEED):
    return {
        "tf_compact": TFMLPClassifier(features=features, arch="compact", seed=seed),
        "tf_medium": TFMLPClassifier(features=features, arch="medium", seed=seed),
        "tf_regularised": TFMLPClassifier(features=features, arch="regularised", seed=seed),
    }
