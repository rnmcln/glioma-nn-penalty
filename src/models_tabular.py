"""FT-Transformer-lite: a compact feature-tokenizer + transformer for tabular data.

Added as a modern tabular baseline so the conclusion is not limited to multilayer
perceptrons. It is deliberately small and prespecified (not tuned outside
cross-validation). Each preprocessed feature (numeric or one-hot column) is
embedded into a token; a CLS token is prepended; a transformer encoder is applied;
the CLS representation drives a sigmoid output. This follows the FT-Transformer
design (Gorishniy et al.) in reduced form.
"""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

from . import config
from .preprocessing import make_preprocessor


def build_ft_transformer(n_features, d_token=32, n_heads=4, n_blocks=2,
                         dropout=0.1, lr=1e-3, seed=config.SEED):
    import tensorflow as tf
    from tensorflow.keras import layers

    class CLSToken(layers.Layer):
        def build(self, input_shape):
            self.cls = self.add_weight(shape=(1, 1, input_shape[-1]),
                                       initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
                                       trainable=True, name="cls")
        def call(self, x):
            b = tf.shape(x)[0]
            return tf.concat([tf.tile(self.cls, [b, 1, 1]), x], axis=1)

    tf.keras.utils.set_random_seed(seed)
    inp = layers.Input(shape=(n_features,))
    # feature tokenizer: x_j -> x_j * W_j + b_j  (per-feature d-dim embedding)
    x = layers.Reshape((n_features, 1))(inp)
    tokens = layers.Dense(d_token)(x)              # (batch, n_features, d_token)
    seq = CLSToken()(tokens)
    for _ in range(n_blocks):
        attn = layers.MultiHeadAttention(num_heads=n_heads, key_dim=d_token, dropout=dropout)(seq, seq)
        seq = layers.LayerNormalization()(layers.Add()([seq, attn]))
        ff = layers.Dense(d_token * 2, activation="gelu")(seq)
        ff = layers.Dense(d_token)(ff)
        seq = layers.LayerNormalization()(layers.Add()([seq, ff]))
    cls_out = seq[:, 0, :]
    cls_out = layers.Dropout(dropout)(cls_out)
    out = layers.Dense(1, activation="sigmoid")(cls_out)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy")
    return model


class FTTransformerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, features=None, d_token=32, n_heads=4, n_blocks=2, dropout=0.1,
                 lr=1e-3, batch_size=64, max_epochs=200, patience=15, val_frac=0.2,
                 seed=config.SEED, verbose=0):
        self.features = features; self.d_token = d_token; self.n_heads = n_heads
        self.n_blocks = n_blocks; self.dropout = dropout; self.lr = lr
        self.batch_size = batch_size; self.max_epochs = max_epochs
        self.patience = patience; self.val_frac = val_frac; self.seed = seed; self.verbose = verbose

    def fit(self, X, y):
        import tensorflow as tf
        y = np.asarray(y).astype("float32")
        feats = self.features or list(X.columns)
        self.features_ = [f for f in feats if f in X.columns]
        self.pre_ = make_preprocessor(X[self.features_], self.features_, scale=True)
        Xt = self.pre_.fit_transform(X[self.features_]).astype("float32")
        strat = y if len(np.unique(y)) > 1 else None
        Xtr, Xv, ytr, yv = train_test_split(Xt, y, test_size=self.val_frac, random_state=self.seed, stratify=strat)
        pos = float(ytr.sum()); neg = float(len(ytr) - pos)
        cw = {0: 1.0, 1: (neg / pos) if pos > 0 else 1.0}
        self.model_ = build_ft_transformer(Xt.shape[1], self.d_token, self.n_heads,
                                           self.n_blocks, self.dropout, self.lr, self.seed)
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=self.patience, restore_best_weights=True)
        self.model_.fit(Xtr, ytr, validation_data=(Xv, yv), epochs=self.max_epochs,
                        batch_size=self.batch_size, class_weight=cw, callbacks=[es], verbose=self.verbose)
        self.classes_ = np.array([0, 1]); self.n_input_ = Xt.shape[1]
        return self

    def predict_proba(self, X):
        Xt = self.pre_.transform(X[self.features_]).astype("float32")
        p = np.asarray(self.model_(Xt, training=False)).ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
