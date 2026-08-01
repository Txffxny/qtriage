"""Anomaly detection with split-conformal calibration.

The calibration day is partitioned into two disjoint halves. One fits the
detector; the other produces the null score distribution. This matters: if the
detector scored its own training points, those scores would be in-sample and
therefore not exchangeable with test scores, silently invalidating every
conformal p-value downstream.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from src import config


def feature_matrix(df):
    """Select model input columns from a cached parquet frame.

    Drops bookkeeping columns (prefixed '_') except the missingness flag, which
    is included only when config.INCLUDE_IMPUTED_FLAG is set.
    """
    cols = [c for c in df.columns if not c.startswith("_")]
    if getattr(config, "INCLUDE_IMPUTED_FLAG", False) and "_was_imputed" in df.columns:
        cols = cols + ["_was_imputed"]
    return df[cols].to_numpy(dtype=np.float32), cols


def split_calibration(n, train_fraction=None, seed=None):
    """Disjoint train/calibration index split over the calibration day."""
    train_fraction = config.TRAIN_FRACTION if train_fraction is None else train_fraction
    seed = config.RANDOM_SEED if seed is None else seed

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round(train_fraction * n))
    return perm[:cut], perm[cut:]


def fit_detector(X_train, seed=None, n_estimators=300, max_samples=256):
    """Fit IsolationForest on benign-only training data."""
    seed = config.RANDOM_SEED if seed is None else seed
    det = IsolationForest(
        n_estimators=n_estimators,
        max_samples=max_samples,
        random_state=seed,
        n_jobs=-1,
    )
    det.fit(X_train)
    return det


def anomaly_scores(det, X):
    """Higher = more anomalous. sklearn's score_samples is the opposite sign."""
    return -det.score_samples(X)
