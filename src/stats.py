"""Conformal p-values and Benjamini-Hochberg FDR control.

Reference: Bates, Candes, Lei, Romano, Sesia (2023). "Testing for outliers
with conformal p-values." Annals of Statistics 51(1), 149-178.
"""

from __future__ import annotations
import numpy as np


def conformal_pvalues(cal_scores, test_scores):
    """Empirical p-values against a benign calibration set.

    Convention: HIGHER score = MORE anomalous, for both arrays.
    p_i = (1 + #{j in cal : s_j >= s_i}) / (n_cal + 1)
    """
    cal_sorted = np.sort(np.asarray(cal_scores, dtype=np.float64))
    s = np.asarray(test_scores, dtype=np.float64)
    n = cal_sorted.size
    if n == 0:
        raise ValueError("Calibration set is empty.")
    n_ge = n - np.searchsorted(cal_sorted, s, side="left")
    return (n_ge + 1.0) / (n + 1.0)


def benjamini_hochberg(p, q=0.05):
    """Step-up BH. Returns (reject mask, adjusted q-values)."""
    p = np.asarray(p, dtype=np.float64)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)

    order = np.argsort(p, kind="mergesort")
    p_sorted = p[order]
    ranks = np.arange(1, m + 1)

    below = np.nonzero(p_sorted <= q * ranks / m)[0]
    k = below[-1] + 1 if below.size else 0
    reject = np.zeros(m, dtype=bool)
    reject[order[:k]] = True

    raw = p_sorted * m / ranks
    adj_sorted = np.minimum.accumulate(raw[::-1])[::-1]
    qvals = np.empty(m, dtype=np.float64)
    qvals[order] = np.clip(adj_sorted, 0.0, 1.0)
    return reject, qvals


def benjamini_yekutieli(p, q=0.05):
    """BH under arbitrary dependence. Conservative fallback."""
    m = np.asarray(p).size
    c_m = np.sum(1.0 / np.arange(1, m + 1)) if m else 1.0
    return benjamini_hochberg(p, q=q / c_m)


def realized_fdp(reject, is_true_anomaly):
    """Fraction of raised alerts that were actually benign."""
    n_rej = int(reject.sum())
    if n_rej == 0:
        return 0.0
    return int((reject & ~is_true_anomaly).sum()) / n_rej


def realized_power(reject, is_true_anomaly):
    """Recall. FDR says nothing about this, which is why we report it too."""
    n_true = int(is_true_anomaly.sum())
    if n_true == 0:
        return float("nan")
    return int((reject & is_true_anomaly).sum()) / n_true