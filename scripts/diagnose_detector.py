"""Diagnose detector separation: is the bottleneck the statistics or the model?

BH over m hypotheses needs p-values near q/m. With conformal p-values floored at
1/(n_cal+1), that demands attacks sit in the extreme upper tail of the benign
score distribution - a far stronger requirement than good AUC. This script
measures where attacks actually land.
"""
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config
from src.detector import feature_matrix, split_calibration, fit_detector, anomaly_scores
from src.stats import conformal_pvalues


def load_cached(filename):
    path = config.PROCESSED / filename.replace(".csv", ".parquet")
    if not path.exists():
        sys.exit("Missing cache " + str(path))
    return pd.read_parquet(path)


def evaluate(cal_df, test_df, drop_cols, tag):
    """Fit, calibrate, score and report separation for one feature configuration."""
    print("\n" + "=" * 68)
    print(tag)
    print("=" * 68)

    cal = cal_df.drop(columns=[c for c in drop_cols if c in cal_df.columns])
    test = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns])

    X_cal, cols = feature_matrix(cal)
    tr, ca = split_calibration(len(X_cal))
    det = fit_detector(X_cal[tr])
    cal_scores = anomaly_scores(det, X_cal[ca])

    X_test, _ = feature_matrix(test)
    y = test_df["_is_attack"].to_numpy().astype(bool)
    scores = anomaly_scores(det, X_test)
    p = conformal_pvalues(cal_scores, scores)

    floor = 1.0 / (len(cal_scores) + 1)
    auc = roc_auc_score(y, scores)

    print(str(len(cols)) + " features | AUC = " + format(auc, ".4f"))
    print("p-value floor = " + format(floor, ".2e"))

    # Where do attacks sit in the benign score distribution?
    pct = 100 * (1 - p)
    print("\nattack score percentile vs benign calibration:")
    for label, val in [("median", np.median(pct[y])), ("90th", np.percentile(pct[y], 90)),
                       ("99th", np.percentile(pct[y], 99)), ("max", pct[y].max())]:
        print("  " + label.ljust(8) + format(val, ".4f") + "%")

    # How many attacks reach useful p-values?
    print("\nattacks reaching each p-value threshold:")
    for thr in [1e-2, 1e-3, 1e-4, 1e-5, floor * 1.001]:
        n_atk = int((p[y] <= thr).sum())
        n_ben = int((p[~y] <= thr).sum())
        print("  p <= " + format(thr, ".1e") + "   attacks "
              + format(n_atk, ",").rjust(7) + "   benign " + format(n_ben, ",").rjust(9))

    # Per-class: which attack types are detectable at all?
    print("\nper-class median p-value (lower = more detectable):")
    labels = test_df["_label"].to_numpy()
    for lab in pd.unique(labels):
        mask = labels == lab
        print("  " + str(lab).ljust(30) + format(np.median(p[mask]), ".3e")
              + "   n=" + format(int(mask.sum()), ","))

    return auc


def main():
    cal_df = load_cached(config.CALIBRATION_FILE)
    test_df = load_cached(config.TEST_FILE)

    # Headline configuration: behaviour only.
    evaluate(cal_df, test_df, drop_cols=[], tag="A. Headline (no Dst Port, no imputed flag)")

    # Does the port carry the signal we deliberately withheld?
    if "Dst Port" not in cal_df.columns:
        print("\n" + "=" * 68)
        print("B. Dst Port ablation UNAVAILABLE: column was dropped at cache time.")
        print("   To run it, remove 'Dst Port' from EXCLUDE_FROM_FEATURES in")
        print("   src/config.py and re-run scripts/build_cache.py.")
        print("=" * 68)


if __name__ == "__main__":
    main()
