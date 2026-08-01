"""Null test: are the conformal p-values valid, or is exchangeability broken?

Channel B produced FDP 0.571 at q=0.05. BH controls FDR regardless of detector
quality, so that is not a detection failure - it means the p-values are not
uniform under the null. This script isolates the cause.

Three regimes, all evaluated on BENIGN-ONLY data so every rejection is by
construction false:

  A. within-day, random split   - calibrate on half of Monday, test the other
                                  half. Random assignment, so units are
                                  exchangeable by construction. If this fails,
                                  the problem is correlated units (windows from
                                  one host are not independent observations).

  B. within-day, temporal split - calibrate on Monday's first half by time,
                                  test the later half. Isolates within-day
                                  temporal drift.

  C. across-day                 - calibrate on Monday, test on another benign
                                  day's benign flows. Isolates day-to-day drift.

Reading the result: A fails -> correlated units, use Mondrian conformal.
A passes but B or C fails -> drift, use rolling recalibration.
"""
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats as sps

from src import config
from src.aggregate import aggregate
from src.detector import fit_detector, anomaly_scores
from src.stats import conformal_pvalues, benjamini_hochberg

Q_GRID = [0.05, 0.10, 0.20]
MIN_BATCH_B = 5


def load(f):
    path = config.PROCESSED / f.replace(".csv", ".parquet")
    if not path.exists():
        sys.exit("Missing cache " + str(path))
    return pd.read_parquet(path)


def feats(agg):
    cols = [c for c in agg.columns if not c.startswith("_")]
    X = agg[cols].to_numpy(dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def null_report(name, X_fit, X_cal, X_test, ts_test, note=""):
    """Fit, calibrate and test on benign-only data. Every rejection is false."""
    det = fit_detector(X_fit, max_samples=min(256, len(X_fit)))
    cal_scores = anomaly_scores(det, X_cal)
    p = conformal_pvalues(cal_scores, anomaly_scores(det, X_test))

    ks = sps.kstest(p, "uniform")
    floor = 1.0 / (len(cal_scores) + 1)

    print("\n" + name)
    if note:
        print("  " + note)
    print("  fit " + format(len(X_fit), ",") + " | cal " + format(len(X_cal), ",")
          + " | test " + format(len(X_test), ","))
    print("  p-values under the null: mean " + format(float(p.mean()), ".4f")
          + "  (valid ~ 0.500)")
    print("  KS vs uniform: D = " + format(ks.statistic, ".4f")
          + ", p = " + format(ks.pvalue, ".3g")
          + ("  UNIFORM" if ks.pvalue > 0.01 else "  NOT UNIFORM"))
    print("  fraction below floor*10 (" + format(floor * 10, ".1e") + "): "
          + format(float((p <= floor * 10).mean()), ".4f")
          + "   (valid ~ " + format(floor * 10, ".4f") + ")")

    bid = ts_test.dt.floor(config.BATCH).astype("int64").to_numpy()
    print("  " + "q".rjust(6) + "false alerts".rjust(14) + "batches firing".rjust(16)
          + "   verdict")
    for q in Q_GRID:
        n_rej = 0
        n_fire = 0
        n_batch = 0
        for b in np.unique(bid):
            idx = np.flatnonzero(bid == b)
            if len(idx) < MIN_BATCH_B:
                continue
            n_batch += 1
            rej, _ = benjamini_hochberg(p[idx], q=q)
            if rej.any():
                n_fire += 1
                n_rej += int(rej.sum())
        frac = n_fire / max(n_batch, 1)
        verdict = "OK" if frac <= q + 0.05 else "VIOLATION"
        print("  " + format(q, ".2f").rjust(6) + format(n_rej, ",").rjust(14)
              + (str(n_fire) + "/" + str(n_batch)).rjust(16)
              + "   " + format(frac, ".3f") + "  " + verdict)


def main():
    print("NULL CALIBRATION TEST - benign only, every rejection is false")
    print("window=" + config.BATCH + "  unit=" + config.AGGREGATION_UNIT)

    mon = aggregate(load(config.CALIBRATION_FILE), verbose=False)
    X_mon = feats(mon)
    ts_mon = pd.to_datetime(mon["_timestamp"])

    # --- A. random split: exchangeable by construction ---------------------
    rng = np.random.default_rng(config.RANDOM_SEED)
    perm = rng.permutation(len(X_mon))
    a, b, c = np.array_split(perm, 3)
    null_report("A. WITHIN-DAY, RANDOM SPLIT", X_mon[a], X_mon[b], X_mon[c],
                ts_mon.iloc[c].reset_index(drop=True),
                note="units assigned at random; failure here means correlated units")

    # --- B. temporal split: same day, later in time ------------------------
    order = np.argsort(ts_mon.to_numpy())
    n = len(order)
    early, mid, late = order[:n // 3], order[n // 3:2 * n // 3], order[2 * n // 3:]
    null_report("B. WITHIN-DAY, TEMPORAL SPLIT", X_mon[early], X_mon[mid], X_mon[late],
                ts_mon.iloc[late].reset_index(drop=True),
                note="calibrate early, test late; isolates within-day drift")

    # --- C. across-day: Monday calibrates, another day's benign tests ------
    others = [p.name.replace(".parquet", ".csv")
              for p in sorted(config.PROCESSED.glob("*.parquet"))
              if p.name.replace(".parquet", ".csv") != config.CALIBRATION_FILE]
    if others:
        target = config.TEST_FILE if config.TEST_FILE in others else others[0]
        other = load(target)
        other = other[~other["_is_attack"].to_numpy().astype(bool)]
        oth_agg = aggregate(other, verbose=False)
        oth_agg = oth_agg[~oth_agg["_is_attack"].to_numpy().astype(bool)]
        half = len(X_mon) // 2
        null_report("C. ACROSS-DAY (" + target.replace(".csv", "") + ", benign only)",
                    X_mon[perm[:half]], X_mon[perm[half:]], feats(oth_agg),
                    pd.to_datetime(oth_agg["_timestamp"]).reset_index(drop=True),
                    note="calibrate Monday, test another day; isolates day-to-day drift")

    print("\nREADING IT:")
    print("  A fails            -> correlated units; windows from one host are")
    print("                        not independent. Fix: Mondrian conformal.")
    print("  A passes, B/C fail -> drift. Fix: rolling recalibration.")
    print("  all pass           -> p-values are valid; channel B's FDP violation")
    print("                        has another cause.")


if __name__ == "__main__":
    main()
