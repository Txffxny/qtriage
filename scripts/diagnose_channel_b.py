"""Why did channel B violate FDR control despite AUC 0.99?

Hypothesis: hosts present on the test day but absent from calibration are not
exchangeable with the calibration set. They have no baseline, so they score as
extreme for a reason unrelated to malice, and their p-values are not valid.
BH sorts by p-value, so these contaminate the rejection set from the top down.

This script partitions the test day into profiled and unprofiled hosts and
recomputes FDR control separately for each.
"""
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config
from src.aggregate import aggregate
from src.detector import split_calibration, fit_detector, anomaly_scores
from src.stats import conformal_pvalues, benjamini_hochberg, realized_fdp, realized_power

Q_GRID = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
MIN_BATCH_B = 5


def load_cached(filename):
    path = config.PROCESSED / filename.replace(".csv", ".parquet")
    if not path.exists():
        sys.exit("Missing cache " + str(path))
    return pd.read_parquet(path)


def host_features(agg):
    cols = [c for c in agg.columns if not c.startswith("_")]
    X = agg[cols].to_numpy(dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), cols


def batched_bh(p, batch_ids, q, mask=None, min_batch=MIN_BATCH_B):
    """BH within batches, optionally restricted to a subset of units."""
    reject = np.zeros(len(p), dtype=bool)
    sel = np.ones(len(p), dtype=bool) if mask is None else mask
    for bid in np.unique(batch_ids[sel]):
        idx = np.flatnonzero((batch_ids == bid) & sel)
        if len(idx) < min_batch:
            continue
        rej_b, _ = benjamini_hochberg(p[idx], q=q)
        reject[idx] = rej_b
    return reject


def main():
    cal_agg = aggregate(load_cached(config.CALIBRATION_FILE), verbose=False)
    test_agg = aggregate(load_cached(config.TEST_FILE), verbose=False)

    X_cal, _ = host_features(cal_agg)
    tr, ca = split_calibration(len(X_cal))
    det = fit_detector(X_cal[tr], max_samples=min(256, len(tr)))
    cal_scores = anomaly_scores(det, X_cal[ca])
    floor = 1.0 / (len(cal_scores) + 1)

    X_test, _ = host_features(test_agg)
    y = test_agg["_is_attack"].to_numpy().astype(bool)
    p = conformal_pvalues(cal_scores, anomaly_scores(det, X_test))
    batch_ids = pd.to_datetime(test_agg["_timestamp"]).dt.floor(config.BATCH).astype("int64").to_numpy()

    cal_hosts = set(cal_agg["_src_ip"].unique())
    test_hosts = test_agg["_src_ip"].to_numpy()
    profiled = np.array([h in cal_hosts for h in test_hosts])

    print("HOST COVERAGE")
    print("  calibration hosts: " + str(len(cal_hosts)))
    print("  test hosts:        " + str(len(set(test_hosts))))
    novel = sorted(set(test_hosts) - cal_hosts)
    print("  novel (unprofiled): " + str(len(novel)) + "  " + str(novel[:6]))
    print("  test windows: " + format(int(profiled.sum()), ",") + " profiled, "
          + format(int((~profiled).sum()), ",") + " unprofiled")
    print()

    print("PER-HOST BEHAVIOUR (sorted by minimum p-value)")
    print("  " + "host".ljust(18) + "prof".rjust(5) + "wins".rjust(6)
          + "mal".rjust(5) + "min p".rjust(11) + "med p".rjust(11) + "  at floor")
    rows = []
    for h in set(test_hosts):
        m = test_hosts == h
        rows.append((float(p[m].min()), h, h in cal_hosts, int(m.sum()),
                     int(y[m].sum()), float(np.median(p[m])),
                     int((p[m] <= floor + 1e-12).sum())))
    for minp, h, prof, n, nm, medp, nfloor in sorted(rows)[:12]:
        print("  " + str(h).ljust(18) + ("Y" if prof else "N").rjust(5)
              + str(n).rjust(6) + str(nm).rjust(5)
              + format(minp, ".2e").rjust(11) + format(medp, ".2e").rjust(11)
              + str(nfloor).rjust(10))
    print()

    print("WHO IS AT THE p-VALUE FLOOR? (floor = " + format(floor, ".2e") + ")")
    at_floor = p <= floor + 1e-12
    print("  total " + str(int(at_floor.sum()))
          + " | malicious " + str(int((at_floor & y).sum()))
          + " | benign " + str(int((at_floor & ~y).sum()))
          + " | unprofiled " + str(int((at_floor & ~profiled).sum())))
    print()

    print("FDR CONTROL, ALL TEST UNITS vs PROFILED HOSTS ONLY")
    print("  " + "q".rjust(6) + " | " + "all: FDP".rjust(10) + "power".rjust(8)
          + " | " + "profiled: FDP".rjust(14) + "power".rjust(8))
    for q in Q_GRID:
        r_all = batched_bh(p, batch_ids, q)
        r_pro = batched_bh(p, batch_ids, q, mask=profiled)
        fdp_a, pow_a = realized_fdp(r_all, y), realized_power(r_all, y)
        fdp_p = realized_fdp(r_pro[profiled], y[profiled])
        pow_p = realized_power(r_pro[profiled], y[profiled])
        print("  " + format(q, ".2f").rjust(6) + " | " + format(fdp_a, ".4f").rjust(10)
              + format(pow_a, ".3f").rjust(8) + " | " + format(fdp_p, ".4f").rjust(14)
              + format(pow_p, ".3f").rjust(8))
    print()

    if (~profiled).any():
        print("AUC among profiled windows only: "
              + format(roc_auc_score(y[profiled], -p[profiled]), ".4f")
              if y[profiled].any() and not y[profiled].all() else "AUC: n/a")


if __name__ == "__main__":
    main()
