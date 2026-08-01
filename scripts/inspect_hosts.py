"""Measure host-hour cardinality and check whether BH can fire at that scale.

Aggregation trades hypothesis count for signal. Fewer hypotheses relax the BH
threshold, but a smaller calibration set raises the conformal p-value floor.
This script reports both so the batching design is chosen from the data rather
than assumed.
"""
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config
from src.aggregate import aggregate


def load_cached(filename):
    path = config.PROCESSED / filename.replace(".csv", ".parquet")
    if not path.exists():
        sys.exit("Missing cache " + str(path) + ". Run scripts/build_cache.py first.")
    return pd.read_parquet(path)


def feasibility(n_cal_hosts, m_per_batch, q=0.05):
    """Can the smallest achievable p-value clear the k=1 BH threshold?"""
    floor = 1.0 / (n_cal_hosts + 1)
    thresh = q / max(m_per_batch, 1)
    k_needed = int(np.ceil(floor / thresh)) if thresh > 0 else 0
    return floor, thresh, k_needed


def main():
    print("Aggregation unit: " + config.AGGREGATION_UNIT
          + " | window: " + config.BATCH
          + " | min flows/host-hour: " + str(config.MIN_FLOWS_PER_HOST_HOUR))
    print()

    # --- calibration day ---------------------------------------------------
    print("CALIBRATION: " + config.CALIBRATION_FILE)
    cal = aggregate(load_cached(config.CALIBRATION_FILE))
    n_cal_total = len(cal)
    n_cal_half = n_cal_total // 2
    print("  " + format(n_cal_total, ",") + " host-hours -> "
          + format(n_cal_half, ",") + " for calibration after the 50/50 split")
    print("  flows per host-hour: median "
          + format(int(cal["n_flows"].median()), ",")
          + ", max " + format(int(cal["n_flows"].max()), ","))
    print()

    # --- test day ----------------------------------------------------------
    print("TEST: " + config.TEST_FILE)
    test = aggregate(load_cached(config.TEST_FILE))
    print("  " + format(len(test), ",") + " host-hours")

    hourly = test["_timestamp"].dt.floor(config.BATCH).value_counts().sort_index()
    print("  " + str(len(hourly)) + " batches, sizes "
          + format(int(hourly.min()), ",") + " - " + format(int(hourly.max()), ",")
          + " (median " + format(int(hourly.median()), ",") + ")")
    print()

    # --- how malicious is a malicious host-hour? ---------------------------
    atk = test[test["_is_attack"]]
    print("MALICIOUS HOST-HOURS: " + format(len(atk), ","))
    if len(atk):
        print("  attack-flow fraction within them:")
        for pct in [10, 25, 50, 75, 90]:
            print("    p" + str(pct).ljust(3) + " "
                  + format(np.percentile(atk["_attack_frac"], pct), ".3f"))
        contaminated = int((atk["_attack_frac"] < 0.05).sum())
        print("  " + format(contaminated, ",") + " are <5% attack flows "
              + "(labelled malicious on very little evidence)")
    print()

    # --- can BH fire at this scale? ----------------------------------------
    print("FEASIBILITY at q=0.05")
    print("  " + "regime".ljust(26) + "m".rjust(8) + "floor".rjust(12)
          + "k=1 thresh".rjust(13) + "  flows at floor needed")
    for name, m in [("per hourly batch (median)", int(hourly.median())),
                    ("largest hourly batch", int(hourly.max())),
                    ("whole day, one family", len(test))]:
        floor, thresh, k = feasibility(n_cal_half, m)
        verdict = "OK" if k <= 3 else ("tight" if k <= 10 else "BLOCKED")
        print("  " + name.ljust(26) + format(m, ",").rjust(8)
              + format(floor, ".2e").rjust(12) + format(thresh, ".2e").rjust(13)
              + "   " + str(k) + "  " + verdict)
    print()
    print("Compare flow level: floor 5.4e-06, threshold 1.3e-06, needed k=4")
    print("(and no attack flow ever came within four orders of magnitude)")


if __name__ == "__main__":
    main()
