"""End-to-end pipeline: conformal p-values, hourly BH, calibration curve.

Fits on half of the benign calibration day, calibrates on the other half, then
tests a target day in hourly batches. Each hour is its own hypothesis family,
which is both the operational unit a SOC works in and the statistically
workable one: a single family of ~360k hypotheses puts the k=1 BH threshold
below the conformal p-value floor, so nothing could ever be rejected.
"""
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src.detector import feature_matrix, split_calibration, fit_detector, anomaly_scores
from src.stats import conformal_pvalues, benjamini_hochberg, realized_fdp, realized_power

Q_GRID = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]


def load_cached(filename):
    path = config.PROCESSED / filename.replace(".csv", ".parquet")
    if not path.exists():
        sys.exit("Missing cache " + str(path) + ". Run scripts/build_cache.py first.")
    return pd.read_parquet(path)


def run_batched_bh(p, y, batch_ids, q):
    """Apply BH within each batch. Returns reject mask and per-batch records."""
    reject = np.zeros(len(p), dtype=bool)
    records = []

    for bid in np.unique(batch_ids):
        idx = np.flatnonzero(batch_ids == bid)
        if len(idx) < config.MIN_BATCH:
            records.append({"batch": bid, "m": len(idx), "skipped": True,
                            "n_reject": 0, "fdp": np.nan, "power": np.nan})
            continue

        rej_b, _ = benjamini_hochberg(p[idx], q=q)
        reject[idx] = rej_b
        records.append({
            "batch": bid,
            "m": len(idx),
            "skipped": False,
            "n_true": int(y[idx].sum()),
            "n_reject": int(rej_b.sum()),
            "fdp": realized_fdp(rej_b, y[idx]),
            "power": realized_power(rej_b, y[idx]),
        })

    return reject, pd.DataFrame(records)


def main():
    print("Calibration day: " + config.CALIBRATION_FILE)
    print("Test day:        " + config.TEST_FILE)
    print("Attempted policy: " + config.ATTEMPTED_POLICY
          + " | imputed flag as feature: " + str(config.INCLUDE_IMPUTED_FLAG))
    print()

    # --- calibration day: split, fit, calibrate ---------------------------
    cal_df = load_cached(config.CALIBRATION_FILE)
    if cal_df["_is_attack"].any():
        sys.exit("Calibration day contains attacks. It must be benign-only.")

    X_cal_all, cols = feature_matrix(cal_df)
    tr_idx, ca_idx = split_calibration(len(X_cal_all))
    print("Calibration day: " + format(len(X_cal_all), ",") + " benign flows, "
          + str(len(cols)) + " features")
    print("  train " + format(len(tr_idx), ",")
          + " | calibrate " + format(len(ca_idx), ","))

    det = fit_detector(X_cal_all[tr_idx])
    cal_scores = anomaly_scores(det, X_cal_all[ca_idx])

    floor = 1.0 / (len(cal_scores) + 1)
    print("  conformal p-value floor: " + format(floor, ".2e"))
    print()

    # --- test day ----------------------------------------------------------
    test_df = load_cached(config.TEST_FILE)
    X_test, _ = feature_matrix(test_df)
    y = test_df["_is_attack"].to_numpy().astype(bool)
    ts = pd.to_datetime(test_df["_timestamp"])
    batch_ids = ts.dt.floor(config.BATCH).astype("int64").to_numpy()

    print("Test day: " + format(len(X_test), ",") + " flows, "
          + format(int(y.sum()), ",") + " attacks ("
          + str(round(100 * y.mean(), 3)) + "%)")

    test_scores = anomaly_scores(det, X_test)
    p = conformal_pvalues(cal_scores, test_scores)

    at_floor = int((p <= floor + 1e-12).sum())
    print("  flows at the p-value floor: " + format(at_floor, ",")
          + " (" + format(int(y[p <= floor + 1e-12].sum()), ",") + " are attacks)")

    n_batches = len(np.unique(batch_ids))
    usable = sum(1 for b in np.unique(batch_ids)
                 if (batch_ids == b).sum() >= config.MIN_BATCH)
    print("  " + str(n_batches) + " batches, " + str(usable)
          + " above MIN_BATCH=" + format(config.MIN_BATCH, ","))
    print()

    # --- sweep q -----------------------------------------------------------
    print("q sweep (BH applied within each hourly batch)")
    print("  " + "q".rjust(6) + "alerts".rjust(10) + "pooled FDP".rjust(13)
          + "mean FDP".rjust(11) + "power".rjust(9) + "  vol. reduction")

    rows = []
    for q in Q_GRID:
        reject, per_batch = run_batched_bh(p, y, batch_ids, q)
        pooled = realized_fdp(reject, y)
        power = realized_power(reject, y)
        mean_fdp = per_batch.loc[~per_batch["skipped"], "fdp"].mean()
        n_alerts = int(reject.sum())
        reduction = 100 * (1 - n_alerts / len(p))

        rows.append({"q": q, "alerts": n_alerts, "pooled_fdp": pooled,
                     "mean_fdp": mean_fdp, "power": power})
        print("  " + format(q, ".2f").rjust(6)
              + format(n_alerts, ",").rjust(10)
              + format(pooled, ".4f").rjust(13)
              + format(mean_fdp, ".4f").rjust(11)
              + format(power, ".3f").rjust(9)
              + "   " + format(reduction, ".2f") + "%")

    results = pd.DataFrame(rows)
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    results.to_csv(config.FIGURES / "q_sweep.csv", index=False)

    # --- the money chart ---------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    lim = max(Q_GRID) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="nominal (y = q)")
    ax.plot(results["q"], results["pooled_fdp"], "o-", label="pooled FDP")
    ax.plot(results["q"], results["mean_fdp"], "s--", alpha=0.7, label="mean per-batch FDP")
    ax.set_xlabel("target FDR (q)")
    ax.set_ylabel("realised false discovery proportion")
    ax.set_title("FDR control holds on real traffic")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(results["pooled_fdp"], results["power"], "o-")
    for _, r in results.iterrows():
        ax.annotate("q=" + format(r["q"], ".2f"),
                    (r["pooled_fdp"], r["power"]),
                    fontsize=7, xytext=(4, -8), textcoords="offset points")
    ax.set_xlabel("realised false discovery proportion")
    ax.set_ylabel("recall (fraction of attacks caught)")
    ax.set_title("The dial a SOC actually sets")
    ax.grid(alpha=0.3)

    fig.suptitle(
        config.TEST_FILE.replace(".csv", "") + "  |  calibrated on "
        + config.CALIBRATION_FILE.replace(".csv", "")
        + "  |  policy=" + config.ATTEMPTED_POLICY,
        fontsize=9,
    )
    fig.tight_layout()
    out = config.FIGURES / ("calibration_" + config.TEST_FILE.replace(".csv", "") + ".png")
    fig.savefig(out, dpi=150)
    print("\nSaved " + str(out))
    print("Saved " + str(config.FIGURES / "q_sweep.csv"))


if __name__ == "__main__":
    main()
