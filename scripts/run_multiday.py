"""Channel B across every attack day: FDP as a distribution, not a point.

FDR is an expectation. A single day's pooled FDP over ~7 rejections estimates
nothing - the realised proportion is either 0 or 1 in most batches, and the
sampling error swamps the signal. The synthetic validation averaged 200 trials
per q for exactly this reason; that discipline was dropped on real data.

This evaluates Tuesday through Friday against the same Monday baseline, giving
independent realisations and enough malicious units to estimate a rate. Reports
mean FDP with spread, and the batch-firing rate under the null.
"""
import sys
import pathlib
import json

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from src import config
from src.aggregate import aggregate
from src.detector import split_calibration, fit_detector, anomaly_scores
from src.stats import conformal_pvalues, benjamini_hochberg, realized_fdp, realized_power

Q_GRID = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
MIN_BATCH_B = 5


def load(f):
    path = config.PROCESSED / f.replace(".csv", ".parquet")
    if not path.exists():
        sys.exit("Missing cache " + str(path))
    return pd.read_parquet(path)


def feats(agg):
    cols = [c for c in agg.columns if not c.startswith("_")]
    X = agg[cols].to_numpy(dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), cols


def batched_bh(p, batch_ids, q):
    """BH within each batch. Returns reject mask and per-batch FDP records."""
    reject = np.zeros(len(p), dtype=bool)
    fired = 0
    n_batch = 0
    for b in np.unique(batch_ids):
        idx = np.flatnonzero(batch_ids == b)
        if len(idx) < MIN_BATCH_B:
            continue
        n_batch += 1
        rej, _ = benjamini_hochberg(p[idx], q=q)
        reject[idx] = rej
        if rej.any():
            fired += 1
    return reject, fired, n_batch


def main():
    outdir = config.FIGURES / "exp03_multiday"
    outdir.mkdir(parents=True, exist_ok=True)

    print("MULTI-DAY EVALUATION - channel B, Monday baseline")
    print("  window=" + config.BATCH + "  unit=" + config.AGGREGATION_UNIT)
    print()

    # --- one detector, fitted once on Monday -------------------------------
    cal_agg = aggregate(load(config.CALIBRATION_FILE), verbose=False)
    X_cal, cols = feats(cal_agg)
    tr, ca = split_calibration(len(X_cal))
    det = fit_detector(X_cal[tr], max_samples=min(256, len(tr)))
    cal_scores = anomaly_scores(det, X_cal[ca])
    floor = 1.0 / (len(cal_scores) + 1)

    print("Baseline: " + format(len(X_cal), ",") + " host-windows from "
          + str(cal_agg["_src_ip"].nunique()) + " hosts")
    print("  fit " + format(len(tr), ",") + " | calibrate " + format(len(ca), ",")
          + " | p-floor " + format(floor, ".2e"))
    print()

    days = [p.name.replace(".parquet", ".csv")
            for p in sorted(config.PROCESSED.glob("*.parquet"))
            if p.name.replace(".parquet", ".csv") != config.CALIBRATION_FILE]

    records = []
    per_day_summary = []

    for day in days:
        agg = aggregate(load(day), verbose=False)
        X, _ = feats(agg)
        y = agg["_is_attack"].to_numpy().astype(bool)
        if not y.any():
            continue
        p = conformal_pvalues(cal_scores, anomaly_scores(det, X))
        bid = pd.to_datetime(agg["_timestamp"]).dt.floor(config.BATCH).astype("int64").to_numpy()
        auc = roc_auc_score(y, -p) if not y.all() else float("nan")

        name = day.replace("-WorkingHours.csv", "")
        print(name + ": " + format(len(X), ",") + " host-windows, "
              + format(int(y.sum()), ",") + " malicious ("
              + str(round(100 * y.mean(), 2)) + "%)  AUC=" + format(auc, ".4f"))
        per_day_summary.append({"day": name, "n_windows": len(X),
                                "n_malicious": int(y.sum()), "auc": float(auc)})

        for q in Q_GRID:
            reject, fired, n_batch = batched_bh(p, bid, q)
            records.append({
                "day": name, "q": q,
                "n_alerts": int(reject.sum()),
                "tp": int((reject & y).sum()),
                "fp": int((reject & ~y).sum()),
                "fdp": realized_fdp(reject, y),
                "power": realized_power(reject, y),
                "batches_fired": fired, "n_batches": n_batch,
                "fire_rate": fired / max(n_batch, 1),
            })

    df = pd.DataFrame(records)
    df.to_csv(outdir / "per_day.csv", index=False)
    pd.DataFrame(per_day_summary).to_csv(outdir / "day_summary.csv", index=False)

    # --- aggregate across days ---------------------------------------------
    print("\nACROSS DAYS (n=" + str(df["day"].nunique()) + " days)")
    print("  " + "q".rjust(6) + "mean FDP".rjust(11) + "sd".rjust(8)
          + "min".rjust(8) + "max".rjust(8) + "mean power".rjust(13)
          + "total alerts".rjust(14))
    summary = []
    for q in Q_GRID:
        sub = df[df["q"] == q]
        row = {"q": q, "mean_fdp": sub["fdp"].mean(), "sd_fdp": sub["fdp"].std(),
               "min_fdp": sub["fdp"].min(), "max_fdp": sub["fdp"].max(),
               "mean_power": sub["power"].mean(),
               "total_alerts": int(sub["n_alerts"].sum()),
               "total_tp": int(sub["tp"].sum()), "total_fp": int(sub["fp"].sum())}
        summary.append(row)
        print("  " + format(q, ".2f").rjust(6)
              + format(row["mean_fdp"], ".4f").rjust(11)
              + format(row["sd_fdp"], ".4f").rjust(8)
              + format(row["min_fdp"], ".3f").rjust(8)
              + format(row["max_fdp"], ".3f").rjust(8)
              + format(row["mean_power"], ".3f").rjust(13)
              + format(row["total_alerts"], ",").rjust(14))

    # Pooled over all days: many more rejections, so a usable estimate.
    print("\nPOOLED over all days and batches")
    print("  " + "q".rjust(6) + "TP".rjust(8) + "FP".rjust(8)
          + "pooled FDP".rjust(13) + "  (n rejections)")
    for row in summary:
        tot = row["total_tp"] + row["total_fp"]
        pooled = row["total_fp"] / tot if tot else 0.0
        print("  " + format(row["q"], ".2f").rjust(6)
              + format(row["total_tp"], ",").rjust(8)
              + format(row["total_fp"], ",").rjust(8)
              + format(pooled, ".4f").rjust(13)
              + "   " + format(tot, ","))

    sm = pd.DataFrame(summary)
    sm.to_csv(outdir / "summary.csv", index=False)

    with open(outdir / "run_meta.json", "w") as fh:
        json.dump({"experiment": "exp03_multiday", "window": config.BATCH,
                   "unit": config.AGGREGATION_UNIT,
                   "attempted_policy": config.ATTEMPTED_POLICY,
                   "calibration_file": config.CALIBRATION_FILE,
                   "days": [d["day"] for d in per_day_summary],
                   "n_cal_windows": int(len(ca)), "p_floor": float(floor),
                   "seed": config.RANDOM_SEED}, fh, indent=2)

    # --- plot ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    lim = max(Q_GRID) * 1.05

    ax = axes[0]
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="nominal (y = q)")
    for day, g in df.groupby("day"):
        ax.plot(g["q"], g["fdp"], "o-", alpha=0.45, lw=1, ms=4, label=day)
    ax.errorbar(sm["q"], sm["mean_fdp"], yerr=sm["sd_fdp"], fmt="s-",
                color="black", lw=2, capsize=3, label="mean +/- sd")
    ax.set_xlabel("target FDR (q)")
    ax.set_ylabel("realised FDP")
    ax.set_title("FDR is an expectation: spread across days")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for day, g in df.groupby("day"):
        ax.plot(g["q"], g["power"], "o-", alpha=0.6, lw=1, ms=4, label=day)
    ax.set_xlabel("target FDR (q)")
    ax.set_ylabel("recall")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Recall by day")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle("Channel B, host-window, Monday baseline, window="
                 + config.BATCH, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "multiday.png", dpi=150)
    print("\nSaved " + str(outdir))


if __name__ == "__main__":
    main()
