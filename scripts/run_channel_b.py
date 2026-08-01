"""Channel B: host-window behavioural detection under FDR control.

Flow-level rarity (channel A) reached AUC 0.80 yet zero power: brute-force
flows are individually unremarkable, so no flow ever landed in the extreme tail
of the benign distribution. Aggregating to (host, window) makes the pattern -
many near-identical connections to one port - the unit of analysis.

Feasibility rule derived from that failure: with H hosts and W windows in the
calibration period, n_cal = H*W/2 and m = H, so BH can reject a single unit iff
W >= 2/q. H cancels; network size is irrelevant, window count is everything.
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

# Channel A's MIN_BATCH of 1000 flows would skip every host-window batch.
MIN_BATCH_B = 5


def load_cached(filename):
    path = config.PROCESSED / filename.replace(".csv", ".parquet")
    if not path.exists():
        sys.exit("Missing cache " + str(path) + ". Run scripts/build_cache.py first.")
    return pd.read_parquet(path)


def host_features(agg):
    """Model matrix from aggregated host-windows: drop '_'-prefixed metadata."""
    cols = [c for c in agg.columns if not c.startswith("_")]
    X = agg[cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.astype(np.float32), cols


def run_batched_bh(p, y, batch_ids, q, min_batch=MIN_BATCH_B):
    reject = np.zeros(len(p), dtype=bool)
    records = []
    for bid in np.unique(batch_ids):
        idx = np.flatnonzero(batch_ids == bid)
        if len(idx) < min_batch:
            records.append({"batch": bid, "m": len(idx), "skipped": True,
                            "n_reject": 0, "fdp": np.nan, "power": np.nan})
            continue
        rej_b, _ = benjamini_hochberg(p[idx], q=q)
        reject[idx] = rej_b
        records.append({"batch": bid, "m": len(idx), "skipped": False,
                        "n_true": int(y[idx].sum()), "n_reject": int(rej_b.sum()),
                        "fdp": realized_fdp(rej_b, y[idx]),
                        "power": realized_power(rej_b, y[idx])})
    return reject, pd.DataFrame(records)


def main():
    outdir = config.FIGURES / "exp02_host_window"
    outdir.mkdir(parents=True, exist_ok=True)

    print("CHANNEL B - host-window behavioural detection")
    print("  window=" + config.BATCH + "  unit=" + config.AGGREGATION_UNIT
          + "  min_flows=" + str(config.MIN_FLOWS_PER_HOST_HOUR))
    print()

    # --- calibration -------------------------------------------------------
    cal_agg = aggregate(load_cached(config.CALIBRATION_FILE), verbose=False)
    if cal_agg["_is_attack"].any():
        sys.exit("Calibration day contains attacks; it must be benign-only.")

    X_cal, cols = host_features(cal_agg)
    tr, ca = split_calibration(len(X_cal))
    det = fit_detector(X_cal[tr], max_samples=min(256, len(tr)))
    cal_scores = anomaly_scores(det, X_cal[ca])

    floor = 1.0 / (len(cal_scores) + 1)
    n_hosts_cal = cal_agg["_src_ip"].nunique()
    print("Calibration: " + format(len(X_cal), ",") + " host-windows from "
          + str(n_hosts_cal) + " distinct hosts, " + str(len(cols)) + " features")
    print("  train " + format(len(tr), ",") + " | calibrate " + format(len(ca), ","))
    print("  p-value floor: " + format(floor, ".2e"))
    print()

    # --- test --------------------------------------------------------------
    test_agg = aggregate(load_cached(config.TEST_FILE), verbose=False)
    X_test, _ = host_features(test_agg)
    y = test_agg["_is_attack"].to_numpy().astype(bool)
    ts = pd.to_datetime(test_agg["_timestamp"])
    batch_ids = ts.dt.floor(config.BATCH).astype("int64").to_numpy()

    scores = anomaly_scores(det, X_test)
    p = conformal_pvalues(cal_scores, scores)
    auc = roc_auc_score(y, scores) if y.any() and not y.all() else float("nan")

    print("Test: " + format(len(X_test), ",") + " host-windows from "
          + str(test_agg["_src_ip"].nunique()) + " hosts, "
          + format(int(y.sum()), ",") + " malicious ("
          + str(round(100 * y.mean(), 3)) + "%)")
    print("  AUC = " + format(auc, ".4f"))
    print("  malicious at the p-value floor: "
          + format(int((p[y] <= floor + 1e-12).sum()), ",") + " / " + str(int(y.sum()))
          + "   (benign at floor: " + format(int((p[~y] <= floor + 1e-12).sum()), ",") + ")")
    print("  median p: malicious " + format(float(np.median(p[y])), ".3e")
          + " | benign " + format(float(np.median(p[~y])), ".3e"))
    print()

    # --- q sweep -----------------------------------------------------------
    print("q sweep (BH within each " + config.BATCH + " batch)")
    print("  " + "q".rjust(6) + "alerts".rjust(9) + "TP".rjust(6) + "FP".rjust(6)
          + "pooled FDP".rjust(13) + "power".rjust(9))

    rows = []
    for q in Q_GRID:
        reject, per_batch = run_batched_bh(p, y, batch_ids, q)
        pooled = realized_fdp(reject, y)
        power = realized_power(reject, y)
        tp = int((reject & y).sum())
        fp = int((reject & ~y).sum())
        mean_fdp = per_batch.loc[~per_batch["skipped"], "fdp"].mean()
        rows.append({"q": q, "alerts": int(reject.sum()), "tp": tp, "fp": fp,
                     "pooled_fdp": pooled, "mean_fdp": mean_fdp, "power": power})
        print("  " + format(q, ".2f").rjust(6) + format(int(reject.sum()), ",").rjust(9)
              + str(tp).rjust(6) + str(fp).rjust(6)
              + format(pooled, ".4f").rjust(13) + format(power, ".3f").rjust(9))

    results = pd.DataFrame(rows)
    results.to_csv(outdir / "q_sweep.csv", index=False)

    # --- provenance --------------------------------------------------------
    meta = {"experiment": "exp02_host_window", "window": config.BATCH,
            "unit": config.AGGREGATION_UNIT,
            "min_flows_per_window": config.MIN_FLOWS_PER_HOST_HOUR,
            "attempted_policy": config.ATTEMPTED_POLICY,
            "calibration_file": config.CALIBRATION_FILE,
            "test_file": config.TEST_FILE,
            "n_cal_windows": int(len(ca)), "n_cal_hosts": int(n_hosts_cal),
            "p_floor": float(floor), "auc": float(auc), "seed": config.RANDOM_SEED}
    with open(outdir / "run_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    # --- plot --------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    lim = max(Q_GRID) * 1.05
    ax = axes[0]
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="nominal (y = q)")
    ax.plot(results["q"], results["pooled_fdp"], "o-", label="pooled FDP")
    ax.plot(results["q"], results["mean_fdp"], "s--", alpha=0.7, label="mean per-batch")
    ax.set_xlabel("target FDR (q)")
    ax.set_ylabel("realised FDP")
    ax.set_title("Does the guarantee hold on host-windows?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(results["q"], results["power"], "o-", color="tab:green")
    ax.set_xlabel("target FDR (q)")
    ax.set_ylabel("recall")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Recall vs tolerance (n=" + str(int(y.sum())) + " malicious)")
    ax.grid(alpha=0.3)

    fig.suptitle("Channel B | " + config.TEST_FILE.replace(".csv", "")
                 + " | calibrated on " + config.CALIBRATION_FILE.replace(".csv", "")
                 + " | window=" + config.BATCH, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "calibration.png", dpi=150)
    print("\nSaved " + str(outdir))


if __name__ == "__main__":
    main()
