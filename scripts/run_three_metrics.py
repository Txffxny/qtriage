"""Three ways to report FDP, and which one an analyst actually means.

Same rejections, three denominators:

  per-batch FDP (all)     mean over every batch of (FP / rejections), counting
                          silent batches as 0. This is the usual reading of
                          "BH controls FDR", but it is flattering: a batch that
                          makes no discoveries has a false discovery proportion
                          of 0 by definition, which is true and empty.

  per-batch FDP (firing)  mean over batches that actually raised something.
                          "When this system alerts, how often is it wrong?"
                          This is the operational question.

  pooled FDP              all FP / all rejections across the day. What the
                          analyst working the whole queue experiences.

BH's guarantee is E[FDP] <= q where the expectation runs over all batches,
silent ones included. The conditional version carries no such guarantee - it is
reported here because it is what a dashboard would have to show.
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
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def evaluate_day(p, y, batch_ids, q):
    reject = np.zeros(len(p), dtype=bool)
    all_fdps = []
    firing_fdps = []
    n_batch = 0

    for b in np.unique(batch_ids):
        idx = np.flatnonzero(batch_ids == b)
        if len(idx) < MIN_BATCH_B:
            continue
        n_batch += 1
        rej, _ = benjamini_hochberg(p[idx], q=q)
        reject[idx] = rej
        fdp_b = realized_fdp(rej, y[idx])
        all_fdps.append(fdp_b)
        if rej.any():
            firing_fdps.append(fdp_b)

    tp = int((reject & y).sum())
    fp = int((reject & ~y).sum())
    return {
        "fdp_all_batches": float(np.mean(all_fdps)) if all_fdps else 0.0,
        "fdp_firing_batches": float(np.mean(firing_fdps)) if firing_fdps else np.nan,
        "fdp_pooled": fp / (tp + fp) if (tp + fp) else 0.0,
        "power": realized_power(reject, y),
        "tp": tp, "fp": fp,
        "n_firing": len(firing_fdps), "n_batches": n_batch,
        "fire_rate": len(firing_fdps) / max(n_batch, 1),
        "mean_rejects_per_firing": (tp + fp) / max(len(firing_fdps), 1),
    }


def main():
    outdir = config.FIGURES / "exp05_three_metrics"
    outdir.mkdir(parents=True, exist_ok=True)

    print("THREE WAYS TO REPORT FDP")
    print("  window=" + config.BATCH + "  unit=" + config.AGGREGATION_UNIT)
    print()

    cal_agg = aggregate(load(config.CALIBRATION_FILE), verbose=False)
    X_cal = feats(cal_agg)
    tr, ca = split_calibration(len(X_cal))
    det = fit_detector(X_cal[tr], max_samples=min(256, len(tr)))
    cal_scores = anomaly_scores(det, X_cal[ca])

    days = [p.name.replace(".parquet", ".csv")
            for p in sorted(config.PROCESSED.glob("*.parquet"))
            if p.name.replace(".parquet", ".csv") != config.CALIBRATION_FILE]

    rows = []
    for day in days:
        agg = aggregate(load(day), verbose=False)
        y = agg["_is_attack"].to_numpy().astype(bool)
        if not y.any():
            continue
        name = day.replace("-WorkingHours.csv", "")
        p = conformal_pvalues(cal_scores, anomaly_scores(det, feats(agg)))
        bid = pd.to_datetime(agg["_timestamp"]).dt.floor(config.BATCH).astype("int64").to_numpy()
        for q in Q_GRID:
            r = evaluate_day(p, y, bid, q)
            r.update({"day": name, "q": q, "prevalence": float(y.mean())})
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "three_metrics.csv", index=False)

    print("  " + "q".rjust(6) + "all batches".rjust(14) + "firing only".rjust(14)
          + "pooled".rjust(10) + "fire rate".rjust(12)
          + "rej/fire".rjust(11) + "power".rjust(9))
    summary = []
    for q in Q_GRID:
        s = df[df["q"] == q]
        row = {"q": q,
               "fdp_all": s["fdp_all_batches"].mean(),
               "fdp_all_sd": s["fdp_all_batches"].std(),
               "fdp_firing": s["fdp_firing_batches"].mean(),
               "fdp_firing_sd": s["fdp_firing_batches"].std(),
               "fdp_pooled": s["fdp_pooled"].mean(),
               "fdp_pooled_sd": s["fdp_pooled"].std(),
               "fire_rate": s["fire_rate"].mean(),
               "rej_per_fire": s["mean_rejects_per_firing"].mean(),
               "power": s["power"].mean()}
        summary.append(row)
        firing_str = ("     n/a" if np.isnan(row["fdp_firing"])
                      else format(row["fdp_firing"], ".4f"))
        print("  " + format(q, ".2f").rjust(6)
              + format(row["fdp_all"], ".4f").rjust(14)
              + firing_str.rjust(14)
              + format(row["fdp_pooled"], ".4f").rjust(10)
              + format(row["fire_rate"], ".3f").rjust(12)
              + format(row["rej_per_fire"], ".2f").rjust(11)
              + format(row["power"], ".3f").rjust(9))

    sm = pd.DataFrame(summary)
    sm.to_csv(outdir / "summary.csv", index=False)

    print("\nHOW THE DENOMINATORS DIFFER (q=0.05)")
    s = df[df["q"] == 0.05]
    for _, r in s.iterrows():
        print("  " + r["day"].ljust(11)
              + str(int(r["n_firing"])) + "/" + str(int(r["n_batches"]))
              + " batches fired, " + str(int(r["tp"] + r["fp"]))
              + " rejections (" + str(int(r["tp"])) + " TP, "
              + str(int(r["fp"])) + " FP)")

    with open(outdir / "run_meta.json", "w") as fh:
        json.dump({"experiment": "exp05_three_metrics", "window": config.BATCH,
                   "unit": config.AGGREGATION_UNIT,
                   "n_cal_windows": int(len(ca)), "seed": config.RANDOM_SEED}, fh, indent=2)

    # --- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    lim = max(Q_GRID) * 1.08

    ax = axes[0]
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, label="nominal (y = q)")
    ax.errorbar(sm["q"], sm["fdp_all"], yerr=sm["fdp_all_sd"], fmt="o-",
                color="tab:green", capsize=3, lw=2,
                label="all batches (what BH bounds)")
    ax.errorbar(sm["q"], sm["fdp_firing"], yerr=sm["fdp_firing_sd"], fmt="^-",
                color="tab:orange", capsize=3, lw=2,
                label="firing batches only (operational)")
    ax.errorbar(sm["q"], sm["fdp_pooled"], yerr=sm["fdp_pooled_sd"], fmt="s-",
                color="tab:red", capsize=3, lw=2,
                label="pooled over the day (the queue)")
    ax.set_xlabel("target FDR (q)")
    ax.set_ylabel("realised FDP")
    ax.set_title("One procedure, three answers")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(sm["q"], sm["fire_rate"], "o-", color="tab:blue", label="fraction of batches firing")
    ax.plot(sm["q"], sm["power"], "s-", color="tab:purple", label="recall")
    ax.set_xlabel("target FDR (q)")
    ax.set_ylabel("fraction")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Most batches stay silent, and that is the point")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Channel B | " + config.BATCH + " batches | baseline "
                 + config.CALIBRATION_FILE.replace(".csv", "")
                 + " | silent batches have FDP 0 by definition", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "three_metrics.png", dpi=150)
    print("\nSaved " + str(outdir / "three_metrics.png"))


if __name__ == "__main__":
    main()
