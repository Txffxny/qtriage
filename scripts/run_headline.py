"""Headline result: which FDP are you reporting?

Benjamini-Hochberg controls the false discovery rate within a hypothesis
family. When the family is a time batch, two different quantities can be
computed from the same rejections:

  per-batch FDP : mean over batches of (false rejections / rejections).
                  This is what BH guarantees.

  pooled FDP    : (all false rejections) / (all rejections), across the day.
                  This is what an analyst working the queue experiences.

They diverge because attacks are concentrated in time. Most batches contain no
attack, so when such a batch fires - which it does at rate q, exactly as
promised - it contributes a false positive with no true positive to offset it.
Averaging weights quiet hours equally; pooling weights them by rejection count.

Stream-level control requires the family to be the whole stream, so m = H*W and
n_cal >= m/q means roughly 2/q days of clean baseline: 40 days at q=0.05.
CICIDS2017 provides one.
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
    """Return both FDP flavours plus recall for one day at one q."""
    reject = np.zeros(len(p), dtype=bool)
    batch_fdps = []
    fired = 0
    n_batch = 0
    empty_batch_fires = 0

    for b in np.unique(batch_ids):
        idx = np.flatnonzero(batch_ids == b)
        if len(idx) < MIN_BATCH_B:
            continue
        n_batch += 1
        rej, _ = benjamini_hochberg(p[idx], q=q)
        reject[idx] = rej
        batch_fdps.append(realized_fdp(rej, y[idx]))
        if rej.any():
            fired += 1
            if not y[idx].any():
                empty_batch_fires += 1

    tp = int((reject & y).sum())
    fp = int((reject & ~y).sum())
    return {
        "per_batch_fdp": float(np.mean(batch_fdps)) if batch_fdps else 0.0,
        "pooled_fdp": fp / (tp + fp) if (tp + fp) else 0.0,
        "power": realized_power(reject, y),
        "tp": tp, "fp": fp,
        "batches_fired": fired, "n_batches": n_batch,
        "fires_in_attack_free_batches": empty_batch_fires,
    }


def main():
    outdir = config.FIGURES / "exp04_headline"
    outdir.mkdir(parents=True, exist_ok=True)

    print("HEADLINE: per-batch vs pooled FDP")
    print("  window=" + config.BATCH + "  unit=" + config.AGGREGATION_UNIT)
    print()

    cal_agg = aggregate(load(config.CALIBRATION_FILE), verbose=False)
    X_cal = feats(cal_agg)
    tr, ca = split_calibration(len(X_cal))
    det = fit_detector(X_cal[tr], max_samples=min(256, len(tr)))
    cal_scores = anomaly_scores(det, X_cal[ca])
    floor = 1.0 / (len(cal_scores) + 1)

    days = [p.name.replace(".parquet", ".csv")
            for p in sorted(config.PROCESSED.glob("*.parquet"))
            if p.name.replace(".parquet", ".csv") != config.CALIBRATION_FILE]

    rows = []
    prevalence = {}
    for day in days:
        agg = aggregate(load(day), verbose=False)
        y = agg["_is_attack"].to_numpy().astype(bool)
        if not y.any():
            continue
        name = day.replace("-WorkingHours.csv", "")
        p = conformal_pvalues(cal_scores, anomaly_scores(det, feats(agg)))
        bid = pd.to_datetime(agg["_timestamp"]).dt.floor(config.BATCH).astype("int64").to_numpy()
        prevalence[name] = float(y.mean())

        for q in Q_GRID:
            r = evaluate_day(p, y, bid, q)
            r.update({"day": name, "q": q, "prevalence": float(y.mean())})
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "per_day_both_metrics.csv", index=False)

    # --- summary table ------------------------------------------------------
    print("  " + "q".rjust(6) + "per-batch FDP".rjust(16) + "pooled FDP".rjust(13)
          + "ratio".rjust(8) + "mean power".rjust(13))
    summary = []
    for q in Q_GRID:
        sub = df[df["q"] == q]
        pb, pl = sub["per_batch_fdp"].mean(), sub["pooled_fdp"].mean()
        ratio = pl / pb if pb > 0 else float("nan")
        summary.append({"q": q, "per_batch_fdp": pb, "per_batch_sd": sub["per_batch_fdp"].std(),
                        "pooled_fdp": pl, "pooled_sd": sub["pooled_fdp"].std(),
                        "ratio": ratio, "power": sub["power"].mean()})
        print("  " + format(q, ".2f").rjust(6) + format(pb, ".4f").rjust(16)
              + format(pl, ".4f").rjust(13)
              + (format(ratio, ".1f") + "x").rjust(8)
              + format(sub["power"].mean(), ".3f").rjust(13))

    sm = pd.DataFrame(summary)
    sm.to_csv(outdir / "summary.csv", index=False)

    # --- why they diverge ---------------------------------------------------
    print("\nWHY: fires in batches containing no attack at all (q=0.05)")
    sub = df[df["q"] == 0.05]
    for _, r in sub.iterrows():
        print("  " + r["day"].ljust(12) + "prevalence "
              + format(100 * r["prevalence"], "6.2f") + "%   fired "
              + str(int(r["batches_fired"])) + "/" + str(int(r["n_batches"]))
              + " batches, " + str(int(r["fires_in_attack_free_batches"]))
              + " of them attack-free")

    # --- baseline requirement ----------------------------------------------
    n_cal = len(ca)
    print("\nSTREAM-LEVEL FEASIBILITY (family = whole day)")
    print("  " + "q".rjust(6) + "m".rjust(8) + "n_cal needed".rjust(14)
          + "have".rjust(8) + "  benign days required")
    m_day = int(df.groupby("day").first()["n_batches"].mean() * 13)
    for q in [0.05, 0.10, 0.20]:
        need = int(np.ceil(m_day / q))
        days_needed = need / max(n_cal, 1)
        print("  " + format(q, ".2f").rjust(6) + format(m_day, ",").rjust(8)
              + format(need, ",").rjust(14) + format(n_cal, ",").rjust(8)
              + "   " + format(days_needed, ".0f"))

    with open(outdir / "run_meta.json", "w") as fh:
        json.dump({"experiment": "exp04_headline", "window": config.BATCH,
                   "unit": config.AGGREGATION_UNIT, "n_cal_windows": int(n_cal),
                   "p_floor": float(floor), "prevalence": prevalence,
                   "seed": config.RANDOM_SEED}, fh, indent=2)

    # --- the figure ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    lim = max(Q_GRID) * 1.08

    ax = axes[0]
    ax.plot([0, lim], [0, lim], "k--", lw=1.2, label="nominal (y = q)")
    ax.errorbar(sm["q"], sm["per_batch_fdp"], yerr=sm["per_batch_sd"],
                fmt="o-", color="tab:green", capsize=3, lw=2,
                label="per-batch FDP (what BH controls)")
    ax.errorbar(sm["q"], sm["pooled_fdp"], yerr=sm["pooled_sd"],
                fmt="s-", color="tab:red", capsize=3, lw=2,
                label="pooled FDP (what the analyst sees)")
    ax.set_xlabel("target FDR (q)")
    ax.set_ylabel("realised FDP")
    ax.set_title("Same rejections, two guarantees")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    sub = df[df["q"] == 0.05].sort_values("prevalence")
    ax.plot(100 * sub["prevalence"], sub["pooled_fdp"], "s-", color="tab:red",
            label="pooled")
    ax.plot(100 * sub["prevalence"], sub["per_batch_fdp"], "o-", color="tab:green",
            label="per-batch")
    ax.axhline(0.05, ls="--", c="k", lw=1, label="q = 0.05")
    for _, r in sub.iterrows():
        ax.annotate(r["day"][:3], (100 * r["prevalence"], r["pooled_fdp"]),
                    fontsize=7, xytext=(3, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("attack prevalence (% of host-windows, log scale)")
    ax.set_ylabel("realised FDP at q = 0.05")
    ax.set_title("The gap closes as attacks get common")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for day, g in df.groupby("day"):
        ax.plot(g["q"], g["power"], "o-", lw=1, ms=4, alpha=0.75, label=day)
    ax.set_xlabel("target FDR (q)")
    ax.set_ylabel("recall")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Recall by day")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.suptitle("Channel B (host-window, " + config.BATCH
                 + " batches) | baseline: " + config.CALIBRATION_FILE.replace(".csv", "")
                 + " | BH controls the green curve, not the red one", fontsize=10)
    fig.tight_layout()
    fig.savefig(outdir / "headline.png", dpi=150)
    print("\nSaved " + str(outdir / "headline.png"))


if __name__ == "__main__":
    main()
