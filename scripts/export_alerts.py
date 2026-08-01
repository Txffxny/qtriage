"""Export per-alert records so the dashboard reads results rather than recomputing.

Writes one row per host-window with its conformal p-value, BH q-value, batch,
label and features. This is also the substrate a constrained-agent layer would
consume: every decision can then cite the evidence that produced it.
"""
import sys
import pathlib
import json

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import config
from src.aggregate import aggregate
from src.detector import split_calibration, fit_detector, anomaly_scores
from src.stats import conformal_pvalues, benjamini_hochberg

MIN_BATCH_B = 5
Q_CERTIFY = [0.01, 0.05, 0.10, 0.20, 0.30]


def load(f):
    path = config.PROCESSED / f.replace(".csv", ".parquet")
    if not path.exists():
        sys.exit("Missing cache " + str(path) + ". Run scripts/build_cache.py first.")
    return pd.read_parquet(path)


def feats(agg):
    cols = [c for c in agg.columns if not c.startswith("_")]
    X = agg[cols].to_numpy(dtype=np.float64)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), cols


def main():
    out = config.PROCESSED / "alerts.parquet"

    cal_agg = aggregate(load(config.CALIBRATION_FILE), verbose=False)
    X_cal, cols = feats(cal_agg)
    tr, ca = split_calibration(len(X_cal))
    det = fit_detector(X_cal[tr], max_samples=min(256, len(tr)))
    cal_scores = anomaly_scores(det, X_cal[ca])
    floor = 1.0 / (len(cal_scores) + 1)
    known_hosts = set(cal_agg["_src_ip"].unique())

    print("Baseline: " + format(len(ca), ",") + " calibration windows, floor "
          + format(floor, ".2e"))

    days = [p.name.replace(".parquet", ".csv")
            for p in sorted(config.PROCESSED.glob("*.parquet"))
            if p.name.replace(".parquet", ".csv") != config.CALIBRATION_FILE
            and p.name != "alerts.parquet"]

    frames = []
    for day in days:
        agg = aggregate(load(day), verbose=False)
        X, _ = feats(agg)
        p = conformal_pvalues(cal_scores, anomaly_scores(det, X))
        ts = pd.to_datetime(agg["_timestamp"])
        batch = ts.dt.floor(config.BATCH)
        bid = batch.astype("int64").to_numpy()

        rec = pd.DataFrame({
            "day": day.replace("-WorkingHours.csv", ""),
            "host": agg["_src_ip"].to_numpy(),
            "window": batch.to_numpy(),
            "p_value": p,
            "at_floor": p <= floor + 1e-12,
            "profiled_host": [h in known_hosts for h in agg["_src_ip"]],
            "is_attack": agg["_is_attack"].to_numpy().astype(bool),
            "n_flows": agg["n_flows"].to_numpy(),
            "n_attack_flows": agg["_n_attack_flows"].to_numpy(),
            "dst_port_entropy": agg["dst_port_entropy"].to_numpy(),
            "top_port_share": agg["top_port_share"].to_numpy(),
            "fwd_bytes_cv": agg["fwd_bytes_cv"].to_numpy(),
            "duration_cv": agg["duration_cv"].to_numpy(),
            "rst_rate": agg["rst_rate"].to_numpy(),
            "n_distinct_dst_port": agg["n_distinct_dst_port"].to_numpy(),
            "batch_size": [int((bid == b).sum()) for b in bid],
        })

        # BH q-value and certification status at each q, computed within batch
        rec["q_value"] = np.nan
        for b in np.unique(bid):
            idx = np.flatnonzero(bid == b)
            if len(idx) < MIN_BATCH_B:
                continue
            _, qv = benjamini_hochberg(p[idx], q=0.05)
            rec.loc[idx, "q_value"] = qv

        for q in Q_CERTIFY:
            col = np.zeros(len(rec), dtype=bool)
            for b in np.unique(bid):
                idx = np.flatnonzero(bid == b)
                if len(idx) < MIN_BATCH_B:
                    continue
                rej, _ = benjamini_hochberg(p[idx], q=q)
                col[idx] = rej
            rec["certified_q" + str(q).replace("0.", "")] = col

        rec["evidence_rank"] = rec["p_value"].rank(method="min").astype(int)
        frames.append(rec)
        print("  " + day.replace("-WorkingHours.csv", "").ljust(11)
              + format(len(rec), ",").rjust(7) + " host-windows, "
              + format(int(rec["is_attack"].sum()), ",") + " malicious")

    alerts = pd.concat(frames, ignore_index=True)
    alerts.to_parquet(out, index=False)

    meta = {"floor": float(floor), "n_cal": int(len(ca)),
            "calibration_file": config.CALIBRATION_FILE,
            "window": config.BATCH, "unit": config.AGGREGATION_UNIT,
            "min_batch": MIN_BATCH_B, "q_levels": Q_CERTIFY,
            "attempted_policy": config.ATTEMPTED_POLICY,
            "n_cal_hosts": int(len(known_hosts))}
    with open(config.PROCESSED / "alerts_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print("\nSaved " + str(out) + "  (" + format(len(alerts), ",") + " rows)")


if __name__ == "__main__":
    main()
