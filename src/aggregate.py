"""Channel B: aggregate flows into host-hour behavioural profiles.

Flow-level outlier detection fails on brute-force attacks because each
individual flow is unremarkable - short, well-formed, ordinary. What is
anomalous is the pattern: thousands of near-identical connections from one
source. Aggregating to (host, hour) makes that pattern the unit of analysis.

This also repairs the dependency problem in the flow-level design. One
host-hour is genuinely one hypothesis; thousands of flows from a single attack
were never independent tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def _entropy(counts):
    """Shannon entropy in bits of a value-count array."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _group_keys(df):
    """Grouping columns for the configured aggregation unit."""
    hour = pd.to_datetime(df["_timestamp"]).dt.floor(config.BATCH)
    if config.AGGREGATION_UNIT == "src":
        return [df["_src_ip"].rename("src_ip"), hour.rename("hour")]
    if config.AGGREGATION_UNIT == "pair":
        return [df["_src_ip"].rename("src_ip"),
                df["_dst_ip"].rename("dst_ip"),
                hour.rename("hour")]
    raise ValueError("Unknown AGGREGATION_UNIT: " + repr(config.AGGREGATION_UNIT))


def aggregate(df, verbose=True):
    """Build host-hour feature rows from a cached flow-level frame.

    Returns a DataFrame with behavioural features plus '_'-prefixed metadata:
    _is_attack, _attack_frac, _n_attack_flows, _timestamp, _src_ip, _label.
    """
    required = {"_src_ip", "_timestamp", "_is_attack"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            "Cache is missing " + str(sorted(missing))
            + ". Re-run scripts/build_cache.py with the updated config."
        )

    work = pd.DataFrame({
        "duration": df["Flow Duration"].to_numpy(dtype=np.float64),
        "fwd_pkts": df["Total Fwd Packet"].to_numpy(dtype=np.float64),
        "bwd_pkts": df["Total Bwd packets"].to_numpy(dtype=np.float64),
        "fwd_bytes": df["Total Length of Fwd Packet"].to_numpy(dtype=np.float64),
        "bwd_bytes": df["Total Length of Bwd Packet"].to_numpy(dtype=np.float64),
        "rst": df["RST Flag Count"].to_numpy(dtype=np.float64),
        "syn": df["SYN Flag Count"].to_numpy(dtype=np.float64),
        "fin": df["FIN Flag Count"].to_numpy(dtype=np.float64),
        "dst_port": df["_dst_port"].to_numpy(),
        "dst_ip": df["_dst_ip"].to_numpy() if "_dst_ip" in df.columns else "na",
        "is_attack": df["_is_attack"].to_numpy().astype(bool),
        "ts": pd.to_datetime(df["_timestamp"]).to_numpy(),
    })

    keys = _group_keys(df)
    for k in keys:
        work[k.name] = k.to_numpy()
    key_names = [k.name for k in keys]

    rows = []
    for key, g in work.groupby(key_names, sort=False):
        n = len(g)
        if n < config.MIN_FLOWS_PER_HOST_HOUR:
            continue

        port_counts = g["dst_port"].value_counts().to_numpy()
        ts_sorted = np.sort(g["ts"].astype("int64").to_numpy())
        gaps = np.diff(ts_sorted) / 1e9 if n > 1 else np.array([0.0])

        n_attack = int(g["is_attack"].sum())
        rec = {
            # --- volume ---
            "n_flows": float(n),
            "n_distinct_dst_ip": float(g["dst_ip"].nunique()),
            "n_distinct_dst_port": float(g["dst_port"].nunique()),
            "flows_per_dst_port": float(n / max(g["dst_port"].nunique(), 1)),
            # --- how concentrated is the targeting? ---
            "dst_port_entropy": _entropy(port_counts),
            "top_port_share": float(port_counts.max() / n),
            # --- how repetitive are the flows? (brute force is regular) ---
            "duration_mean": float(g["duration"].mean()),
            "duration_cv": float(g["duration"].std() / (g["duration"].mean() + 1e-9)),
            "fwd_bytes_mean": float(g["fwd_bytes"].mean()),
            "fwd_bytes_cv": float(g["fwd_bytes"].std() / (g["fwd_bytes"].mean() + 1e-9)),
            "pkts_mean": float((g["fwd_pkts"] + g["bwd_pkts"]).mean()),
            "pkts_cv": float((g["fwd_pkts"] + g["bwd_pkts"]).std()
                             / ((g["fwd_pkts"] + g["bwd_pkts"]).mean() + 1e-9)),
            # --- timing regularity ---
            "gap_median": float(np.median(gaps)),
            "gap_cv": float(gaps.std() / (gaps.mean() + 1e-9)),
            "flows_per_second": float(n / max(np.ptp(ts_sorted) / 1e9, 1.0)),
            # --- connection outcomes: failed auth leaves a signature ---
            "rst_rate": float(g["rst"].mean()),
            "syn_rate": float(g["syn"].mean()),
            "fin_rate": float(g["fin"].mean()),
            "bwd_fwd_byte_ratio": float(g["bwd_bytes"].sum()
                                        / (g["fwd_bytes"].sum() + 1.0)),
            # --- metadata ---
            "_n_attack_flows": n_attack,
            "_attack_frac": n_attack / n,
            "_is_attack": (n_attack / n) > config.HOST_HOUR_ATTACK_MIN_FRAC
                          if config.HOST_HOUR_ATTACK_MIN_FRAC > 0 else n_attack > 0,
            "_timestamp": pd.Timestamp(g["ts"].min()),
        }
        key_tuple = key if isinstance(key, tuple) else (key,)
        for name, val in zip(key_names, key_tuple):
            rec["_" + name] = val
        rows.append(rec)

    out = pd.DataFrame(rows)
    if verbose and len(out):
        print("  " + format(len(out), ",") + " host-hours, "
              + format(int(out["_is_attack"].sum()), ",") + " malicious ("
              + str(round(100 * out["_is_attack"].mean(), 3)) + "%)")
    return out
