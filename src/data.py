"""Loading, cleaning and label handling for the corrected CICIDS2017 release."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config


def _strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_timestamps(s: pd.Series) -> pd.Series:
    """Parse the Timestamp column, trying day-first then month-first."""
    for dayfirst in (True, False):
        out = pd.to_datetime(s, format="mixed", errors="coerce", dayfirst=dayfirst)
        if out.notna().mean() > 0.99:
            return out
    raise ValueError(
        "Could not parse timestamps. Sample values: " + repr(list(s.head(3)))
    )


def binarise_labels(labels: pd.Series, policy: str):
    """Map raw label strings to (is_attack, keep_mask) under an Attempted policy.

    Returns
    -------
    is_attack : bool Series, True for genuine attack traffic
    keep      : bool Series, False for rows to drop entirely
    """
    lab = labels.astype(str).str.strip()
    is_benign = lab == config.BENIGN_LABEL
    is_attempted = lab.str.endswith(config.ATTEMPTED_SUFFIX)

    if policy == "exclude":
        keep = ~is_attempted
        is_attack = ~is_benign & ~is_attempted
    elif policy == "attack":
        keep = pd.Series(True, index=lab.index)
        is_attack = ~is_benign
    elif policy == "benign":
        keep = pd.Series(True, index=lab.index)
        is_attack = ~is_benign & ~is_attempted
    else:
        raise ValueError("Unknown ATTEMPTED_POLICY: " + repr(policy))

    return is_attack, keep


def feature_columns(df: pd.DataFrame) -> list:
    """Numeric behavioural columns only: no identity, timing or label."""
    drop = set(config.EXCLUDE_FROM_FEATURES) | {"Label"}
    return [
        c for c in df.columns
        if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]


def load_day(filename: str, medians: dict = None, verbose: bool = True):
    """Load one day, clean it, and return everything downstream code needs.

    medians : imputation values fitted on the CALIBRATION day only. Pass None
              when loading Monday (they get fitted); pass the fitted dict for
              every other day. Never fit medians on test data - that is leakage.

    Returns (X, y, timestamps, raw_labels, metadata, medians).
    """
    path = config.RAW / filename
    if verbose:
        print("Loading " + filename + " ...")

    df = _strip_columns(pd.read_csv(path, low_memory=False))

    timestamps = parse_timestamps(df["Timestamp"])
    raw_labels = df["Label"].astype(str).str.strip()
    is_attack, keep = binarise_labels(raw_labels, config.ATTEMPTED_POLICY)

    feats = feature_columns(df)
    X = df[feats].astype(np.float64)

    # Inf arises on zero-duration flows; treat as missing, not as data.
    X = X.replace([np.inf, -np.inf], np.nan)

    # A missing IAT is itself informative (single-packet flow), so flag it
    # before imputing rather than silently filling.
    had_missing = X.isna().any(axis=1)

    if medians is None:
        fitted = X.median(numeric_only=True).to_dict()
        medians = {k: (0.0 if pd.isna(v) else float(v)) for k, v in fitted.items()}
        if verbose:
            print("  fitted imputation medians on this file")

    X = X.fillna(value=medians)
    X["_was_imputed"] = had_missing.astype(np.float64)

    # Metadata: never model features, but needed for aggregation and auditing.
    meta = pd.DataFrame(index=df.index)
    for col in getattr(config, "METADATA_COLUMNS", []):
        if col in df.columns:
            meta["_" + col.lower().replace(" ", "_")] = df[col].to_numpy()

    # Apply the keep mask last so every array stays aligned.
    X = X.loc[keep].reset_index(drop=True).astype(np.float32)
    y = is_attack.loc[keep].reset_index(drop=True).to_numpy()
    ts = timestamps.loc[keep].reset_index(drop=True)
    labels = raw_labels.loc[keep].reset_index(drop=True)
    meta = meta.loc[keep].reset_index(drop=True)

    if verbose:
        dropped = int((~keep).sum())
        print(
            "  " + format(len(X), ",") + " rows kept, "
            + format(dropped, ",") + " dropped (policy="
            + config.ATTEMPTED_POLICY + ")"
        )
        print(
            "  " + format(int(y.sum()), ",") + " attacks ("
            + str(round(100 * float(y.mean()), 3)) + "%), "
            + str(X.shape[1]) + " features"
        )

    return X, y, ts, labels, meta, medians


def save_medians(medians: dict, path) -> None:
    with open(path, "w") as fh:
        json.dump(medians, fh, indent=2)


def load_medians(path) -> dict:
    with open(path) as fh:
        return json.load(fh)
