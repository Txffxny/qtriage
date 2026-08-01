"""Load every day once, cache to parquet, fit imputation medians on Monday."""
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.data import load_day, save_medians


def main():
    config.PROCESSED.mkdir(parents=True, exist_ok=True)

    days = sorted(p.name for p in config.RAW.glob("*.csv"))
    cal = config.CALIBRATION_FILE

    if not days:
        sys.exit("No CSVs found in " + str(config.RAW))
    if cal not in days:
        sys.exit("Calibration file " + cal + " not found. Present: " + str(days))

    # Monday first: it fits the medians every other day reuses.
    ordered = [cal] + [d for d in days if d != cal]
    medians = None

    for day in ordered:
        X, y, ts, labels, meta, medians = load_day(day, medians=medians)

        out = X.copy()
        for col in meta.columns:
            out[col] = meta[col].to_numpy()
        out["_is_attack"] = y
        out["_timestamp"] = ts.to_numpy()
        out["_label"] = labels.to_numpy()

        dest = config.PROCESSED / day.replace(".csv", ".parquet")
        out.to_parquet(dest, index=False)
        print(
            "  -> " + dest.name + "  ("
            + str(round(dest.stat().st_size / 1e6, 1)) + " MB)"
        )

        print("  time span: " + str(ts.min()) + "  ->  " + str(ts.max()))
        hourly = ts.dt.floor("h").value_counts().sort_index()
        print(
            "  " + str(len(hourly)) + " hourly batches, sizes "
            + format(int(hourly.min()), ",") + " - "
            + format(int(hourly.max()), ",")
            + " (median " + format(int(hourly.median()), ",") + ")"
        )
        print()

        if day == cal:
            save_medians(medians, config.PROCESSED / "medians.json")
            print("  saved medians.json\n")

    print("Done.")


if __name__ == "__main__":
    main()
