"""Enumerate the raw dataset: files, schema, labels, data-quality issues."""
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import RAW

files = sorted(RAW.rglob("*.csv"))
if not files:
    sys.exit("No CSVs found under " + str(RAW))

print("Found " + str(len(files)) + " CSV file(s)")
print("=" * 72)

for f in files:
    print("\n" + str(f.relative_to(RAW)) + "   (" + str(round(f.stat().st_size / 1e6, 1)) + " MB)")

    header = pd.read_csv(f, nrows=0)
    cols = [c.strip() for c in header.columns]
    print("  " + str(len(cols)) + " columns")

    dupes = sorted({c for c in cols if cols.count(c) > 1})
    if dupes:
        print("  DUPLICATE COLUMN NAMES: " + str(dupes))

    label_raw = None
    for orig in header.columns:
        if orig.strip().lower() in ("label", "attack", "class", "attack_cat"):
            label_raw = orig
            break
    print("  label column: " + repr(label_raw))

    if label_raw is not None:
        labels = pd.read_csv(f, usecols=[label_raw])[label_raw].astype(str).str.strip()
        total = len(labels)
        print("  rows: " + format(total, ","))
        for lab, n in labels.value_counts().items():
            pct = round(100 * n / total, 3)
            print("     " + lab.ljust(34) + format(n, ",").rjust(10) + "  (" + str(pct) + "%)")

    sample = pd.read_csv(f, nrows=20000, low_memory=False)
    sample.columns = [c.strip() for c in sample.columns]
    num = sample.select_dtypes(include=[np.number])
    inf_cols = [c for c in num.columns if np.isinf(num[c]).any()]
    nan_cols = [c for c in sample.columns if sample[c].isna().any()]
    if inf_cols:
        print("  Inf values in: " + str(inf_cols))
    if nan_cols:
        print("  NaN values in: " + str(nan_cols[:8]))

print("\n" + "=" * 72)
print("FULL COLUMN LIST (" + files[0].name + "):")
for i, c in enumerate(pd.read_csv(files[0], nrows=0).columns):
    print("  " + str(i).rjust(3) + "  " + repr(c.strip()))
