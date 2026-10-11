"""Convert the per-cell CSV tables to .npy structured arrays.

WHAT THESE FILES ARE
    One row per grid cell (9,300 = 93 x 100), written by
    scripts/make_variant_outputs.py so the store's contents can be inspected
    without opening zarr. They are NOT what the model reads — training and
    evaluation go through data_store_sc/*/dataset.zarr. Converting them changes
    nothing about the pipeline; it just makes the tables faster and easier to
    load for analysis.

WHY A STRUCTURED ARRAY RATHER THAN A PLAIN 2-D FLOAT ARRAY
    A plain array would force every column to one dtype and drop the column
    names into a side file that can drift from the data. A structured dtype
    keeps the names with the values, loads without allow_pickle, and lets
    integer columns stay integers:

        a = np.load("training_store_cells.npy")
        a["dem"]            # one column by name
        a["lat"][a["is_land"] == 1]

WHY SOME COLUMNS STAY FLOAT
    A column is stored as int32 only if every value is integral AND none is
    missing. Sea cells carry NaN in the climatology and target columns, and NaN
    has no integer representation, so those stay float64 even where the finite
    values happen to be whole numbers.

Usage:
    python scripts/csv_to_npy.py                       # every CSV under outputs/
    python scripts/csv_to_npy.py path/to/one.csv ...   # specific files
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

DEFAULT_GLOBS = ["outputs/*.csv", "outputs/variant_stores/*.csv"]


def convert(csv_path: Path, out_path: Path | None = None, verify: bool = True):
    df = pd.read_csv(csv_path)
    out_path = out_path or csv_path.with_suffix(".npy")

    bad = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise SystemExit(f"{csv_path.name}: non-numeric columns {bad}; "
                         f"this converter assumes an all-numeric table")

    fields, n_int = [], 0
    for c in df.columns:
        col = df[c]
        integral = not col.isna().any() and bool((col % 1 == 0).all())
        fields.append((c, "<i4" if integral else "<f8"))
        n_int += integral

    arr = np.empty(len(df), dtype=np.dtype(fields))
    for c in df.columns:
        arr[c] = df[c].values

    np.save(out_path, arr, allow_pickle=False)

    if verify:
        back = np.load(out_path, allow_pickle=False)
        if list(back.dtype.names) != list(df.columns):
            raise SystemExit(f"{out_path.name}: column names did not round-trip")
        for c in df.columns:
            a, b = back[c], df[c].values
            # equal_nan so the sea cells compare as equal rather than as a
            # spurious mismatch.
            if not np.allclose(a, b, rtol=0, atol=0, equal_nan=True):
                raise SystemExit(f"{out_path.name}: column {c!r} did not round-trip")

    return out_path, arr, n_int


def main(paths, verify):
    if not paths:
        paths = sorted({p for g in DEFAULT_GLOBS for p in glob.glob(str(REPO / g))})
    if not paths:
        raise SystemExit("no CSV files found; pass paths explicitly")

    total_csv = total_npy = 0
    for p in paths:
        src = Path(p)
        out, arr, n_int = convert(src, verify=verify)
        c_sz, n_sz = src.stat().st_size, out.stat().st_size
        total_csv += c_sz
        total_npy += n_sz
        print(f"  {src.name:<34} {len(arr):>6} rows x {len(arr.dtype.names):>2} cols"
              f"   {n_int:>2} int / {len(arr.dtype.names) - n_int:>2} float"
              f"   {c_sz/1e6:5.2f} -> {n_sz/1e6:5.2f} MB"
              f"   {'verified' if verify else ''}")
    print(f"\n  {len(paths)} file(s)   {total_csv/1e6:.2f} MB CSV -> "
          f"{total_npy/1e6:.2f} MB npy")
    print("  load with:  a = np.load(path); a['dem']; a.dtype.names")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="CSV files; default is everything under outputs/")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the round-trip check")
    a = ap.parse_args()
    sys.exit(main(a.paths, not a.no_verify))
