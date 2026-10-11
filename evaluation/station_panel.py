"""Score gridded products against every ISD station in the held-out box.

WHY THIS EXISTS
    The station comparison carried the most weight of any result in this
    project — it is the only judge neither gridded product controls — and it
    rested on two stations in one year. Two points cannot separate "AORC is
    closer to reality" from "AORC happens to suit Austin-Bergstrom". This
    widens it to every ISD station whose nearest grid cell falls inside the
    held-out box and which reported through 2019-2023.

WHY PRODUCTS GET FIVE YEARS AND MODELS GET ONE
    Comparing ERA5-Land and AORC to a thermometer involves no model, so every
    year is fair game and all five are used. A MODEL may only be scored on the
    test split (2023) — its other years were training data. The two are
    reported separately and must not be merged.

WHY NEAREST CELL, AND WHAT THAT COSTS
    A thermometer is a point; a cell is a 100 km^2 average. The resulting
    representativeness error inflates every product's score equally, so the
    RANKING is meaningful even though the absolute numbers are not an accuracy
    ceiling. The spread between stations reported here is the honest measure of
    how much that matters.

Usage:
    python -m evaluation.station_panel --fetch          # download, then score
    python -m evaluation.station_panel --models dynaorc_s1337 dynera_s1337
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

if os.environ.get("DOWNSCALE_MAX_WORKERS"):
    import dask
    import zarr
    zarr.config.set({"threading.max_workers": int(os.environ["DOWNSCALE_MAX_WORKERS"])})
    dask.config.set(scheduler="synchronous")

ISD_URL = ("https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/"
           "{year}/{usaf}-{wban}-{year}.gz")
PANEL = REPO / "evaluation" / "station_panel.json"
CACHE = REPO / "data_store_sc" / "raw" / "isd"
MIN_HOURS = 20          # a day needs this many hourly obs to count as a daily mean


def fetch(usaf: str, wban: str, year: int) -> pd.Series | None:
    """Daily mean temperature (degC) from one ISD-Lite station-year."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / f"{usaf}-{wban}-{year}.gz"
    if not dest.exists():
        try:
            urllib.request.urlretrieve(
                ISD_URL.format(year=year, usaf=usaf, wban=wban), dest)
        except Exception:
            return None
    try:
        with gzip.open(dest, "rt") as fh:
            df = pd.read_fwf(fh, header=None,
                             colspecs=[(0, 4), (5, 7), (8, 11), (11, 13), (13, 19)],
                             names=["y", "m", "d", "h", "tmp"])
    except Exception:
        return None
    if df.empty:
        return None
    # ISD-Lite stores tenths of degC and -9999 for missing.
    df = df[df.tmp > -9000]
    if df.empty:
        return None
    df["t"] = pd.to_datetime(dict(year=df.y, month=df.m, day=df.d), errors="coerce")
    df = df.dropna(subset=["t"])
    g = (df.tmp / 10.0).groupby(df.t)
    return g.mean()[g.count() >= MIN_HOURS]


def load_panel() -> list[dict]:
    if not PANEL.exists():
        raise SystemExit(f"{PANEL} is missing; regenerate it with --build-panel")
    return json.loads(PANEL.read_text())


def score(pred: np.ndarray, obs: np.ndarray) -> dict:
    d = pred - obs
    return {"n": int(np.isfinite(d).sum()),
            "rmse": float(np.sqrt(np.nanmean(d ** 2))),
            "mae": float(np.nanmean(np.abs(d))),
            "bias": float(np.nanmean(d))}


def main(years, models, do_fetch, out_dir, device):
    from evaluation.fields import Store, predict_all

    panel = load_panel()
    st_e = Store("data_store_sc/ix_era/dataset.zarr") if \
        (REPO / "data_store_sc/ix_era/dataset.zarr").exists() else \
        Store("data_store_sc/super/dataset.zarr")
    st_a = Store("data_store_sc/aorc/dataset.zarr")
    times = pd.DatetimeIndex(st_a.ds["time"].values)
    era = st_e.to_absolute(st_e.truth())
    aorc = st_a.to_absolute(st_a.truth())
    test_mask = st_a.split == "test"

    # ---- products, every requested year, no model involved -----------------
    prod_rows, obs_cache = [], {}
    for s in panel:
        series = []
        for y in years:
            o = fetch(s["usaf"], s["wban"], y) if do_fetch else None
            if o is not None and len(o):
                series.append(o)
        if not series:
            continue
        obs = pd.concat(series).sort_index()
        obs = obs[~obs.index.duplicated()]
        sel = times.isin(obs.index)
        if sel.sum() < 200:
            continue
        o = obs.reindex(times[sel]).values
        obs_cache[s["key"]] = (sel, o)
        i, j = s["row"], s["col"]
        r = {"station": s["key"], "lat": s["lat"], "lon": s["lon"],
             "elev_m": s["elev_m"], "days": int(sel.sum())}
        for lab, F in [("era5_land", era), ("aorc", aorc)]:
            r[lab] = score(F[sel, i, j], o)
        prod_rows.append(r)
        print(f"  {s['key'][:38]:<38} {r['days']:>5}d  "
              f"ERA5-Land {r['era5_land']['rmse']:.3f}  AORC {r['aorc']['rmse']:.3f}",
              flush=True)

    # ---- models, TEST SPLIT ONLY -------------------------------------------
    model_rows = []
    for spec in models:
        # "checkpoint=store" so the pairing is explicit. Guessing the store
        # from the checkpoint name breaks the moment an arm is named for its
        # channels rather than its target, and a model silently scored against
        # the wrong store's climatology produces plausible nonsense.
        name, _, store_name = spec.partition("=")
        if store_name:
            store = Store(f"data_store_sc/{store_name}/dataset.zarr")
        else:
            store = st_a if "aorc" in name else st_e
        if not (REPO / "checkpoints" / name / "last.pt").exists():
            print(f"  {name}: no checkpoint, skipping")
            continue
        pred = store.to_absolute(predict_all(store, name, "last", device=device))
        for s in panel:
            if s["key"] not in obs_cache:
                continue
            sel, o = obs_cache[s["key"]]
            keep = sel & test_mask
            if keep.sum() < 100:
                continue
            # line the observation up with the test-only subset
            idx = np.flatnonzero(keep[sel])
            model_rows.append({"model": name, "station": s["key"],
                               **score(pred[keep, s["row"], s["col"]], o[idx])})

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "products.json").write_text(json.dumps(prod_rows, indent=1))
    (out / "models.json").write_text(json.dumps(model_rows, indent=1))

    # ---- summary -----------------------------------------------------------
    print(f"\n=== PRODUCTS vs {len(prod_rows)} stations, {min(years)}-{max(years)} ===")
    for lab in ("era5_land", "aorc"):
        v = np.array([r[lab]["rmse"] for r in prod_rows])
        b = np.array([r[lab]["bias"] for r in prod_rows])
        print(f"  {lab:<10} RMSE {v.mean():.4f} +/- {v.std():.4f}   "
              f"median {np.median(v):.4f}   |bias| mean {np.abs(b).mean():.4f}")
    if prod_rows:
        wins = sum(r["aorc"]["rmse"] < r["era5_land"]["rmse"] for r in prod_rows)
        print(f"  AORC closer at {wins}/{len(prod_rows)} stations")

    if model_rows:
        print(f"\n=== MODELS vs the same stations, test split only ===")
        md = pd.DataFrame(model_rows)
        for name, g in md.groupby("model"):
            print(f"  {name:<18} RMSE {g.rmse.mean():.4f} +/- {g.rmse.std():.4f} "
                  f"over {len(g)} stations")
    print(f"\nwrote {out/'products.json'} and {out/'models.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, nargs="+", default=[2019, 2020, 2021, 2022, 2023])
    ap.add_argument("--models", nargs="*", default=[],
                    help="checkpoint, or checkpoint=store (e.g. aorc_s1337=aorc)")
    ap.add_argument("--fetch", action="store_true", default=True,
                    help="download any missing station-years (cached afterwards)")
    ap.add_argument("--out", default="outputs/station_panel")
    # predict_all defaults to CPU. Scoring 18 checkpoints over 1,826 days each
    # is 33k forward passes and does not fit a 90-minute allocation on CPU; on
    # the GPU the job already holds it is minutes.
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    main(a.years, a.models, a.fetch, a.out, a.device)
