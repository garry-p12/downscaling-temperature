"""Score the model — and the README 5.7 offset correction — against AORC.

WHY A SECOND TRUTH PRODUCT
    Every number in this project is measured against ERA5-Land, which is a
    reanalysis produced by lapse-rate-downscaling coarse ERA5 with a land
    surface model. A network given a DEM can learn much of that deterministic
    correction, so scoring well against it is not the same as being right.
    README §6 put a number on this: ERA5-Land is itself 0.90-0.96 degC from
    station thermometers, and after the offset correction our model is level
    with it. At that point further gains against ERA5-Land may be fitting the
    reanalysis rather than reality.

    The station check answers this with TWO POINTS. AORC answers it over all
    1295 held-out cells with an observation-informed product, which is the
    difference between an anecdote and a field.

WHAT THIS CAN AND CANNOT CONCLUDE
    RMSE is NOT comparable across truth products — a bilinear floor of 2.209
    degC against ERA5-Land was 2.608 against AORC on the Colorado domain. So
    the absolute numbers here mean nothing next to §5.2's. What IS comparable
    is the RELATIVE effect of the offset correction under each truth:

      correction still helps under AORC  -> it removes a real POWER bias
      correction helps much less         -> part of it was ERA5-Land-specific

    That is the question this script exists to answer.

    AORC is a CONUS analysis. Over this domain it is ~92% finite, but only 74%
    south of 29 N, so cells it does not cover are dropped and reported.

Usage:
    DOWNSCALE_CONFIG_data=configs/data_sc_super.yaml \\
        python -m evaluation.aorc_eval --archs v2_landcov_s1337 of0_s1337
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from common import load_config                       # noqa: E402
from common.grid import build_target_grid            # noqa: E402
from common.normalize import Normalizer              # noqa: E402
from evaluation.offset_correction import (predict_all,   # noqa: E402
                                          ridge_fit_predict, subregion_means)
from training.dataset import holdout_bounds          # noqa: E402


def load_aorc_on_target(cfg, tgrid, year: int) -> xr.DataArray:
    """Daily-mean AORC 2 m temperature (degC) block-averaged onto the grid.

    Block mean, not nearest: one 10 km cell covers ~100 AORC pixels, and the
    nearest-neighbour fallback this code used to hit returned a single pixel —
    a point sample, not the cell mean the coarse product reports.
    """
    from data.regrid import field_to_target

    src = cfg["sources"]["aorc"]
    p = Path(src["out_dir"]) / f"aorc_{cfg['domain']['name']}_{year}_daily.zarr"
    if not p.exists():
        raise SystemExit(f"{p} not found — run data.download_aorc first")
    ds = xr.open_zarr(p, consolidated=False)
    var = src["variables"][0]
    da = ds[var]
    if float(da.max()) > 200:                     # AORC ships Kelvin
        da = da - 273.15
    if "latitude" in da.dims:
        da = da.rename({"latitude": "lat", "longitude": "lon"})
    print(f"[aorc] {p.name}: {da.sizes['time']} days, "
          f"{da.sizes['lat']}x{da.sizes['lon']} native")
    out = field_to_target(da, tgrid, "conservative")
    print(f"[aorc] on target grid: "
          f"{100 * np.isfinite(out.values).mean():.1f}% of cells covered")
    return out


def main(archs, zarr, year, device, out):
    cfg = load_config("data")
    ds = xr.open_zarr(zarr, consolidated=True)
    nz = Normalizer.load(str(Path(zarr).parent / "norm_stats.json"))
    tgrid = build_target_grid(cfg["domain"], cfg["grid"])
    stored = ds["channel_in"].values.tolist()

    i0, i1, j0, j1 = holdout_bounds(ds, cfg["holdout"])
    land = ds["input"].values[0, stored.index("land_mask")] > 0.5
    hold = np.zeros_like(land); hold[i0:i1, j0:j1] = True
    m_hold, m_train = hold & land, (~hold) & land

    times = pd.DatetimeIndex(ds["time"].values)
    doy = times.dayofyear.values - 1
    clim_t = ds["clim_tmp"].values[doy]
    clim_c = ds["clim_coarse_tmp"].values[doy]
    split = ds["split"].values
    tr_idx = np.where(split == "train")[0]

    aorc = load_aorc_on_target(cfg, tgrid, year)
    at = pd.DatetimeIndex(aorc["time"].values).normalize()
    sel = np.isin(times.normalize(), at)
    if sel.sum() == 0:
        raise SystemExit("no overlap between the store and the AORC year")
    order = pd.Series(np.arange(len(at)), index=at).reindex(
        times.normalize()[sel]).values
    A = aorc.values[order]                          # (n, y, x) degC, may be NaN

    era5 = ds["target"].values[:, 0][sel] + clim_t[sel]
    power = ds["input"].values[:, stored.index("coarse_tmp")][sel] + clim_c[sel]
    # Score only where BOTH truths exist, so every row sees the same cells.
    ok = np.isfinite(A) & np.isfinite(era5) & m_hold[None]
    print(f"[aorc] {sel.sum()} shared days; scoring "
          f"{ok.sum() / max(sel.sum(), 1):.0f} holdout cells/day "
          f"({100 * ok.sum() / max(sel.sum() * m_hold.sum(), 1):.1f}% of the "
          f"held-out land cells AORC covers)\n")

    def rmse(pred):
        return float(np.sqrt(np.nanmean((pred - A)[ok] ** 2)))

    res = {"interpolated_POWER": rmse(power),
           "ERA5-Land (the training target)": rmse(era5)}

    for arch in archs:
        ck = REPO / "checkpoints" / arch / "last.pt"
        if not ck.exists():
            print(f"{arch}: no last.pt, skipping"); continue
        pa = nz.inverse("out::tmp", predict_all(ck, ds, nz, stored, device))
        res[arch] = rmse(pa[sel] + clim_t[sel])
        # The 5.7 correction, rebuilt exactly as it is there: from the model's
        # residual against ERA5-Land over the TRAINING region. It never sees
        # AORC, so AORC is a genuinely independent judge of it.
        resid = np.nan_to_num(ds["target"].values[:, 0]) - pa
        est = ridge_fit_predict(
            subregion_means(resid, m_train, *land.shape, k=6),
            resid[:, m_hold].mean(1), tr_idx)
        res[f"{arch} + offset correction"] = rmse(pa[sel] + clim_t[sel]
                                                  + est[sel, None, None])

    print(f"{'method':<44}{'RMSE vs AORC':>14}")
    print("-" * 58)
    for k, v in sorted(res.items(), key=lambda kv: kv[1]):
        print(f"{k:<44}{v:>14.4f}")
    print("\nAbsolute values are NOT comparable with the ERA5-Land tables —")
    print("different truth products have different floors. What is comparable")
    print("is whether the offset correction still helps under a second truth.")

    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(res, indent=2))
        print(f"\n[aorc] wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archs", nargs="+", required=True)
    ap.add_argument("--zarr", default="data_store_sc/super/dataset.zarr")
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="outputs/aorc_eval.json")
    a = ap.parse_args()
    main(a.archs, a.zarr, a.year, a.device, a.out)
