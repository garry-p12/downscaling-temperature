"""Export the offset-correction result as a compact JSON bundle for a web demo.

The static figures show the correction on one day and on the year mean. Neither
lets you scrub through the test year and watch the error map flatten, which is
the thing that makes the mechanism obvious. This writes everything a browser
needs to do that, and nothing it does not.

SIZE IS THE WHOLE DESIGN CONSTRAINT. A published artifact is capped at 16 MB,
and 365 days x 9300 cells x two float32 fields is 27 MB before encoding. Three
choices bring it down:

  * Only the ERROR field is stored per day, not the corrected field. The
    correction subtracts one number, so the browser computes
    corrected = error - offset itself. That also makes the page honest: the
    slider really is applying the estimator, not replaying a second render.
  * Everything is quantised to uint8 and base64'd. The error range is about
    +/- 4 degC, so a byte buys 0.03 degC — an order of magnitude finer than the
    0.24 degC RMSE change being shown.
  * Days are subsampled by --stride. The per-day SCALARS (offsets, RMSE) are
    still exported for all 365 days, so the scatter and the time series are
    complete even though the maps are not.

TWO TARGETS, same computation.

    --target artifact   one self-contained data.js of base64 strings, for a
                        published Claude artifact where every byte must ship
                        inside the page.
    --target webapp     manifest.json plus a raw fields.bin, for the Next.js
                        app in webapp/. A browser fetching an ArrayBuffer and
                        taking typed-array views over it needs no decode step
                        and no base64 tax — about 25% smaller and it streams.

Usage:
    python scripts/export_offset_demo.py --target webapp --arch of0_s1337
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from evaluation.fields import Store, decompose, predict_all      # noqa: E402
from evaluation.offset_correction import (REGIONS,               # noqa: E402
                                          ridge_fit_predict, subregion_means)
from training.dataset import holdout_bounds                      # noqa: E402


def b64(arr) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def quant8(field, lo, hi):
    """Clip to [lo, hi] and map onto 0..255."""
    z = (np.clip(field, lo, hi) - lo) / (hi - lo)
    return np.round(z * 255).astype("uint8")


def quant16(field, lo, hi):
    z = (np.clip(field, lo, hi) - lo) / (hi - lo)
    return np.round(z * 65535).astype("uint16")


def offsets_for(resid, m_hold, m_train, shape, i_train, k):
    """(k-block estimate, single global mean, truth) for every day."""
    true = resid[:, m_hold].mean(1)
    glob = resid[:, m_train].mean(1)
    X = subregion_means(resid, m_train, *shape, k=k)
    return ridge_fit_predict(X, true, i_train), glob, true


def main(arch, zarr, ckpt, k, stride, out, target):
    st = Store(zarr)
    te, land, hold = st.i_test, st.land, st.m_hold
    ny, nx = st.shape

    truth_a = st.truth()
    resid = truth_a - predict_all(st, arch, ckpt)
    est, glob, true = offsets_for(resid, hold, st.m_train, st.shape, st.i_train, k)

    # --- per-day scalars, ALL 365 test days --------------------------------
    r_te = resid[te]
    rm = np.sqrt((r_te[:, hold] ** 2).mean(1))
    rc = np.sqrt(((r_te - est[te, None, None])[:, hold] ** 2).mean(1))
    rg = np.sqrt(((r_te - glob[te, None, None])[:, hold] ** 2).mean(1))
    ro = np.sqrt(((r_te - true[te, None, None])[:, hold] ** 2).mean(1))

    R = lambda v: float(np.sqrt((v ** 2).mean()))                # noqa: E731
    year = dict(
        model=R(r_te[:, hold]),
        corrected=R((r_te - est[te, None, None])[:, hold]),
        globalmean=R((r_te - glob[te, None, None])[:, hold]),
        oracle=R((r_te - true[te, None, None])[:, hold]),
        interp=R((truth_a - st.coarse())[te][:, hold]),
    )

    # --- per-day MAPS, subsampled ------------------------------------------
    keep = np.arange(0, len(te), stride)
    days = te[keep]
    err = r_te[keep]                                  # truth - model
    # TWO colour limits, and the distinction matters for whether the demo
    # shows anything. The domain-wide 99.7th percentile is ~3.1 degC, driven by
    # the coastal front and the western terrain; on that scale the ~0.4 degC
    # offset being corrected is a shade of the same pink and the whole point is
    # invisible. `elimBox` is the 98th percentile over the HELD-OUT BOX, which
    # is the region every metric on the page is quoted over. Storage still uses
    # the wide range so no value is destroyed; the page draws with the narrow
    # one and says so, and cells outside it clip rather than lie.
    elim = float(np.percentile(np.abs(err[:, land]), 99.7))
    # Three selectable display ranges, taken from the box's own |error|
    # distribution. The median |error| over the box is about 0.50 degC and the
    # median daily offset about 0.36, so the p70 range is the one on which a
    # typical correction is actually legible; the wider two are there for
    # anyone who wants to see the tail instead of having it clipped.
    elimBox = [float(np.percentile(np.abs(err[:, hold]), q)) for q in (70, 90, 98)]
    tabs = st.to_absolute(truth_a)[days]
    tlo = float(np.percentile(tabs[:, land], 0.5))
    thi = float(np.percentile(tabs[:, land], 99.5))

    # --- static maps --------------------------------------------------------
    nanl = lambda f: np.where(land, f, 0.0)                      # noqa: E731
    # Per-cell RMSE for EVERY estimator, not just k=6: the year view lets the
    # reader switch between them and the map has to follow, or the segmented
    # control would silently show the same picture under four labels.
    pc = dict(
        pcModel=np.sqrt((r_te ** 2).mean(0)),
        pcGlobalmean=np.sqrt(((r_te - glob[te, None, None]) ** 2).mean(0)),
        pcCorrected=np.sqrt(((r_te - est[te, None, None]) ** 2).mean(0)),
        pcOracle=np.sqrt(((r_te - true[te, None, None]) ** 2).mean(0)),
    )
    pcmax = float(np.percentile(pc["pcModel"][land], 99.5))

    task = (truth_a - st.coarse())[te]
    _, _, _, sh_box = decompose(task, hold)
    _, _, _, sh_dom = decompose(task, land)
    _, _, _, sh_mod = decompose(r_te, hold)

    # --- leave-one-region-out ----------------------------------------------
    loro = []
    for name, hcfg in REGIONS.items():
        i0, i1, j0, j1 = holdout_bounds(st.ds, hcfg)
        h = np.zeros_like(land)
        h[i0:i1, j0:j1] = True
        mh, mt = h & land, (~h) & land
        e2, _, t2 = offsets_for(resid, mh, mt, st.shape, st.i_train, k)
        r2 = resid[te][:, mh]
        a = R(r2)
        b = R(r2 - e2[te, None])
        c = R(r2 - t2[te, None])
        loro.append(dict(name=name, box=[i0, i1, j0, j1], cells=int(mh.sum()),
                         model=a, corrected=b, oracle=c,
                         pct=(b - a) / a * 100, ofOracle=(b - a) / (c - a) * 100))
    loro.sort(key=lambda r: -r["ofOracle"])

    # Everything below is target-independent: the same arrays, described the
    # same way. Only the packing differs.
    fields = dict(
        lat=quant16(st.LAT, st.LAT.min(), st.LAT.max()),
        lon=quant16(st.LON, st.LON.min(), st.LON.max()),
        land=land.astype("uint8"),
        truth=quant8(nanl(tabs), tlo, thi),
        err=quant8(nanl(err), -elim, elim),
        **{k_: quant8(nanl(v_), 0, pcmax) for k_, v_ in pc.items()},
    )
    scales = dict(
        lat=dict(lo=float(st.LAT.min()), hi=float(st.LAT.max())),
        lon=dict(lo=float(st.LON.min()), hi=float(st.LON.max())),
        truth=dict(lo=tlo, hi=thi),
        err=dict(lo=-elim, hi=elim, lim=elim, limBox=elimBox, limBoxQ=[70, 90, 98]),
        **{k_: dict(lo=0.0, hi=pcmax) for k_ in pc},
    )

    meta = dict(
        arch=arch, ckpt=ckpt, k=k, nx=nx, ny=ny,
        box=[int(v) for v in st.box],
        landCells=int(land.sum()), boxCells=int(hold.sum()),
        nDays=len(days), nTestDays=len(te), stride=stride, blocks=k * k,
    )
    scalars = dict(
        meta=meta,
        dates=[str(d)[:10] for d in st.times[days].values],
        allDates=[str(d)[:10] for d in st.times[te].values],
        offsets=dict(true=[round(float(v), 4) for v in true[te]],
                     est=[round(float(v), 4) for v in est[te]],
                     glob=[round(float(v), 4) for v in glob[te]]),
        daily=dict(model=[round(float(v), 4) for v in rm],
                   corrected=[round(float(v), 4) for v in rc],
                   globalmean=[round(float(v), 4) for v in rg],
                   oracle=[round(float(v), 4) for v in ro]),
        mapDayIndex=[int(v) for v in keep],
        # Open on a TYPICAL day, not the most flattering one: the map day whose
        # holdout offset is the median in magnitude over the test year.
        defaultDay=int(np.argsort(np.abs(true[days]))[len(days) // 2]),
        year=year,
        bias=dict(model=float(r_te[:, hold].mean()),
                  corrected=float((r_te - est[te, None, None])[:, hold].mean())),
        corr=dict(glob=float(np.corrcoef(glob[te], true[te])[0, 1]),
                  est=float(np.corrcoef(est[te], true[te])[0, 1])),
        budget=dict(box={kk: sh_box[kk] for kk in ("offset", "static", "remainder")},
                    domain={kk: sh_dom[kk] for kk in ("offset", "static", "remainder")},
                    model={kk: sh_mod[kk] for kk in ("offset", "static", "remainder")},
                    totalVar=sh_box["total_var"]),
        loro=loro,
    )

    if target == "artifact":
        bundle = dict(scalars)
        bundle["lat"] = dict(**scales["lat"], d=b64(fields["lat"]))
        bundle["lon"] = dict(**scales["lon"], d=b64(fields["lon"]))
        bundle["land"] = b64(fields["land"])
        bundle["truth"] = dict(**scales["truth"], d=b64(fields["truth"]))
        bundle["err"] = dict(lim=elim, limBox=elimBox, d=b64(fields["err"]))
        bundle["perCell"] = dict(
            max=pcmax,
            model=b64(fields["pcModel"]), globalmean=b64(fields["pcGlobalmean"]),
            corrected=b64(fields["pcCorrected"]), oracle=b64(fields["pcOracle"]))
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("window.OFFSET_DEMO = "
                       + json.dumps(bundle, separators=(",", ":")) + ";")
        emitted = [out]
    else:
        # Binary layout. The uint16 fields go FIRST so that a Uint16Array view
        # over the fetched ArrayBuffer is 2-byte aligned; a uint8 field ahead of
        # them would make that view throw in some browsers.
        d = Path(out)
        d.mkdir(parents=True, exist_ok=True)
        order = ["lat", "lon", "land", "truth", "err",
                 "pcModel", "pcGlobalmean", "pcCorrected", "pcOracle"]
        blob, layout, pos = bytearray(), {}, 0
        for name in order:
            a_ = np.ascontiguousarray(fields[name])
            layout[name] = dict(offset=pos, length=int(a_.size),
                                dtype=str(a_.dtype), **scales.get(name, {}))
            blob += a_.tobytes()
            pos += a_.nbytes
        (d / "fields.bin").write_bytes(bytes(blob))
        scalars["layout"] = layout
        scalars["binBytes"] = pos
        # Content hash, used by the page as a cache-busting query on the
        # fields.bin URL. Without it a browser holding a cached copy from an
        # earlier export pairs old pixels with a new manifest, and the only
        # reason that is caught at all is the byte-length check in useFields.
        scalars["version"] = hashlib.sha1(bytes(blob)).hexdigest()[:12]
        (d / "manifest.json").write_text(json.dumps(scalars, separators=(",", ":")))
        emitted = [d / "manifest.json", d / "fields.bin"]

    mb = sum(f.stat().st_size for f in emitted) / 1e6
    for f in emitted:
        print(f"  {f}  {f.stat().st_size / 1e6:.2f} MB")
    print(f"  {len(days)} map days of {len(te)} test days, {mb:.2f} MB total")
    print(f"  year RMSE  model {year['model']:.4f} -> corrected {year['corrected']:.4f} "
          f"({(year['corrected'] - year['model']) / year['model']:+.1%}), "
          f"oracle {year['oracle']:.4f} "
          f"({(year['corrected'] - year['model']) / (year['oracle'] - year['model']) * 100:.0f}%)")
    print(f"  offset corr  global {scalars['corr']['glob']:.2f} -> k={k} "
          f"{scalars['corr']['est']:.2f}")
    if target == "artifact" and mb > 14:
        print("  WARNING: over 14 MB, raise --stride (artifact cap is 16 MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arch", default="of0_s1337")
    ap.add_argument("--zarr", default="data_store_sc/super/dataset.zarr")
    ap.add_argument("--ckpt", default="last", choices=["best", "last"])
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--stride", type=int, default=3, help="keep every Nth test day's map")
    ap.add_argument("--target", default="webapp", choices=["artifact", "webapp"])
    ap.add_argument("--out", default=None,
                    help="webapp: a directory. artifact: a .js file.")
    a = ap.parse_args()
    default = (str(REPO / "webapp" / "public" / "data") if a.target == "webapp"
               else str(REPO / "image_outputs" / "offset_demo" / "data.js"))
    main(a.arch, a.zarr, a.ckpt, a.k, a.stride, a.out or default, a.target)
