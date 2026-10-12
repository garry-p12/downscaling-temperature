"""Score the target x interaction 2x2, keeping the two truth products apart.

THE COMPARISON THAT IS NOT ALLOWED
    dynera's RMSE against ERA5-Land and dynaorc's RMSE against AORC are not
    comparable numbers. They are distances to different references with
    different variance (ERA5-Land anomaly sd 3.73, AORC 3.88 over the shared
    mask), so the arm with the smoother target wins by default. Every
    cross-target claim in this report therefore goes through a COMMON JUDGE:
    AORC for the gridded comparison, NOAA ISD thermometers for the physical
    one. Within a target the direct comparison is fine, and that is what
    answers "do the interaction channels help".

WHAT IS REPORTED, AND WHY EACH
    own-truth      each arm against the product it was trained on. Tells you
                   whether the model fit its own objective; says nothing across
                   targets. Written to a separate section per product.
    vs AORC        every arm against AORC. The cross-target judge for the grid.
    effective res  the diagnostic that motivated the experiment. ERA5-Land
                   holds 26% of AORC's sub-60 km amplitude at ~zero coherence,
                   so an ERA5-trained model cannot resolve what it was never
                   shown. If the target was the ceiling, THIS is what moves.
    offset-corr    the 5.7 correction applied per arm, since a raw RMSE
                   comparison inherits whatever product bias each arm carries.

Usage:
    python -m evaluation.interaction_report --out outputs/interaction
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

if os.environ.get("DOWNSCALE_MAX_WORKERS"):
    import dask
    import zarr
    zarr.config.set({"threading.max_workers": int(os.environ["DOWNSCALE_MAX_WORKERS"])})
    dask.config.set(scheduler="synchronous")

SEEDS = [1337, 7, 42, 2024, 31337]

# (arm, store it trained on, which product that store's target is)
ARMS = [
    ("dynera",  "ix_era",  "ERA5-Land"),
    ("ixera",   "ix_era",  "ERA5-Land"),
    ("dynaorc", "ix_aorc", "AORC"),
    ("ixaorc",  "ix_aorc", "AORC"),
]

# Between-seed noise floors measured in this project (README 5.1). A difference
# smaller than these is not a result, however many decimals it has.
NOISE = {"rmse": 0.0097, "ssim": 0.0039, "resid_corr": 0.0092}


def welch(a, b):
    """Welch t and dof for two small samples. Returns (t, dof, mean_diff)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return 0.0, 0.0, float(b.mean() - a.mean())
    t = (b.mean() - a.mean()) / se
    dof = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return float(t), float(dof), float(b.mean() - a.mean())


def main(out_dir, ckpt, k, seeds, arms, device):
    from evaluation.fields import Store, predict_all
    from evaluation.offset_correction import ridge_fit_predict, subregion_means
    from evaluation.spectral import effective_resolution, radial_spectrum

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    stores = {}
    for _, s, _ in arms:
        if s not in stores:
            stores[s] = Store(f"data_store_sc/{s}/dataset.zarr")
    ref = stores[arms[0][1]]
    te, hold = ref.i_test, ref.m_hold
    i0, i1, j0, j1 = ref.box

    # The cross-target judge. Taken from the AORC store's target, on the shared
    # mask, in absolute degC so both arms are measured against the same field.
    aorc_abs = stores["ix_aorc"].to_absolute(stores["ix_aorc"].truth())

    results = {}
    for arm, store_name, product in arms:
        st = stores[store_name]
        own_abs = st.to_absolute(st.truth())
        per_seed = {"own_rmse": [], "aorc_rmse": [], "own_rmse_corr": [],
                    "aorc_rmse_corr": [], "eff_res": [], "coh60": [], "amp60": []}
        present = []

        for seed in seeds:
            name = f"{arm}_s{seed}"
            if not (REPO / "checkpoints" / name / f"{ckpt}.pt").exists():
                print(f"  {name}: no {ckpt}.pt, skipping")
                continue
            present.append(seed)
            pred_anom = predict_all(st, name, ckpt, device=device)
            pred_abs = st.to_absolute(pred_anom)

            r_own = float(np.sqrt(((pred_abs - own_abs)[te][:, hold] ** 2).mean()))
            r_aorc = float(np.sqrt(((pred_abs - aorc_abs)[te][:, hold] ** 2).mean()))

            # Offset correction, estimated from the training region of the arm's
            # OWN product — the same rule 5.7 uses. Applied before scoring
            # against both judges, so neither number carries a product bias
            # that the correction is known to remove.
            resid = st.truth() - pred_anom
            X = subregion_means(resid, st.m_train, *st.shape, k=k)
            est = ridge_fit_predict(X, resid[:, hold].mean(1), st.i_train)
            corr_abs = pred_abs + est[:, None, None]
            r_own_c = float(np.sqrt(((corr_abs - own_abs)[te][:, hold] ** 2).mean()))
            r_aorc_c = float(np.sqrt(((corr_abs - aorc_abs)[te][:, hold] ** 2).mean()))

            # Effective resolution is measured against AORC for EVERY arm, not
            # against each arm's own target. Against ERA5-Land an ERA5-trained
            # model would be scored on reproducing a field that has no real
            # fine structure, which is the confound the experiment exists to
            # expose.
            crop = lambda f: f[te][:, i0:i1, j0:j1]                 # noqa: E731
            p = crop(pred_abs); a = crop(aorc_abs)
            p = p - p.mean((1, 2))[:, None, None]
            a = a - a.mean((1, 2))[:, None, None]
            rows = radial_spectrum(p, a, dx_km=st.dx_km)
            fine = [r for r in rows if r["wavelength_km"] <= 60]

            per_seed["own_rmse"].append(r_own)
            per_seed["aorc_rmse"].append(r_aorc)
            per_seed["own_rmse_corr"].append(r_own_c)
            per_seed["aorc_rmse_corr"].append(r_aorc_c)
            per_seed["eff_res"].append(effective_resolution(rows) or float("nan"))
            per_seed["amp60"].append(float(np.mean([r["amplitude_ratio"] for r in fine])))
            per_seed["coh60"].append(float(np.mean([r["coherence"] for r in fine])))
            print(f"  {name}: own {r_own:.4f} -> {r_own_c:.4f} | "
                  f"vs AORC {r_aorc:.4f} -> {r_aorc_c:.4f} | "
                  f"eff res {per_seed['eff_res'][-1]:.0f} km", flush=True)

        if present:
            results[arm] = {"product": product, "store": store_name,
                            "seeds": present, **per_seed}

    (out / "raw.json").write_text(json.dumps(results, indent=2))
    write_report(out / "report.md", results, ckpt, k)
    print(f"\nwrote {out/'report.md'} and {out/'raw.json'}")
    return results


def _stat(v):
    v = [x for x in v if np.isfinite(x)]
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


def write_report(path, R, ckpt, k):
    L = [f"# Target x interaction channels ({ckpt}.pt, k={k} offset correction)\n",
         "Both truth products are reported separately. A cross-target claim is",
         "only made through a common judge — see the module docstring.\n"]

    for product in ("ERA5-Land", "AORC"):
        arms = [a for a in R if R[a]["product"] == product]
        if not arms:
            continue
        L += [f"\n## Trained on {product} — scored against {product}\n",
              "Within-target only. Do NOT compare these rows to the other section.\n",
              "| arm | seeds | RMSE | +offset corr |", "|---|---|---|---|"]
        for a in arms:
            m, s = _stat(R[a]["own_rmse"]); mc, sc = _stat(R[a]["own_rmse_corr"])
            L.append(f"| `{a}` | {len(R[a]['seeds'])} | {m:.4f} ± {s:.4f} "
                     f"| {mc:.4f} ± {sc:.4f} |")
        dyn = [a for a in arms if a.startswith("dyn")]
        ix = [a for a in arms if a.startswith("ix")]
        if dyn and ix:
            t, dof, d = welch(R[dyn[0]]["own_rmse"], R[ix[0]]["own_rmse"])
            verdict = ("below the between-seed noise floor" if abs(d) < NOISE["rmse"]
                       else f"{abs(d) / NOISE['rmse']:.1f}x the noise floor")
            L.append(f"\n**Interaction channels, {product}:** ΔRMSE {d:+.4f} "
                     f"(Welch t = {t:+.2f}, dof {dof:.1f}) — {verdict}.")

    L += ["\n## Every arm against AORC — the common judge\n",
          "The only table in which an ERA5-trained arm and an AORC-trained arm",
          "may be compared directly.\n",
          "| arm | trained on | RMSE vs AORC | +offset corr |", "|---|---|---|---|"]
    for a in R:
        m, s = _stat(R[a]["aorc_rmse"]); mc, sc = _stat(R[a]["aorc_rmse_corr"])
        L.append(f"| `{a}` | {R[a]['product']} | {m:.4f} ± {s:.4f} "
                 f"| {mc:.4f} ± {sc:.4f} |")
    if "dynera" in R and "dynaorc" in R:
        t, dof, d = welch(R["dynera"]["aorc_rmse"], R["dynaorc"]["aorc_rmse"])
        L.append(f"\n**Target swap (dyn channels):** ΔRMSE vs AORC {d:+.4f} "
                 f"(Welch t = {t:+.2f}, dof {dof:.1f}).")

    L += ["\n## Effective resolution and fine-scale skill\n",
          "Measured against AORC for every arm, including the ERA5-trained ones:",
          "scoring an ERA5-trained model on reproducing ERA5-Land's own fine",
          "scales would hide exactly the confound this experiment tests.",
          "Published baseline is 165 km on a 10 km grid.\n",
          "| arm | trained on | eff. res. (km) | amplitude <60 km | coherence <60 km |",
          "|---|---|---|---|---|"]
    for a in R:
        e, es = _stat(R[a]["eff_res"]); am, _ = _stat(R[a]["amp60"])
        ch, chs = _stat(R[a]["coh60"])
        L.append(f"| `{a}` | {R[a]['product']} | {e:.0f} ± {es:.0f} "
                 f"| {am:.3f} | {ch:.3f} ± {chs:.3f} |")

    L += ["\n---\n",
          f"Between-seed noise floors used: RMSE {NOISE['rmse']}, "
          f"resid corr {NOISE['resid_corr']}. A difference smaller than the",
          "floor is not a result.\n"]
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="outputs/interaction")
    ap.add_argument("--ckpt", default="last", choices=["best", "last"])
    ap.add_argument("-k", type=int, default=6)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--arms", nargs="+", default=None,
                    help="subset of arm names; default all four")
    # 20 checkpoints x 1826 days is 36,520 forward passes. On CPU that does not
    # reliably fit a one-hour allocation; on the GPU already attached to the
    # job it is minutes.
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    arms = ARMS if not a.arms else [x for x in ARMS if x[0] in a.arms]
    main(a.out, a.ckpt, a.k, a.seeds, arms, a.device)
