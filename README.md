# Temperature Downscaling: NASA POWER 50 km → 10 km

Statistical downscaling of daily 2 m air temperature from **NASA POWER**
(MERRA-2 native 0.5° × 0.625°, ≈ 50 km) to a **10 km** grid, with **ERA5-Land**
(0.1°) as the training target.

| | |
|---|---|
| **Training domain** | South-Central US, 10° × 8° (27–35°N, 104–94°W) → 93 × 100 cells |
| **Evaluation** | Austin, held out **spatially** — 1295 land cells, zero training gradient |
| **Period** | 2019–2023 daily · train 2019–21 · val 2022 · **test 2023** |
| **Projection** | EPSG:5070 CONUS Albers, 10 km |
| **Scope** | Temperature only |

This is the single reference document for the project. It replaces the earlier
`RESULTS.md`, `REPORT.md`, `SLIDES.md`, `DATASETS_SLIDES.md`,
`DATASETS_VISUAL_GUIDE.md` and `DESIGN_DECISIONS_SLIDES.md`; code comments that
cited those files now cite sections here.

**Primary metrics are SSIM and residual correlation, not RMSE.** A model can win
RMSE by over-smoothing; SSIM cannot be won that way. See §5.4 for why RMSE is
structurally incapable of moving on this task.

---

## 1. Summary

Ten architectures, five covariate sets, and five random seeds per arm, all
evaluated on a region no model ever trained on.

1. **The method works on unseen terrain.** 0.772 °C on the Austin spatial
   holdout against a 1.014 °C interpolation floor — a 1.31× skill ratio, a 23%
   cut in error.
2. **Architecture does not matter for accuracy.** Across 6 architectures × 5
   seeds the between-architecture spread (0.0233 °C) is **1.22×** the
   between-seed sd (0.0191 °C) — a measured tie, not merely overlapping
   intervals. On SSIM the models *do* separate: Swin and Restormer produce
   better-structured fields.
3. **Engineered covariates do not move RMSE either.** Five channel sets, counts
   from 7 to 13, every delta inside a pre-registered 0.0100 °C bar (§5.2).
   Land-cover composition is the one real win, and it shows up on SSIM.
4. **We now know why.** Over the holdout, **50.8% of the residual variance is a
   spatially uniform daily offset** between POWER and ERA5-Land, and only
   **2.8%** is the fixed spatial pattern that any static covariate can address
   (§5.4). Every null result above follows from that.
5. **ERA5-Land is the accuracy ceiling.** It is itself 0.90–0.96 °C from station
   observations — further than the gap between our best and worst model.

---

## 2. Method

**Perfect-prognosis residual correction.** The coarse field is interpolated onto
the target grid *first*, and the network learns the correction. It never changes
resolution: input and output are both 10 km, H and W preserved exactly.

1. **POWER is cubically interpolated** onto the 10 km grid — a genuine bicubic
   via `scipy.RegularGridInterpolator` tensor product, not
   `griddata(method='cubic')`, which builds a Clough–Tocher triangulation over
   scattered points and is both slower and not what the SR literature means.
   Cubic over bilinear because the network learns a residual, and a smoother
   interpolant leaves fewer of its own artefacts to undo.
2. **ERA5-Land is reprojected bilinearly, not aggregated.** Its native 0.1°
   (≈ 11 km lat / 8.7 km lon) already matches the target scale, so this is a
   reprojection between comparable grids. Area-weighted averaging here would
   smooth the truth below its own native resolution and quietly make the task
   easier.
3. **Anomalies, not absolute temperature.** A 15-day-smoothed day-of-year
   climatology, fit on **train years only**, is subtracted from both input and
   target. The network learns fine-scale residual structure instead of
   relearning the seasonal cycle. All metrics are reported in absolute °C with
   the climatology added back.
4. **Land mask from the regridded truth.** ERA5-Land is a land reanalysis; ocean
   cells are excluded from the loss and from every metric.
5. **Loss:** L1 + 0.2·(1 − SSIM), masked to land ∩ outside-holdout. L1 rather
   than MSE because both regress toward the conditional mean but MSE's quadratic
   penalty blurs harder, and fine structure is the product.
6. **Splits are contiguous blocks, never random days.** Daily fields are
   strongly autocorrelated; a random day split leaks.
7. **Spatial holdout by loss masking.** Patches cover the whole domain, but every
   cell inside the Austin box (+1° buffer) is zeroed in the loss mask. Rejecting
   overlapping patches instead is impossible here — the box sits mid-domain
   (rows 36–73 of 93), leaving no room for a 48-px patch above or below.

---

## 3. Data and on-disk formats

Five source formats converge on one array.

| Source | Role | Format | Native resolution |
|---|---|---|---|
| NASA POWER `T2M` | coarse predictor | NetCDF4, ~166 KB/yr | 0.5° × 0.625° |
| ERA5-Land `t2m` | truth | NetCDF4, ~11 MB/file, **Kelvin** | 0.1°, hourly |
| Copernicus DEM | terrain | GeoTIFF, `float32`, EPSG:4326 | 0.00083° ≈ 90 m |
| NLCD | land cover | GeoTIFF, `uint8`, nodata 0 | 0.0019° ≈ 208 m |
| Natural Earth coastline | coastal distance | ESRI Shapefile (vector) | — |

Alternatives wired but not used in the headline: **AORC** (hourly 1 km,
observation-informed, daily mean of hourly is the same statistic as POWER T2M)
and **PRISM** (4 km `tmean` = (tmin+tmax)/2 — a *different* statistic, so it
carries a systematic offset; this is why it was dropped).

**RMSE is not comparable across truth products.** The bilinear floor on the
Colorado domain was 2.209 °C against ERA5-Land and 2.608 °C against AORC. Report
skill ratios against the matching floor, never raw RMSE across the two.

### Two download traps worth knowing

- **POWER must be mosaicked.** The regional endpoint caps at ~4.5° per side, so a
  10° × 8° domain arrives as 9 spatial tiles × 5 years. Stitch with `xr.merge`,
  not `combine_first` — the latter silently yields ~70% NaN with no error.
- **NLCD tiles all declare `nodata=0`.** `merge_arrays` then treats real class
  codes as empty and the mosaic comes back 100% nodata, with training looking
  entirely normal.

---

## 4. The training store

Everything lands in one Zarr v3 store, Zstd-compressed:

```
data_store_sc/super/dataset.zarr
  input    (time=1826, channel_in=21, y=93, x=100)   float32
  target   (time=1826, channel_out=1,  y=93, x=100)  float32
  split    (time=1826,)                              <U5   train|val|test
  clim_tmp, clim_coarse_tmp  (doy=366, y, x)         float32
  x, y                                               float64, projected metres
  attrs: crs=EPSG:5070, res_m=10000.0, anomaly=1
```

**One superset store, subsets selected at load time.** Rather than build a
dataset per experiment, the store holds all 21 candidate channels and each
variant selects a subset via `dataset.use_channels`. Every variant therefore
reads bit-identical data, so a difference in score cannot be an artefact of the
build — a different GDAL tie-break, NaN fill, or resampling nondeterminism.

### The default channel set — `configs/data_sc_v2.yaml` (13 channels)

```
coarse_tmp  dem  coastal_dist  land_mask  doy_sin  doy_cos
lc_water  lc_developed  lc_forest  lc_shrub  lc_herbaceous  lc_cultivated  lc_wetland
```

`coarse_tmp` must stay first — several call sites recover the interpolation
baseline via `in_names.index("coarse_tmp")`.

| Channel | Meaning | Unit / range |
|---|---|---|
| `coarse_tmp` | interpolated POWER, anomaly | °C, ≈ ±4 |
| `dem` | mean elevation | 0–2173 m |
| `coastal_dist` | distance to coastline | 0.9–962 km |
| `land_mask` | where ERA5-Land has truth | 0/1, **unnormalised** |
| `doy_sin`, `doy_cos` | day-of-year, cyclic pair | −1…1 |
| `lc_*` | fraction of cell per NLCD Level-1 group | 0–1, **unnormalised** |

Land-cover fractions are left unnormalised because z-scoring a near-absent class
amplifies its rounding noise into apparent signal. Everything else is z-scored on
training-split statistics persisted to `norm_stats.json`.

### Channels built and stored but deliberately unused

`landcover` (majority-class integer) and `urban_frac` are retained so the V0
baseline stays reproducible. The terrain block — `elev_std`, `slope_mean`, `tpi`,
`northness`, `eastness` — and `lc_barren` are stored so a future test needs no
rebuild. All were tested and rejected (§5.2).

Terrain derivatives are computed at the DEM's **native ~90 m** and area-averaged
to 10 km; computing slope from the already-aggregated field would describe the
large-scale gradient the coarse predictor already implies. `tpi` is the
deliberate exception, computed at 10 km because "is this cell low relative to its
neighbours" is a target-scale question. Aspect is averaged as a *vector* —
a plain mean of angles treats 359° and 1° as opposite — and slope-weighted so
flat cells contribute zero rather than random directions.

---

## 5. Results

All figures: Austin spatial holdout, 365 test days, 1295 land cells, metrics in
physical °C after climatology add-back, 95% intervals from a 500-sample bootstrap
over days.

### 5.1 Architecture does not matter

Six architectures × five seeds, 20 epochs each.

| Arch | mean RMSE | seed sd | min | max |
|---|---|---|---|---|
| restormer | **0.7733** | 0.0251 | 0.7452 | 0.8000 |
| swin | 0.7757 | 0.0103 | 0.7650 | 0.7890 |
| deepsd | 0.7848 | 0.0095 | 0.7764 | 0.8000 |
| esrt | 0.7900 | 0.0191 | 0.7727 | 0.8118 |
| edsr | 0.7967 | 0.0193 | 0.7809 | 0.8288 |
| convnext | 0.8370 | 0.0281 | 0.8122 | 0.8697 |

Excluding ConvNeXt, between-architecture spread is 0.0233 °C against a median
between-seed sd of 0.0191 °C — a ratio of **1.22×**. A single-seed table ranked
DeepSD first; across five seeds it ranks third. **The tie is real; the identity
of the leader was noise.**

### 5.2 Covariate ablation

Five channel sets, five seeds each, one evaluation pass. A decision rule was
fixed **before any run**: a variant wins only if its 5-seed mean beats the
baseline by more than the pooled between-seed sd, floored at 0.01 °C. The pooled
sd came out at 0.0097 °C, so the floor governed.

| Variant | Ch. | RMSE | Δ | SSIM | Δ SSIM | resid | Δ resid | p95 bias |
|---|---|---|---|---|---|---|---|---|
| v0_base | 8 | 0.7817 | — | 0.7923 | — | 0.506 | — | −0.097 |
| v1_terrain | 12 | 0.7879 | +0.0062 | 0.7912 | −0.0011 | 0.493 | **−0.0132** | −0.108 |
| v2_landcov | 13 | 0.7834 | +0.0017 | **0.8007** | **+0.0084** | 0.500 | −0.0060 | −0.051 |
| v3_nocoast | 7 | 0.7903 | +0.0086 | 0.7892 | −0.0031 | 0.481 | **−0.0255** | **+0.074** |
| v4_elevstd | 9 | 0.7896 | +0.0079 | 0.7898 | −0.0025 | 0.487 | **−0.0193** | +0.018 |
| *interpolation* | — | 1.0136 | +0.232 | 0.6863 | −0.106 | 0.000 | — | +0.882 |

Per-metric seed sd: RMSE 0.0097, SSIM 0.0039, residual correlation 0.0092,
p95 bias 0.0721. Bold entries clear it.

**No covariate change moved RMSE.** Per-seed values make it plain — V0 spans
0.767–0.802, V2 spans 0.764–0.797. The same distribution drawn twice.

**Land-cover composition is the one win**, +0.0084 SSIM at 2.2× the noise. The
majority-class channel it replaces was a categorical code (11–95) fed as a
continuous ramp *and* a statistic that discarded everything but the winner:
shrub was the mode in 47.6% of land cells, while developed land was the mode in
under 2% despite reaching 98.6% cover somewhere.

**Terrain derivatives hurt**, tested twice — as a four-channel block and with
`elev_std` alone. Both cost residual correlation. The mechanism is the anomaly
framing: terrain's effect on temperature is largely seasonal-mean structure the
per-cell climatology has already removed from the target.

**`coastal_dist` earns its place invisibly.** Removing it left RMSE unmoved but
cost the most residual correlation of any variant and flipped the hot-tail bias
from −0.097 to +0.074. It is r = 0.740 with `dem`, so elevation absorbs the
accuracy-relevant part while the coast channel carries the tails.

Redundancies measured *before* spending compute: `elev_std` ↔ `slope_mean`
r = 0.869 (slope dropped), `lc_developed` ↔ `urban_frac` r = 0.996 (urban_frac
dropped), `coastal_dist` ↔ `dem` r = 0.740.

The 0.996 figure was measured where NLCD has coverage. Over the full grid it is
**0.910**: NLCD is CONUS-only (limitation 9 below) and the two channels treat the
gap differently — `urban_frac` counts nodata as "not developed" and is hard-zero
south of the border, while `lc_developed` is nearest-neighbour filled and reaches
0.71 there. The redundancy that justified dropping `urban_frac` therefore holds
on 79% of the grid, not all of it. `scripts/make_variant_outputs.py` emits
`nlcd_valid_frac` per cell so this is checkable.

### 5.3 The land-cover gain generalises past DeepSD

Seven attention architectures on the V2 channel set, single seed, against the
5-seed DeepSD reference.

| Model | RMSE | 95% CI | SSIM | resid | params |
|---|---|---|---|---|---|
| restormer | 0.7528 | [0.713, 0.791] | **0.8063** | 0.548 | 23.00 M |
| swin | 0.7703 | [0.731, 0.811] | 0.8062 | 0.528 | 41.45 M |
| esrt | 0.7763 | [0.738, 0.818] | 0.8050 | 0.527 | 0.69 M |
| *deepsd (5 seeds)* | *0.7834* | *sd 0.0136* | *0.8007* | *0.500* | *0.21 M* |
| segformer | 0.7860 | [0.750, 0.822] | 0.7900 | 0.500 | 3.31 M |
| swinir_light | 0.7906 | [0.749, 0.832] | 0.7999 | 0.498 | 0.66 M |
| maxvit | 0.7926 | [0.750, 0.835] | 0.7961 | 0.481 | 1.31 M |
| vit | 0.7944 | [0.755, 0.832] | 0.7993 | 0.496 | 15.21 M |

Restormer, Swin and ESRT all exceed the best SSIM ever recorded under the
majority-class channel (0.8033), so the gain is a property of the **data**, not
of one small model. Every RMSE interval overlaps every other.

Restormer's apparent 0.0307 RMSE edge was **seed luck**: five seeds give 0.7719
vs DeepSD's 0.7834, a gap of 0.57× the pooled noise, with Restormer's range
(0.7400–0.8119) fully containing DeepSD's. The same seed scored 0.7528 in one run
and 0.7400 in another, purely from non-deterministic GPU scheduling. Its
*structural* advantage is real: residual correlation +0.0363 at **3.6×** noise,
the largest single effect measured in the project.

### 5.4 The error budget — why RMSE will not move

Decomposing the residual the model must learn, over the holdout test split, in
anomaly space (total variance 0.6905 °C²):

| Component | Share | sd |
|---|---|---|
| **Spatially uniform daily offset** | **50.8%** | 0.592 °C |
| Time-invariant spatial pattern | **2.8%** | 0.139 °C |
| Space–time remainder | 46.4% | — |

Half the error is *one number per day applied to the whole region* — a
product-level disagreement between POWER and ERA5-Land. No spatial model can
address it. Meanwhile the slice any static covariate could ever reach is 2.8%.

That single fact explains §5.1, §5.2 and §5.3 at once: architecture, parameters
and covariates all compete for 2.8% while half the error is a bias term.

**The offset is partly predictable, but not from anything spatial.**

| Predictor | Out-of-sample R² |
|---|---|
| Coarse field state + season | **−0.227** (worse than the mean) |
| Residual seasonal cycle | 0.005 |
| Fitted AR(1) on the offset itself | **+0.221** |
| AR(3) | +0.235 |

Lag-1 autocorrelation is +0.566 — the signature of synoptic weather persisting
across days. In anomaly space over the holdout, the interpolated field scores
0.8898; subtracting an AR(1)-predicted offset gives 0.7829, and subtracting the
*true* offset gives 0.5829. A two-parameter time-series model recovers more than
any covariate change tested.

---

## 6. Station validation and the ceiling

Every number above is measured against ERA5-Land, which is a **reanalysis, not
observations**. Scored instead against NOAA ISD at two Austin stations inside the
holdout, 20 km apart:

| Method | KAUS RMSE | KATT RMSE |
|---|---|---|
| **ERA5-Land (our target)** | **0.964** | **0.897** |
| best model | 1.039 | 0.987 |
| *interpolated POWER* | *1.334* | *1.185* |

**The improvement is real** — roughly 20% and 17% against data appearing in no
training objective. **But ERA5-Land is the ceiling**: at 0.90–0.96 °C from the
stations, the target is further from reality than the entire spread between our
best and worst model. If you need better than ~1 °C at a point, the lever is the
truth product, not the network.

**Station data cannot rank these models.** The two stations produce *opposite*
orderings — Restormer 9th at one and 1st at the other. The between-station spread
(±0.15 °C and a sign flip in bias) exceeds the between-model spread. Any paper
ranking downscaling models on one station is measuring noise.

---

## 7. What did not work

**Zero-shot transfer failed, and it reshaped the project.** Colorado-trained
models applied directly to Austin were *all worse than doing nothing*:

| Method | RMSE | SSIM |
|---|---|---|
| bilinear (do nothing) | **1.007** | 0.663 |
| restormer | 1.448 | 0.533 |
| swin | 1.653 | 0.369 |
| segformer | 4.664 | 0.330 |

They had learned "add terrain-driven detail" from 1173–3757 m relief and applied
it to a 16–692 m plain. Training on the wide South-Central domain with a proper
spatial holdout is what converted this into the working result in §5.

`configs/data.yaml` (Colorado) and `configs/data_austin.yaml` are retained to
reproduce this, not as active domains.

---

## 8. Limitations

1. **Multi-seed covers 6 of 10 architectures.** segformer, swinir_light, vit and
   maxvit are single-run; their exact ranks should not be trusted.
2. **Two stations, one year, one metro.** Enough to retract a one-station
   conclusion; not enough to establish a positive one.
3. **SSIM is inside the training objective.** Quantified by ablation: DeepSD with
   pure L1 still reaches SSIM 0.7700 versus 0.7897 with the term and 0.6863 for
   interpolation — so **81% of the structural gain is earned by the task**, ~19%
   by optimising the metric. The term also improved RMSE slightly
   (0.7809 → 0.7723), so it is not a structure-versus-accuracy trade-off.
   Across three architectures: deepsd 0.7700/0.7897, esrt 0.7854/0.7965,
   edsr 0.7837/0.7953 (L1-only / with SSIM).
4. **The SSIM in the loss is mis-parameterised.** `training/losses.py` applies
   `data_range=1.0` to z-scored anomalies whose actual range is ≈ 6.6, so the
   stabilising constants are far too small. `training/metrics.py` uses the
   per-sample target range. The reported and optimised SSIM are therefore not the
   same quantity.
5. **One region, one test year.** Austin is n = 1 for spatial generalisation and
   2023 is n = 1 for years.
6. **The skill ratio is measured against a mis-specified floor.**
   `evaluation/holdout_eval.py` adds the *fine* climatology to the model's
   prediction but the *coarse* climatology to the baseline, inflating skill by
   ≈ 1.14×. Comparisons between variants are unaffected (all are inflated
   identically); absolute skill numbers are overstated.
7. **The spatial holdout is a holdout of the loss, not of the data.** Patches
   overlapping the box are still trained on as inputs, and `clim_tmp` is fit
   per-cell at *all* cells including the holdout.
8. **20 epochs, one hyperparameter setting.** MaxViT's best checkpoint was epoch
   2 of 20, suggesting instability rather than convergence.
9. **NLCD is CONUS-only**; ~24% of the domain has `landcover`/`lc_*` imputed by
   nearest neighbour.

---

## 9. Defects found

Each was silent and would have produced plausible-looking wrong results.

| Defect | Consequence if missed |
|---|---|
| **bf16 vs fp32** — autocast is gated on `device.type=="cuda"`, so laptop runs were fp32 and GPU runs bf16 | Swin scored SSIM **0.44 vs 0.89** on the same model and data. A later sweep reproduced it exactly: 0.4627 vs 0.7923, with the baseline scoring *worse than interpolation*. 20 runs discarded |
| **`coastal_dist` silently all-zero** when geopandas is missing | V0 and V3 would have had identical inputs — "removing the coast has no effect" |
| **NLCD tiles merged to 100% nodata** — every tile declared `nodata=0` | Constant land-cover channels; training looks normal |
| **Restart keyed on `best.pt`** — written mid-run whenever validation improves | A run killed mid-training folded into a seed mean as if complete |
| **Training imported rasterio** at module scope | Every cluster run dies on import |
| **Land mask ≠ truth** — mask built on the native grid, but interpolation smears NaN | 63 land cells with NaN climatology |
| **`spatial_corr` not NaN-aware** | Both correlation metrics silently returned NaN once masking existed |
| **Stale checkpoints in globbed paths** | Colorado 4-channel results silently mixed into South-Central tables |

Guards now in place: `data/sanity_check.py` gates training; `check_no_dead_channels`
fails any build with a constant input channel; the restart predicate uses
`last.pt`, written only after the final epoch; `train.py` prints the resolved
precision at startup; checkpoints record their own channel list so evaluation
rebuilds the right input stack.

---

## 10. Compute

| Where | What | Time |
|---|---|---|
| MacBook Air (MPS) | 6 architectures | 10.2 h |
| TACC Vista (GH200) | 4 architectures | **1.1 h** |

~30× faster per model on the GPU. Per-run reference on a GH200: DeepSD ~3.5 min,
Restormer ~31 min, both at 20 epochs.

Cluster notes: `gh-dev` is 20 nodes with a **2 h wall cap and one running job per
user** but starts in seconds; `gh` is 576 nodes but routinely 340 jobs deep. The
`gb` (Grace-Blackwell) partition looks empty because the standard torch build has
no kernels for compute capability 10.0 — jobs fail ~35 s in.

---

## 11. Reproducing

```bash
# credentials: ~/.cdsapirc for ERA5-Land (free CDS account + licence accepted)
export DOWNSCALE_CONFIG_data=configs/data_sc_v2.yaml

python -m data.download_power     --years 2019 2020 2021 2022 2023
python -m data.download_era5_land --years 2019 2020 2021 2022 2023   # --months to split
python -m data.download_covariates
python -m data.build_dataset          # builds all 21 channels
python -m data.sanity_check           # gates the sweep; exits non-zero on hard failures

python -m training.train --arch deepsd --epochs 20 --seed 1337 --precision fp32 \
    --zarr data_store_sc/super/dataset.zarr --run-name deepsd_v2
python -m evaluation.holdout_eval --archs deepsd_v2 --n-boot 500
```

**Always pass `--precision fp32` on GPU** — `configs/train.yaml` commits `bf16`,
and bf16 halves SSIM on this loss (§9).

Cluster: `sbatch slurm/vista_covariate_ablation.sh`. The experiment bodies live in
`slurm/*_body.sh` and are sourced by thin wrappers, so a laptop run and a cluster
run cannot drift apart.

A self-contained smoke test needing no credentials:

```bash
export DOWNSCALE_CONFIG_data=configs/data_synthetic.yaml
python -m data.build_dataset --synthetic --n 64 --size 100
python -m training.train --epochs 60 --zarr data_store_synthetic/dataset.zarr \
                         --run-name synthetic
```

Tests: `pytest` — 122 tests, entirely offline, needing no store, GPU or
credentials. Lint: `ruff check .`

### Inspecting the data

```bash
python power_vis.py <file.nc> --list        # POWER daily T2M: overview, values, map
python xarr.py <era5_dir> --save            # ERA5-Land monthly compare + stitch
python scripts/generate_dataset_gallery.py  # regenerate docs/dataset_gallery/
```

---

## 12. Repository layout

```
configs/     data_base.yaml (shared) + overlays that `extends:` it —
             data_sc_v2.yaml (DEFAULT), data_sc_super.yaml (21-channel build
             manifest), data_southcentral.yaml, data.yaml (Colorado, archived),
             data_austin.yaml (archived), data_synthetic.yaml; model.yaml, train.yaml
common/      config loader (extends + ${var} interpolation), target grid, normalization
data/        download_{power,era5_land,aorc,prism,covariates}.py, covariates.py
             (terrain + land-cover derivation), regrid.py, build_dataset.py, sanity_check.py
models/      model.py (arch registry + load_checkpoint), encoder_swin.py, unet.py,
             deepsd.py, zoo.py
training/    dataset.py (spatial holdout, channel selection), losses.py, metrics.py,
             train.py, run_all.py
evaluation/  holdout_eval.py, station_check.py, export_panels.py, baselines.py,
             tree_baselines.py, evaluate.py, compare.py, transfer_eval.py
tests/       122 offline tests
slurm/       *_body.sh (the experiments) + vista_*.sh (thin environment wrappers)
docs/        dataset_gallery/ — preview images and captured xarray reprs
```

Everything a run produces — data stores, checkpoints, figures, reports, wandb —
is gitignored. Only source, configs and this README are tracked.

---

## 13. Next steps

The evidence points away from the model and toward the target and the bias term.

1. **Dynamic coarse predictors.** Every covariate tested so far is *static*, which
   structurally caps them at the 2.8% slice. POWER already serves `WS2M`, `RH2M`,
   `T2MDEW` and `ALLSKY_SFC_SW_DWN` from the same endpoint with no credentials.
   The daily offset behaves like a synoptic signal; this is the first channel
   class that could touch the 50.8%.
2. **An explicit bias-correction term.** A lagged-offset channel, or post-hoc AR
   correction. Worth ~12% off the interpolation error immediately (§5.4).
3. **Report the error budget as a standard diagnostic**, so spatial skill is not
   buried under a bias term.
4. **Fix the skill floor** (§8.6) — three lines, and it corrects every published
   skill number.
5. **Leave-one-region-out holdouts.** Austin is n = 1; four or five crops of the
   same store cost no new data.
6. **Tmin/Tmax rather than daily mean.** Terrain effects — cold-air pooling,
   aspect-driven heating — live in the diurnal extremes, and averaging over 24 h
   washes them out. The hourly ERA5-Land files are retained, so this needs no new
   download.
7. **AORC as truth.** ~1.33 GB on disk after daily aggregation, ~21 GB of S3
   reads. Note `field_to_target(..., 'conservative')` is effectively
   unimplemented — xesmf is absent and it silently falls back to nearest — so
   this needs a real block-mean first.
