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
6. **The loss was part of the problem.** Trained under a pixel loss the output is
   16× smoother than its grid — effective resolution 165 km at 10 km (§5.5).
   Splitting the loss along the §5.4 budget, so the model stops being charged
   for a daily bias it cannot fix, gains **+0.0415 residual correlation at 4.5×
   seed noise** (§5.6). It buys spatial structure, not fine-scale phase: the
   remaining deficit looks like an information limit.
7. **Acting on the error budget beats modelling harder.** The daily bias is not
   one number but a smooth spatial FIELD. Regressing it on a 6x6 grid of
   training-region block means — using no holdout truth at all — cuts RMSE
   **30.4% across 8 checkpoints**, which is **89% of the oracle** (§5.7). A
   single domain mean gets only 38%. This is by far the largest effect in the
   project, and it comes from arithmetic on the error budget rather than from
   any model change.
8. **A linear control says capacity is not the constraint.** A 2.2k-parameter
   affine model reaches within 4.7× seed noise of DeepSD overall, and *beats* it
   below 60 km (coherence 0.156 vs 0.088, 11.5× noise): the deep model
   manufactures fine-scale structure less correlated with truth than a linear
   one does. Nonlinearity earns only +0.010 coherence, in the 60–165 km band.

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

### 5.5 Effective resolution — the field is 16× smoother than its grid

None of the metrics above reports whether the output has the right *amount* of
fine-scale variance. It does not. Measured over the holdout on 365 test days
with a windowed 2D DFT (`python -m evaluation.spectral`):

| Wavelength | Amplitude ratio | Coherence |
|---|---|---|
| 495 km | 0.958 | 0.985 |
| 165 km | 0.800 | 0.568 |
| 99 km | **0.629** | 0.408 |
| 55 km | **0.472** | 0.247 |
| 38 km | **0.421** | 0.108 |

Amplitude ratio is `sqrt(PSD_model / PSD_truth)`; 1.0 means correct variance.
Taking the standard threshold of `sqrt(0.75)` — a quarter of the per-wavenumber
energy lost — **effective resolution is ~165 km on a 10 km grid.**

This is not a bug, it is what the loss asks for. Under a pixel loss, if a scale
has correlation ρ < 1 the loss-optimal amplitude is ρ, so the model minimises
its loss by *shrinking* what it cannot predict rather than predicting it
(Subich et al. 2025, ICML, arXiv:2501.19374). The measured amplitude ratio
tracks the measured coherence down the whole column, which is that prediction.
L1 blurs less than MSE but still blurs — the paper's own ablation (appendix B.5)
finds the same.

Two consequences worth stating:

- **It caps the covariate results.** §5.2 attributes the null results to static
  covariates reaching only the 2.8% spatial slice. That holds, but the loss was
  *also* suppressing the fine-scale amplitude those metrics measure, so some of
  the structural headroom was never on offer to any channel set.
- **V0's apparent sharpness is noise.** Run the same diagnostic on `v0_base`
  and the amplitude ratio rises to 2.0–6.7 below 30 km — with coherence at 0.0
  to *negative*. That is uncorrelated noise, not recovered signal, and it is
  consistent with V2 winning on SSIM: V2 is smoother but clean.

`training/losses.py` implements AMSE, the paper's fix, adapted from spherical
harmonics to a windowed 2D DFT for this limited-area grid (`--pixel-loss amse`).
Whether it helps *this* variable is an open question and deliberately not
assumed: the paper's own appendix B.3 finds essentially no effect on 2 m
temperature, which it attributes to elevation anchoring the fine scales — the
exact setup used here. The counter-argument is that their 2 m temperature was
not measurably smooth, and ours is.

Note before reading any result from it: **sharpening should make RMSE slightly
worse.** Loss-optimal smoothing minimises RMSE by construction. Given §5.4, that
is a trade worth making, but it is pre-registered here rather than discovered
afterwards.

### 5.6 Splitting the loss along the error budget — the largest effect measured

§5.4 says half the holdout residual is a spatially uniform daily offset that no
spatial model can fix, and that it is separately predictable from its own lag-1.
The loss charged for it anyway, so gradient that could have bought spatial
structure was spent chasing a bias. `training/losses.py decomposed_loss` makes
the split explicit:

```
loss = spatial(pred − μ_p, target − μ_t) + offset_weight · |μ_p − μ_t|
```

Three arms, DeepSD, V2 channels, five seeds each. The full-field control is not
optional: without it, the offset arm would confound two changes at once.

| Arm | RMSE °C | SSIM | resid corr |
|---|---|---|---|
| `v2_landcov` (published baseline, patched) | 0.7834 ±0.0122 | 0.8007 ±0.0026 | 0.5003 ±0.0061 |
| `of1` — full fields, offset_weight 1.0 | 0.7906 ±0.0056 | 0.7959 ±0.0021 | 0.4846 ±0.0052 |
| `of0` — full fields, offset_weight 0.0 | 0.7869 ±0.0210 | **0.8028** ±0.0008 | **0.5261** ±0.0055 |

| Contrast | RMSE | SSIM | resid corr |
|---|---|---|---|
| full fields alone (`of1` − base) | +0.0072 | −0.0047 | **−0.0157** |
| offset removed (`of0` − `of1`) | −0.0038 | +0.0069 | **+0.0415** |

**Removing the offset penalty gains +0.0415 residual correlation at 4.5× the
between-seed sd** (0.0092) — the largest single effect in this project, ahead of
Restormer's +0.0363 at 3.6× (§5.3). Against the published baseline directly it
is +0.0258, still 2.8× noise. Full-field training *by itself* slightly hurt,
which is exactly why the control arm exists: quoting `of0` against the baseline
alone would misattribute the mechanism.

**A pre-registered prediction that was wrong.** §5.5 committed to `of0`'s raw
RMSE getting worse, since it has no offset term by construction. It did not —
0.7869 against 0.7834, inside the seed sd. The reason is that `coarse_tmp`
carries the offset in the *input*: removing the penalty stopped the model being
charged for the bias without stopping it reproducing one (bias 0.0983 vs 0.1174).
The capacity was freed at no RMSE cost.

**The spectra say the gain is not fine-scale phase.** Mean over the 22–99 km band:

| | baseline | `of1` | `of0` |
|---|---|---|---|
| amplitude ratio | 0.552 | 0.641 | **0.658** |
| coherence | 0.144 | 0.129 | 0.131 |

Amplitude improves substantially — at 22 km it goes 0.638 → 0.961 — while
coherence is flat (−0.0135, 1.1× noise). Effective resolution stays 165 km for
all three arms. At 22–26 km `of0` reaches amplitude ~0.96 at coherence ~0.02:
matching variance while uncorrelated is realistic-looking noise, not skill, and
is the "noise-based effective resolution" the source paper warns about. So the
result supports the fine-scale deficit being an **information limit rather than
a loss limit** — a deterministic model whose only time-varying input is a smooth
interpolated field has no source for correct-amplitude fine-scale anomalies.

**Two caveats.** `of0` is less stable across seeds than the baseline: RMSE sd
0.0210 vs 0.0122, p95 bias sd 0.2437 vs 0.0459. The mean improved, the variance
got worse. And a single-seed pilot of this work pointed the *opposite* way
(coherence up, amplitude flat); five seeds reversed it, which is §5.1's lesson
applying to our own method work.

Reproduce with `slurm/vista_ofs_of1.sh` and `slurm/vista_ofs_of0.sh`, then
`slurm/vista_ofs_eval.sh`. The decomposition requires full fields: the offset has
sd 0.463 °C while the gap between a 48×48 patch mean and the true domain mean has
sd 0.376, so splitting on a patch mean removes real structure rather than bias.
`--offset-weight` refuses to run without `--patch-size 0` for that reason.

### 5.7 Correcting the offset — the largest RMSE gain in the project

§5.6 stopped the model being *charged* for the uniform daily bias. This estimates
that bias and subtracts it. `evaluation/offset_correction.py`.

**The estimators are separated by what they assume, because that is the whole
argument.** §5.4 quoted an AR(1) fit on the holdout's own offset history, which
silently requires yesterday's truth *in the region being predicted* — strictly
more information than an uncorrected model gets, so it is not a fair head-to-head.
The alternative uses the 50.8% finding directly: if the offset is spatially
uniform, measure it over the **training region on the same day** and apply it to
the holdout. That uses no holdout truth at all and needs no time-series model.

| Arm | seeds | uncorrected | **spatial** | gain | ar(1) | *oracle* | offset corr |
|---|---|---|---|---|---|---|---|
| `v0_base` | 1 | 0.8024 | 0.7085 | −11.7% | 0.7340 | *0.5376* | 0.601 |
| `v2_landcov` | 3 | 0.8040 | 0.7144 | −11.1% | 0.7380 | *0.5443* | 0.594 |
| `of1` | 1 | 0.7966 | 0.7091 | −11.0% | 0.7288 | *0.5343* | 0.590 |
| **`of0`** | 5 | 0.8065 | **0.6867** | **−14.8%** | 0.7204 | *0.5163* | **0.678** |

Across all 10 checkpoints: **−13.0% RMSE, sd 2.1** — and a better estimator of
the same quantity more than doubles that, see below.

**The assumption-free estimator beats AR(1) on every single checkpoint**, using
strictly less information. That reverses §5.4's framing: the offset is better
recovered from *elsewhere in space on the same day* than from *the same place
yesterday*. Spatial sharing beats temporal persistence here.

**§5.6 and this correction compound.** `of0` — the arm trained with the offset
penalty removed — starts worst uncorrected (0.8065) and finishes best corrected
(0.6867), with the highest train↔holdout offset correlation (0.678 against
~0.59). Training the model to stop fitting the daily bias left a residual bias
that is *more spatially uniform*, and therefore more correctable from elsewhere.
The two interventions were designed independently and reinforce each other.

For scale: covariate engineering moved RMSE by ~0 (§5.2), architecture by ~0
(§5.1), and the §5.6 loss moved structure but not RMSE. This moves RMSE by
**13%** with no new data, no retraining and no extra information.

**The bias is a FIELD, not a number — and that is worth most of the ceiling.**
A single domain mean throws away the spatial structure of the bias. Splitting
the training region into a k x k grid of blocks and regressing the holdout
offset on those block means (ridge, fit on the train split only, still using no
holdout truth of any kind) recovers most of the gap. Mean over 8 checkpoints:

| Estimator | features | gain | % of oracle |
|---|---|---|---|
| single global mean | 1 | −12.9% | 38% |
| sub-regions k=3 | 9 | −25.5% | 75% |
| **sub-regions k=6** | **36** | **−30.4%** (sd 1.8) | **89%** |
| *oracle (true holdout offset)* | — | *−34.1%* | *100%* |

The per-checkpoint "% of oracle" runs 88–90% on every one of the eight, across
two training recipes — this is not a single-checkpoint artefact. A k-sweep puts
the knee at k=6: monotonic improvement to k=6, a plateau through k=12 (−35.2%
on `of0_s1337`), then degradation at k=16 (−32.7%) as the blocks start fitting
train-split noise. Adding lag-1, season and coarse-field state **on top** of the
sub-regions made it worse (−28.8% vs −29.5% at k=3): once spatial structure is
in the model those features are redundant and only cost degrees of freedom.

This revises §5.4's framing. "50.8% spatially uniform daily offset" understates
what is there: it is a smooth spatial bias field, and treating it as one number
was discarding most of the recoverable signal. What remains after k=6 is ~11%
of the offset variance — the genuinely unpredictable part.

**Caveats.** All numbers use `last.pt`, so the uncorrected baselines sit above
§5.2's `best.pt` figures; the comparison within this table is like-for-like. A
single-seed reading of `of0` gave −17.1%; five seeds give −14.8%, so the pilot
overstated it — the same lesson as §5.1 and §5.6. `v0_base` and `of1` are
single-seed and included for spread, not as estimates.


### 5.8 A linear control — is nonlinear capacity the constraint?

Before designing a new architecture, a prior question: how much of DeepSD's
skill actually needs its nonlinearity? `models/linear_probe.py` is the control —
a residual corrector through a single 13×13 conv over all 13 channels with **no
activation anywhere**, initialised to zero so epoch 0 *is* the interpolation
baseline. **2,198 parameters against DeepSD's 210,000.** Identical protocol to
the published `v2_landcov` arm; the only difference is the nonlinearity.

| | linear (5 seeds) | DeepSD (3 seeds) | Welch t |
|---|---|---|---|
| RMSE | 0.8244 ±0.0262 | **0.7841** ±0.0133 | −2.9 |
| SSIM | 0.7973 ±0.0019 | 0.7988 ±0.0038 | +0.6 (tied) |
| resid corr | 0.4591 ±0.0154 | **0.4951** ±0.0085 | +4.3 |
| p95 bias | +0.5160 ±0.1177 | **−0.0702** ±0.0654 | −9.1 |

Overall the nonlinearity earns its place — but the per-scale picture inverts.

| Band | linear | DeepSD | Δ coherence | Welch t | |
|---|---|---|---|---|---|
| 165–495 km | 0.984 | 0.984 | +0.001 | +3.1 | statistically real, practically nil |
| **60–165 km** | 0.414 | **0.424** | **+0.010** | +4.3 | the only band it helps |
| **20–60 km** | **0.156** | 0.088 | **−0.068** | **−9.3** | **linear is better** |

**Below 60 km the deep model is worse than an affine one.** The amplitudes say
why: in that band DeepSD produces amplitude ratio 0.536 against linear's 0.163 —
more than three times the variance — at *lower* coherence. It is not resolving
fine structure, it is manufacturing plausible-looking structure that is less
correlated with truth than the little a linear model emits.

Note the coarse band as a caution on reading significance: t = +3.1 on a
difference of +0.001. The seed spreads are so small there that a trivial
difference is statistically resolvable. Statistical and practical significance
are not the same thing, and this table contains an example of each.

**What this means for architecture work.** The total value of nonlinearity, in
the only band where it helps, is +0.010 coherence. A new architecture competes
for a slice of that — against §5.7's 13% RMSE, which came from arithmetic on the
error budget rather than from modelling. It also suggests a cheaper experiment
than a new model: if capacity below 60 km is actively harmful, *suppressing*
generation there (a spectral penalty, or low-passing the increment at inference)
should help, and costs a day rather than weeks. That test is untried.

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
   correction. Worth ~12% off the interpolation error immediately (§5.4). §5.6
   is the first half of this and it worked: removing the offset from the LOSS
   bought spatial skill at no RMSE cost. The other half — predicting the offset
   with AR(1) and adding it back at inference — is untested and is now the
   single highest-value open item.
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
