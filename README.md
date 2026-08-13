# Temperature Downscaling Pipeline — NASA POWER 50 km → 10 km

Downscales coarse daily 2 m air temperature from **NASA POWER** (MERRA-2 native
0.5° × 0.625°, ~50 km) to a **10 km** grid, with **ERA5-Land** (0.1°) as the
training target. Trained over the South-Central US with **Austin held out
spatially**, then validated against **NOAA station observations**.

Ten architectures spanning every major attention family are compared against
CNN, per-pixel-ML and interpolation baselines. **SSIM and spatial correlation
are the primary metrics** — RMSE can be won by over-smoothing.

See **RESULTS.md** (kept local — write-ups are gitignored; README is the only
tracked document). Three headlines:

- **0.772 °C on terrain never trained on**, vs a 1.014 °C interpolation floor.
- **Architecture doesn't matter for accuracy** — confirmed by 5 seeds × 6
  architectures: between-model spread is 1.22× the between-seed noise. On SSIM
  the models *do* separate (Swin/Restormer lead).
- **Station data cannot rank these models** — two stations 20 km apart give
  opposite orderings. ERA5-Land itself is ~0.9 °C from the stations.

**Scope: temperature only.** The precipitation heads were removed — they were
never trained and there is no `ppt` channel in the dataset. Checkpoints written
while they existed still load: `models.model.load_checkpoint` drops their
weights. Git history has the code if precip is ever revived.

## Locked decisions (this build)

| Decision | Choice |
|---|---|
| Task | **Temperature only** (`T2M` → 10 km daily mean) |
| Coarse predictor | **NASA POWER** daily `T2M`, 0.5°×0.625° ≈ 55×50 km (no credentials) |
| Downscaling factor | **5×** (50 km → 10 km) |
| Training domain | South-Central US 10°×8° (`configs/data_southcentral.yaml`) |
| Evaluation | **Austin, spatially held out** (1295 cells, zero training gradient) |
| Grid projection | **EPSG:5070** CONUS Albers (equal-area) |
| Ground truth | **ERA5-Land** 0.1° (needs a free CDS key); AORC / PRISM selectable |
| Independent check | **NOAA ISD stations** (`evaluation/station_check.py`) |
| Target space | **Anomalies** vs a train-only day-of-year climatology |
| Primary metrics | **SSIM**, spatial correlation, residual correlation |
| Precip heads | **Disabled** — stubs only |

## Architecture

```
coarse_tmp ──┐  (POWER T2M, CUBIC → 10 km, minus climatology)
dem          │
landcover    │
coastal_dist ├─► one of ten backbones ──────► 10 km temperature ANOMALY
urban_frac   │   (--arch selects; one I/O contract)      + climatology = °C
land_mask    │
doy_sin/cos ─┘
```

| `--arch` | Mechanism | Params | Austin holdout RMSE |
|---|---|---|---|
| `deepsd` | stacked SRCNN + elevation (Vandal 2017) | **0.13 M** | **0.772** |
| `esrt` | efficient hybrid CNN + reduced-token attention | 0.69 M | 0.773 |
| `edsr` | deep residual CNN, no normalization | 4.88 M | 0.778 |
| `restormer` | channel / transposed attention | 23.00 M | 0.779 |
| `segformer` | global attention, spatial-reduction keys | 3.31 M | 0.788 |
| `swinir_light` | flat windowed attention, no hierarchy | 0.66 M | 0.792 |
| `vit` | isotropic ViT, full global attention | 15.17 M | 0.797 |
| `swin` | hierarchical windowed attention | 41.45 M | 0.801 |
| `maxvit` | block (local) + grid (sparse global) attention | 1.31 M | 0.818 |
| `convnext` | modern conv U-Net | 24.18 M | 0.865 |
| *interpolated POWER* | — | — | *1.014* |

Also available: `unet`, `swinir` (full). Nine of ten models above tie within
95% CIs — see RESULTS.md §3.

Every architecture obeys the same contract — `(B,C,H,W) → {"temp": (B,1,H,W)}`
with H, W preserved exactly — so the trainer, loss, metrics and eval are shared
and the comparison is apples-to-apples by construction.

SR-style corrector: the coarse POWER field is bilinearly interpolated onto the
10 km grid and stacked with static covariates as model input. ERA5-Land truth is
**reprojected** (not aggregated) — its native 0.1° already sits at the target
scale, so area-averaging would smooth the truth below its own resolution and
make the task artificially easier. The network's job is the fine-scale
correction the 50 km field cannot carry — mostly terrain-driven at 5×, hence
the DEM channel.

## Layout

```
configs/        data_base.yaml (shared) + per-domain overlays that `extends:` it:
                data.yaml (Colorado), data_southcentral.yaml (ACTIVE),
                data_austin.yaml, data_synthetic.yaml; model.yaml, train.yaml
common/         config loader (extends + ${var}), target-grid, normalization
data/           download_{power,era5_land,aorc,prism,covariates}.py, regrid.py,
                build_dataset.py, sanity_check.py
models/         model.py (arch registry + load_checkpoint), encoder_swin.py,
                unet.py, deepsd.py, zoo.py
training/       dataset.py (spatial holdout), losses.py, metrics.py, train.py, run_all.py
evaluation/     holdout_eval.py, station_check.py, export_panels.py,
                baselines.py, tree_baselines.py, evaluate.py, compare.py
tests/          offline unit tests — no store, no GPU, no network
slurm/          vista_tierb.sh (TACC GH200 batch job)
pyproject.toml  packaging, pytest and ruff config
```

Everything a run produces — data stores, checkpoints, figures, reports, wandb —
is gitignored. Only source, configs and this README are tracked.

## Setup

```bash
conda activate downscale
pip install -r requirements.txt
# Or, for the test suite and linter as well:
pip install -e ".[dev]"
# Area-weighted regridding (recommended, conda-forge only):
conda install -c conda-forge xesmf esmpy
```

```bash
pytest        # offline; needs no dataset, GPU or credentials
ruff check .
```

Credentials:
- **NASA POWER**: none — open REST API, no account.
- **ERA5-Land**: free Copernicus CDS account. Create `~/.cdsapirc` (mode 600):
  ```
  url: https://cds.climate.copernicus.eu/api
  key: <your-key>
  ```
  Then **accept the licence** at the dataset page → Terms of use. Registering is
  not enough; without it every request returns `403 required licences not accepted`.
- **AORC**: none (anonymous S3). **PRISM**: none (free web service; rate-limited).

## Quickstart — synthetic smoke test (no downloads)

Exercises the full model + training + eval path on generated data. The
synthetic coarse field is a 5×5 block-average of the truth, mirroring the real
50 km → 10 km factor:

```bash
export DOWNSCALE_CONFIG_data=configs/data_synthetic.yaml
python -m data.build_dataset --synthetic --n 64 --size 100
python -m training.train --epochs 60 --zarr data_store_synthetic/dataset.zarr \
                         --run-name synthetic
python -m evaluation.evaluate --ckpt checkpoints/synthetic/best.pt
```

`data_synthetic.yaml` exists so this path cannot touch a real store: it writes
to its own `data_store_synthetic/`, and `--zarr` / `--run-name` keep the trained
weights out of the real checkpoint tree. Its channel list matches
`model.in_channels`, which the shipped Colorado config does not.

Only `numpy/scipy/torch/xarray/zarr/matplotlib` are needed for this path —
no netCDF, rasterio, or S3.

## Real run

Extra dependencies beyond the synthetic path — `pip install netcdf4 rasterio
rioxarray s3fs` (netCDF to read POWER, rasterio/rioxarray for DEM/NLCD/PRISM,
s3fs for AORC).

```bash
export DOWNSCALE_CONFIG_data=configs/data_southcentral.yaml   # broad domain

# Phase 0 — acquire + build + VERIFY
python -m data.download_power     --years 2019 2020 2021 2022 2023   # ~10 min
python -m data.download_era5_land --years 2019 2020 2021 2022 2023   # hours; --months to split
python -m data.download_covariates                                   # ~15 min, no auth
python -m data.build_dataset
python -m data.sanity_check          # gates the sweep; exits non-zero on hard failures

# Phase 1 — cheap models first
python -m training.run_all --tier-a --epochs 20    # ~7 h on a laptop

# Phase 2 — evaluate on the SPATIAL HOLDOUT, then against real stations
python -m evaluation.holdout_eval --archs deepsd edsr vit esrt segformer \
    swinir_light swin restormer maxvit convnext --n-boot 500
python -m evaluation.station_check --station KAUS_Austin_Bergstrom --year 2023

# Phase 3 — export to the Umbra surrogate (confirm downstream grid/format)
```

GPU: `sbatch slurm/vista_tierb.sh` on TACC Vista — ~30× faster per model. The
script pins `precision: fp32`; bf16 autocast silently halves SSIM (RESULTS.md §8).

Checkpoints go to `checkpoints/<arch>/` and reports to
`outputs/eval_report_<arch>.json`, so sweeps never clobber each other.

Swap the truth source with `truth_source:` in `configs/data.yaml`
(`era5_land` | `aorc` | `prism`) and run the matching downloader.

**RMSE is not comparable across truth sources.** ERA5-Land is resolution-matched
(0.1°) but is itself a reanalysis product; AORC is observation-informed 1 km but
must be aggregated 10×. Measured on the same method: the bilinear floor is
2.209 °C against ERA5-Land and 2.608 °C against AORC. Report skill ratios
against the matching floor, never raw RMSE across the two.

### Static covariates (handled in `data/download_covariates.py`)

Both auto-fetch with no credentials; `--dem` / `--landcover` override with local
GeoTIFFs, and `--dem-source py3dep` switches to USGS 3DEP 30 m.

- **DEM** — Copernicus GLO-90 from `s3://copernicus-dem-90m` (anonymous),
  1°×1° COG tiles, cached under `data_store/raw/dem/tiles/`, mosaicked and
  clipped. 90 m not 30 m on purpose: target cells are 10 km.
- **Land cover** — NLCD 2021 via the MRLC GeoServer WMS. WMS returns a
  *paletted* raster (indices 1–21 + colormap), **not** class codes, so the
  palette RGB is matched to the official NLCD legend and real codes
  (11…95) are written. All 16 CONUS classes recovered on the default domain.
- Both are fetched with a **0.75° margin** (`COV_PAD`). The target grid is
  built in Albers and snapped outward, and Albers rotates ~6° against lat/lon
  at 105°W, so an exact-bbox fetch leaves ~18% of target cells NaN. Any
  residual NaN is nearest-neighbour filled in `load_statics`, with a warning.
- Regridded to 10 km with `average` (DEM) and `mode` (land cover), not
  `bilinear`/`nearest` — at 90 m → 10 km, point-sampling picks one arbitrary
  ridge or pixel per cell.

### ERA5-Land notes (handled in `data/download_era5_land.py`)

- Requires `~/.cdsapirc` **and** licence acceptance (see Setup). Authentication
  succeeding is not sufficient — the licence is a separate gate.
- One request per **month**; a full year of hourly data times out.
- Hourly → **daily mean in flight**, matching POWER's `T2M` statistic.
- Arrives as `t2m(valid_time, latitude, longitude)` in **Kelvin**; coords and
  units are normalized on read.
- Regridded **bilinearly**, not conservatively — native 0.1° is already at the
  10 km target scale (see Architecture).
- 60 months ≈ 230 MB for a 4°×4° domain with a 0.5° margin.

### AORC notes (handled in `data/download_aorc.py`)

- The bucket root is **not** a Zarr group — one store per year,
  `s3://noaa-nws-aorc-v1-1-1km/<year>.zarr`, opened separately.
- Hourly data is aggregated to a **daily mean in flight** and cast to float32:
  ~53 MB/year written for a 4°×4° domain vs ~16 GB/year raw. `--keep-hourly`
  opts out. Reads still stream every intersecting hourly chunk — measured
  ~10 min per year over home broadband.
- Source encoding (`numcodecs.Zstd`, Zarr v2 chunk grid) is stripped before
  writing, and the bbox subset is rechunked; otherwise the Zarr v3 writer
  raises `Expected a BytesBytesCodec` / `uniform chunk sizes`.
- Years are written to a `.partial` store and renamed on success, and an
  existing store is re-downloaded unless its time span actually **covers** the
  requested window — a short test pull must not masquerade as a full year.
- Temperature arrives in **Kelvin**; `build_dataset` converts.

### NASA POWER service limits (handled in `data/download_power.py`)

- One parameter per regional request → we only need `T2M`.
- Bounding box capped at ~4.5° per side → the padded domain is auto-tiled into
  ≤4° boxes (each grown to ≥2°, the service's lower bound) and re-stitched by
  `open_power`.
- Long windows time out → requests are chunked one calendar year per tile.
- Missing values arrive as `-999` and are converted to NaN on read.

## Weights & Biases

Enabled by default (`train.yaml` → `logging.wandb.enabled: true`, project
`downscaling-temperature`). Log in once with `wandb login`.

Streamed **during training**, every epoch, in physical °C:

| Panel | Meaning |
|---|---|
| `train/loss`, `train/lr` | MSE + 0.2·(1−SSIM), cosine schedule |
| `val/rmse`, `val/mae`, `val/bias`, `val/spatial_corr` | val-split temperature skill |
| `val/bilinear_rmse` | the coarse POWER field scored as-is |
| `val/skill_rmse` | `bilinear / model` — **>1 means the model beats interpolation** |
| `val/resid_corr` | fine-scale skill after the bilinear floor is removed — the honest number, since raw `spatial_corr` sits near 0.99 for free |
| `pred/tmp` | coarse / prediction / truth / \|error\| image panels |

Pushed **after evaluation** with `--wandb`, as a separate `<ckpt>-eval` run
(test split, so it does not overwrite training history): `test/model_temp/*`,
`test/bilinear_temp/*`, `test/lapse_temp/*`, `test/bcsd_temp/*`,
`test/edge_disambiguation/*`, plus a log-log `test/power_spectrum_tmp` figure.

`mode` in `train.yaml` overrides the `WANDB_MODE` env var, since `Tracker`
passes it to `wandb.init` explicitly. To go offline set `mode: offline` there,
or `enabled: false` to turn tracking off entirely.

## Configuration

All knobs live in `configs/`. Shared values live once in `data_base.yaml`;
each domain file `extends:` it and overrides only what makes it that domain.
`${store_root}` is substituted into every `out_dir`, so pointing a domain at a
new directory is a one-line change.

- `data_base.yaml` → grid, temporal splits, source list, channel order. Editing
  here changes **every** domain.
- `data_*.yaml` → `domain` bbox, `store_root`, and (South-Central) the spatial
  `holdout` box and anomaly settings.
- `model.yaml` → Swin dims/depths/heads; `in_channels` must equal
  `len(dataset.input_channels)` (currently **8**) — `tests/test_config.py`
  enforces this.
- `train.yaml` → loss weights, LR schedule, device, checkpoint/output dirs.

Run a different domain without editing anything:

```bash
DOWNSCALE_CONFIG_data=configs/data_southcentral.yaml python -m ...
```

## Evaluation

`evaluation/evaluate.py` reports, on the held-out test split:
- **SSIM** — the primary metric. Structural similarity over local means,
  variance and covariance, per day with that day's truth range as `data_range`.
  Unlike RMSE it cannot be won by blurring (a smoothed field scores ≈0.20).
- Temperature: RMSE, MAE, bias, spatial correlation.
- Residual spatial correlation vs both floors — the discriminating number, since
  raw spatial corr sits near 0.93+ for free even for raw interpolation.
- Edge/interior disambiguation, so a boundary artifact can't masquerade as a
  skill deficit.
- Radially-averaged power spectrum vs truth (over-smoothing check — the main
  failure mode of a deterministic 5× corrector).
- Baselines to beat: bilinear (the coarse channel as-is), **lapse-rate**
  (bilinear + fitted elevation correction — the real floor), and BCSD.

## Notes / open items

- **All nine architectures are statistically tied** (SSIM 0.804–0.832 across
  590× in parameters). Architecture selection is not where the remaining gains
  are — see RESULTS.md §6 for the recommendation (Restormer or SegFormer, not
  Swin) and §8 for what to try next.
- **No spatial holdout yet.** Train and test share grid cells, so these numbers
  do not predict transfer to a new region. Largest open question.
- The neural models receive **no time encoding**, yet day-of-year accounts for
  35% of the tree models' feature importance. Adding `doy_sin`/`doy_cos` input
  channels is the cheapest available improvement.
- Domain size matters more than it did with ERA5: at 0.5°, a 2°×2° box holds
  only ~5×3 coarse cells (verified live). The default was widened to 4°×4°.
- `POWER T2M` is a daily mean with `cell_methods = "time: mean"`, matching the
  AORC hourly→daily-mean truth, so coarse and truth share a definition.
- `netcdf4` (or `h5netcdf`) must be importable to read POWER files —
  `pip install netcdf4`. It is in `requirements.txt` but was missing from the
  `downscale` conda env at the time of writing.
- The 4°×4° domain is only **49×40 cells** at 10 km. `DownscalingModel.forward`
  pads to a multiple of `patch_size` and crops back, because the strided
  patch-embed floors an odd 49 to 24 and the head returns 48 — a silent
  one-row misalignment against the target. Odd grids are the common case.
- `coastal_dist` is zero for inland prototype domains; provide a coastline
  shapefile in `regrid.coastal_distance` for coastal regions.
- Residual-vs-direct prediction, per-season normalization, and spatial holdout
  are left as tunables — see inline comments.
- Prior results against **AORC** truth are archived in `outputs/aorc_truth/`.
  They are a separate experiment, not a baseline for the current numbers.
