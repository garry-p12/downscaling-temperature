# Offset Correction Bench

An interactive view of §5.7 of the downscaling project: half the held-out error
is a single number per day, and this shows what happens when you estimate that
number from the training region and subtract it.

```bash
npm install
npm run dev          # http://localhost:3000
```

The app needs `public/data/{manifest.json,fields.bin}`. They are build products,
not source — regenerate them from the repository root:

```bash
python scripts/export_offset_demo.py --target webapp            # of0_s1337
python scripts/export_offset_demo.py --target webapp --arch v2_landcov_s1337
python scripts/export_offset_demo.py --target webapp --stride 1 # all 365 days, 7 MB
```

That script reads a checkpoint under `checkpoints/`, runs it over every day in
the store, fits the k=6 ridge estimator on the training split only, and writes
the result. Nothing is computed in the browser except `error − offset`.

## What is where

Three views behind a bottom tab bar, each reachable by URL hash — `#day`,
`#year`, `#budget`.

| path | role |
|---|---|
| `app/page.tsx` | Server Component. Reads `manifest.json` with `fs`, so the numbers exist before any client JavaScript runs. |
| `components/Shell.tsx` | Tab state, hash routing, and the state the tabs share — the estimator choice carries from the day view to the year view. |
| `components/DayView.tsx` | Two error maps on one day, the estimator picker, the scrubber, the offset scatter. |
| `components/YearView.tsx` | Per-cell RMSE before/after for the selected estimator, the RMSE ladder, the leave-one-region-out table. |
| `components/BudgetView.tsx` | The three-way error budget, toggling between the task residual and what the trained model left. |
| `components/FieldMap.tsx` | One 93×100 field on a canvas, with the true lat/lon graticule and the held-out box. |
| `components/Info.tsx` | The circled "i". All the explanatory prose lives behind these so the numbers have room. |
| `lib/useFields.ts` | Fetches `fields.bin` once and takes typed-array views over it — no copy, no base64. |
| `lib/grid.ts` | Graticule tracing. The store is EPSG:5070 Albers, so lines of constant latitude curve; these are interpolated from the shipped lat/lon fields rather than drawn as a straight grid. |
| `lib/color.ts` | The two ramps from `evaluation/mapping.py`, so this page and the published PNG figures render the same fields in the same colours. |

## Design

Single light theme, committed to deliberately — every colour is painted
explicitly, so the page holds whatever ground the browser puts behind it.

**No web fonts.** Georgia for display, the system sans stack for text, the
system mono stack for every figure. Nothing to download, nothing to fall back
from, no flash of unstyled text.

**The orange is not decoration.** `#cf5f2e` is the holdout box's orange in
every figure in the repository. Same object, same colour.

## Things worth knowing before changing it

**The colour range defaults to the narrowest option on purpose.** The domain-wide
99.7th percentile of |error| is about 3.1 °C, and on that scale the ~0.36 °C
offset being corrected is a shade of the same pink — the page would demonstrate
nothing. The default is the 70th percentile over the held-out box (±0.80 °C);
the wider presets are there for anyone who wants the tail back.

**The scrubber track is scaled on the 98th percentile, not the maximum.** One
−2.28 °C day was otherwise flattening every other day onto the centreline. Days
beyond the range clamp to the edge.

**Sign convention.** The store holds `truth − model`. The maps display
`model − truth`, so red reads as "too warm here", matching the paper figures.
The offset is reported with the same sign as the correction added to the
prediction, so `true offset` and `applied` always point the same way.

**Data quantisation.** 8 bits over ±3.1 °C, about 0.03 °C per step — roughly an
order of magnitude finer than the 0.28 °C RMSE effect being shown. lat/lon are
16-bit and dequantised into `Float32Array`s at load because the graticule tracer
reads them thousands of times per redraw.

**The uint16 fields come first in `fields.bin`** so a `Uint16Array` view over
the fetched `ArrayBuffer` is 2-byte aligned. Reordering the layout in the
exporter without keeping that property will throw in the browser.

## Numbers this build shows

Checkpoint `of0_s1337`, 365 held-out test days in 2023, over the 1,295 land
cells inside the Austin box:

| estimator | year RMSE (°C) |
|---|---|
| interpolation only | 0.890 |
| model | 0.831 |
| + one global mean | 0.688 |
| + k = 6 sub-regions | **0.547** (−34.2 %, 91 % of oracle) |
| oracle offset | 0.517 |

Mean error collapses from −0.265 to +0.000 °C. Across the five leave-one-region-out
boxes the estimator recovers 82 % ± 5 of the oracle.

## Deploying

The site is a pure static export — `app/page.tsx` is `force-static`, there are
no API routes, no server actions and no dynamic params — so there is nothing
for a serverless function to do.

```bash
npm run build:static     # -> out/, 3.1 MB
npm run preview:static   # serve it locally on :4321
```

**Netlify.** `netlify.toml` is committed and sets everything: base directory
`webapp` (this app lives inside a Python research repository, so npm must not
run from the repo root), `npm run build:static`, publish `out`. Connect the
repo and it builds with no plugin and no functions. `public/_headers` carries
the same cache rules for hosts that read that instead (Cloudflare Pages,
Netlify Drop).

Netlify's build needs `public/data/*` to exist in the repo, which is why those
two files are tracked — the build machine has neither the zarr store nor the
checkpoints, so it cannot run the exporter itself.

Any other static host works the same way: drop `out/` on it. The only thing
that matters is that `/data/fields.bin` is served with its bytes intact and,
ideally, the cache headers above.

## Note on AGENTS.md / CLAUDE.md

`next dev` writes both files into this directory itself and re-creates them if
deleted. They are Next's own doing, not part of this project.
