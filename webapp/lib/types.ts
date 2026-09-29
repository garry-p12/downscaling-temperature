/**
 * Shapes written by `scripts/export_offset_demo.py --target webapp`.
 *
 * The split is deliberate. `Manifest` is ~30 kB of JSON that the server reads
 * at request time, so the tables, the budget and every quoted number are in
 * the first HTML response. `Fields` is 2.3 MB of quantised pixels that only
 * the canvases need, fetched once on the client as an ArrayBuffer.
 */

export interface Scale {
  lo: number;
  hi: number;
  lim?: number;
  /** Selectable display half-ranges, from the held-out box's |error| spread. */
  limBox?: number[];
  /** The percentile each entry of `limBox` corresponds to. */
  limBoxQ?: number[];
}

export interface FieldLayout extends Partial<Scale> {
  offset: number;
  length: number;
  dtype: "uint8" | "uint16";
}

export type PerCellName =
  | "pcModel"
  | "pcGlobalmean"
  | "pcCorrected"
  | "pcOracle";

export type FieldName = "lat" | "lon" | "land" | "truth" | "err" | PerCellName;

export interface Meta {
  arch: string;
  ckpt: string;
  k: number;
  nx: number;
  ny: number;
  /** [i0, i1, j0, j1] — the held-out box in grid indices. */
  box: [number, number, number, number];
  landCells: number;
  boxCells: number;
  /** Days with a stored map. */
  nDays: number;
  /** Days in the test split — always 365 here. */
  nTestDays: number;
  stride: number;
  blocks: number;
}

export interface Series {
  true: number[];
  est: number[];
  glob: number[];
}

export interface DailyRmse {
  model: number[];
  corrected: number[];
  globalmean: number[];
  oracle: number[];
}

export interface YearRmse {
  model: number;
  corrected: number;
  globalmean: number;
  oracle: number;
  interp: number;
}

export interface BudgetShares {
  offset: number;
  static: number;
  remainder: number;
}

export interface LoroRow {
  name: string;
  box: [number, number, number, number];
  cells: number;
  model: number;
  corrected: number;
  oracle: number;
  pct: number;
  ofOracle: number;
}

export interface Manifest {
  meta: Meta;
  /** Dates of the days that have a stored map. */
  dates: string[];
  /** All 365 test dates. */
  allDates: string[];
  offsets: Series;
  daily: DailyRmse;
  /** mapDayIndex[m] is the index into the 365-day arrays for map day m. */
  mapDayIndex: number[];
  defaultDay: number;
  year: YearRmse;
  bias: { model: number; corrected: number };
  corr: { glob: number; est: number };
  budget: {
    box: BudgetShares;
    domain: BudgetShares;
    model: BudgetShares;
    totalVar: number;
  };
  loro: LoroRow[];
  layout: Record<FieldName, FieldLayout>;
  binBytes: number;
  /** Content hash of fields.bin, used to cache-bust its URL. */
  version: string;
}

/** Decoded views over fields.bin, plus the dequantisers that go with them. */
export interface Fields {
  lat: Float32Array;
  lon: Float32Array;
  land: Uint8Array;
  truth: Uint8Array;
  err: Uint8Array;
  perCell: Record<PerCellName, Uint8Array>;
  /** Model error in degC at cell `i` on map day `d`, sign-flipped so positive = too warm. */
  errAt: (d: number, i: number) => number;
  truthAt: (d: number, i: number) => number;
  /** Per-cell year RMSE in degC for one estimator, keyed by its `rmseKey`. */
  pcAt: (which: keyof DailyRmse, i: number) => number;
  /** Storage range — the widest value any cell can hold. */
  errLim: number;
  /** Display half-ranges to choose between, narrow first. */
  errLevels: number[];
  errLevelQ: number[];
  pcMax: number;
}

export type MethodId = "none" | "glob" | "est" | "oracle";

export interface Method {
  id: MethodId;
  label: string;
  sub: string;
  /** Key into `offsets`; null means no correction. */
  key: keyof Series | null;
  /** Key into `daily`, for the RMSE this method achieves. */
  rmseKey: keyof DailyRmse;
}

export const METHODS: Method[] = [
  {
    id: "none",
    label: "No correction",
    sub: "leave the model alone",
    key: null,
    rmseKey: "model",
  },
  {
    id: "glob",
    label: "One global mean",
    sub: "one average, taken from the rest of Texas",
    key: "glob",
    rmseKey: "globalmean",
  },
  {
    id: "est",
    label: "k = 6 sub-regions",
    sub: "36 squares, weighted by which ones predict best",
    key: "est",
    rmseKey: "corrected",
  },
  {
    id: "oracle",
    label: "Oracle offset",
    sub: "the real answer — cheating, shown for comparison",
    key: "true",
    rmseKey: "oracle",
  },
];
