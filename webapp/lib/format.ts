export const fixed = (v: number, d = 3) => v.toFixed(d);

export const signed = (v: number, d = 3) => (v >= 0 ? "+" : "") + v.toFixed(d);

export const pct = (from: number, to: number) => ((to - from) / from) * 100;

export const signedPct = (from: number, to: number, d = 1) => {
  const p = pct(from, to);
  return (p >= 0 ? "+" : "") + p.toFixed(d) + "%";
};

/** How much of the oracle's headroom a method captured. */
export const shareOfOracle = (model: number, got: number, oracle: number) =>
  ((got - model) / (oracle - model)) * 100;

export const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

export const sd = (xs: number[]) => {
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
};
