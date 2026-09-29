/**
 * The two colour ramps from `evaluation/mapping.py`, transcribed.
 *
 * They are the same stops the published PNG figures use. A demo that showed
 * the same fields in a different palette would invite the reader to compare
 * two things that only look different.
 */
export type RGB = readonly [number, number, number];

export const DIVERGING: RGB[] = [
  [42, 120, 214],
  [158, 197, 244],
  [240, 239, 236],
  [240, 163, 162],
  [227, 73, 72],
];

export const SEQUENTIAL: RGB[] = [
  [232, 241, 253],
  [134, 182, 239],
  [42, 120, 214],
  [24, 79, 149],
  [13, 54, 107],
];

export function sample(stops: RGB[], t: number): RGB {
  const u = t <= 0 ? 0 : t >= 1 ? 1 : t;
  const n = stops.length - 1;
  const k = Math.min(n - 1, Math.floor(u * n));
  const f = u * n - k;
  const a = stops[k];
  const b = stops[k + 1];
  return [
    a[0] + (b[0] - a[0]) * f,
    a[1] + (b[1] - a[1]) * f,
    a[2] + (b[2] - a[2]) * f,
  ];
}

export function gradientCss(stops: RGB[]): string {
  const parts = stops.map(
    (s, i) => `rgb(${s[0]},${s[1]},${s[2]}) ${(i / (stops.length - 1)) * 100}%`,
  );
  return `linear-gradient(90deg,${parts.join(",")})`;
}

/** Read a CSS custom property off the document root. */
export function token(name: string): string {
  if (typeof window === "undefined") return "#000";
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export function hexToRgb(hex: string): RGB {
  const h = hex.replace("#", "").trim();
  const full =
    h.length === 3
      ? h
          .split("")
          .map((c) => c + c)
          .join("")
      : h;
  return [
    parseInt(full.slice(0, 2), 16),
    parseInt(full.slice(2, 4), 16),
    parseInt(full.slice(4, 6), 16),
  ];
}
