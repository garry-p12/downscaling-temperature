/**
 * Graticule drawing for a grid in a projected CRS.
 *
 * The store is EPSG:5070 Albers, so lines of constant latitude CURVE across
 * the image and a lat/lon rectangle is not an index rectangle. Drawing a
 * regular grid of straight lines would be a lie that happens to look tidy.
 * Instead, for each level, walk the rows (or columns) of the shipped lat/lon
 * fields and interpolate where the field crosses it — the same contours the
 * matplotlib figures draw, computed the same way.
 */

export const LAT_LEVELS = [30, 32, 34];
export const LON_LEVELS = [-102, -100, -98, -96];

export interface GridSpec {
  nx: number;
  ny: number;
  lat: Float32Array;
  lon: Float32Array;
}

/** Where `field` crosses `level` down column 0 (for latitude row ticks). */
export function crossingRow(g: GridSpec, level: number): number | null {
  for (let b = 0; b < g.ny - 1; b++) {
    const v0 = g.lat[b * g.nx];
    const v1 = g.lat[(b + 1) * g.nx];
    if ((v0 - level) * (v1 - level) <= 0 && v0 !== v1) {
      return b + (level - v0) / (v1 - v0);
    }
  }
  return null;
}

/** Where `field` crosses `level` along the bottom row (for longitude ticks). */
export function crossingCol(g: GridSpec, level: number): number | null {
  const row = (g.ny - 1) * g.nx;
  for (let b = 0; b < g.nx - 1; b++) {
    const v0 = g.lon[row + b];
    const v1 = g.lon[row + b + 1];
    if ((v0 - level) * (v1 - level) <= 0 && v0 !== v1) {
      return b + (level - v0) / (v1 - v0);
    }
  }
  return null;
}

function trace(
  ctx: CanvasRenderingContext2D,
  g: GridSpec,
  field: Float32Array,
  level: number,
  horizontal: boolean,
  sx: number,
  sy: number,
) {
  ctx.beginPath();
  let started = false;
  const outer = horizontal ? g.nx : g.ny;
  const inner = horizontal ? g.ny : g.nx;

  for (let a = 0; a < outer; a++) {
    let hit: number | null = null;
    for (let b = 0; b < inner - 1; b++) {
      const i0 = horizontal ? b * g.nx + a : a * g.nx + b;
      const i1 = horizontal ? (b + 1) * g.nx + a : a * g.nx + b + 1;
      const v0 = field[i0];
      const v1 = field[i1];
      if ((v0 - level) * (v1 - level) <= 0 && v0 !== v1) {
        hit = b + (level - v0) / (v1 - v0);
        break;
      }
    }
    if (hit === null) {
      started = false;
      continue;
    }
    const x = horizontal ? a * sx : hit * sx;
    const y = horizontal ? hit * sy : a * sy;
    if (started) ctx.lineTo(x, y);
    else {
      ctx.moveTo(x, y);
      started = true;
    }
  }
  ctx.stroke();
}

export function drawGraticule(
  ctx: CanvasRenderingContext2D,
  g: GridSpec,
  w: number,
  h: number,
  ink: string,
  mono: string,
  halo = "#ffffff",
) {
  const sx = w / g.nx;
  const sy = h / g.ny;

  ctx.save();
  ctx.strokeStyle = ink;
  ctx.globalAlpha = 0.34;
  ctx.lineWidth = 0.6;
  ctx.setLineDash([2, 3]);
  for (const L of LAT_LEVELS) trace(ctx, g, g.lat, L, true, sx, sy);
  for (const L of LON_LEVELS) trace(ctx, g, g.lon, L, false, sx, sy);
  ctx.restore();

  // Labels sit on top of saturated reds and blues, so they get a halo in the
  // card colour. Flat grey text was unreadable over half the map.
  ctx.save();
  ctx.font = `10px ${mono}`;
  ctx.lineWidth = 2.6;
  ctx.lineJoin = "round";
  ctx.strokeStyle = halo;
  ctx.fillStyle = ink;
  const label = (text: string, x: number, y: number) => {
    ctx.strokeText(text, x, y);
    ctx.fillText(text, x, y);
  };
  for (const L of LAT_LEVELS) {
    const r = crossingRow(g, L);
    if (r === null) continue;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    label(`${L}N`, 5, r * sy);
  }
  for (const L of LON_LEVELS) {
    const c = crossingCol(g, L);
    if (c === null) continue;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    label(`${Math.abs(L)}W`, c * sx, h - 4);
  }
  ctx.restore();
}

export function drawHoldoutBox(
  ctx: CanvasRenderingContext2D,
  box: [number, number, number, number],
  nx: number,
  ny: number,
  w: number,
  h: number,
  color: string,
) {
  const [i0, i1, j0, j1] = box;
  const sx = w / nx;
  const sy = h / ny;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.strokeRect(j0 * sx, i0 * sy, (j1 - j0) * sx, (i1 - i0) * sy);
  ctx.restore();
}

/** Set up a canvas for the device pixel ratio and return a CSS-pixel context. */
export function prepare(canvas: HTMLCanvasElement, cssHeight: number) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = canvas.clientWidth || 300;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  canvas.style.height = `${cssHeight}px`;
  const ctx = canvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, cssHeight);
  return { ctx, w, h: cssHeight };
}
