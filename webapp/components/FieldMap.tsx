"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { hexToRgb, sample, token, type RGB } from "@/lib/color";
import { drawGraticule, drawHoldoutBox, type GridSpec } from "@/lib/grid";
import type { Fields, Meta } from "@/lib/types";

interface Props {
  fields: Fields;
  meta: Meta;
  /** Value in physical units at cell index `i`. */
  valueAt: (i: number) => number;
  ramp: RGB[];
  lo: number;
  hi: number;
  /** Tooltip body for cell `i`; newlines become separate lines. */
  readout: (i: number) => string;
  /** Bump to force a repaint when a dependency outside `valueAt` changes. */
  revision?: number;
}

/**
 * One 93x100 field, painted a cell at a time into an ImageData and blitted up
 * with smoothing off, so a grid cell stays a visible square rather than being
 * interpolated into a smooth blur. The grid is the resolution of the product;
 * hiding it would misrepresent what the model actually outputs.
 */
export default function FieldMap({
  fields,
  meta,
  valueAt,
  ramp,
  lo,
  hi,
  readout,
  revision = 0,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const bufferRef = useRef<HTMLCanvasElement | null>(null);
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);

  const { nx, ny } = meta;
  const nc = nx * ny;

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    if (!bufferRef.current) {
      const b = document.createElement("canvas");
      b.width = nx;
      b.height = ny;
      bufferRef.current = b;
    }
    const buf = bufferRef.current;
    const bctx = buf.getContext("2d")!;
    const img = bctx.createImageData(nx, ny);
    const px = img.data;
    const nodata = hexToRgb(token("--nodata") || "#e0e0d8");
    const span = hi - lo || 1;

    for (let i = 0; i < nc; i++) {
      const c = fields.land[i] ? sample(ramp, (valueAt(i) - lo) / span) : nodata;
      const o = i * 4;
      px[o] = c[0];
      px[o + 1] = c[1];
      px[o + 2] = c[2];
      px[o + 3] = 255;
    }
    bctx.putImageData(img, 0, 0);

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth || 300;
    const h = Math.round((w * ny) / nx);
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.height = `${h}px`;

    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(buf, 0, 0, w, h);

    const grid: GridSpec = { nx, ny, lat: fields.lat, lon: fields.lon };
    const mono = token("--font-mono") || "monospace";
    drawGraticule(ctx, grid, w, h, token("--ink2") || "#6c6d66", mono,
                  token("--card") || "#ffffff");
    drawHoldoutBox(ctx, meta.box, nx, ny, w, h, token("--accent") || "#eb6834");
  }, [fields, meta, nx, ny, nc, valueAt, ramp, lo, hi]);

  useEffect(() => {
    paint();
  }, [paint, revision]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => paint());
    ro.observe(canvas);

    return () => ro.disconnect();
  }, [paint]);

  const onMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const j = Math.floor(((e.clientX - r.left) / r.width) * nx);
    const i = Math.floor(((e.clientY - r.top) / r.height) * ny);
    if (i < 0 || j < 0 || i >= ny || j >= nx) {
      setTip(null);
      return;
    }
    setTip({
      x: e.clientX - r.left,
      y: e.clientY - r.top,
      text: readout(i * nx + j),
    });
  };

  return (
    <div className="mapwrap">
      <canvas
        ref={canvasRef}
        onMouseMove={onMove}
        onMouseLeave={() => setTip(null)}
        role="img"
        aria-label="Gridded field over south-central Texas, 10 km cells"
      />
      {tip && (
        <div
          className="readout"
          style={{
            left: Math.max(4, Math.min((canvasRef.current?.clientWidth ?? 300) - 140, tip.x + 14)),
            top: Math.max(2, tip.y - 38),
          }}
        >
          {tip.text}
        </div>
      )}
    </div>
  );
}
