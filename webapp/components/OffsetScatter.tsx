"use client";

import { useCallback, useEffect, useRef } from "react";
import { token } from "@/lib/color";
import { prepare } from "@/lib/grid";
import type { Manifest, Method } from "@/lib/types";

interface Props {
  manifest: Manifest;
  method: Method;
  dayIndex: number;
}

/**
 * Estimated offset against the true one, every test day.
 *
 * This is the panel that answers "why does this work at all". Both estimators
 * are plotted together: the single global mean tracks the truth loosely
 * (r 0.68), the 36-block ridge tracks it tightly (r 0.95), and neither ever
 * reads held-out truth. The current day is circled so scrubbing the year moves
 * a point you can follow.
 */
export default function OffsetScatter({ manifest, method, dayIndex }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const { true: truth, est, glob } = manifest.offsets;

  const draw = useCallback(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const height = Math.min(300, Math.max(210, canvas.clientWidth || 300));
    const { ctx, w, h } = prepare(canvas, height);

    let lim = 0;
    for (let i = 0; i < truth.length; i++) {
      lim = Math.max(lim, Math.abs(truth[i]), Math.abs(est[i]), Math.abs(glob[i]));
    }
    lim *= 1.05;

    const pad = 26;
    const size = Math.min(w, h) - pad - 8;
    const X = (v: number) => pad + ((v + lim) / (2 * lim)) * size;
    const Y = (v: number) => h - pad - ((v + lim) / (2 * lim)) * size;

    ctx.strokeStyle = token("--line");
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, h - pad - size, size, size);

    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = token("--ink3");
    ctx.beginPath();
    ctx.moveTo(X(-lim), Y(-lim));
    ctx.lineTo(X(lim), Y(lim));
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = token("--ink3");
    ctx.globalAlpha = 0.3;
    for (let i = 0; i < truth.length; i++) {
      ctx.beginPath();
      ctx.arc(X(glob[i]), Y(truth[i]), 1.7, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = token("--bar-blue");
    ctx.globalAlpha = 0.62;
    for (let i = 0; i < truth.length; i++) {
      ctx.beginPath();
      ctx.arc(X(est[i]), Y(truth[i]), 1.9, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    const xv = method.key ? manifest.offsets[method.key][dayIndex] : 0;
    ctx.strokeStyle = token("--accent");
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(X(xv), Y(truth[dayIndex]), 5.2, 0, Math.PI * 2);
    ctx.stroke();

    const mono = token("--font-mono");
    ctx.fillStyle = token("--ink3");
    ctx.font = `10px ${mono}`;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText("estimated  →", pad + size / 2, h - 5);
    ctx.save();
    ctx.translate(10, h - pad - size / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = "top";
    ctx.fillText("true  →", 0, 0);
    ctx.restore();

  }, [truth, est, glob, method, dayIndex, manifest]);

  useEffect(() => {
    draw();
  }, [draw]);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => draw());
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [draw]);

  return <canvas ref={ref} className="chart" />;
}
