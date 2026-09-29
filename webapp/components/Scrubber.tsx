"use client";

import { useCallback, useEffect, useRef } from "react";
import { prepare } from "@/lib/grid";
import { token } from "@/lib/color";
import Info from "./Info";
import Segmented from "./Segmented";
import type { Manifest } from "@/lib/types";

interface Props {
  manifest: Manifest;
  mapDay: number;
  onMapDay: (d: number) => void;
  playing: boolean;
  onTogglePlay: () => void;
  appliedOffset: number;
  methodApplies: boolean;
  /** Selectable display half-ranges for the error maps, narrow first. */
  levels: number[] | null;
  scaleIdx: number;
  onScale: (i: number) => void;
  scaleQ: number;
}

/**
 * The scrubber's track is the real daily-offset series over the test year, so
 * moving through the year is navigating the data rather than an abstract
 * index. All 365 days are drawn even though only every `stride`-th day has a
 * stored map — hiding the days without maps would misrepresent the series.
 */
export default function Scrubber({
  manifest,
  mapDay,
  onMapDay,
  playing,
  onTogglePlay,
  appliedOffset,
  methodApplies,
  levels,
  scaleIdx,
  onScale,
  scaleQ,
}: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const trueOffsets = manifest.offsets.true;
  const dayIndex = manifest.mapDayIndex[mapDay];

  const draw = useCallback(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const { ctx, w, h } = prepare(canvas, 44);
    const n = trueOffsets.length;

    // Robust scale. Using the maximum let a single -2.28 degC day flatten
    // every other day onto the centreline; the p98 keeps the shape of the
    // series and clamps the two or three days that exceed it.
    const sorted = [...trueOffsets].map(Math.abs).sort((a, b) => a - b);
    const lim = (sorted[Math.floor(sorted.length * 0.98)] || 1) * 1.05;
    const y = (v: number) =>
      h / 2 - (Math.max(-lim, Math.min(lim, v)) / lim) * (h / 2 - 4);

    ctx.strokeStyle = token("--track");
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    ctx.strokeStyle = token("--ink3");
    ctx.globalAlpha = 0.75;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = (i / (n - 1)) * w;
      if (i) ctx.lineTo(x, y(trueOffsets[i]));
      else ctx.moveTo(x, y(trueOffsets[i]));
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    const x = (dayIndex / (n - 1)) * w;
    ctx.strokeStyle = token("--accent");
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    ctx.fillStyle = token("--accent");
    ctx.beginPath();
    ctx.arc(x, y(trueOffsets[dayIndex]), 3.4, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = token("--ink3");
    ctx.font = `10px ${token("--font-mono")}`;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(
      `how far off the model was, each day of the year · scale ±${lim.toFixed(1)} °C`,
      2,
      2,
    );
  }, [trueOffsets, dayIndex]);

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

  /** Click the track: jump to the nearest day that actually has a map. */
  const onTrackClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const target = Math.round(
      ((e.clientX - r.left) / r.width) * (trueOffsets.length - 1),
    );
    let best = 0;
    let bestDist = Infinity;
    manifest.mapDayIndex.forEach((v, i) => {
      const d = Math.abs(v - target);
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    });
    onMapDay(best);
  };

  return (
    <div className="card">
      <div className="scrubtop">
        <button
          type="button"
          className="play"
          onClick={onTogglePlay}
          aria-pressed={playing}
        >
          {playing ? "\u275a\u275a PAUSE" : "\u25b6 PLAY YEAR"}
        </button>
        <span className="daylabel">{manifest.dates[mapDay]}</span>
        <span className="dayoff">
          {/*
            Direction in words, not a signed number.

            The store holds `truth - model`, because that is the quantity you
            ADD to the prediction to correct it. The maps display the opposite,
            `model - truth`, so that red reads as "too warm". Printing the raw
            stored number beside the maps put a "+0.437" next to a blue box and
            told the reader the model ran warm when it ran cold. No sign
            convention is self-evident to someone reading a label, so this says
            which way round it is.
          */}
          model ran <b>{Math.abs(trueOffsets[dayIndex]).toFixed(2)} °C</b> too{" "}
          {trueOffsets[dayIndex] >= 0 ? "cold" : "warm"}
          {methodApplies
            ? ` \u00b7 we ${appliedOffset >= 0 ? "warmed" : "cooled"} it by ${Math.abs(appliedOffset).toFixed(2)}`
            : " \u00b7 nothing applied"}
        </span>
      </div>

      <div className="scrubrow">
        <span className="rangelab">range</span>
        <Info label="Colour range">
          <p>
            How big a temperature difference the strongest red and blue stand
            for, in °C.
          </p>
          <p>
            Keep it small (±0.80) and small changes are easy to spot. Make it
            large and everything washes out to pale pink, because a handful of
            extreme spots stretch the scale and flatten everything else.
          </p>
          <p>
            Anything past the limit just shows as the strongest colour. Both maps
            always use the same setting, so they stay comparable.
          </p>
        </Info>
        {levels && (
          <Segmented
            mono
            ariaLabel="Error map colour range"
            value={String(scaleIdx)}
            onChange={(v) => onScale(Number(v))}
            items={levels.map((v, i) => ({
              id: String(i),
              label: `\u00b1${v.toFixed(2)}`,
            }))}
          />
        )}
      </div>

      <canvas ref={ref} onClick={onTrackClick} className="track" />
      <input
        type="range"
        id="day-slider"
        min={0}
        max={manifest.meta.nDays - 1}
        step={1}
        value={mapDay}
        onChange={(e) => onMapDay(Number(e.target.value))}
        aria-label="Test day"
      />
    </div>
  );
}
