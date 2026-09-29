"use client";

import { useCallback } from "react";
import FieldMap from "./FieldMap";
import Scrubber from "./Scrubber";
import OffsetScatter from "./OffsetScatter";
import Info from "./Info";
import { DIVERGING, gradientCss } from "@/lib/color";
import { fixed, signed } from "@/lib/format";
import { METHODS, type Fields, type Manifest, type MethodId } from "@/lib/types";

interface Props {
  manifest: Manifest;
  fields: Fields | null;
  methodId: MethodId;
  onMethod: (m: MethodId) => void;
  mapDay: number;
  onMapDay: (d: number) => void;
  playing: boolean;
  onTogglePlay: () => void;
  scaleIdx: number;
  onScale: (i: number) => void;
}

export default function DayView(p: Props) {
  const { manifest, fields, methodId } = p;
  const method = METHODS.find((m) => m.id === methodId)!;
  const dayIndex = manifest.mapDayIndex[p.mapDay];
  const offset = method.key ? manifest.offsets[method.key][dayIndex] : 0;

  const before = useCallback(
    (i: number) => (fields ? fields.errAt(p.mapDay, i) : 0),
    [fields, p.mapDay],
  );
  const after = useCallback(
    (i: number) => (fields ? fields.errAt(p.mapDay, i) + offset : 0),
    [fields, p.mapDay, offset],
  );

  const rmseBefore = manifest.daily.model[dayIndex];
  const rmseAfter = manifest.daily[method.rmseKey][dayIndex];
  const eLim = fields ? (fields.errLevels[p.scaleIdx] ?? fields.errLim) : 1;
  const eQ = fields ? (fields.errLevelQ[p.scaleIdx] ?? 100) : 100;

  const bar = (
    <div className="cbar">
      <span>{fields ? `−${eLim.toFixed(2)} °C` : "—"}</span>
      <div className="ramp" style={{ background: gradientCss(DIVERGING) }} />
      <span>{fields ? `+${eLim.toFixed(2)}` : "—"}</span>
    </div>
  );

  return (
    <>
      <div className="sechead">
        <h2>The correction, day by day</h2>
        <Info label="The correction, day by day">
          <p>
            <b>Left:</b> how wrong the model was on this day.{" "}
            <b>Right:</b> the same day, after adding one number to every cell.
          </p>
          <p>
            Red means the model was too warm there. Blue means too cold. White
            means it got that spot about right.
          </p>
          <p>
            The orange box is the area we test on. The model never saw it while
            training, and every score on this page is measured inside it &mdash;
            that&rsquo;s {manifest.meta.boxCells.toLocaleString()} squares of land.
          </p>
          <p>
            <b>Why one number can help:</b> on most days the model is off by
            roughly the same amount everywhere. If that&rsquo;s true, adding the
            right number should wash most of the colour out of the box.
          </p>
        </Info>
      </div>

      <div className="dayGrid">
        <div className="stack">
          <div className="mapPair">
            <figure className="card" style={{ margin: 0 }}>
              <figcaption className="cardhead">
                <span className="cardtitle">
                  Model error
                  <span className="cardsub">uncorrected</span>
                </span>
                <span className="stat">
                  {fixed(rmseBefore)}
                  <small>RMSE °C</small>
                </span>
              </figcaption>
              {fields ? (
                <FieldMap
                  fields={fields}
                  meta={manifest.meta}
                  valueAt={before}
                  ramp={DIVERGING}
                  lo={-eLim}
                  hi={eLim}
                  revision={p.scaleIdx}
                  readout={(i) =>
                    `${place(fields, i)}\nerror ${signed(fields.errAt(p.mapDay, i), 2)} °C\ntruth ${fields.truthAt(p.mapDay, i).toFixed(1)} °C`
                  }
                />
              ) : (
                <div className="skeleton" aria-busy="true" />
              )}
              {bar}
            </figure>

            <figure className="card lead" style={{ margin: 0 }}>
              <figcaption className="cardhead">
                <span className="cardtitle">
                  After correction
                  <span className="cardsub">{method.label.toLowerCase()}</span>
                </span>
                <span
                  className={`stat${rmseAfter < rmseBefore - 1e-9 ? " good" : ""}`}
                >
                  {fixed(rmseAfter)}
                  <small>RMSE °C</small>
                </span>
              </figcaption>
              {fields ? (
                <FieldMap
                  fields={fields}
                  meta={manifest.meta}
                  valueAt={after}
                  ramp={DIVERGING}
                  lo={-eLim}
                  hi={eLim}
                  revision={p.scaleIdx * 10 + METHODS.indexOf(method)}
                  readout={(i) =>
                    `${place(fields, i)}\nerror ${signed(fields.errAt(p.mapDay, i) + offset, 2)} °C\ntruth ${fields.truthAt(p.mapDay, i).toFixed(1)} °C`
                  }
                />
              ) : (
                <div className="skeleton" aria-busy="true" />
              )}
              {bar}
            </figure>
          </div>

          <Scrubber
            manifest={manifest}
            mapDay={p.mapDay}
            onMapDay={p.onMapDay}
            playing={p.playing}
            onTogglePlay={p.onTogglePlay}
            appliedOffset={offset}
            methodApplies={method.key !== null}
            levels={fields?.errLevels ?? null}
            scaleIdx={p.scaleIdx}
            onScale={p.onScale}
            scaleQ={eQ}
          />
        </div>

        <aside className="rail">
          <div className="card">
            <div className="sechead" style={{ marginBottom: 0 }}>
              <h3>Where the number comes from</h3>
              <Info label="Where the number comes from">
                <p>
                  We need to guess how far off the model is today &mdash; without
                  looking at the answer inside the box. So we look at how wrong it
                  is everywhere <em>else</em> in Texas, where we do have answers.
                </p>
                <p>
                  <b>No correction</b> &mdash; leave the model alone.
                </p>
                <p>
                  <b>One global mean</b> &mdash; average how wrong it is across the
                  rest of Texas. One number for the whole day.
                </p>
                <p>
                  <b>k = 6</b> &mdash; split the rest of Texas into{" "}
                  {manifest.meta.blocks} squares and learn which of them best
                  predict what happens in the box. Better, because the model
                  isn&rsquo;t off by quite the same amount everywhere.
                </p>
                <p>
                  <b>Oracle</b> &mdash; the real answer, looked up. This is
                  cheating. It&rsquo;s here to show the best any of this could
                  possibly do.
                </p>
                <p>
                  The number beside each option is the error it leaves behind on
                  this day. Smaller is better.
                </p>
              </Info>
            </div>
            <div className="methods" role="radiogroup" aria-label="Offset estimator">
              {METHODS.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  role="radio"
                  aria-checked={m.id === methodId}
                  className="mbtn"
                  onClick={() => p.onMethod(m.id)}
                >
                  <span className="dot" aria-hidden="true" />
                  <span>
                    <span className="lab">{m.label}</span>
                    <span className="sub">{m.sub}</span>
                  </span>
                  <span className="val">
                    {fixed(manifest.daily[m.rmseKey][dayIndex])}
                  </span>
                </button>
              ))}
            </div>
          </div>

          <div className="card">
            <div className="sechead" style={{ marginBottom: 12 }}>
              <h3>Estimated vs. true offset</h3>
              <Info label="Estimated vs. true offset">
                <p>
                  One dot per day, for all {manifest.meta.nTestDays} test days.
                  Across = what we guessed. Up = what it actually turned out to be.
                  Dots on the dashed line are perfect guesses.
                </p>
                <p>
                  <b>This is why the method works.</b> The blue dots
                  (splitting Texas into squares) hug the line. The grey dots (one
                  plain average) scatter much wider.
                </p>
                <p>The circled dot is the day shown on the maps.</p>
              </Info>
            </div>
            <div className="legendrow">
              <span>
                <i className="swatch" style={{ background: "var(--bar-blue)" }} />
                k = {manifest.meta.k} · r = {manifest.corr.est.toFixed(2)}
              </span>
              <span>
                <i className="swatch" style={{ background: "var(--ink3)" }} />
                global · r = {manifest.corr.glob.toFixed(2)}
              </span>
            </div>
            <OffsetScatter manifest={manifest} method={method} dayIndex={dayIndex} />
          </div>
        </aside>
      </div>
    </>
  );
}

function place(f: Fields, i: number) {
  return `${f.lat[i].toFixed(2)}°N  ${Math.abs(f.lon[i]).toFixed(2)}°W`;
}
