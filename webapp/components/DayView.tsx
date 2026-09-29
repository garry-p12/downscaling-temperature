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
            Both maps are the model&rsquo;s error over the same day &mdash; output
            minus ERA5-Land truth &mdash; on one shared colour scale. Red is too
            warm, blue too cold. The orange box is the held-out region: it was
            never in training, and every RMSE quoted here is taken over the{" "}
            {manifest.meta.boxCells.toLocaleString()} land cells inside it.
          </p>
          <p>
            <b>The correction adds one number to every cell.</b> What changes
            between the two maps is a constant, so if half the error really is a
            uniform daily bias, subtracting one well-chosen number should visibly
            flatten the box.
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
                  Each option is a different way of guessing today&rsquo;s offset,
                  and the figure beside it is the RMSE it achieves on this day.
                </p>
                <p>
                  <b>One global mean</b> averages the model&rsquo;s residual over
                  the whole training region &mdash; one number. <b>k = 6</b> splits
                  that region into a 6×6 grid, takes {manifest.meta.blocks} block
                  means and learns which combination best predicts the held-out
                  region&rsquo;s offset; the ridge is fitted on training days only,
                  so every test-day estimate is out of sample. Neither reads
                  held-out truth.
                </p>
                <p>
                  <b>Oracle</b> is the true value, computed by looking at the
                  answer. It is not a method &mdash; it is the ceiling, there so the
                  others have something to be judged against.
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
                  Every one of the {manifest.meta.nTestDays} test days. The
                  horizontal axis is what the estimator guessed from the training
                  region; the vertical axis is the offset that was actually there.
                  Points on the dashed line are perfect guesses.
                </p>
                <p>
                  This is the panel that answers <b>why it works at all</b>: the
                  single global mean tracks the truth loosely (r ={" "}
                  {manifest.corr.glob.toFixed(2)}), the {manifest.meta.blocks}-block
                  ridge tracks it tightly (r = {manifest.corr.est.toFixed(2)}). The
                  circled point is the day on the maps.
                </p>
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
