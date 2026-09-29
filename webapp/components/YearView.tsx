"use client";

import { useCallback } from "react";
import FieldMap from "./FieldMap";
import Bars from "./Bars";
import Info from "./Info";
import Segmented from "./Segmented";
import { SEQUENTIAL, gradientCss } from "@/lib/color";
import { mean, sd, signed, signedPct } from "@/lib/format";
import { METHODS, type Fields, type Manifest, type MethodId } from "@/lib/types";

export default function YearView({
  manifest,
  fields,
  methodId,
  onMethod,
}: {
  manifest: Manifest;
  fields: Fields | null;
  methodId: MethodId;
  onMethod: (m: MethodId) => void;
}) {
  const method = METHODS.find((m) => m.id === methodId)!;
  const { year } = manifest;

  const corrected = useCallback(
    (i: number) => (fields ? fields.pcAt(method.rmseKey, i) : 0),
    [fields, method.rmseKey],
  );
  const modelPc = useCallback(
    (i: number) => (fields ? fields.pcAt("model", i) : 0),
    [fields],
  );

  const rows = [
    { key: "interp", name: "Interpolation only", value: year.interp, color: "var(--bar-pale)" },
    { key: "model", name: "Model", value: year.model, color: "var(--bar-grey)", active: methodId === "none" },
    { key: "glob", name: "+ one global mean", value: year.globalmean, color: "var(--bar-blue-soft)", active: methodId === "glob" },
    { key: "k6", name: `+ k = ${manifest.meta.k} sub-regions`, value: year.corrected, color: "var(--bar-blue)", active: methodId === "est" },
    { key: "oracle", name: "Oracle offset", value: year.oracle, color: "var(--bar-accent)", active: methodId === "oracle" },
  ];
  const shares = manifest.loro.map((r) => r.ofOracle);

  const bar = (
    <div className="cbar">
      <span>0</span>
      <div className="ramp" style={{ background: gradientCss(SEQUENTIAL) }} />
      <span>{fields ? `${fields.pcMax.toFixed(2)} °C` : "—"}</span>
    </div>
  );

  return (
    <>
      <div className="sechead">
        <h2>Over the whole year</h2>
        <Info label="Over the whole year">
          <p>
            A single day can be lucky. These maps use all{" "}
            {manifest.meta.nTestDays} test days at once. Each square shows how
            wrong the model typically is there across the year &mdash; darker is
            worse.
          </p>
          <p>
            The bars on the right are the score for the whole year, one per
            method. Lower is better, and the orange bar is the best anyone could
            do.
          </p>
          <p>
            <b>The two numbers underneath</b> are worth reading together.{" "}
            <em>Mean error</em> is whether the model runs warm or cold on average.{" "}
            <em>RMSE change</em> is how much smaller its mistakes got. The
            correction almost wipes out the first and only partly reduces the
            second &mdash; exactly what should happen if the problem really is a
            once-a-day bias rather than a general lack of detail.
          </p>
        </Info>
        <span className="spacer" />
        <Segmented
          ariaLabel="Offset estimator"
          value={methodId}
          onChange={onMethod}
          items={METHODS.map((m) => ({ id: m.id, label: m.label }))}
        />
      </div>

      <div className="yearGrid">
        <figure className="card" style={{ margin: 0 }}>
          <figcaption className="cardhead">
            <span className="cardtitle">
              Per-cell RMSE, model
              <span className="cardsub">all 365 test days</span>
            </span>
          </figcaption>
          {fields ? (
            <FieldMap
              fields={fields}
              meta={manifest.meta}
              valueAt={modelPc}
              ramp={SEQUENTIAL}
              lo={0}
              hi={fields.pcMax}
              readout={(i) =>
                `${place(fields, i)}\nRMSE ${fields.pcAt("model", i).toFixed(3)} °C`
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
              Per-cell RMSE, corrected
              <span className="cardsub">{method.label.toLowerCase()}</span>
            </span>
          </figcaption>
          {fields ? (
            <FieldMap
              fields={fields}
              meta={manifest.meta}
              valueAt={corrected}
              ramp={SEQUENTIAL}
              lo={0}
              hi={fields.pcMax}
              revision={METHODS.indexOf(method)}
              readout={(i) =>
                `${place(fields, i)}\nRMSE ${fields.pcAt(method.rmseKey, i).toFixed(3)} °C`
              }
            />
          ) : (
            <div className="skeleton" aria-busy="true" />
          )}
          {bar}
        </figure>

        <div className="card">
          <div className="cardhead">
            <span className="cardtitle">
              Year RMSE, held-out box
              <span className="cardsub">lower is better</span>
            </span>
          </div>
          <Bars rows={rows} max={Math.max(...rows.map((r) => r.value)) * 1.06} />
          <dl className="statline">
            <div>
              <dt>MEAN ERROR</dt>
              <dd>
                {signed(manifest.bias.model)} → {signed(manifest.bias.corrected)}
              </dd>
            </div>
            <div>
              <dt>RMSE CHANGE</dt>
              <dd className="good">{signedPct(year.model, year.corrected)}</dd>
            </div>
          </dl>
        </div>
      </div>

      <div className="card" style={{ marginTop: 22 }}>
        <div className="sechead" style={{ marginBottom: 14 }}>
          <h3>Does it only work for Austin?</h3>
          <Info label="Does it only work for Austin?">
            <p>
              We ran the same method five times, each time hiding a different part
              of Texas and predicting it from everywhere else.
            </p>
            <p>
              <b>% of oracle is the column to read.</b> It says how much of the
              best possible improvement the method actually got. It is fair to
              compare across regions because it is measured inside each one
              separately &mdash; rough terrain is harder everywhere, and this
              cancels that out.
            </p>
            <p>
              <b>One catch.</b> The model was only ever trained with Austin hidden.
              For the other four regions it had already seen the answers while
              training, so their &ldquo;model&rdquo; column looks better than it
              should. What this table really tests is the <em>guessing method</em>,
              not the model.
            </p>
          </Info>
          <span className="spacer" />
          <span className="hint">
            averages {mean(shares).toFixed(0)}% of the best possible, across five
            regions
          </span>
        </div>
        <div className="tablewrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Region</th>
                <th scope="col">Land cells</th>
                <th scope="col">Model</th>
                <th scope="col">Corrected</th>
                <th scope="col">Oracle</th>
                <th scope="col">Δ RMSE</th>
                <th scope="col">% of oracle</th>
              </tr>
            </thead>
            <tbody>
              {manifest.loro.map((r) => (
                <tr key={r.name} className={r.name === "austin" ? "hl" : undefined}>
                  <th scope="row">
                    {r.name}
                    {r.name === "austin" && <em>held out</em>}
                  </th>
                  <td className="n">{r.cells.toLocaleString()}</td>
                  <td className="n">{r.model.toFixed(3)}</td>
                  <td className="n">{r.corrected.toFixed(3)}</td>
                  <td className="n">{r.oracle.toFixed(3)}</td>
                  <td className="n">{r.pct.toFixed(1)}%</td>
                  <td className="n">{r.ofOracle.toFixed(0)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

function place(f: Fields, i: number) {
  return `${f.lat[i].toFixed(2)}°N  ${Math.abs(f.lon[i]).toFixed(2)}°W`;
}
