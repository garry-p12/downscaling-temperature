"use client";

import { useState } from "react";
import Info from "./Info";
import Segmented from "./Segmented";
import type { Manifest } from "@/lib/types";

type Side = "task" | "model";

export default function BudgetView({ manifest }: { manifest: Manifest }) {
  const [side, setSide] = useState<Side>("task");
  const shares = side === "task" ? manifest.budget.box : manifest.budget.model;

  const rows = [
    { key: "offset", name: "Uniform daily offset", sub: "μ(t)", v: shares.offset, color: "var(--bar-blue)" },
    { key: "static", name: "Static spatial pattern", sub: "s(x)", v: shares.static, color: "var(--bar-blue-soft)" },
    { key: "rem", name: "Space–time remainder", sub: "everything else", v: shares.remainder, color: "var(--bar-grey)" },
  ];

  return (
    <>
      <div className="sechead">
        <h2>Why this was worth doing at all</h2>
        <Info label="Why this was worth doing at all">
          <p>
            Split the <b>task residual</b> &mdash; truth minus the interpolated NASA
            POWER input, which is the signal any downscaler is asked to produce
            &mdash; into three parts that cannot overlap, so the shares sum to 100%.
            Total variance over the box is{" "}
            {manifest.budget.totalVar.toFixed(4)} °C².
          </p>
          <p>
            Only the middle bar is reachable by a map of terrain or land cover:
            those inputs are fixed in time, so they can only address the part of the
            error that is also fixed in time. That is{" "}
            {(manifest.budget.box.static * 100).toFixed(1)}% of it, which is why ten
            architectures and five covariate sets all tied.
          </p>
          <p>
            <b>What the model left</b> switches to the trained model&rsquo;s own
            residual. It moved the static term and left the offset at{" "}
            {(manifest.budget.model.offset * 100).toFixed(1)}%, essentially where it
            found it.
          </p>
        </Info>
        <span className="spacer" />
        <Segmented
          ariaLabel="Which residual"
          value={side}
          onChange={setSide}
          items={[
            { id: "task", label: "The task" },
            { id: "model", label: "What the model left" },
          ]}
        />
      </div>

      <div className="card" style={{ padding: "30px 32px" }}>
        <div className="bars" style={{ gap: 26 }}>
          {rows.map((r) => (
            <div key={r.key} className="budgetrow">
              <span className="bname">
                {r.name}
                <span className="bsub">{r.sub}</span>
              </span>
              <span className="bartrack">
                <span
                  className="barfill"
                  style={{ width: `${(r.v / 0.62) * 100}%`, background: r.color }}
                />
              </span>
              <span className="bval">{(r.v * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>

      <p className="hint" style={{ marginTop: 18, maxWidth: "76ch" }}>
        Bars are the {manifest.meta.boxCells.toLocaleString()} land cells inside the
        held-out box. The offset is close to uniform <em>inside</em> the box (
        {(manifest.budget.box.offset * 100).toFixed(1)}%) but not across the whole
        domain ({(manifest.budget.domain.offset * 100).toFixed(1)}%), which is
        exactly why {manifest.meta.blocks} block means beat one global average.
      </p>
    </>
  );
}
