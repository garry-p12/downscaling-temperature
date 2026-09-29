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
            Take the whole job the model is asked to do and split it into three
            parts that don&rsquo;t overlap. They add up to 100%.
          </p>
          <p>
            <b>Uniform daily offset</b> &mdash; the whole map is off by the same
            amount today, a different amount tomorrow. Half the job.
          </p>
          <p>
            <b>Static spatial pattern</b> &mdash; certain places are always wrong,
            in the same way, every single day.
          </p>
          <p>
            <b>Space&ndash;time remainder</b> &mdash; everything else.
          </p>
          <p>
            Here&rsquo;s the catch. Maps of terrain and land cover never change
            from one day to the next, so they can only help with the{" "}
            <em>middle</em> part &mdash; just{" "}
            {(manifest.budget.box.static * 100).toFixed(1)}% of the job. That is
            why ten different neural networks all scored the same: they were
            competing over a sliver.
          </p>
          <p>
            <b>What the model left</b> shows the same split for the errors still
            there after training. The model shrank the middle part and barely
            touched the big one &mdash; it is still{" "}
            {(manifest.budget.model.offset * 100).toFixed(1)}%.
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
        Measured over the {manifest.meta.boxCells.toLocaleString()} squares of land
        inside the test box. Within that box the daily error really is close to one
        single number ({(manifest.budget.box.offset * 100).toFixed(1)}%). Across all
        of Texas it isn&rsquo;t ({(manifest.budget.domain.offset * 100).toFixed(1)}%)
        &mdash; it drifts from place to place. That is why splitting the map into{" "}
        {manifest.meta.blocks} squares beats taking one average.
      </p>
    </>
  );
}
