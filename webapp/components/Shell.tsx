"use client";

import { useEffect, useState } from "react";
import TabBar, { type TabId } from "./TabBar";
import DayView from "./DayView";
import YearView from "./YearView";
import BudgetView from "./BudgetView";
import { useFields } from "@/lib/useFields";
import type { Manifest, MethodId } from "@/lib/types";

const FRAME_MS = 110;

/**
 * Tab state, and the state the tabs share.
 *
 * The estimator choice is deliberately shared between the day view and the year
 * view: picking "one global mean" on one day and then switching to the year
 * should show that same estimator's year maps, not silently reset to k=6.
 */
export default function Shell({ manifest }: { manifest: Manifest }) {
  const { status, fields, error } = useFields(manifest);
  const [tab, setTab] = useState<TabId>("day");

  // Tabs are reflected in the URL hash, so a view can be linked to and the
  // back button steps through them. Read on mount rather than in useState, or
  // the server-rendered markup and the first client render disagree.
  useEffect(() => {
    const read = () => {
      const h = window.location.hash.replace("#", "") as TabId;
      if (h === "day" || h === "year" || h === "budget") setTab(h);
    };
    read();
    window.addEventListener("hashchange", read);
    return () => window.removeEventListener("hashchange", read);
  }, []);

  const selectTab = (id: TabId) => {
    setTab(id);
    if (window.location.hash !== `#${id}`) {
      window.history.pushState(null, "", `#${id}`);
    }
  };
  const [methodId, setMethodId] = useState<MethodId>("est");
  const [mapDay, setMapDay] = useState(manifest.defaultDay);
  const [playing, setPlaying] = useState(false);
  const [scaleIdx, setScaleIdx] = useState(0);

  useEffect(() => {
    if (!playing || tab !== "day") return;
    const id = window.setInterval(
      () => setMapDay((d) => (d + 1) % manifest.meta.nDays),
      FRAME_MS,
    );
    return () => window.clearInterval(id);
  }, [playing, tab, manifest.meta.nDays]);

  // Arrow keys and space scrub from anywhere on the day view — the maps are
  // what people look at, not the slider.
  useEffect(() => {
    if (tab !== "day") return;
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && ["INPUT", "BUTTON", "TEXTAREA"].includes(t.tagName)) return;
      if (e.key === "ArrowRight") {
        setMapDay((d) => Math.min(manifest.meta.nDays - 1, d + 1));
      } else if (e.key === "ArrowLeft") {
        setMapDay((d) => Math.max(0, d - 1));
      } else if (e.key === " ") {
        setPlaying((p) => !p);
      } else {
        return;
      }
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [tab, manifest.meta.nDays]);

  if (status === "error") {
    return (
      <div className="card problem">
        <h3>Could not load the field data</h3>
        <p className="hint" style={{ marginTop: 8 }}>
          {error}
        </p>
        <p className="hint" style={{ marginTop: 8 }}>
          Regenerate it from the repository root:{" "}
          <code>python scripts/export_offset_demo.py --target webapp</code>
        </p>
      </div>
    );
  }

  return (
    <>
      <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`}>
        {tab === "day" && (
          <DayView
            manifest={manifest}
            fields={fields}
            methodId={methodId}
            onMethod={setMethodId}
            mapDay={mapDay}
            onMapDay={setMapDay}
            playing={playing}
            onTogglePlay={() => setPlaying((p) => !p)}
            scaleIdx={scaleIdx}
            onScale={setScaleIdx}
          />
        )}
        {tab === "year" && (
          <YearView
            manifest={manifest}
            fields={fields}
            methodId={methodId}
            onMethod={setMethodId}
          />
        )}
        {tab === "budget" && <BudgetView manifest={manifest} />}
      </div>
      <TabBar value={tab} onChange={selectTab} />
    </>
  );
}
