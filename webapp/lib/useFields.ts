"use client";

import { useEffect, useState } from "react";
import type { Fields, Manifest } from "./types";

type State =
  | { status: "loading"; fields: null; error: null }
  | { status: "ready"; fields: Fields; error: null }
  | { status: "error"; fields: null; error: string };

/**
 * Fetch fields.bin once and take typed-array views over it.
 *
 * No copying and no base64: the exporter writes the uint16 fields first so a
 * Uint16Array view is 2-byte aligned, and the rest are bytes, so every field
 * is a view into the one buffer the browser already downloaded.
 *
 * lat/lon are the exception — they are dequantised into Float32Arrays up
 * front, because the graticule tracer reads them thousands of times per redraw
 * and doing the arithmetic per access showed up in profiling.
 */
export function useFields(manifest: Manifest): State {
  const [state, setState] = useState<State>({
    status: "loading",
    fields: null,
    error: null,
  });

  useEffect(() => {
    let alive = true;
    const controller = new AbortController();

    (async () => {
      try {
        // The query is the content hash. fields.bin is served with a long
        // Cache-Control, so without this a browser keeps an older export's
        // pixels and pairs them with a newer manifest.
        const res = await fetch(`/data/fields.bin?v=${manifest.version}`, {
          signal: controller.signal,
        });
        if (!res.ok) throw new Error(`fields.bin responded ${res.status}`);
        const buf = await res.arrayBuffer();
        if (!alive) return;

        if (buf.byteLength !== manifest.binBytes) {
          throw new Error(
            `fields.bin is ${buf.byteLength} bytes but the manifest expects ` +
              `${manifest.binBytes}. They came from different exporter runs. ` +
              `The manifest is read at build time, so after re-exporting you ` +
              `also need to rebuild if you are running \`npm run start\`.`,
          );
        }

        const L = manifest.layout;
        const view = (name: keyof typeof L) => {
          const f = L[name];
          return f.dtype === "uint16"
            ? new Uint16Array(buf, f.offset, f.length)
            : new Uint8Array(buf, f.offset, f.length);
        };

        const deq16 = (q: Uint16Array, lo: number, hi: number) => {
          const out = new Float32Array(q.length);
          const s = (hi - lo) / 65535;
          for (let i = 0; i < q.length; i++) out[i] = lo + q[i] * s;
          return out;
        };

        const latQ = view("lat") as Uint16Array;
        const lonQ = view("lon") as Uint16Array;
        const truth = view("truth") as Uint8Array;
        const err = view("err") as Uint8Array;
        const perCell = {
          pcModel: view("pcModel") as Uint8Array,
          pcGlobalmean: view("pcGlobalmean") as Uint8Array,
          pcCorrected: view("pcCorrected") as Uint8Array,
          pcOracle: view("pcOracle") as Uint8Array,
        };
        // The estimator buttons are keyed by `rmseKey` (model / globalmean /
        // corrected / oracle); the binary fields are keyed by pcXxx. One map
        // here rather than a switch at every call site.
        const PC_OF = {
          model: "pcModel",
          globalmean: "pcGlobalmean",
          corrected: "pcCorrected",
          oracle: "pcOracle",
        } as const;

        const nc = manifest.meta.nx * manifest.meta.ny;
        const eLim = L.err.lim ?? L.err.hi ?? 4;
        const tLo = L.truth.lo ?? 0;
        const tHi = L.truth.hi ?? 1;
        const pcMax = L.pcModel.hi ?? 1;

        setState({
          status: "ready",
          error: null,
          fields: {
            lat: deq16(latQ, L.lat.lo ?? 0, L.lat.hi ?? 1),
            lon: deq16(lonQ, L.lon.lo ?? 0, L.lon.hi ?? 1),
            land: view("land") as Uint8Array,
            truth,
            err,
            perCell,
            // Stored residual is truth - model. Flip the sign so positive
            // reads as "the model is too warm here", matching the published
            // figures, where red is a warm bias.
            errAt: (d, i) => -((err[d * nc + i] / 255) * 2 * eLim - eLim),
            truthAt: (d, i) => tLo + (truth[d * nc + i] / 255) * (tHi - tLo),
            pcAt: (which, i) => (perCell[PC_OF[which]][i] / 255) * pcMax,
            errLim: eLim,
            errLevels: L.err.limBox ?? [eLim],
            errLevelQ: L.err.limBoxQ ?? [100],
            pcMax,
          },
        });
      } catch (e) {
        if (!alive || (e as Error).name === "AbortError") return;
        setState({ status: "error", fields: null, error: (e as Error).message });
      }
    })();

    return () => {
      alive = false;
      controller.abort();
    };
  }, [manifest]);

  return state;
}
