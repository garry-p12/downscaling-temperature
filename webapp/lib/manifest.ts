import { promises as fs } from "node:fs";
import path from "node:path";
import type { Manifest } from "./types";

/**
 * Read the manifest on the SERVER.
 *
 * Every number the page quotes outside the canvases — the ladder, the LORO
 * table, the budget, the footer — comes from here, so it is in the first HTML
 * response rather than appearing after a client fetch. Only the 2.3 MB of
 * pixels waits, and only the canvases wait for it.
 */
export async function loadManifest(): Promise<Manifest> {
  const p = path.join(process.cwd(), "public", "data", "manifest.json");
  try {
    return JSON.parse(await fs.readFile(p, "utf8")) as Manifest;
  } catch (err) {
    throw new Error(
      `Could not read ${p}. Generate it from the repository root with:\n` +
        `  python scripts/export_offset_demo.py --target webapp\n` +
        `Original error: ${(err as Error).message}`,
    );
  }
}
