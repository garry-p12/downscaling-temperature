import Shell from "@/components/Shell";
import { loadManifest } from "@/lib/manifest";

export const dynamic = "force-static";

export default async function Page() {
  const manifest = await loadManifest();
  return (
    <main className="wrap">
      <h1>Downscaling temperature</h1>
      <Shell manifest={manifest} />
    </main>
  );
}
