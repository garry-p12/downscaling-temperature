/** @type {import('next').NextConfig} */
// Set NEXT_STATIC_EXPORT=1 to emit a pure static site in out/ (Netlify, GitHub
// Pages, any bucket). `headers()` is not supported by the static exporter —
// there is no server to send them — so in that mode the cache rules come from
// public/_headers instead, which Netlify reads directly.
const isExport = process.env.NEXT_STATIC_EXPORT === "1";

const nextConfig = {
  reactStrictMode: true,
  ...(isExport ? { output: "export", images: { unoptimized: true } } : {}),
  // fields.bin is requested with `?v=<content hash>` (see lib/useFields.ts), so
  // a new export is a new URL and a long cache is safe. It was NOT safe before
  // that query existed: a cached copy of an older export got paired with a
  // freshly built manifest, and only the byte-length check caught it.
  ...(isExport ? {} : { async headers() {
    return [
      {
        source: "/data/fields.bin",
        headers: [
          { key: "Cache-Control", value: "public, max-age=31536000, immutable" },
        ],
      },
      {
        source: "/data/manifest.json",
        headers: [{ key: "Cache-Control", value: "no-cache" }],
      },
    ];
  } }),
};
export default nextConfig;
