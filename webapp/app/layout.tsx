import type { Metadata, Viewport } from "next";
import "./globals.css";

/**
 * No web fonts. Georgia ships with every desktop and mobile OS, and the sans
 * and mono stacks resolve to whatever the reader's system already uses, so
 * there is nothing to download, nothing to fall back from, and no flash of
 * unstyled text.
 */
export const metadata: Metadata = {
  title: "Downscaling temperature",
  description:
    "Scrub the 2023 test year and watch a single estimated daily offset flatten a " +
    "downscaling model's error map over held-out central Texas.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: "#f4f4ef",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
