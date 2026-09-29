"use client";

import { useEffect, useRef, useState } from "react";

/**
 * The circled "i" next to a heading.
 *
 * This design keeps the explanatory prose out of the page and behind these,
 * so the numbers have room. That only works if the prose is genuinely one
 * click away and dismisses on Escape or an outside click, like any popover.
 */
export default function Info({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (!wrap.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <span className="infowrap" ref={wrap}>
      <button
        type="button"
        className="info"
        aria-expanded={open}
        aria-label={`About: ${label}`}
        onClick={() => setOpen((o) => !o)}
      >
        i
      </button>
      {open && <span className="pop">{children}</span>}
    </span>
  );
}
