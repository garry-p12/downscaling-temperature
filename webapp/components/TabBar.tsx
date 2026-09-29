"use client";

export type TabId = "day" | "year" | "budget";

const TABS: { id: TabId; label: string }[] = [
  { id: "day", label: "Day by day" },
  { id: "year", label: "Whole year" },
  { id: "budget", label: "Error budget" },
];

/**
 * The three views are a sequence — one day, then the year, then why any of it
 * was worth doing — so they carry numbers. Ordinary tabs would not earn them.
 */
export default function TabBar({
  value,
  onChange,
}: {
  value: TabId;
  onChange: (id: TabId) => void;
}) {
  return (
    <nav className="tabbar" role="tablist" aria-label="Views">
      {TABS.map((t, i) => (
        <button
          key={t.id}
          type="button"
          role="tab"
          aria-selected={t.id === value}
          aria-controls={`panel-${t.id}`}
          id={`tab-${t.id}`}
          onClick={() => onChange(t.id)}
        >
          <span className="n">{String(i + 1).padStart(2, "0")}</span>
          {t.label}
        </button>
      ))}
    </nav>
  );
}
