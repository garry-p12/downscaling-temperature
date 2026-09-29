"use client";

export interface SegItem<T extends string> {
  id: T;
  label: string;
}

export default function Segmented<T extends string>({
  items,
  value,
  onChange,
  mono = false,
  ariaLabel,
}: {
  items: SegItem<T>[];
  value: T;
  onChange: (id: T) => void;
  mono?: boolean;
  ariaLabel: string;
}) {
  return (
    <div className={mono ? "seg mono" : "seg"} role="group" aria-label={ariaLabel}>
      {items.map((it) => (
        <button
          key={it.id}
          type="button"
          aria-pressed={it.id === value}
          onClick={() => onChange(it.id)}
        >
          {it.label}
        </button>
      ))}
    </div>
  );
}
