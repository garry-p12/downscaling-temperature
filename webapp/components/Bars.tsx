"use client";

export interface BarRow {
  key: string;
  name: string;
  value: number;
  color: string;
  /** Printed at the right of the row; defaults to the value at 3 dp. */
  display?: string;
  active?: boolean;
}

/** Label + value above a rounded track. Used for the year RMSE ladder. */
export default function Bars({ rows, max }: { rows: BarRow[]; max: number }) {
  return (
    <div className="bars">
      {rows.map((r) => (
        <div key={r.key} className={r.active ? "barrow on" : "barrow"}>
          <div className="barhead">
            <span className="barname">{r.name}</span>
            <span className="barval">{r.display ?? r.value.toFixed(3)}</span>
          </div>
          <div className="bartrack">
            <div
              className="barfill"
              style={{
                width: `${Math.max(1, (r.value / max) * 100)}%`,
                background: r.color,
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
