/**
 * A single headline number.
 *
 * The right form for one current value — a one-bar bar chart would be worse.
 * The value wears text ink, never a series colour; the optional icon carries the
 * accent instead.
 */

import type { ReactNode } from "react";
import { Skeleton } from "../ui/States";
import "./StatTile.css";

export interface StatTileProps {
  label: string;
  value: ReactNode;
  /** Small qualifier under the value, e.g. "of 24 enrolled". */
  meta?: string;
  icon?: ReactNode;
  tone?: "default" | "good" | "warning" | "critical";
  loading?: boolean;
  /** Renders the value at hero size — use for the one number that leads. */
  hero?: boolean;
}

export function StatTile({
  label,
  value,
  meta,
  icon,
  tone = "default",
  loading = false,
  hero = false,
}: StatTileProps) {
  return (
    <div className={`stat card stat--${tone}`}>
      <div className="stat__head">
        <span className="stat__label">{label}</span>
        {icon && <span className="stat__icon">{icon}</span>}
      </div>

      {loading ? (
        <Skeleton height={hero ? 40 : 30} width="60%" />
      ) : (
        <p className={`stat__value ${hero ? "stat__value--hero" : ""}`}>{value}</p>
      )}

      {meta && !loading && <p className="stat__meta">{meta}</p>}
    </div>
  );
}
