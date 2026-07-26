/**
 * Status pill.
 *
 * Status tones always ship with a label (and, for attendance outcomes, an icon)
 * — colour never carries the meaning on its own, which is what keeps the UI
 * readable for colour-blind users and in forced-colors mode.
 */

import type { ReactNode } from "react";
import "./Badge.css";

export type BadgeTone = "good" | "warning" | "critical" | "neutral" | "accent";

export interface BadgeProps {
  tone?: BadgeTone;
  icon?: ReactNode;
  children: ReactNode;
  className?: string;
}

export function Badge({
  tone = "neutral",
  icon,
  children,
  className = "",
}: BadgeProps) {
  return (
    <span className={`badge badge--${tone} ${className}`}>
      {icon && (
        <span className="badge__icon" aria-hidden="true">
          {icon}
        </span>
      )}
      {children}
    </span>
  );
}
