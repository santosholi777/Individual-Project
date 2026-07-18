/**
 * Formatting helpers.
 *
 * The AI service speaks UTC ISO-8601; a lecturer reads local time. Every
 * conversion between the two happens here so it is done once and consistently.
 */

/** Format an ISO timestamp as local wall-clock time, e.g. "09:41". */
export function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Format an ISO timestamp as local date and time. */
export function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Format a YYYY-MM-DD date as e.g. "Wed 15 Jul 2026". */
export function shortDate(iso: string): string {
  return new Date(`${iso}T00:00:00`).toLocaleDateString(undefined, {
    weekday: "short",
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

/** Render a Date as YYYY-MM-DD in the *local* calendar.
 *
 * `toISOString()` is deliberately avoided: it converts to UTC first, which
 * silently shifts the date by a day either side of midnight in most timezones.
 */
export function isoDate(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

/** Today as YYYY-MM-DD, local. */
export function today(): string {
  return isoDate(new Date());
}

/** N days before today as YYYY-MM-DD, local. */
export function daysAgo(days: number): string {
  const date = new Date();
  date.setDate(date.getDate() - days);
  return isoDate(date);
}

/** Cosine similarity (0–1) as a percentage string, e.g. "98%". */
export function percent(value: number, decimals = 0): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

/** Relative time for the live feed, e.g. "just now", "4m ago". */
export function timeAgo(iso: string): string {
  const seconds = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 10) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}
