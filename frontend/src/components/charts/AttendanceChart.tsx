/**
 * Daily attendance column chart.
 *
 * Form: the job is "compare magnitude across days", so a column chart with a
 * single sequential hue — not categorical, because the days are not identities.
 * A single series needs no legend: the card title says what is plotted.
 *
 * Mark specs followed: columns capped at 24px with the band's leftover left as
 * air, a 4px rounded cap with a square baseline end, hairline recessive
 * gridlines, labels only on the extreme (never a number on every column), and a
 * per-column hover tooltip. A table view is always available, so nothing is
 * gated behind colour or hover.
 */

import { useId, useState } from "react";
import "./AttendanceChart.css";

export interface DayPoint {
  /** ISO date, YYYY-MM-DD. */
  date: string;
  present: number;
}

export interface AttendanceChartProps {
  data: DayPoint[];
  /** Enrolled student count, drawn as the reference ceiling. */
  totalStudents: number;
}

const WIDTH = 720;
const HEIGHT = 240;
const PAD = { top: 24, right: 16, bottom: 34, left: 38 };
const MAX_BAR = 24;
const RADIUS = 4;

/** Column path: rounded at the data end, square at the baseline. */
function columnPath(x: number, y: number, width: number, height: number): string {
  if (height <= 0) return "";
  const r = Math.min(RADIUS, width / 2, height);
  return [
    `M${x},${y + height}`,
    `L${x},${y + r}`,
    `Q${x},${y} ${x + r},${y}`,
    `L${x + width - r},${y}`,
    `Q${x + width},${y} ${x + width},${y + r}`,
    `L${x + width},${y + height}`,
    "Z",
  ].join(" ");
}

/** Round a maximum up to a clean axis top (0 / 5 / 10 / 20 …). */
function niceMax(value: number): number {
  if (value <= 5) return 5;
  const magnitude = 10 ** Math.floor(Math.log10(value));
  return Math.ceil(value / (magnitude / 2)) * (magnitude / 2);
}

function weekday(iso: string): string {
  return new Date(`${iso}T00:00:00`).toLocaleDateString(undefined, {
    weekday: "short",
  });
}

function longDate(iso: string): string {
  return new Date(`${iso}T00:00:00`).toLocaleDateString(undefined, {
    weekday: "long",
    day: "numeric",
    month: "short",
  });
}

export function AttendanceChart({ data, totalStudents }: AttendanceChartProps) {
  const [hovered, setHovered] = useState<number | null>(null);
  const [showTable, setShowTable] = useState(false);
  const clipId = useId();

  const plotWidth = WIDTH - PAD.left - PAD.right;
  const plotHeight = HEIGHT - PAD.top - PAD.bottom;

  const peak = Math.max(totalStudents, ...data.map((point) => point.present), 1);
  const axisTop = niceMax(peak);
  const band = plotWidth / Math.max(data.length, 1);
  const barWidth = Math.min(MAX_BAR, band * 0.55);

  const yOf = (value: number) => PAD.top + plotHeight * (1 - value / axisTop);
  const xOf = (index: number) => PAD.left + band * index + band / 2;

  const ticks = [0, axisTop / 2, axisTop];
  const peakIndex = data.reduce(
    (best, point, index) => (point.present > data[best].present ? index : best),
    0,
  );

  if (showTable) {
    return (
      <div className="chart">
        <div className="chart__toolbar">
          <button className="chart__toggle" onClick={() => setShowTable(false)}>
            View chart
          </button>
        </div>
        <div className="chart__table-wrap">
          <table className="chart__table">
            <caption className="sr-only">
              Students present per day over the last {data.length} days
            </caption>
            <thead>
              <tr>
                <th scope="col">Date</th>
                <th scope="col">Present</th>
                <th scope="col">Rate</th>
              </tr>
            </thead>
            <tbody>
              {data.map((point) => (
                <tr key={point.date}>
                  <td>{longDate(point.date)}</td>
                  <td className="tabular">{point.present}</td>
                  <td className="tabular">
                    {totalStudents
                      ? `${Math.round((point.present / totalStudents) * 100)}%`
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  return (
    <div className="chart">
      <div className="chart__toolbar">
        <button className="chart__toggle" onClick={() => setShowTable(true)}>
          View as table
        </button>
      </div>

      <div className="chart__scroll">
        <svg
          className="chart__svg"
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="img"
          aria-label={`Column chart: students present per day over the last ${data.length} days. Peak ${data[peakIndex]?.present ?? 0} on ${data[peakIndex] ? longDate(data[peakIndex].date) : "n/a"}.`}
          onMouseLeave={() => setHovered(null)}
        >
          <defs>
            <clipPath id={clipId}>
              <rect x={PAD.left} y={0} width={plotWidth} height={HEIGHT} />
            </clipPath>
          </defs>

          {/* Gridlines: hairline, solid, one step off surface, recessive. */}
          {ticks.map((tick) => (
            <g key={tick}>
              <line
                x1={PAD.left}
                x2={WIDTH - PAD.right}
                y1={yOf(tick)}
                y2={yOf(tick)}
                className={tick === 0 ? "chart__baseline" : "chart__grid"}
              />
              <text x={PAD.left - 8} y={yOf(tick) + 4} className="chart__tick">
                {Math.round(tick)}
              </text>
            </g>
          ))}

          {/* Enrolled ceiling — context for how far each day fell short. */}
          {totalStudents > 0 && totalStudents <= axisTop && (
            <g clipPath={`url(#${clipId})`}>
              <line
                x1={PAD.left}
                x2={WIDTH - PAD.right}
                y1={yOf(totalStudents)}
                y2={yOf(totalStudents)}
                className="chart__reference"
              />
              <text
                x={WIDTH - PAD.right}
                y={yOf(totalStudents) - 6}
                className="chart__reference-label"
                textAnchor="end"
              >
                {totalStudents} enrolled
              </text>
            </g>
          )}

          {data.map((point, index) => {
            const x = xOf(index) - barWidth / 2;
            const y = yOf(point.present);
            const height = PAD.top + plotHeight - y;
            const isPeak = index === peakIndex && point.present > 0;

            return (
              <g key={point.date}>
                {/* Hit target spans the whole band, so hovering is forgiving —
                    the 24px column would be a fiddly target on its own. */}
                <rect
                  x={PAD.left + band * index}
                  y={PAD.top}
                  width={band}
                  height={plotHeight}
                  fill="transparent"
                  onMouseEnter={() => setHovered(index)}
                />
                {hovered === index && (
                  <rect
                    x={PAD.left + band * index + 2}
                    y={PAD.top}
                    width={band - 4}
                    height={plotHeight}
                    className="chart__hover-band"
                    rx={4}
                  />
                )}
                {point.present > 0 && (
                  <path
                    d={columnPath(x, y, barWidth, height)}
                    className={`chart__bar ${hovered === index ? "chart__bar--hot" : ""}`}
                  />
                )}
                {/* Selective direct label: the extreme only. The axis and the
                    tooltip carry every other value. */}
                {isPeak && (
                  <text x={xOf(index)} y={y - 7} className="chart__value" textAnchor="middle">
                    {point.present}
                  </text>
                )}
                <text
                  x={xOf(index)}
                  y={HEIGHT - 12}
                  className="chart__axis-label"
                  textAnchor="middle"
                >
                  {weekday(point.date)}
                </text>
              </g>
            );
          })}
        </svg>

        {hovered !== null && data[hovered] && (
          <div
            className="chart__tooltip"
            style={{ left: `${(xOf(hovered) / WIDTH) * 100}%` }}
          >
            <p className="chart__tooltip-date">{longDate(data[hovered].date)}</p>
            <p className="chart__tooltip-value">
              <span className="chart__swatch" aria-hidden="true" />
              {data[hovered].present} present
              {totalStudents > 0 && (
                <span className="chart__tooltip-rate">
                  {Math.round((data[hovered].present / totalStudents) * 100)}%
                </span>
              )}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
