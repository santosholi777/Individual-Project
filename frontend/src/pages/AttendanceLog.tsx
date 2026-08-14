/**
 * Attendance log: filter the register and export it.
 *
 * Filters sit in one row above the table and are applied server-side by the AI
 * service, so the browser never holds more records than it is showing. CSV
 * export is generated client-side from the current filtered view — what you see
 * is exactly what you get.
 */

import { useMemo, useState } from "react";
import { api } from "../api/client";
import type { AttendanceRecord } from "../api/types";
import { PageHeader } from "../components/layout/PageHeader";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import {
  IconCheckCircle,
  IconDownload,
  IconList,
  IconRefresh,
} from "../components/ui/icons";
import { EmptyState, ErrorState, Skeleton } from "../components/ui/States";
import { useToast } from "../components/ui/toast-context";
import { useQuery } from "../hooks/useApi";
import { daysAgo, formatTime, percent, shortDate, today } from "../utils/format";
import "./Students.css";
import "./AttendanceLog.css";

type RangePreset = "today" | "7d" | "30d" | "custom";

const PRESETS: { key: RangePreset; label: string }[] = [
  { key: "today", label: "Today" },
  { key: "7d", label: "Last 7 days" },
  { key: "30d", label: "Last 30 days" },
  { key: "custom", label: "Custom" },
];

/** Turn the current view into a CSV file and trigger a download. */
function exportCsv(records: AttendanceRecord[]): void {
  const columns = [
    "student_id",
    "name",
    "date",
    "timestamp",
    "session",
    "confidence",
    "source",
  ] as const;

  const escape = (value: unknown) => {
    const text = String(value ?? "");
    // Quote anything that could break the column structure.
    return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
  };

  const csv = [
    columns.join(","),
    ...records.map((record) => columns.map((key) => escape(record[key])).join(",")),
  ].join("\n");

  const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = `attendance-${today()}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

export function AttendanceLog() {
  const toast = useToast();
  const [preset, setPreset] = useState<RangePreset>("7d");
  const [customFrom, setCustomFrom] = useState(daysAgo(7));
  const [customTo, setCustomTo] = useState(today());
  const [session, setSession] = useState("");
  const [student, setStudent] = useState("");

  const range = useMemo(() => {
    switch (preset) {
      case "today":
        return { from: today(), to: today() };
      case "7d":
        return { from: daysAgo(6), to: today() };
      case "30d":
        return { from: daysAgo(29), to: today() };
      case "custom":
        return { from: customFrom, to: customTo };
    }
  }, [preset, customFrom, customTo]);

  const records = useQuery(
    () =>
      api.listAttendance({
        date_from: range.from,
        date_to: range.to,
        session: session.trim() || undefined,
        student_id: student.trim() || undefined,
      }),
    [range.from, range.to, session, student],
  );

  const rows = useMemo(
    () => [...(records.data?.records ?? [])].reverse(),
    [records.data],
  );

  const sessions = useMemo(
    () => [...new Set((records.data?.records ?? []).map((r) => r.session))].sort(),
    [records.data],
  );

  const download = () => {
    if (!rows.length) {
      toast.info("Nothing to export", "No records match the current filters.");
      return;
    }
    exportCsv(rows);
    toast.success(`Exported ${rows.length} records`, "Saved as a CSV file.");
  };

  return (
    <div className="page">
      <PageHeader
        title="Attendance Log"
        description="Every attendance record, with the confidence score behind each decision."
        actions={
          <>
            <Button icon={<IconRefresh size={15} />} onClick={records.refetch}>
              Refresh
            </Button>
            <Button
              variant="primary"
              icon={<IconDownload size={15} />}
              onClick={download}
            >
              Export CSV
            </Button>
          </>
        }
      />

      <div className="card">
        {/* Filters: one row above the table. */}
        <div className="filters">
          <div className="segmented" role="group" aria-label="Date range">
            {PRESETS.map((option) => (
              <button
                key={option.key}
                className={`segmented__btn ${preset === option.key ? "segmented__btn--active" : ""}`}
                onClick={() => setPreset(option.key)}
                aria-pressed={preset === option.key}
              >
                {option.label}
              </button>
            ))}
          </div>

          {preset === "custom" && (
            <div className="filters__dates">
              <input
                type="date"
                className="input"
                value={customFrom}
                max={customTo}
                onChange={(event) => setCustomFrom(event.target.value)}
                aria-label="From date"
              />
              <span className="filters__dash">–</span>
              <input
                type="date"
                className="input"
                value={customTo}
                min={customFrom}
                onChange={(event) => setCustomTo(event.target.value)}
                aria-label="To date"
              />
            </div>
          )}

          <select
            className="input filters__select"
            value={session}
            onChange={(event) => setSession(event.target.value)}
            aria-label="Filter by session"
          >
            <option value="">All sessions</option>
            {sessions.map((value) => (
              <option key={value} value={value}>
                {value}
              </option>
            ))}
          </select>

          <input
            className="input filters__select"
            value={student}
            onChange={(event) => setStudent(event.target.value)}
            placeholder="Student ID…"
            aria-label="Filter by student ID"
          />

          <div className="spacer" />

          <span className="table__count">
            {records.loading ? "Loading…" : `${rows.length} record${rows.length === 1 ? "" : "s"}`}
          </span>
        </div>

        {records.error ? (
          <ErrorState error={records.error} onRetry={records.refetch} />
        ) : records.loading ? (
          <div className="table__skeleton">
            {[0, 1, 2, 3, 4].map((key) => (
              <Skeleton key={key} height={48} radius="var(--radius)" />
            ))}
          </div>
        ) : rows.length === 0 ? (
          <EmptyState
            icon={<IconList size={22} />}
            title="No records in this range"
            description={`Nothing between ${shortDate(range.from)} and ${shortDate(range.to)}. Try widening the date range, or take attendance from the kiosk.`}
          />
        ) : (
          <div className="table__wrap">
            <table className="table">
              <caption className="sr-only">
                Attendance records from {shortDate(range.from)} to {shortDate(range.to)}
              </caption>
              <thead>
                <tr>
                  <th scope="col">Student</th>
                  <th scope="col">ID</th>
                  <th scope="col">Date</th>
                  <th scope="col">Time</th>
                  <th scope="col">Session</th>
                  <th scope="col">Confidence</th>
                  <th scope="col">Source</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((record) => (
                  <tr key={`${record.student_id}-${record.timestamp}`}>
                    <td>
                      <div className="cell-user">
                        <span className="cell-user__avatar" aria-hidden="true">
                          {record.name.charAt(0).toUpperCase()}
                        </span>
                        <span className="cell-user__name">{record.name}</span>
                      </div>
                    </td>
                    <td className="tabular secondary">{record.student_id}</td>
                    <td className="secondary">{shortDate(record.date)}</td>
                    <td className="tabular secondary">{formatTime(record.timestamp)}</td>
                    <td>
                      <Badge tone="neutral">{record.session}</Badge>
                    </td>
                    <td>
                      <ConfidenceCell value={record.confidence} />
                    </td>
                    <td>
                      <Badge
                        tone={record.source === "manual" ? "warning" : "good"}
                        icon={record.source === "auto" ? <IconCheckCircle /> : undefined}
                      >
                        {record.source === "auto" ? "Recognised" : "Manual"}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Confidence as a meter plus its number.
 *
 * A meter, not a chart: it is one ratio against a limit. The number is always
 * present, so the bar is a reinforcement rather than the only encoding.
 */
function ConfidenceCell({ value }: { value: number }) {
  return (
    <div className="conf">
      <div className="conf__track">
        <div className="conf__fill" style={{ width: `${Math.max(value, 0) * 100}%` }} />
      </div>
      <span className="conf__value tabular">{percent(value)}</span>
    </div>
  );
}
