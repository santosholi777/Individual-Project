/**
 * Admin dashboard: today's headline numbers, a 7-day trend and the absentee list.
 *
 * The numbers come from /attendance/summary; the trend is aggregated client-side
 * from a single ranged /attendance query rather than seven separate calls.
 */

import { useMemo } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { AttendanceChart, type DayPoint } from "../components/charts/AttendanceChart";
import { StatTile } from "../components/charts/StatTile";
import { PageHeader } from "../components/layout/PageHeader";
import { Badge } from "../components/ui/Badge";
import { Button } from "../components/ui/Button";
import {
  IconCheckCircle,
  IconRefresh,
  IconScan,
  IconUsers,
  IconX,
} from "../components/ui/icons";
import { EmptyState, ErrorState, Skeleton } from "../components/ui/States";
import { useQuery } from "../hooks/useApi";
import { formatTime, isoDate, shortDate } from "../utils/format";

const TREND_DAYS = 7;

/** Build the last N calendar dates, oldest first. */
function recentDates(days: number): string[] {
  const today = new Date();
  return Array.from({ length: days }, (_, index) => {
    const day = new Date(today);
    day.setDate(today.getDate() - (days - 1 - index));
    return isoDate(day);
  });
}

export function Dashboard() {
  const dates = useMemo(() => recentDates(TREND_DAYS), []);
  const today = dates[dates.length - 1];

  const summary = useQuery(() => api.attendanceSummary(today), [today]);
  const trend = useQuery(
    () => api.listAttendance({ date_from: dates[0], date_to: today }),
    [dates[0], today],
  );

  // One student can appear several times a day across sessions; the trend counts
  // distinct students per day, which is what "present" means on a register.
  const trendData: DayPoint[] = useMemo(() => {
    const byDate = new Map<string, Set<string>>();
    for (const date of dates) byDate.set(date, new Set());
    for (const record of trend.data?.records ?? []) {
      byDate.get(record.date)?.add(record.student_id);
    }
    return dates.map((date) => ({
      date,
      present: byDate.get(date)?.size ?? 0,
    }));
  }, [dates, trend.data]);

  const loading = summary.loading || trend.loading;
  const error = summary.error ?? trend.error;

  const refetch = () => {
    summary.refetch();
    trend.refetch();
  };

  const rate = summary.data?.attendance_rate ?? 0;
  const rateTone = rate >= 75 ? "good" : rate >= 50 ? "warning" : "critical";

  return (
    <div className="page">
      <PageHeader
        title="Dashboard"
        description={`Attendance overview for ${shortDate(today)}.`}
        actions={
          <>
            <Button icon={<IconRefresh size={15} />} onClick={refetch}>
              Refresh
            </Button>
            <Link to="/kiosk" className="unstyled-link">
              <Button variant="primary" icon={<IconScan size={15} />}>
                Take attendance
              </Button>
            </Link>
          </>
        }
      />

      {error ? (
        <div className="card">
          <ErrorState error={error} onRetry={refetch} />
        </div>
      ) : (
        <>
          <section className="dash__stats" aria-label="Today's totals">
            <StatTile
              label="Attendance rate"
              value={`${rate}%`}
              meta={`${summary.data?.present ?? 0} of ${summary.data?.total_students ?? 0} students`}
              tone={rateTone}
              icon={<IconCheckCircle size={15} />}
              loading={loading}
              hero
            />
            <StatTile
              label="Present today"
              value={summary.data?.present ?? 0}
              meta={summary.data?.records.length ? `${summary.data.records.length} check-ins` : "No check-ins yet"}
              tone="good"
              icon={<IconCheckCircle size={15} />}
              loading={loading}
            />
            <StatTile
              label="Absent today"
              value={summary.data?.absent ?? 0}
              meta="Not yet recognised"
              tone={summary.data?.absent ? "critical" : "default"}
              icon={<IconX size={15} />}
              loading={loading}
            />
            <StatTile
              label="Enrolled students"
              value={summary.data?.total_students ?? 0}
              meta="Registered faces"
              icon={<IconUsers size={15} />}
              loading={loading}
            />
          </section>

          <section className="dash__grid">
            <div className="card dash__panel">
              <div className="dash__panel-head">
                <div>
                  <h2 className="dash__panel-title">Students present</h2>
                  <p className="dash__panel-sub">Distinct students per day, last 7 days</p>
                </div>
              </div>
              {loading ? (
                <Skeleton height={240} radius="var(--radius)" />
              ) : (
                <AttendanceChart
                  data={trendData}
                  totalStudents={summary.data?.total_students ?? 0}
                />
              )}
            </div>

            <div className="card dash__panel">
              <div className="dash__panel-head">
                <div>
                  <h2 className="dash__panel-title">Recent check-ins</h2>
                  <p className="dash__panel-sub">Today, most recent first</p>
                </div>
              </div>

              {loading ? (
                <div className="stack">
                  {[0, 1, 2].map((key) => (
                    <Skeleton key={key} height={44} radius="var(--radius)" />
                  ))}
                </div>
              ) : summary.data?.records.length ? (
                <ul className="feed">
                  {[...summary.data.records]
                    .reverse()
                    .slice(0, 6)
                    .map((record) => (
                      <li key={`${record.student_id}-${record.timestamp}`} className="feed__item">
                        <span className="feed__avatar" aria-hidden="true">
                          {record.name.charAt(0).toUpperCase()}
                        </span>
                        <div className="feed__text">
                          <span className="feed__name">{record.name}</span>
                          <span className="feed__meta">
                            {record.student_id} · {record.session}
                          </span>
                        </div>
                        <div className="feed__right">
                          <span className="feed__time tabular">
                            {formatTime(record.timestamp)}
                          </span>
                          <Badge tone={record.source === "manual" ? "neutral" : "accent"}>
                            {record.source === "manual"
                              ? "Manual"
                              : `${Math.round(record.confidence * 100)}%`}
                          </Badge>
                        </div>
                      </li>
                    ))}
                </ul>
              ) : (
                <EmptyState
                  title="No check-ins yet today"
                  description="Open the kiosk and let students look at the camera."
                  action={
                    <Link to="/kiosk">
                      <Button variant="primary" icon={<IconScan size={15} />}>
                        Take attendance
                      </Button>
                    </Link>
                  }
                />
              )}
            </div>
          </section>

          {!loading && (summary.data?.absentees.length ?? 0) > 0 && (
            <section className="card dash__panel dash__absentees">
              <div className="dash__panel-head">
                <div>
                  <h2 className="dash__panel-title">Absentees</h2>
                  <p className="dash__panel-sub">
                    {summary.data?.absentees.length} enrolled students with no record today
                  </p>
                </div>
              </div>
              <ul className="chips">
                {summary.data?.absentees.map((student) => (
                  <li key={student.student_id} className="chip">
                    <span className="chip__name">{student.name}</span>
                    <span className="chip__id">{student.student_id}</span>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </>
      )}
    </div>
  );
}
