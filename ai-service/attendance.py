"""Command line attendance reporting.

Query, summarise and export the attendance log without going through the API.

Usage::

    python attendance.py list                              # today's records
    python attendance.py list --student-id CS2021001       # one student
    python attendance.py list --from 2026-07-01 --to 2026-07-15
    python attendance.py summary                           # today's totals
    python attendance.py summary --date 2026-07-14
    python attendance.py export --output report.csv        # CSV for the report
    python attendance.py mark --student-id CS2021001       # manual entry
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date, datetime
from pathlib import Path

from config import get_settings
from dependencies import get_container
from domain import AttendanceRecord
from exceptions import DeepVisionAttendError
from logging_config import get_logger, setup_logging

logger = get_logger(__name__)

_CSV_COLUMNS = ["student_id", "name", "date", "timestamp", "session", "confidence", "source"]


def _parse_date(value: str) -> date:
    """Parse a ``YYYY-MM-DD`` command line date.

    Args:
        value: The raw argument.

    Returns:
        The parsed date.

    Raises:
        argparse.ArgumentTypeError: If the format is wrong.
    """
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{value}'. Expected the format YYYY-MM-DD."
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    """Define the command line interface."""
    parser = argparse.ArgumentParser(
        prog="attendance.py",
        description="Query and export DeepVisionAttend attendance records.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_filters(target: argparse.ArgumentParser) -> None:
        """Attach the filter flags shared by list and export."""
        target.add_argument("--student-id", default=None, help="Filter by student")
        target.add_argument("--session", default=None, help="Filter by session label")
        target.add_argument(
            "--from", dest="date_from", type=_parse_date, default=None,
            help="Earliest date, inclusive (YYYY-MM-DD)",
        )
        target.add_argument(
            "--to", dest="date_to", type=_parse_date, default=None,
            help="Latest date, inclusive (YYYY-MM-DD)",
        )

    list_parser = subparsers.add_parser("list", help="List attendance records")
    add_filters(list_parser)
    list_parser.add_argument(
        "--all", action="store_true", help="Include every date, not just today"
    )

    summary_parser = subparsers.add_parser("summary", help="Daily attendance summary")
    summary_parser.add_argument(
        "--date", dest="on_date", type=_parse_date, default=None,
        help="Day to summarise (defaults to today)",
    )

    export_parser = subparsers.add_parser("export", help="Export records to CSV")
    add_filters(export_parser)
    export_parser.add_argument(
        "--output", type=Path, required=True, help="Destination .csv file"
    )

    mark_parser = subparsers.add_parser("mark", help="Manually mark a student present")
    mark_parser.add_argument("--student-id", required=True, help="Student to mark")
    mark_parser.add_argument("--session", default=None, help="Session label")

    return parser


def _print_table(records: list[AttendanceRecord]) -> None:
    """Print attendance records as an aligned table."""
    if not records:
        print("\nNo attendance records match this query.")
        return

    header = f"{'STUDENT ID':<14} {'NAME':<24} {'DATE':<12} {'TIME':<10} {'SESSION':<14} {'CONF':>6} {'SOURCE':<8}"
    print("\n" + header)
    print("-" * len(header))
    for record in records:
        local_time = record.timestamp.astimezone()
        print(
            f"{record.student_id:<14} {record.name[:23]:<24} "
            f"{record.date.isoformat():<12} {local_time.strftime('%H:%M:%S'):<10} "
            f"{record.session[:13]:<14} {record.confidence:>6.3f} {record.source:<8}"
        )
    print("-" * len(header))
    print(f"{len(records)} record(s)\n")


def command_list(args: argparse.Namespace) -> int:
    """Run the ``list`` subcommand.

    Defaults to today so a lecturer's most common query needs no flags; ``--all``
    or an explicit date range widens it.
    """
    service = get_container().attendance_service
    today = date.today()

    date_from = args.date_from
    date_to = args.date_to
    if not args.all and date_from is None and date_to is None:
        date_from = date_to = today

    records = service.list_records(
        student_id=args.student_id,
        date_from=date_from,
        date_to=date_to,
        session=args.session,
    )
    _print_table(records)
    return 0


def command_summary(args: argparse.Namespace) -> int:
    """Run the ``summary`` subcommand."""
    service = get_container().attendance_service
    summary = service.daily_summary(args.on_date)

    print("\n" + "=" * 52)
    print(f"  Attendance summary for {summary['date']}")
    print("=" * 52)
    print(f"  Total students : {summary['total_students']}")
    print(f"  Present        : {summary['present']}")
    print(f"  Absent         : {summary['absent']}")
    print(f"  Attendance rate: {summary['attendance_rate']}%")
    print("=" * 52)

    absentees = summary["absentees"]
    if absentees:
        print("\n  Absent students:")
        for student in absentees:  # type: ignore[union-attr]
            print(f"    - {student['name']} ({student['student_id']})")
    print()
    return 0


def command_export(args: argparse.Namespace) -> int:
    """Run the ``export`` subcommand, writing a CSV report."""
    service = get_container().attendance_service
    records = service.list_records(
        student_id=args.student_id,
        date_from=args.date_from,
        date_to=args.date_to,
        session=args.session,
    )

    output: Path = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            payload = record.to_dict()
            writer.writerow({column: payload[column] for column in _CSV_COLUMNS})

    print(f"\nExported {len(records)} record(s) to {output}\n")
    return 0


def command_mark(args: argparse.Namespace) -> int:
    """Run the ``mark`` subcommand for manual staff entry."""
    service = get_container().attendance_service
    outcome = service.mark(
        student_id=args.student_id,
        confidence=1.0,
        session=args.session,
        source="manual",
    )

    if outcome.marked and outcome.record is not None:
        print(f"\nMarked {outcome.record.name} ({outcome.record.student_id}) present.\n")
        return 0

    print(f"\nNot marked [{outcome.status.value}]: {outcome.reason}\n")
    return 1


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python attendance.py``.

    Args:
        argv: Command line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    args = build_parser().parse_args(argv)
    settings = get_settings()
    setup_logging(
        level="WARNING",  # keep report output clean; details still go to the log file
        log_dir=settings.logs_dir,
        log_to_file=settings.log_to_file,
    )

    handlers = {
        "list": command_list,
        "summary": command_summary,
        "export": command_export,
        "mark": command_mark,
    }

    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        return 130
    except DeepVisionAttendError as exc:
        print(f"\nError: {exc.message}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"\nUnexpected error: {exc}", file=sys.stderr)
        logger.exception("Unexpected error in the attendance CLI")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
