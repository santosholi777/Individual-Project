"""Attendance business rules.

Owns three decisions and nothing else: is this recognition confident enough to
count, has this student already been marked, and what does the record look like.
It knows nothing about cameras, models or HTTP, so the same rules apply whether
attendance arrives from the live CLI, the REST API or the future Node backend.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Sequence

from config import Settings
from database import AttendanceRepository, StudentRepository
from domain import (
    AttendanceOutcome,
    AttendanceRecord,
    AttendanceStatus,
    RecognitionResult,
    utc_now,
)
from exceptions import StudentNotFoundError
from logging_config import get_logger

logger = get_logger(__name__)


class AttendanceService:
    """Applies attendance policy and persists attendance records.

    Args:
        settings: Supplies the attendance threshold and duplicate policy.
        attendance_repository: Where records are stored.
        student_repository: Used to resolve and validate student identities.
    """

    def __init__(
        self,
        settings: Settings,
        attendance_repository: AttendanceRepository,
        student_repository: StudentRepository,
    ) -> None:
        self._settings = settings
        self._attendance = attendance_repository
        self._students = student_repository

    # ------------------------------------------------------------------
    # Marking
    # ------------------------------------------------------------------
    def mark(
        self,
        student_id: str,
        confidence: float,
        session: str | None = None,
        source: str = "auto",
    ) -> AttendanceOutcome:
        """Mark attendance for a student, subject to policy.

        Args:
            student_id: The student to mark.
            confidence: Cosine similarity of the recognition, in ``[0, 1]``.
                Manual entries should pass ``1.0``.
            session: Session label, e.g. ``"lecture-1"``. Defaults to the
                configured session.
            source: ``"auto"`` for recognition, ``"manual"`` for staff entry.

        Returns:
            An outcome describing whether a record was written, skipped as a
            duplicate, or rejected for low confidence. Duplicates and rejections
            are ordinary results, not errors — a camera pointed at a classroom
            re-recognises the same student many times per minute.

        Raises:
            StudentNotFoundError: If the student is not registered.
        """
        student = self._students.get(student_id)
        if student is None:
            raise StudentNotFoundError(
                f"Cannot mark attendance: no student registered with id '{student_id}'",
                details={"student_id": student_id},
            )

        session_label = session or self._settings.default_session
        now = utc_now()

        if source == "auto" and confidence < self._settings.attendance_threshold:
            reason = (
                f"Confidence {confidence:.3f} is below the attendance threshold "
                f"{self._settings.attendance_threshold:.2f}"
            )
            logger.info("Attendance rejected for %s: %s", student_id, reason)
            return AttendanceOutcome(status=AttendanceStatus.REJECTED, reason=reason)

        duplicate_reason = self._duplicate_reason(student_id, session_label, now.date())
        if duplicate_reason is not None:
            logger.debug("Attendance skipped for %s: %s", student_id, duplicate_reason)
            return AttendanceOutcome(
                status=AttendanceStatus.DUPLICATE,
                record=self._attendance.latest_for(student_id, session_label),
                reason=duplicate_reason,
            )

        record = AttendanceRecord(
            student_id=student.student_id,
            name=student.name,
            timestamp=now,
            confidence=float(confidence),
            session=session_label,
            source=source,
        )
        self._attendance.add(record)
        return AttendanceOutcome(status=AttendanceStatus.MARKED, record=record)

    def mark_from_recognition(
        self, result: RecognitionResult, session: str | None = None
    ) -> AttendanceOutcome:
        """Mark attendance directly from a recognition result.

        Args:
            result: The recognition outcome for one face.
            session: Optional session label.

        Returns:
            The attendance outcome; ``REJECTED`` when the face was not
            recognised at all.
        """
        if not result.recognized or result.student_id is None:
            return AttendanceOutcome(
                status=AttendanceStatus.REJECTED,
                reason="Face was not recognised as a registered student",
            )
        return self.mark(
            student_id=result.student_id,
            confidence=result.confidence,
            session=session,
            source="auto",
        )

    def mark_many(
        self, results: Sequence[RecognitionResult], session: str | None = None
    ) -> list[AttendanceOutcome]:
        """Mark attendance for every recognised face in a group photo.

        Args:
            results: Recognition results, typically one classroom frame's worth.
            session: Optional session label.

        Returns:
            One outcome per input result, in the same order.
        """
        return [self.mark_from_recognition(result, session) for result in results]

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def list_records(
        self,
        student_id: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        session: str | None = None,
    ) -> list[AttendanceRecord]:
        """Return attendance records matching every supplied filter.

        Args:
            student_id: Restrict to one student.
            date_from: Earliest date to include (inclusive).
            date_to: Latest date to include (inclusive).
            session: Restrict to one session label.

        Returns:
            Matching records, oldest first.
        """
        return self._attendance.list_records(
            student_id=student_id,
            date_from=date_from,
            date_to=date_to,
            session=session,
        )

    def daily_summary(self, on_date: date | None = None) -> dict[str, object]:
        """Summarise attendance for a single day.

        Args:
            on_date: Day to summarise; defaults to today (UTC).

        Returns:
            Present/absent counts and the attendance rate, ready for the admin
            dashboard.
        """
        target = on_date or utc_now().date()
        records = self._attendance.list_records(date_from=target, date_to=target)

        present_ids = {record.student_id for record in records}
        all_students = self._students.list_all()
        absent = [
            student for student in all_students if student.student_id not in present_ids
        ]
        total = len(all_students)

        return {
            "date": target.isoformat(),
            "total_students": total,
            "present": len(present_ids),
            "absent": len(absent),
            "attendance_rate": round(len(present_ids) / total * 100, 2) if total else 0.0,
            "records": [record.to_dict() for record in records],
            "absentees": [
                {"student_id": student.student_id, "name": student.name}
                for student in absent
            ],
        }

    def is_marked(
        self, student_id: str, session: str | None = None, on_date: date | None = None
    ) -> bool:
        """Whether a student is already marked for a date and session.

        Args:
            student_id: Student to check.
            session: Session label; defaults to the configured session.
            on_date: Date to check; defaults to today (UTC).

        Returns:
            True when a matching record exists.
        """
        return self._attendance.exists_for(
            student_id=student_id,
            on_date=on_date or utc_now().date(),
            session=session or self._settings.default_session,
        )

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------
    def _duplicate_reason(
        self, student_id: str, session: str, today: date
    ) -> str | None:
        """Return why this mark is a duplicate, or ``None`` if it is not.

        Two independent rules, either of which can suppress a record:

        * ``once_per_session_per_day`` — the intended classroom behaviour: one
          record per student per session per day.
        * ``duplicate_cooldown_minutes`` — a time-based guard for continuous
          camera feeds, and the rule that still applies when the once-per-day
          rule is switched off (e.g. entry/exit logging).
        """
        if self._settings.once_per_session_per_day and self._attendance.exists_for(
            student_id, today, session
        ):
            return (
                f"Attendance already marked for {student_id} in session "
                f"'{session}' on {today.isoformat()}"
            )

        cooldown = self._settings.duplicate_cooldown_minutes
        if cooldown > 0:
            latest = self._attendance.latest_for(student_id, session)
            if latest is not None:
                elapsed = utc_now() - latest.timestamp
                if elapsed < timedelta(minutes=cooldown):
                    remaining = timedelta(minutes=cooldown) - elapsed
                    return (
                        f"Marked {int(elapsed.total_seconds())}s ago; "
                        f"{int(remaining.total_seconds())}s of the {cooldown}-minute "
                        "cooldown remain"
                    )
        return None
