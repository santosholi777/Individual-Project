"""Tests for the attendance policy rules."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from config import Settings
from database import LocalFileDatabase
from domain import AttendanceStatus, RecognitionResult, Student
from exceptions import StudentNotFoundError
from services.attendance_service import AttendanceService
from tests.conftest import make_face


@pytest.fixture()
def enrolled(database: LocalFileDatabase) -> LocalFileDatabase:
    """A database with two registered students."""
    database.students.add(Student(student_id="S1", name="Alice"))
    database.students.add(Student(student_id="S2", name="Bob"))
    return database


class TestMark:
    """Behaviour of :meth:`AttendanceService.mark`."""

    def test_marks_a_confident_recognition(
        self, attendance_service: AttendanceService, enrolled: LocalFileDatabase
    ) -> None:
        """A confident match is written to the log."""
        outcome = attendance_service.mark("S1", confidence=0.9)

        assert outcome.status is AttendanceStatus.MARKED
        assert outcome.record is not None
        assert outcome.record.name == "Alice"
        assert enrolled.attendance.count() == 1

    def test_records_a_timestamp(
        self, attendance_service: AttendanceService, enrolled: LocalFileDatabase
    ) -> None:
        """Every record carries the time it was taken."""
        outcome = attendance_service.mark("S1", confidence=0.9)

        assert outcome.record is not None
        assert outcome.record.date == date.today()
        assert outcome.record.timestamp.tzinfo is not None

    def test_rejects_low_confidence(
        self, attendance_service: AttendanceService, enrolled: LocalFileDatabase
    ) -> None:
        """Below the attendance threshold, nothing is written."""
        outcome = attendance_service.mark("S1", confidence=0.3)

        assert outcome.status is AttendanceStatus.REJECTED
        assert enrolled.attendance.count() == 0

    def test_manual_entry_bypasses_the_confidence_gate(
        self, attendance_service: AttendanceService, enrolled: LocalFileDatabase
    ) -> None:
        """Staff overrides are not subject to a model's confidence."""
        outcome = attendance_service.mark("S1", confidence=0.0, source="manual")

        assert outcome.status is AttendanceStatus.MARKED

    def test_unknown_student_raises(
        self, attendance_service: AttendanceService, enrolled: LocalFileDatabase
    ) -> None:
        """Attendance cannot be attributed to somebody unregistered."""
        with pytest.raises(StudentNotFoundError):
            attendance_service.mark("ghost", confidence=0.9)


class TestDuplicatePrevention:
    """The duplicate rules that stop a live camera flooding the log."""

    def test_second_mark_same_session_is_a_duplicate(
        self, attendance_service: AttendanceService, enrolled: LocalFileDatabase
    ) -> None:
        """One record per student per session per day."""
        attendance_service.mark("S1", confidence=0.9, session="lecture-1")
        outcome = attendance_service.mark("S1", confidence=0.95, session="lecture-1")

        assert outcome.status is AttendanceStatus.DUPLICATE
        assert enrolled.attendance.count() == 1

    def test_a_duplicate_returns_the_original_record(
        self, attendance_service: AttendanceService, enrolled: LocalFileDatabase
    ) -> None:
        """The caller can still show when the student was actually marked."""
        first = attendance_service.mark("S1", confidence=0.9)
        duplicate = attendance_service.mark("S1", confidence=0.9)

        assert duplicate.record is not None
        assert first.record is not None
        assert duplicate.record.timestamp == first.record.timestamp

    def test_different_sessions_are_marked_independently(
        self, attendance_service: AttendanceService, enrolled: LocalFileDatabase
    ) -> None:
        """A student attends several lectures a day."""
        assert attendance_service.mark("S1", 0.9, session="lecture-1").marked
        assert attendance_service.mark("S1", 0.9, session="lecture-2").marked
        assert enrolled.attendance.count() == 2

    def test_different_students_are_marked_independently(
        self, attendance_service: AttendanceService, enrolled: LocalFileDatabase
    ) -> None:
        """One student's record does not block another's."""
        assert attendance_service.mark("S1", 0.9).marked
        assert attendance_service.mark("S2", 0.9).marked

    def test_cooldown_blocks_a_rapid_remark(
        self, settings: Settings, enrolled: LocalFileDatabase
    ) -> None:
        """With the daily rule off, the cooldown still suppresses re-marking."""
        settings.once_per_session_per_day = False
        settings.duplicate_cooldown_minutes = 30
        service = AttendanceService(settings, enrolled.attendance, enrolled.students)

        assert service.mark("S1", 0.9).marked
        assert service.mark("S1", 0.9).status is AttendanceStatus.DUPLICATE

    def test_both_rules_off_allows_repeat_marks(
        self, settings: Settings, enrolled: LocalFileDatabase
    ) -> None:
        """Entry/exit logging is possible when both guards are disabled."""
        settings.once_per_session_per_day = False
        settings.duplicate_cooldown_minutes = 0
        service = AttendanceService(settings, enrolled.attendance, enrolled.students)

        assert service.mark("S1", 0.9).marked
        assert service.mark("S1", 0.9).marked
        assert enrolled.attendance.count() == 2


class TestMarkFromRecognition:
    """Bridging recognition results into attendance."""

    def test_marks_a_recognised_face(
        self, attendance_service: AttendanceService, enrolled: LocalFileDatabase
    ) -> None:
        """A recognised result becomes a record."""
        result = RecognitionResult(
            face=make_face(), recognized=True, student_id="S1", name="Alice", similarity=0.88
        )
        assert attendance_service.mark_from_recognition(result).marked

    def test_unknown_face_is_rejected(
        self, attendance_service: AttendanceService, enrolled: LocalFileDatabase
    ) -> None:
        """An unrecognised face cannot be attributed to anyone."""
        result = RecognitionResult(face=make_face(), recognized=False, similarity=0.2)
        outcome = attendance_service.mark_from_recognition(result)

        assert outcome.status is AttendanceStatus.REJECTED
        assert enrolled.attendance.count() == 0

    def test_mark_many_returns_one_outcome_per_face(
        self, attendance_service: AttendanceService, enrolled: LocalFileDatabase
    ) -> None:
        """Group photos yield an outcome per detected face, in order."""
        results = [
            RecognitionResult(face=make_face(), recognized=True, student_id="S1", name="Alice", similarity=0.9),
            RecognitionResult(face=make_face(), recognized=False, similarity=0.1),
            RecognitionResult(face=make_face(), recognized=True, student_id="S2", name="Bob", similarity=0.85),
        ]
        outcomes = attendance_service.mark_many(results)

        assert [outcome.status for outcome in outcomes] == [
            AttendanceStatus.MARKED,
            AttendanceStatus.REJECTED,
            AttendanceStatus.MARKED,
        ]


class TestQueries:
    """Reporting helpers used by the dashboard and the CLI."""

    def test_daily_summary_counts_present_and_absent(
        self, attendance_service: AttendanceService, enrolled: LocalFileDatabase
    ) -> None:
        """Absentees are everyone registered who has no record today."""
        attendance_service.mark("S1", confidence=0.9)
        summary = attendance_service.daily_summary()

        assert summary["total_students"] == 2
        assert summary["present"] == 1
        assert summary["absent"] == 1
        assert summary["attendance_rate"] == 50.0
        assert summary["absentees"] == [{"student_id": "S2", "name": "Bob"}]

    def test_daily_summary_of_an_empty_day(
        self, attendance_service: AttendanceService, enrolled: LocalFileDatabase
    ) -> None:
        """A day with no records reports a 0% rate, not a division error."""
        summary = attendance_service.daily_summary(date.today() - timedelta(days=1))

        assert summary["present"] == 0
        assert summary["attendance_rate"] == 0.0

    def test_is_marked_reflects_the_log(
        self, attendance_service: AttendanceService, enrolled: LocalFileDatabase
    ) -> None:
        """The idempotency check the kiosk uses before re-marking."""
        assert attendance_service.is_marked("S1") is False
        attendance_service.mark("S1", confidence=0.9)
        assert attendance_service.is_marked("S1") is True
