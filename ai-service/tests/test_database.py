"""Tests for the local-file persistence layer."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from config import Settings
from database import LocalFileDatabase, validate_student_id
from domain import AttendanceRecord, Student, utc_now
from exceptions import RepositoryError, StudentNotFoundError


class TestValidateStudentId:
    """Behaviour of :func:`database.validate_student_id`."""

    @pytest.mark.parametrize("student_id", ["CS2021001", "abc", "a.b_c-1", "1"])
    def test_accepts_safe_identifiers(self, student_id: str) -> None:
        """Alphanumerics, dot, underscore and hyphen are allowed."""
        assert validate_student_id(student_id) == student_id

    @pytest.mark.parametrize(
        "student_id",
        ["", "   ", "../etc/passwd", "a/b", "a\\b", "id with spaces", "x" * 65],
    )
    def test_rejects_unsafe_identifiers(self, student_id: str) -> None:
        """Ids name files on disk, so path characters must be refused."""
        with pytest.raises(RepositoryError, match="Invalid student_id"):
            validate_student_id(student_id)

    def test_path_traversal_cannot_escape_storage(
        self, database: LocalFileDatabase
    ) -> None:
        """A crafted id must not write outside the vectors directory."""
        with pytest.raises(RepositoryError):
            database.embeddings.save("../../evil", np.zeros((1, 512), dtype=np.float32))


class TestJsonStudentRepository:
    """Behaviour of the JSON-backed student registry."""

    def test_add_and_get_round_trip(self, database: LocalFileDatabase) -> None:
        """A stored student comes back with its fields intact."""
        database.students.add(Student(student_id="S1", name="Alice", embedding_count=3))

        stored = database.students.get("S1")
        assert stored is not None
        assert stored.name == "Alice"
        assert stored.embedding_count == 3

    def test_get_unknown_returns_none(self, database: LocalFileDatabase) -> None:
        """An unknown id is a normal 'not found', not an error."""
        assert database.students.get("nope") is None

    def test_survives_a_restart(
        self, settings: Settings, database: LocalFileDatabase
    ) -> None:
        """The registry is durable: a fresh instance re-reads it from disk."""
        database.students.add(Student(student_id="S1", name="Alice"))

        reopened = LocalFileDatabase(settings)
        stored = reopened.students.get("S1")
        assert stored is not None
        assert stored.name == "Alice"

    def test_add_replaces_an_existing_student(
        self, database: LocalFileDatabase
    ) -> None:
        """Re-adding the same id updates rather than duplicates."""
        database.students.add(Student(student_id="S1", name="Alice"))
        database.students.add(Student(student_id="S1", name="Alice Smith"))

        assert database.students.count() == 1
        assert database.students.get("S1").name == "Alice Smith"  # type: ignore[union-attr]

    def test_list_all_is_sorted_by_id(self, database: LocalFileDatabase) -> None:
        """Listing is deterministic so the dashboard order is stable."""
        for student_id in ("S3", "S1", "S2"):
            database.students.add(Student(student_id=student_id, name=student_id))

        assert [s.student_id for s in database.students.list_all()] == ["S1", "S2", "S3"]

    def test_delete_removes_the_student(self, database: LocalFileDatabase) -> None:
        """Deletion takes effect immediately."""
        database.students.add(Student(student_id="S1", name="Alice"))
        database.students.delete("S1")

        assert database.students.get("S1") is None
        assert database.students.count() == 0

    def test_delete_unknown_raises(self, database: LocalFileDatabase) -> None:
        """Deleting a non-existent student is an error worth surfacing."""
        with pytest.raises(StudentNotFoundError):
            database.students.delete("ghost")


class TestNumpyEmbeddingRepository:
    """Behaviour of the ``.npy``-backed embedding store."""

    def test_save_and_get_round_trip(self, database: LocalFileDatabase) -> None:
        """Vectors survive the round trip unchanged."""
        vectors = np.random.default_rng(0).normal(size=(3, 512)).astype(np.float32)
        database.embeddings.save("S1", vectors)

        loaded = database.embeddings.get("S1")
        assert loaded is not None
        assert loaded.shape == (3, 512)
        assert np.allclose(loaded, vectors)

    def test_get_unknown_returns_none(self, database: LocalFileDatabase) -> None:
        """A student with no embeddings yields None."""
        assert database.embeddings.get("nobody") is None

    def test_a_single_vector_is_stored_as_a_matrix(
        self, database: LocalFileDatabase
    ) -> None:
        """1-D input is reshaped, so callers need not care about the shape."""
        database.embeddings.save("S1", np.zeros(512, dtype=np.float32))
        assert database.embeddings.get("S1").shape == (1, 512)  # type: ignore[union-attr]

    def test_wrong_dimensionality_raises(self, database: LocalFileDatabase) -> None:
        """A vector of the wrong width means the wrong model was used."""
        with pytest.raises(RepositoryError, match=r"shape \(n, 512\)"):
            database.embeddings.save("S1", np.zeros((2, 128), dtype=np.float32))

    def test_append_accumulates(self, database: LocalFileDatabase) -> None:
        """Appending adds to, rather than replaces, the stored matrix."""
        database.embeddings.save("S1", np.zeros((2, 512), dtype=np.float32))
        total = database.embeddings.append("S1", np.ones((3, 512), dtype=np.float32))

        assert total == 5
        assert database.embeddings.count("S1") == 5

    def test_append_to_a_new_student_creates_the_matrix(
        self, database: LocalFileDatabase
    ) -> None:
        """Appending without a prior save is allowed."""
        assert database.embeddings.append("New", np.zeros((2, 512), dtype=np.float32)) == 2

    def test_enforces_the_per_student_cap(self, settings: Settings) -> None:
        """Old embeddings are dropped once the cap is exceeded."""
        settings.max_embeddings_per_student = 3
        database = LocalFileDatabase(settings)
        database.embeddings.save("S1", np.ones((10, 512), dtype=np.float32))

        assert database.embeddings.count("S1") == 3

    def test_load_all_returns_every_student(self, database: LocalFileDatabase) -> None:
        """The gallery load sees each stored student exactly once."""
        database.embeddings.save("S1", np.zeros((2, 512), dtype=np.float32))
        database.embeddings.save("S2", np.zeros((1, 512), dtype=np.float32))

        gallery = database.embeddings.load_all()
        assert set(gallery) == {"S1", "S2"}
        assert gallery["S1"].shape == (2, 512)

    def test_delete_is_idempotent(self, database: LocalFileDatabase) -> None:
        """Deleting absent embeddings is a no-op, not an error."""
        database.embeddings.delete("never-existed")
        assert database.embeddings.count("never-existed") == 0


class TestJsonlAttendanceRepository:
    """Behaviour of the append-only attendance log."""

    @staticmethod
    def _record(
        student_id: str = "S1", session: str = "general", days_ago: int = 0
    ) -> AttendanceRecord:
        """Build an attendance record for tests."""
        return AttendanceRecord(
            student_id=student_id,
            name=f"Student {student_id}",
            timestamp=utc_now() - timedelta(days=days_ago),
            confidence=0.9,
            session=session,
        )

    def test_add_and_list_round_trip(self, database: LocalFileDatabase) -> None:
        """A written record is immediately queryable."""
        database.attendance.add(self._record())

        records = database.attendance.list_records()
        assert len(records) == 1
        assert records[0].student_id == "S1"

    def test_survives_a_restart(
        self, settings: Settings, database: LocalFileDatabase
    ) -> None:
        """The log is durable across process restarts."""
        database.attendance.add(self._record())
        assert LocalFileDatabase(settings).attendance.count() == 1

    def test_exists_for_detects_a_duplicate(self, database: LocalFileDatabase) -> None:
        """The duplicate key is (student, date, session)."""
        database.attendance.add(self._record(session="lecture-1"))
        today = date.today()

        assert database.attendance.exists_for("S1", today, "lecture-1") is True
        assert database.attendance.exists_for("S1", today, "lecture-2") is False
        assert database.attendance.exists_for("S2", today, "lecture-1") is False

    def test_filters_by_student(self, database: LocalFileDatabase) -> None:
        """Filtering narrows to one student."""
        database.attendance.add(self._record("S1"))
        database.attendance.add(self._record("S2"))

        assert len(database.attendance.list_records(student_id="S1")) == 1

    def test_filters_by_date_range(self, database: LocalFileDatabase) -> None:
        """A date range excludes records outside it, inclusive at both ends."""
        database.attendance.add(self._record(days_ago=0))
        database.attendance.add(self._record(student_id="S2", days_ago=10))

        today = date.today()
        recent = database.attendance.list_records(
            date_from=today - timedelta(days=2), date_to=today
        )
        assert len(recent) == 1

    def test_filters_by_session(self, database: LocalFileDatabase) -> None:
        """Sessions partition the same day's records."""
        database.attendance.add(self._record(session="lecture-1"))
        database.attendance.add(self._record(session="lab-2"))

        assert len(database.attendance.list_records(session="lab-2")) == 1

    def test_latest_for_returns_the_newest(self, database: LocalFileDatabase) -> None:
        """The cooldown rule depends on this being the most recent record."""
        database.attendance.add(self._record(days_ago=5))
        database.attendance.add(self._record(days_ago=0))

        latest = database.attendance.latest_for("S1", "general")
        assert latest is not None
        assert latest.date == date.today()

    def test_delete_for_student_erases_their_history(
        self, database: LocalFileDatabase
    ) -> None:
        """Erasure removes only the named student's records."""
        database.attendance.add(self._record("S1"))
        database.attendance.add(self._record("S2"))

        removed = database.attendance.delete_for_student("S1")  # type: ignore[attr-defined]
        assert removed == 1
        assert database.attendance.count() == 1
        assert database.attendance.exists_for("S1", date.today(), "general") is False
