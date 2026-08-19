"""Persistence layer for students, face embeddings and attendance.

The abstract repositories below define *what* the service stores; the JSON/NumPy
classes define *how* it is stored today. Nothing above this module imports a
concrete class — everything depends on the abstract interfaces — so the planned
MongoDB backend only has to implement the same three interfaces and be wired up
in ``dependencies.py``. No other file changes.

Storage layout::

    storage/embeddings/students.json          # student registry
    storage/embeddings/vectors/<id>.npy       # (n, 512) float32 matrix per student
    storage/attendance/attendance.jsonl       # append-only attendance log

Writes are guarded by a re-entrant lock (FastAPI runs sync endpoints in a thread
pool) and go through a temp file + atomic replace, so an interrupted write can
never leave a half-written registry behind.
"""

from __future__ import annotations

import json
import os
import re
import threading
from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

import numpy as np

from config import Settings
from domain import AttendanceRecord, Student, utc_now
from exceptions import RepositoryError, StudentNotFoundError
from logging_config import get_logger

logger = get_logger(__name__)

#: Student ids must be filesystem- and URL-safe: they name the ``.npy`` files.
_SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def validate_student_id(student_id: str) -> str:
    """Validate and normalise a student identifier.

    Rejecting path separators here is what stops a crafted id such as
    ``../../etc/passwd`` from escaping the storage directory.

    Args:
        student_id: Raw identifier from a caller.

    Returns:
        The trimmed identifier.

    Raises:
        RepositoryError: If the identifier is empty or contains characters that
            are unsafe in a filename.
    """
    candidate = (student_id or "").strip()
    if not _SAFE_ID_PATTERN.match(candidate):
        raise RepositoryError(
            "Invalid student_id. Use 1-64 characters limited to letters, "
            "digits, dot, underscore or hyphen.",
            details={"student_id": student_id},
        )
    return candidate


# ======================================================================
# Abstract interfaces
# ======================================================================
class StudentRepository(ABC):
    """Stores the registry of enrolled students."""

    @abstractmethod
    def add(self, student: Student) -> None:
        """Insert or replace a student record."""

    @abstractmethod
    def get(self, student_id: str) -> Student | None:
        """Return a student, or ``None`` when the id is unknown."""

    @abstractmethod
    def list_all(self) -> list[Student]:
        """Return every registered student."""

    @abstractmethod
    def exists(self, student_id: str) -> bool:
        """Whether a student with this id is registered."""

    @abstractmethod
    def delete(self, student_id: str) -> None:
        """Remove a student.

        Raises:
            StudentNotFoundError: If the id is unknown.
        """

    @abstractmethod
    def count(self) -> int:
        """Number of registered students."""


class EmbeddingRepository(ABC):
    """Stores the face embeddings belonging to each student."""

    @abstractmethod
    def save(self, student_id: str, embeddings: np.ndarray) -> None:
        """Replace the stored embedding matrix for a student."""

    @abstractmethod
    def append(self, student_id: str, embeddings: np.ndarray) -> int:
        """Add embeddings to a student and return the new total."""

    @abstractmethod
    def get(self, student_id: str) -> np.ndarray | None:
        """Return a student's ``(n, dim)`` matrix, or ``None`` if absent."""

    @abstractmethod
    def load_all(self) -> dict[str, np.ndarray]:
        """Return every student's embedding matrix, keyed by student id."""

    @abstractmethod
    def delete(self, student_id: str) -> None:
        """Remove a student's embeddings. Silent when there are none."""

    @abstractmethod
    def count(self, student_id: str) -> int:
        """Number of embeddings stored for a student."""


class AttendanceRepository(ABC):
    """Stores attendance records."""

    @abstractmethod
    def add(self, record: AttendanceRecord) -> None:
        """Append an attendance record."""

    @abstractmethod
    def list_records(
        self,
        student_id: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        session: str | None = None,
    ) -> list[AttendanceRecord]:
        """Return records matching every supplied filter, newest last."""

    @abstractmethod
    def exists_for(self, student_id: str, on_date: date, session: str) -> bool:
        """Whether this student is already marked for this date and session."""

    @abstractmethod
    def latest_for(self, student_id: str, session: str) -> AttendanceRecord | None:
        """Return the most recent record for a student in a session."""

    @abstractmethod
    def count(self) -> int:
        """Total number of attendance records."""


# ======================================================================
# Local file implementations
# ======================================================================
class JsonStudentRepository(StudentRepository):
    """Student registry backed by a single JSON file.

    The registry is small (one entry per student), so it is cached in memory and
    rewritten atomically on every change.

    Args:
        settings: Supplies the storage paths.
    """

    def __init__(self, settings: Settings) -> None:
        self._path: Path = settings.students_file
        self._lock = threading.RLock()
        self._cache: dict[str, Student] = {}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        """Read the registry from disk into memory."""
        if not self._path.is_file():
            self._cache = {}
            return

        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RepositoryError(
                f"Could not read the student registry at {self._path}: {exc}",
                details={"path": str(self._path)},
            ) from exc

        students: dict[str, Student] = {}
        for entry in raw.get("students", []):
            try:
                student = Student.from_dict(entry)
            except (KeyError, TypeError, ValueError) as exc:
                # One malformed row must not make every student unreadable.
                logger.warning("Skipping a malformed student entry: %s", exc)
                continue
            students[student.student_id] = student

        self._cache = students
        logger.info("Loaded %s student(s) from %s", len(students), self._path)

    def _flush(self) -> None:
        """Write the in-memory registry to disk atomically."""
        payload = {
            "version": 1,
            "updated_at": utc_now().isoformat(),
            "students": [student.to_dict() for student in self._cache.values()],
        }
        _atomic_write_text(self._path, json.dumps(payload, indent=2))

    def add(self, student: Student) -> None:
        """Insert or replace a student record."""
        with self._lock:
            self._cache[student.student_id] = student
            self._flush()
            logger.info("Saved student %s (%s)", student.student_id, student.name)

    def get(self, student_id: str) -> Student | None:
        """Return a student, or ``None`` when the id is unknown."""
        with self._lock:
            return self._cache.get(student_id)

    def list_all(self) -> list[Student]:
        """Return every registered student, ordered by id."""
        with self._lock:
            return sorted(self._cache.values(), key=lambda student: student.student_id)

    def exists(self, student_id: str) -> bool:
        """Whether a student with this id is registered."""
        with self._lock:
            return student_id in self._cache

    def delete(self, student_id: str) -> None:
        """Remove a student.

        Raises:
            StudentNotFoundError: If the id is unknown.
        """
        with self._lock:
            if student_id not in self._cache:
                raise StudentNotFoundError(
                    f"No student registered with id '{student_id}'",
                    details={"student_id": student_id},
                )
            del self._cache[student_id]
            self._flush()
            logger.info("Deleted student %s", student_id)

    def count(self) -> int:
        """Number of registered students."""
        with self._lock:
            return len(self._cache)


class NumpyEmbeddingRepository(EmbeddingRepository):
    """Embedding store keeping one ``.npy`` matrix per student.

    Raw face images are never persisted — only the 512-D vectors — which is the
    privacy-by-design measure the project commits to: an embedding cannot be
    viewed as a photograph.

    Args:
        settings: Supplies the vector directory, embedding dimension and the
            per-student embedding cap.
    """

    def __init__(self, settings: Settings) -> None:
        self._dir: Path = settings.vectors_dir
        self._dim: int = settings.embedding_dim
        self._max_per_student: int = settings.max_embeddings_per_student
        self._lock = threading.RLock()
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, student_id: str) -> Path:
        """Return the ``.npy`` path for a validated student id."""
        return self._dir / f"{validate_student_id(student_id)}.npy"

    def _validate_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Coerce input to a ``(n, dim)`` float32 matrix.

        Raises:
            RepositoryError: If the dimensionality does not match the model's.
        """
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        if matrix.ndim != 2 or matrix.shape[1] != self._dim:
            raise RepositoryError(
                f"Embeddings must have shape (n, {self._dim}), got {matrix.shape}",
                details={"expected_dim": self._dim},
            )
        return matrix

    def save(self, student_id: str, embeddings: np.ndarray) -> None:
        """Replace the stored embedding matrix for a student."""
        matrix = self._validate_matrix(embeddings)
        with self._lock:
            matrix = self._enforce_cap(matrix)
            path = self._path_for(student_id)
            try:
                _atomic_write_npy(path, matrix)
            except OSError as exc:
                raise RepositoryError(
                    f"Could not write embeddings for '{student_id}': {exc}",
                    details={"student_id": student_id},
                ) from exc
            logger.info("Stored %s embedding(s) for %s", len(matrix), student_id)

    def append(self, student_id: str, embeddings: np.ndarray) -> int:
        """Add embeddings to a student and return the new total."""
        new_matrix = self._validate_matrix(embeddings)
        with self._lock:
            existing = self.get(student_id)
            combined = (
                new_matrix if existing is None else np.vstack([existing, new_matrix])
            )
            self.save(student_id, combined)
            return int(min(len(combined), self._max_per_student))

    def get(self, student_id: str) -> np.ndarray | None:
        """Return a student's ``(n, dim)`` matrix, or ``None`` if absent."""
        path = self._path_for(student_id)
        with self._lock:
            if not path.is_file():
                return None
            try:
                matrix = np.load(path)
            except (OSError, ValueError) as exc:
                raise RepositoryError(
                    f"Could not read embeddings for '{student_id}': {exc}",
                    details={"student_id": student_id, "path": str(path)},
                ) from exc
        return np.asarray(matrix, dtype=np.float32).reshape(-1, self._dim)

    def load_all(self) -> dict[str, np.ndarray]:
        """Return every student's embedding matrix, keyed by student id.

        A single unreadable file is logged and skipped rather than failing the
        whole gallery load, so one corrupt student cannot take recognition down.
        """
        gallery: dict[str, np.ndarray] = {}
        with self._lock:
            for path in sorted(self._dir.glob("*.npy")):
                student_id = path.stem
                try:
                    matrix = np.load(path)
                except (OSError, ValueError) as exc:
                    logger.error("Skipping unreadable embeddings %s: %s", path, exc)
                    continue
                gallery[student_id] = np.asarray(matrix, dtype=np.float32).reshape(
                    -1, self._dim
                )
        logger.info("Loaded embeddings for %s student(s)", len(gallery))
        return gallery

    def delete(self, student_id: str) -> None:
        """Remove a student's embeddings. Silent when there are none."""
        path = self._path_for(student_id)
        with self._lock:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                raise RepositoryError(
                    f"Could not delete embeddings for '{student_id}': {exc}",
                    details={"student_id": student_id},
                ) from exc
        logger.info("Deleted embeddings for %s", student_id)

    def count(self, student_id: str) -> int:
        """Number of embeddings stored for a student."""
        matrix = self.get(student_id)
        return 0 if matrix is None else int(matrix.shape[0])

    def _enforce_cap(self, matrix: np.ndarray) -> np.ndarray:
        """Keep only the most recent ``max_embeddings_per_student`` rows."""
        if len(matrix) <= self._max_per_student:
            return matrix
        logger.debug(
            "Trimming embeddings from %s to the %s most recent",
            len(matrix),
            self._max_per_student,
        )
        return matrix[-self._max_per_student :]


class JsonlAttendanceRepository(AttendanceRepository):
    """Append-only attendance log stored as JSON Lines.

    An append-only file suits an audit trail: marking attendance is a single
    ``write`` of one line, and existing records are never rewritten. Records are
    mirrored in memory so duplicate checks and queries need no disk reads.

    Args:
        settings: Supplies the attendance log path.
    """

    def __init__(self, settings: Settings) -> None:
        self._path: Path = settings.attendance_file
        self._lock = threading.RLock()
        self._records: list[AttendanceRecord] = []
        #: (student_id, isoformat date, session) tuples for O(1) duplicate checks.
        self._keys: set[tuple[str, str, str]] = set()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        """Read the log from disk into memory."""
        if not self._path.is_file():
            return

        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise RepositoryError(
                f"Could not read the attendance log at {self._path}: {exc}",
                details={"path": str(self._path)},
            ) from exc

        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = AttendanceRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                logger.warning("Skipping malformed attendance line %s: %s", number, exc)
                continue
            self._records.append(record)
            self._keys.add(self._key(record.student_id, record.date, record.session))

        logger.info("Loaded %s attendance record(s) from %s", len(self._records), self._path)

    @staticmethod
    def _key(student_id: str, on_date: date, session: str) -> tuple[str, str, str]:
        """Build the duplicate-detection key for a record."""
        return student_id, on_date.isoformat(), session

    def add(self, record: AttendanceRecord) -> None:
        """Append an attendance record to the log."""
        with self._lock:
            try:
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record.to_dict()) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                raise RepositoryError(
                    f"Could not write the attendance record: {exc}",
                    details={"student_id": record.student_id},
                ) from exc

            self._records.append(record)
            self._keys.add(self._key(record.student_id, record.date, record.session))
            logger.info(
                "Attendance recorded: %s (%s) session=%s confidence=%.3f",
                record.student_id,
                record.name,
                record.session,
                record.confidence,
            )

    def list_records(
        self,
        student_id: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
        session: str | None = None,
    ) -> list[AttendanceRecord]:
        """Return records matching every supplied filter, oldest first."""
        with self._lock:
            results = list(self._records)

        if student_id is not None:
            results = [r for r in results if r.student_id == student_id]
        if session is not None:
            results = [r for r in results if r.session == session]
        if date_from is not None:
            results = [r for r in results if r.date >= date_from]
        if date_to is not None:
            results = [r for r in results if r.date <= date_to]

        results.sort(key=lambda record: record.timestamp)
        return results

    def exists_for(self, student_id: str, on_date: date, session: str) -> bool:
        """Whether this student is already marked for this date and session."""
        with self._lock:
            return self._key(student_id, on_date, session) in self._keys

    def latest_for(self, student_id: str, session: str) -> AttendanceRecord | None:
        """Return the most recent record for a student in a session."""
        with self._lock:
            matching = [
                record
                for record in self._records
                if record.student_id == student_id and record.session == session
            ]
        if not matching:
            return None
        return max(matching, key=lambda record: record.timestamp)

    def count(self) -> int:
        """Total number of attendance records."""
        with self._lock:
            return len(self._records)

    def delete_for_student(self, student_id: str) -> int:
        """Erase a student's attendance history and rewrite the log.

        Supports the right to erasure: unlike :meth:`add`, this rewrites the
        file, since removing history is the whole point of the operation.

        Args:
            student_id: Student whose records are removed.

        Returns:
            The number of records removed.
        """
        with self._lock:
            remaining = [r for r in self._records if r.student_id != student_id]
            removed = len(self._records) - len(remaining)
            if removed == 0:
                return 0

            payload = "".join(
                json.dumps(record.to_dict()) + "\n" for record in remaining
            )
            _atomic_write_text(self._path, payload)
            self._records = remaining
            self._keys = {
                self._key(r.student_id, r.date, r.session) for r in remaining
            }
            logger.info("Removed %s attendance record(s) for %s", removed, student_id)
            return removed


# ======================================================================
# Facade
# ======================================================================
class LocalFileDatabase:
    """Groups the three local-file repositories behind one object.

    Exists so ``dependencies.py`` can swap the entire storage backend (for the
    planned ``MongoDatabase``) by changing a single construction site.

    Args:
        settings: Service configuration.
    """

    def __init__(self, settings: Settings) -> None:
        settings.ensure_directories()
        self._settings = settings
        self.students: StudentRepository = JsonStudentRepository(settings)
        self.embeddings: EmbeddingRepository = NumpyEmbeddingRepository(settings)
        self.attendance: AttendanceRepository = JsonlAttendanceRepository(settings)
        logger.info("Local file database initialised at %s", settings.storage_dir)

    def stats(self) -> dict[str, object]:
        """Return storage counters for the ``/health`` endpoint."""
        return {
            "backend": "local-files",
            "students": self.students.count(),
            "attendance_records": self.attendance.count(),
            "storage_dir": str(self._settings.storage_dir),
        }


# ======================================================================
# Internal helpers
# ======================================================================
def _atomic_write_text(path: Path, content: str) -> None:
    """Write text via a temp file and atomic rename.

    Raises:
        RepositoryError: If the write fails.
    """
    temp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        raise RepositoryError(
            f"Could not write {path}: {exc}", details={"path": str(path)}
        ) from exc


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    """Write a NumPy array via a temp file and atomic rename.

    Raises:
        RepositoryError: If the write fails.
    """
    temp_path = path.with_suffix(".npy.tmp")
    try:
        with temp_path.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        raise RepositoryError(
            f"Could not write {path}: {exc}", details={"path": str(path)}
        ) from exc
