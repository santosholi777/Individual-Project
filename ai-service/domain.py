"""Core domain types shared across the DeepVisionAttend AI service.

These are plain, framework-free dataclasses. Detection, embedding, matching,
persistence and the API layer all speak in terms of these types, which is what
keeps the storage backend (local files today, MongoDB later) and the transport
(CLI today, HTTP now, anything later) swappable without touching the pipeline.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

import numpy as np


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """An axis-aligned face box in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Box width in pixels."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Box height in pixels."""
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        return self.width * self.height

    @property
    def min_side(self) -> float:
        """Length of the shorter side — used for minimum-face-size checks."""
        return min(self.width, self.height)

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        """Return ``(x1, y1, x2, y2)`` rounded to integers for drawing/cropping."""
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    def to_dict(self) -> dict[str, float]:
        """Serialise for JSON responses."""
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @classmethod
    def from_array(cls, values: np.ndarray) -> "BoundingBox":
        """Build a box from a detector's ``[x1, y1, x2, y2]`` output."""
        x1, y1, x2, y2 = (float(v) for v in np.asarray(values).flatten()[:4])
        return cls(x1=x1, y1=y1, x2=x2, y2=y2)


@dataclass(slots=True)
class DetectedFace:
    """A single face located in an image by the detector.

    Attributes:
        bbox: Face bounding box in pixel coordinates.
        det_score: Detector confidence in ``[0, 1]``.
        keypoints: The five ArcFace landmarks (eyes, nose, mouth corners) with
            shape ``(5, 2)``. Required to align the crop before embedding.
    """

    bbox: BoundingBox
    det_score: float
    keypoints: np.ndarray

    def __post_init__(self) -> None:
        self.keypoints = np.asarray(self.keypoints, dtype=np.float32)
        if self.keypoints.shape != (5, 2):
            raise ValueError(
                f"keypoints must have shape (5, 2), got {self.keypoints.shape}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON responses."""
        return {"bbox": self.bbox.to_dict(), "det_score": round(self.det_score, 4)}


@dataclass(slots=True)
class Student:
    """A registered student and the metadata describing their enrolment."""

    student_id: str
    name: str
    embedding_count: int = 0
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible mapping."""
        return {
            "student_id": self.student_id,
            "name": self.name,
            "embedding_count": self.embedding_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Student":
        """Rebuild a student from its serialised form."""
        return cls(
            student_id=str(data["student_id"]),
            name=str(data["name"]),
            embedding_count=int(data.get("embedding_count", 0)),
            created_at=_parse_datetime(data.get("created_at")),
            updated_at=_parse_datetime(data.get("updated_at")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class FaceEmbedding:
    """An L2-normalised ArcFace embedding tied to a student."""

    student_id: str
    vector: np.ndarray

    def __post_init__(self) -> None:
        self.vector = np.asarray(self.vector, dtype=np.float32).flatten()


@dataclass(slots=True)
class MatchCandidate:
    """One student's similarity to a probe embedding."""

    student_id: str
    name: str
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON responses."""
        return {
            "student_id": self.student_id,
            "name": self.name,
            "similarity": round(self.similarity, 4),
        }


@dataclass(slots=True)
class RecognitionResult:
    """The outcome of matching one detected face against the gallery.

    A result is always returned, even when nobody matched, so that callers can
    render "Unknown" boxes and inspect the runner-up scores. ``recognized``
    reports whether :attr:`similarity` cleared the recognition threshold.
    """

    face: DetectedFace
    recognized: bool
    student_id: str | None = None
    name: str | None = None
    similarity: float = 0.0
    #: Gap between the best and second-best identity — low values mean ambiguity.
    margin: float = 0.0
    candidates: list[MatchCandidate] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """Cosine similarity of the best match, clipped to ``[0, 1]``.

        Exposed under the name the API contract uses ("confidence score").
        """
        return float(min(max(self.similarity, 0.0), 1.0))

    @property
    def label(self) -> str:
        """Short human-readable label for drawing on video frames."""
        if not self.recognized:
            return "Unknown"
        return f"{self.name} ({self.student_id})"

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON responses."""
        return {
            "recognized": self.recognized,
            "student_id": self.student_id,
            "name": self.name,
            "confidence": round(self.confidence, 4),
            "margin": round(self.margin, 4),
            "bbox": self.face.bbox.to_dict(),
            "det_score": round(self.face.det_score, 4),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


class AttendanceStatus(str, enum.Enum):
    """Outcome of an attempt to mark attendance."""

    #: A new attendance record was written.
    MARKED = "marked"
    #: The student was already marked for this date/session; nothing was written.
    DUPLICATE = "duplicate"
    #: Confidence was below the attendance threshold; nothing was written.
    REJECTED = "rejected"


@dataclass(slots=True)
class AttendanceRecord:
    """A single attendance entry."""

    student_id: str
    name: str
    timestamp: datetime
    confidence: float
    session: str
    #: How the record was created: ``auto`` (recognition) or ``manual``.
    source: str = "auto"

    @property
    def date(self) -> date:
        """Calendar date of the record (UTC)."""
        return self.timestamp.date()

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible mapping."""
        return {
            "student_id": self.student_id,
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "date": self.date.isoformat(),
            "confidence": round(self.confidence, 4),
            "session": self.session,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AttendanceRecord":
        """Rebuild a record from its serialised form."""
        return cls(
            student_id=str(data["student_id"]),
            name=str(data["name"]),
            timestamp=_parse_datetime(data.get("timestamp")),
            confidence=float(data.get("confidence", 0.0)),
            session=str(data.get("session", "general")),
            source=str(data.get("source", "auto")),
        )


@dataclass(slots=True)
class AttendanceOutcome:
    """The result of :meth:`services.attendance_service.AttendanceService.mark`.

    Distinguishes "written" from "already there" from "not confident enough"
    without using exceptions for ordinary control flow.
    """

    status: AttendanceStatus
    record: AttendanceRecord | None = None
    reason: str | None = None

    @property
    def marked(self) -> bool:
        """True when a new record was persisted by this call."""
        return self.status is AttendanceStatus.MARKED

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON responses."""
        return {
            "status": self.status.value,
            "reason": self.reason,
            "record": self.record.to_dict() if self.record else None,
        }


def _parse_datetime(value: Any) -> datetime:
    """Parse a stored timestamp, always returning a UTC-aware datetime.

    Accepts ``datetime`` objects and ISO-8601 strings (including the ``Z``
    suffix). Naive values are assumed to be UTC, which is what this service
    writes.
    """
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    elif value is None:
        return utc_now()
    else:
        raise TypeError(f"Cannot parse datetime from {type(value).__name__}")

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
