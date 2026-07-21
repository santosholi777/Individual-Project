"""Pydantic request/response models — the public HTTP contract.

Keeping the wire format in its own module means the React frontend and Node
backend depend on these shapes, not on internal domain classes, so the two can
evolve independently. The examples attached here are what renders in the
generated OpenAPI docs at ``/docs``.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ======================================================================
# Shared
# ======================================================================
class ErrorResponse(BaseModel):
    """The body returned for every 4xx/5xx raised by this service."""

    error_code: str = Field(description="Stable, machine-readable error identifier")
    message: str = Field(description="Human readable explanation")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Structured context about the failure"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error_code": "no_face_detected",
                "message": "No face was detected in the supplied image.",
                "details": {"min_det_score": 0.5},
            }
        }
    )


class BoundingBoxSchema(BaseModel):
    """A face bounding box in pixel coordinates of the submitted image."""

    x1: float
    y1: float
    x2: float
    y2: float


class HealthResponse(BaseModel):
    """Liveness and readiness information for monitoring."""

    status: Literal["ok", "degraded"] = Field(description="Overall service health")
    service: str
    version: str
    models_ready: bool = Field(description="Whether both models are loaded in memory")
    model_info: dict[str, Any] = Field(default_factory=dict)
    index: dict[str, Any] = Field(default_factory=dict)
    storage: dict[str, Any] = Field(default_factory=dict)
    auth: dict[str, Any] = Field(
        default_factory=dict,
        description="Whether auth is enforced and whether MongoDB is reachable",
    )
    timestamp: datetime


# ======================================================================
# Students / registration
# ======================================================================
class StudentSchema(BaseModel):
    """A registered student."""

    student_id: str = Field(description="Unique identifier, e.g. the roll number")
    name: str
    embedding_count: int = Field(description="Face embeddings stored for this student")
    created_at: datetime
    updated_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "student_id": "CS2021001",
                "name": "Aditi Sharma",
                "embedding_count": 5,
                "created_at": "2026-07-15T09:30:00+00:00",
                "updated_at": "2026-07-15T09:30:00+00:00",
                "metadata": {"department": "Computer Science", "year": "3"},
            }
        }
    )


class StudentListResponse(BaseModel):
    """Every registered student."""

    count: int
    students: list[StudentSchema]


class RegisterBase64Request(BaseModel):
    """JSON enrolment payload for browser clients.

    The React frontend captures frames with ``canvas.toDataURL()``; those
    strings can be posted here directly, data-URI prefix and all.
    """

    student_id: str = Field(
        min_length=1, max_length=64, description="Unique student identifier"
    )
    name: str = Field(min_length=1, max_length=128, description="Student display name")
    images: list[str] = Field(
        min_length=1, description="Base64 images, with or without a data-URI prefix"
    )
    overwrite: bool = Field(
        default=False, description="Replace an existing enrolment for this id"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional extra fields (department, year, …)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "student_id": "CS2021001",
                "name": "Aditi Sharma",
                "images": ["data:image/jpeg;base64,/9j/4AAQSkZJRg..."],
                "overwrite": False,
                "metadata": {"department": "Computer Science"},
            }
        }
    )


class RegisterResponse(BaseModel):
    """The outcome of an enrolment request."""

    success: bool
    student: StudentSchema
    accepted_images: int = Field(description="Images that produced a usable embedding")
    rejected_images: int = Field(description="Images discarded by the quality gate")
    rejections: list[str] = Field(
        default_factory=list, description="Why each rejected image was discarded"
    )
    total_embeddings: int = Field(description="Embeddings now stored for the student")


# ======================================================================
# Recognition
# ======================================================================
class MatchCandidateSchema(BaseModel):
    """A runner-up identity and its similarity to the probe."""

    student_id: str
    name: str
    similarity: float = Field(description="Cosine similarity in [0, 1]")


class RecognitionResultSchema(BaseModel):
    """The identity decision for a single detected face."""

    recognized: bool = Field(description="Whether the confidence cleared the threshold")
    student_id: str | None = Field(default=None, description="Null when not recognised")
    name: str | None = Field(default=None, description="Null when not recognised")
    confidence: float = Field(description="Cosine similarity of the best match, 0–1")
    margin: float = Field(
        description="Gap to the runner-up; small values indicate an ambiguous match"
    )
    bbox: BoundingBoxSchema
    det_score: float = Field(description="Detector confidence for this face")
    candidates: list[MatchCandidateSchema] = Field(default_factory=list)


class RecognizeBase64Request(BaseModel):
    """JSON recognition payload for browser clients."""

    image: str = Field(description="Base64 image, with or without a data-URI prefix")
    mark_attendance: bool = Field(
        default=False, description="Also mark attendance for recognised students"
    )
    session: str | None = Field(
        default=None, description="Session label to record attendance against"
    )
    max_faces: int | None = Field(
        default=None,
        ge=1,
        description="Process at most this many faces, largest first",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
                "mark_attendance": True,
                "session": "lecture-1",
                "max_faces": 1,
            }
        }
    )


class RecognizeResponse(BaseModel):
    """The outcome of a recognition request."""

    success: bool
    faces_detected: int
    recognized_count: int
    elapsed_ms: float = Field(description="Server-side pipeline time")
    results: list[RecognitionResultSchema]
    attendance: list["AttendanceOutcomeSchema"] = Field(
        default_factory=list,
        description="Populated only when attendance marking was requested",
    )


# ======================================================================
# Attendance
# ======================================================================
class AttendanceRecordSchema(BaseModel):
    """A single attendance entry."""

    student_id: str
    name: str
    timestamp: datetime
    date: date
    confidence: float
    session: str
    source: Literal["auto", "manual"] = Field(
        description="'auto' from recognition, 'manual' from staff entry"
    )


class AttendanceOutcomeSchema(BaseModel):
    """What happened when attendance was attempted."""

    status: Literal["marked", "duplicate", "rejected"]
    reason: str | None = Field(
        default=None, description="Why the record was skipped, when it was"
    )
    record: AttendanceRecordSchema | None = None


class MarkAttendanceRequest(BaseModel):
    """Mark attendance for a known student without an image.

    Covers manual entry by staff and re-marking after a recognition failure.
    """

    student_id: str = Field(min_length=1, max_length=64)
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Use 1.0 for manual entries"
    )
    session: str | None = Field(default=None, description="Defaults to the configured session")
    source: Literal["auto", "manual"] = Field(default="manual")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "student_id": "CS2021001",
                "confidence": 1.0,
                "session": "lecture-1",
                "source": "manual",
            }
        }
    )


class AttendanceListResponse(BaseModel):
    """Attendance records matching a query."""

    count: int
    records: list[AttendanceRecordSchema]


class AttendanceSummaryResponse(BaseModel):
    """A day's attendance totals for the admin dashboard."""

    date: date
    total_students: int
    present: int
    absent: int
    attendance_rate: float = Field(description="Percentage of students present")
    records: list[AttendanceRecordSchema]
    absentees: list[dict[str, str]]


class DeleteResponse(BaseModel):
    """Confirmation that a resource was deleted."""

    success: bool
    message: str


# Resolves the forward reference to AttendanceOutcomeSchema above.
RecognizeResponse.model_rebuild()
