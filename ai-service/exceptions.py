"""Exception hierarchy for the DeepVisionAttend AI service.

Every error raised deliberately by this service derives from
:class:`DeepVisionAttendError`, which carries an HTTP status code and a stable
machine-readable ``error_code``. The API layer maps these to JSON responses in
one place (see ``app.py``), so services and repositories never import FastAPI.
"""

from __future__ import annotations

from typing import Any


class DeepVisionAttendError(Exception):
    """Base class for all domain errors raised by this service.

    Args:
        message: Human readable description, safe to return to a client.
        details: Optional structured context (field names, ids, scores).
    """

    #: HTTP status the API layer should respond with.
    status_code: int = 500
    #: Stable identifier clients can branch on without parsing prose.
    error_code: str = "internal_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialise the error into the service's standard payload shape."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(message={self.message!r}, details={self.details!r})"


# ----------------------------------------------------------------------
# Configuration and model lifecycle
# ----------------------------------------------------------------------
class ConfigurationError(DeepVisionAttendError):
    """The service is misconfigured and cannot operate."""

    status_code = 500
    error_code = "configuration_error"


class ModelLoadError(DeepVisionAttendError):
    """A pre-trained model could not be downloaded, found or initialised."""

    status_code = 503
    error_code = "model_load_error"


class InferenceError(DeepVisionAttendError):
    """A model executed but failed to produce a usable result."""

    status_code = 500
    error_code = "inference_error"


# ----------------------------------------------------------------------
# Image / camera input
# ----------------------------------------------------------------------
class ImageDecodeError(DeepVisionAttendError):
    """Uploaded bytes could not be decoded into an image."""

    status_code = 400
    error_code = "image_decode_error"


class CameraError(DeepVisionAttendError):
    """The webcam could not be opened or a frame could not be read."""

    status_code = 503
    error_code = "camera_error"


# ----------------------------------------------------------------------
# Face pipeline
# ----------------------------------------------------------------------
class FaceQualityError(DeepVisionAttendError):
    """A face was found but is not good enough to enrol or recognise."""

    status_code = 422
    error_code = "face_quality_error"


class NoFaceDetectedError(FaceQualityError):
    """No face passed the detector's confidence and size thresholds."""

    status_code = 422
    error_code = "no_face_detected"


class MultipleFacesDetectedError(FaceQualityError):
    """More than one face was present where exactly one is required."""

    status_code = 422
    error_code = "multiple_faces_detected"


class LowQualityImageError(FaceQualityError):
    """The face is too small, too blurry or too poorly lit to be used."""

    status_code = 422
    error_code = "low_quality_image"


# ----------------------------------------------------------------------
# Persistence / domain rules
# ----------------------------------------------------------------------
class RepositoryError(DeepVisionAttendError):
    """The storage backend could not complete an operation."""

    status_code = 500
    error_code = "repository_error"


class StudentNotFoundError(DeepVisionAttendError):
    """No student exists with the supplied identifier."""

    status_code = 404
    error_code = "student_not_found"


class StudentAlreadyExistsError(DeepVisionAttendError):
    """A student with the supplied identifier is already registered."""

    status_code = 409
    error_code = "student_already_exists"


class RegistrationError(DeepVisionAttendError):
    """Enrolment failed (for example, too few usable images were supplied)."""

    status_code = 422
    error_code = "registration_error"


class DuplicateAttendanceError(DeepVisionAttendError):
    """Attendance for this student, date and session has already been marked."""

    status_code = 409
    error_code = "duplicate_attendance"


class RecognitionError(DeepVisionAttendError):
    """A face was processed but could not be matched to a known student."""

    status_code = 404
    error_code = "recognition_failed"


class EmptyGalleryError(RecognitionError):
    """Recognition was attempted before any student had been registered."""

    status_code = 409
    error_code = "empty_gallery"
