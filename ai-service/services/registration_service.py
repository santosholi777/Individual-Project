"""Student enrolment: images in, stored embeddings out.

The enrolment pipeline is: detect exactly one face → check quality → align →
embed → persist. Raw images are deliberately discarded once embedded; only the
512-D vectors are stored, so the system holds no photographs of students.

Shared by the webcam CLI (``register.py``) and the ``POST /register`` endpoint,
which is why it takes plain image arrays and knows nothing about either.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from config import Settings
from database import EmbeddingRepository, StudentRepository, validate_student_id
from domain import Student, utc_now
from exceptions import (
    FaceQualityError,
    LowQualityImageError,
    RegistrationError,
    StudentAlreadyExistsError,
    StudentNotFoundError,
)
from logging_config import get_logger
from services.detector import FaceDetectorProtocol
from services.embedder import FaceEmbedderProtocol
from services.matcher import CosineSimilarityMatcher
from utils.image_utils import blur_score, brightness_score, resize_max_side

logger = get_logger(__name__)

#: Mean intensity outside this band is treated as unusable lighting.
_MIN_BRIGHTNESS: float = 40.0
_MAX_BRIGHTNESS: float = 235.0


@dataclass(slots=True)
class RegistrationResult:
    """Report of an enrolment attempt.

    Attributes:
        student: The stored student record.
        accepted_images: How many images produced a usable embedding.
        rejected_images: How many images were discarded.
        rejections: Human-readable reason per rejected image.
        total_embeddings: Total embeddings held for the student afterwards.
    """

    student: Student
    accepted_images: int
    rejected_images: int
    rejections: list[str] = field(default_factory=list)
    total_embeddings: int = 0

    def to_dict(self) -> dict[str, object]:
        """Serialise for JSON responses."""
        return {
            "student": self.student.to_dict(),
            "accepted_images": self.accepted_images,
            "rejected_images": self.rejected_images,
            "rejections": self.rejections,
            "total_embeddings": self.total_embeddings,
        }


class RegistrationService:
    """Enrols students by turning face images into stored embeddings.

    Args:
        settings: Supplies quality thresholds and the minimum image count.
        detector: Locates the face in each image.
        embedder: Aligns and embeds the located face.
        student_repository: Stores student metadata.
        embedding_repository: Stores the embedding vectors.
        matcher: Refreshed after each change so new students are recognisable
            immediately, without a service restart.
    """

    def __init__(
        self,
        settings: Settings,
        detector: FaceDetectorProtocol,
        embedder: FaceEmbedderProtocol,
        student_repository: StudentRepository,
        embedding_repository: EmbeddingRepository,
        matcher: CosineSimilarityMatcher,
    ) -> None:
        self._settings = settings
        self._detector = detector
        self._embedder = embedder
        self._students = student_repository
        self._embeddings = embedding_repository
        self._matcher = matcher

    # ------------------------------------------------------------------
    # Enrolment
    # ------------------------------------------------------------------
    def register(
        self,
        student_id: str,
        name: str,
        images: Sequence[np.ndarray],
        *,
        overwrite: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> RegistrationResult:
        """Register a student from a set of face images.

        Images are processed independently: a blurred or empty frame is reported
        and skipped rather than failing the whole enrolment, as long as enough
        usable images remain.

        Args:
            student_id: Unique identifier, e.g. a college roll number.
            name: Display name.
            images: BGR images, each showing exactly one face.
            overwrite: Replace an existing student's embeddings instead of
                failing on a duplicate id.
            metadata: Optional extra fields (department, year, …).

        Returns:
            A report of what was accepted, rejected and stored.

        Raises:
            RegistrationError: If ``name`` is blank, no images were supplied, or
                too few produced usable embeddings.
            StudentAlreadyExistsError: If the id exists and ``overwrite`` is off.
        """
        identifier = validate_student_id(student_id)
        display_name = (name or "").strip()
        if not display_name:
            raise RegistrationError(
                "Student name must not be empty", details={"student_id": identifier}
            )
        if not images:
            raise RegistrationError(
                "At least one face image is required to register a student",
                details={"student_id": identifier},
            )

        existing = self._students.get(identifier)
        if existing is not None and not overwrite:
            raise StudentAlreadyExistsError(
                f"Student '{identifier}' is already registered. "
                "Pass overwrite=true to replace their enrolment.",
                details={"student_id": identifier, "name": existing.name},
            )

        vectors, rejections = self._embed_images(images)
        accepted = len(vectors)

        if accepted < self._settings.registration_min_images:
            raise RegistrationError(
                f"Only {accepted} of {len(images)} image(s) produced a usable face "
                f"embedding; at least {self._settings.registration_min_images} are "
                "required. Ensure the face is centred, well lit and unobstructed.",
                details={
                    "student_id": identifier,
                    "accepted": accepted,
                    "required": self._settings.registration_min_images,
                    "rejections": rejections,
                },
            )

        matrix = np.vstack(vectors).astype(np.float32)
        self._embeddings.save(identifier, matrix)

        now = utc_now()
        student = Student(
            student_id=identifier,
            name=display_name,
            embedding_count=int(matrix.shape[0]),
            created_at=existing.created_at if existing else now,
            updated_at=now,
            metadata=dict(metadata or (existing.metadata if existing else {})),
        )
        self._students.add(student)
        self._matcher.refresh()

        logger.info(
            "Registered %s (%s) with %s embedding(s); %s image(s) rejected",
            identifier,
            display_name,
            matrix.shape[0],
            len(rejections),
        )
        return RegistrationResult(
            student=student,
            accepted_images=accepted,
            rejected_images=len(rejections),
            rejections=rejections,
            total_embeddings=int(matrix.shape[0]),
        )

    def add_images(
        self, student_id: str, images: Sequence[np.ndarray]
    ) -> RegistrationResult:
        """Add more face images to a student who is already registered.

        Useful for improving accuracy over time — extra poses, lighting or a new
        hairstyle — without discarding the original enrolment.

        Args:
            student_id: An existing student's identifier.
            images: Additional BGR face images.

        Returns:
            A report of what was accepted, rejected and stored.

        Raises:
            StudentNotFoundError: If the student is not registered.
            RegistrationError: If no supplied image produced an embedding.
        """
        identifier = validate_student_id(student_id)
        student = self._students.get(identifier)
        if student is None:
            raise StudentNotFoundError(
                f"No student registered with id '{identifier}'",
                details={"student_id": identifier},
            )

        vectors, rejections = self._embed_images(images)
        if not vectors:
            raise RegistrationError(
                "None of the supplied images produced a usable face embedding",
                details={"student_id": identifier, "rejections": rejections},
            )

        total = self._embeddings.append(identifier, np.vstack(vectors))
        student.embedding_count = total
        student.updated_at = utc_now()
        self._students.add(student)
        self._matcher.refresh()

        logger.info("Added %s embedding(s) to %s", len(vectors), identifier)
        return RegistrationResult(
            student=student,
            accepted_images=len(vectors),
            rejected_images=len(rejections),
            rejections=rejections,
            total_embeddings=total,
        )

    def delete(self, student_id: str) -> None:
        """Delete a student and their embeddings.

        Supports the consent-withdrawal requirement: enrolment must be as
        reversible as it is voluntary.

        Args:
            student_id: The student to remove.

        Raises:
            StudentNotFoundError: If the id is unknown.
        """
        identifier = validate_student_id(student_id)
        # Delete embeddings first: an orphaned registry row is harmless, whereas
        # orphaned biometric data is exactly what must not survive.
        self._embeddings.delete(identifier)
        self._students.delete(identifier)
        self._matcher.refresh()
        logger.info("Deleted student %s and their embeddings", identifier)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _embed_images(
        self, images: Sequence[np.ndarray]
    ) -> tuple[list[np.ndarray], list[str]]:
        """Embed each image, collecting rejection reasons for the failures.

        Args:
            images: BGR images to process.

        Returns:
            A ``(vectors, rejections)`` pair.
        """
        vectors: list[np.ndarray] = []
        rejections: list[str] = []

        for index, image in enumerate(images, start=1):
            try:
                vectors.append(self.embed_single_face(image))
            except FaceQualityError as exc:
                rejections.append(f"Image {index}: {exc.message}")
                logger.debug("Image %s rejected: %s", index, exc.message)
            except Exception as exc:  # noqa: BLE001 - one bad frame must not abort
                rejections.append(f"Image {index}: unexpected error - {exc}")
                logger.exception("Unexpected error while embedding image %s", index)

        return vectors, rejections

    def embed_single_face(self, image: np.ndarray) -> np.ndarray:
        """Detect, quality-check, align and embed the one face in an image.

        Args:
            image: A BGR image containing exactly one face.

        Returns:
            The L2-normalised embedding, shape ``(dim,)``.

        Raises:
            NoFaceDetectedError: If no face is found.
            MultipleFacesDetectedError: If more than one face is found.
            LowQualityImageError: If the face is too blurry or badly lit.
        """
        prepared = resize_max_side(image, max_side=1920)
        face = self._detector.detect_single(prepared)
        aligned = self._embedder.align(prepared, face)
        self._assert_quality(aligned)
        return self._embedder.embed_aligned(aligned)

    def _assert_quality(self, aligned_face: np.ndarray) -> None:
        """Reject an aligned crop that is too blurry or badly exposed.

        Quality is judged on the aligned crop rather than the full frame, so the
        scores describe the face itself and not the background.

        Raises:
            LowQualityImageError: If the crop fails a quality check.
        """
        sharpness = blur_score(aligned_face)
        if sharpness < self._settings.blur_threshold:
            raise LowQualityImageError(
                f"Face is too blurry (sharpness {sharpness:.1f} < "
                f"{self._settings.blur_threshold:.1f}). Hold still and refocus.",
                details={
                    "sharpness": round(sharpness, 2),
                    "required": self._settings.blur_threshold,
                },
            )

        brightness = brightness_score(aligned_face)
        if brightness < _MIN_BRIGHTNESS:
            raise LowQualityImageError(
                f"Face is too dark (brightness {brightness:.1f}). Add more light.",
                details={"brightness": round(brightness, 2)},
            )
        if brightness > _MAX_BRIGHTNESS:
            raise LowQualityImageError(
                f"Face is over-exposed (brightness {brightness:.1f}). "
                "Reduce direct light or move away from the window.",
                details={"brightness": round(brightness, 2)},
            )
