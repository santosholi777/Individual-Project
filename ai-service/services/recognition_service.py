"""Orchestration of the face recognition pipeline.

Composes detector, embedder and matcher into the one operation callers actually
want: image in, identities out. This is the only class that knows the pipeline's
running order, which keeps that knowledge out of both the API layer and the CLI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from config import Settings
from domain import AttendanceOutcome, RecognitionResult
from exceptions import EmptyGalleryError
from logging_config import get_logger
from services.attendance_service import AttendanceService
from services.detector import FaceDetectorProtocol
from services.embedder import FaceEmbedderProtocol
from services.matcher import CosineSimilarityMatcher
from utils.image_utils import resize_max_side

logger = get_logger(__name__)


@dataclass(slots=True)
class RecognitionReport:
    """Everything one recognition request produced.

    Attributes:
        results: One result per detected face, largest face first.
        attendance: Attendance outcomes, aligned with :attr:`results`, or empty
            when marking was not requested.
        elapsed_ms: Wall-clock time for the whole pipeline.
    """

    results: list[RecognitionResult]
    attendance: list[AttendanceOutcome] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def faces_detected(self) -> int:
        """Number of faces the detector accepted."""
        return len(self.results)

    @property
    def recognized(self) -> list[RecognitionResult]:
        """Only the results that cleared the recognition threshold."""
        return [result for result in self.results if result.recognized]

    @property
    def best(self) -> RecognitionResult | None:
        """The highest-confidence result, or ``None`` when no face was found."""
        if not self.results:
            return None
        return max(self.results, key=lambda result: result.similarity)

    def to_dict(self) -> dict[str, object]:
        """Serialise for JSON responses."""
        return {
            "faces_detected": self.faces_detected,
            "recognized_count": len(self.recognized),
            "elapsed_ms": round(self.elapsed_ms, 2),
            "results": [result.to_dict() for result in self.results],
            "attendance": [outcome.to_dict() for outcome in self.attendance],
        }


class RecognitionService:
    """Runs the detect → align → embed → match pipeline.

    Args:
        settings: Service configuration.
        detector: Locates faces.
        embedder: Aligns and embeds located faces.
        matcher: Matches embeddings against the enrolled gallery.
        attendance_service: Applies attendance policy when marking is requested.
    """

    def __init__(
        self,
        settings: Settings,
        detector: FaceDetectorProtocol,
        embedder: FaceEmbedderProtocol,
        matcher: CosineSimilarityMatcher,
        attendance_service: AttendanceService,
    ) -> None:
        self._settings = settings
        self._detector = detector
        self._embedder = embedder
        self._matcher = matcher
        self._attendance = attendance_service

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------
    def recognize(
        self,
        image: np.ndarray,
        *,
        top_k: int = 3,
        max_faces: int | None = None,
        require_gallery: bool = True,
    ) -> RecognitionReport:
        """Recognise every face in an image.

        Args:
            image: BGR image array.
            top_k: Runner-up candidates to report per face, for diagnostics.
            max_faces: Cap on faces processed, largest first. ``None`` processes
                all of them; ``1`` gives single-subject behaviour at a door.
            require_gallery: Raise if no student is registered yet, instead of
                reporting every face as unknown.

        Returns:
            A report covering every processed face.

        Raises:
            EmptyGalleryError: If ``require_gallery`` is set and nobody is
                registered.
        """
        started = time.perf_counter()

        if require_gallery and self._matcher.is_empty:
            raise EmptyGalleryError(
                "No students are registered yet, so recognition cannot run. "
                "Register at least one student first."
            )

        prepared = resize_max_side(image, max_side=1920)
        faces = self._detector.detect(prepared)
        if max_faces is not None:
            faces = faces[:max_faces]

        results: list[RecognitionResult] = []
        for face in faces:
            embedding = self._embedder.embed(prepared, face)
            results.append(self._matcher.match(embedding, face, top_k=top_k))

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "Recognition finished: %s face(s), %s recognised, %.1fms",
            len(results),
            sum(1 for result in results if result.recognized),
            elapsed_ms,
        )
        return RecognitionReport(results=results, elapsed_ms=elapsed_ms)

    def recognize_one(self, image: np.ndarray, *, top_k: int = 3) -> RecognitionResult:
        """Recognise only the largest face in an image.

        Args:
            image: BGR image array.
            top_k: Runner-up candidates to report.

        Returns:
            The result for the largest detected face.

        Raises:
            NoFaceDetectedError: If no face is found.
            EmptyGalleryError: If no student is registered.
        """
        if self._matcher.is_empty:
            raise EmptyGalleryError(
                "No students are registered yet, so recognition cannot run. "
                "Register at least one student first."
            )

        prepared = resize_max_side(image, max_side=1920)
        face = self._detector.detect_primary(prepared)
        embedding = self._embedder.embed(prepared, face)
        return self._matcher.match(embedding, face, top_k=top_k)

    def recognize_and_mark(
        self,
        image: np.ndarray,
        *,
        session: str | None = None,
        top_k: int = 3,
        max_faces: int | None = None,
    ) -> RecognitionReport:
        """Recognise every face and mark attendance for the ones identified.

        The single call the kiosk and the ``POST /recognize?mark_attendance=true``
        endpoint use.

        Args:
            image: BGR image array.
            session: Session label to record against.
            top_k: Runner-up candidates to report per face.
            max_faces: Cap on faces processed, largest first.

        Returns:
            A report whose ``attendance`` list is aligned with ``results``.

        Raises:
            EmptyGalleryError: If no student is registered.
        """
        report = self.recognize(image, top_k=top_k, max_faces=max_faces)
        report.attendance = self._attendance.mark_many(report.results, session=session)
        return report

    def embed_faces(
        self, image: np.ndarray
    ) -> tuple[list[np.ndarray], Sequence[object]]:
        """Detect and embed every face without matching.

        Exposed for evaluation scripts (accuracy sweeps, threshold tuning) that
        need raw vectors rather than identities.

        Args:
            image: BGR image array.

        Returns:
            A ``(embeddings, faces)`` pair, aligned by index.
        """
        prepared = resize_max_side(image, max_side=1920)
        faces = self._detector.detect(prepared)
        embeddings = [self._embedder.embed(prepared, face) for face in faces]
        return embeddings, faces

    def warm_up(self) -> None:
        """Load both models and build the gallery index ahead of traffic.

        Called during application start-up so the first real request does not
        absorb several seconds of model initialisation.
        """
        logger.info("Warming up the recognition pipeline")
        started = time.perf_counter()

        # Blank frames are enough to force ONNX Runtime to allocate and to run
        # each graph once; no face is expected in them.
        self._detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))
        self._embedder.embed_aligned(np.zeros((112, 112, 3), dtype=np.uint8))
        self._matcher.refresh()

        logger.info(
            "Pipeline warm: %s embedding(s) across %s student(s) in %.0fms",
            self._matcher.size,
            self._matcher.student_count,
            (time.perf_counter() - started) * 1000.0,
        )
