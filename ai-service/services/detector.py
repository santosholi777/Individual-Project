"""Face detection built on InsightFace's pre-trained SCRFD detector.

The detector is responsible for *locating* faces only. Turning a face into a
vector is the embedder's job (:mod:`services.embedder`), and the two are kept
apart so either model can be replaced without touching the other.
"""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

import numpy as np

from config import Settings
from domain import BoundingBox, DetectedFace
from exceptions import (
    InferenceError,
    ModelLoadError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from logging_config import get_logger
from services.engine import ModelPackManager

logger = get_logger(__name__)


@runtime_checkable
class FaceDetectorProtocol(Protocol):
    """The contract every detector implementation must satisfy.

    Depending on this Protocol rather than on :class:`InsightFaceDetector` lets
    the pipeline be unit tested with a stub, and lets the detection model be
    swapped without changes upstream.
    """

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Return every acceptable face in ``image``, largest first."""
        ...

    def detect_single(self, image: np.ndarray) -> DetectedFace:
        """Return exactly one face, raising if there are zero or several."""
        ...


class InsightFaceDetector:
    """SCRFD face detector from the pre-trained InsightFace model pack.

    The underlying ONNX session is created lazily on first use and reused for
    the process lifetime, since model loading costs seconds and inference costs
    milliseconds.

    Args:
        settings: Service configuration (thresholds, detector input size).
        model_pack: Manager that resolves and downloads the pre-trained weights.
    """

    def __init__(self, settings: Settings, model_pack: ModelPackManager) -> None:
        self._settings = settings
        self._model_pack = model_pack
        self._app: object | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load the detection model. Idempotent and thread-safe.

        Raises:
            ModelLoadError: If InsightFace is missing or the model cannot be
                prepared.
        """
        if self._app is not None:
            return

        with self._lock:
            if self._app is not None:
                return

            try:
                from insightface.app import FaceAnalysis
            except ImportError as exc:  # pragma: no cover - environment issue
                raise ModelLoadError(
                    "The 'insightface' package is not installed. "
                    "Run: pip install -r requirements.txt",
                    details={"import_error": str(exc)},
                ) from exc

            # Ensure the pack exists before FaceAnalysis looks for it, so a
            # download failure surfaces as a clear ModelLoadError.
            self._model_pack.ensure_pack()

            logger.info(
                "Loading SCRFD detector from pack '%s' (det_size=%s, device=%s)",
                self._model_pack.pack_name,
                self._settings.det_size,
                "cuda" if self._settings.ctx_id >= 0 else "cpu",
            )
            try:
                # allowed_modules keeps this instance detection-only: the
                # recognition network is owned by the embedder.
                app = FaceAnalysis(
                    name=self._model_pack.pack_name,
                    root=str(self._settings.models_dir),
                    allowed_modules=["detection"],
                    providers=self._model_pack.providers(),
                )
                app.prepare(
                    ctx_id=self._settings.ctx_id,
                    det_size=tuple(self._settings.det_size),
                    det_thresh=self._settings.det_score_threshold,
                )
            except Exception as exc:  # noqa: BLE001 - upstream raises bare Exception
                raise ModelLoadError(
                    f"Failed to initialise the SCRFD detector: {exc}",
                    details={"pack": self._model_pack.pack_name},
                ) from exc

            self._app = app
            logger.info("Face detector ready")

    @property
    def is_loaded(self) -> bool:
        """Whether the detection model has been loaded."""
        return self._app is not None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Detect every acceptable face in an image.

        Faces below the configured detector confidence or smaller than
        ``min_face_size`` are dropped, so callers only ever see faces worth
        embedding.

        Args:
            image: BGR image array.

        Returns:
            Accepted faces sorted by area, largest first. Empty when none pass.

        Raises:
            ModelLoadError: If the model cannot be loaded.
            InferenceError: If the input is not a valid image array or the
                detector fails.
        """
        self.load()
        self._validate_image(image)

        try:
            raw_faces = self._app.get(image)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 - upstream raises bare Exception
            raise InferenceError(f"Face detection failed: {exc}") from exc

        faces: list[DetectedFace] = []
        for raw in raw_faces:
            score = float(getattr(raw, "det_score", 0.0))
            if score < self._settings.det_score_threshold:
                continue

            keypoints = getattr(raw, "kps", None)
            if keypoints is None:
                # Alignment needs the five landmarks; a face without them
                # cannot be embedded reliably, so it is not worth returning.
                logger.debug("Skipping a face with no landmarks (score=%.3f)", score)
                continue

            bbox = BoundingBox.from_array(raw.bbox)
            if bbox.min_side < self._settings.min_face_size:
                logger.debug(
                    "Skipping face smaller than min_face_size (%.0fpx < %spx)",
                    bbox.min_side,
                    self._settings.min_face_size,
                )
                continue

            faces.append(
                DetectedFace(bbox=bbox, det_score=score, keypoints=np.asarray(keypoints))
            )

        faces.sort(key=lambda face: face.bbox.area, reverse=True)
        logger.debug("Detected %s usable face(s)", len(faces))
        return faces

    def detect_single(self, image: np.ndarray) -> DetectedFace:
        """Detect exactly one face, as required during student registration.

        Args:
            image: BGR image array.

        Returns:
            The only detected face.

        Raises:
            NoFaceDetectedError: If no face passes the thresholds.
            MultipleFacesDetectedError: If more than one face is present, which
                would make the enrolled identity ambiguous.
        """
        faces = self.detect(image)

        if not faces:
            raise NoFaceDetectedError(
                "No face was detected. Ensure the face is clearly visible, "
                "well lit and close enough to the camera.",
                details={
                    "min_det_score": self._settings.det_score_threshold,
                    "min_face_size": self._settings.min_face_size,
                },
            )
        if len(faces) > 1:
            raise MultipleFacesDetectedError(
                f"Expected exactly one face but detected {len(faces)}. "
                "Only the student being registered should be in frame.",
                details={"faces_detected": len(faces)},
            )
        return faces[0]

    def detect_primary(self, image: np.ndarray) -> DetectedFace:
        """Return the largest face, ignoring anyone else in frame.

        The forgiving counterpart to :meth:`detect_single`, for recognition at a
        classroom door where bystanders may drift into shot.

        Args:
            image: BGR image array.

        Returns:
            The largest detected face.

        Raises:
            NoFaceDetectedError: If no face passes the thresholds.
        """
        faces = self.detect(image)
        if not faces:
            raise NoFaceDetectedError(
                "No face was detected in the supplied image.",
                details={"min_det_score": self._settings.det_score_threshold},
            )
        return faces[0]

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """Reject arrays the detector cannot process.

        Raises:
            InferenceError: If the input is not a non-empty 3-channel image.
        """
        if not isinstance(image, np.ndarray):
            raise InferenceError(
                f"Expected a numpy array, got {type(image).__name__}"
            )
        if image.size == 0:
            raise InferenceError("Received an empty image array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise InferenceError(
                f"Expected a 3-channel BGR image, got shape {image.shape}"
            )
