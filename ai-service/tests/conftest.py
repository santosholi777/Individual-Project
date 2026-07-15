"""Shared pytest fixtures.

The suite substitutes the two ONNX models with deterministic fakes. That keeps
the tests fast and offline while still exercising the real detector-to-attendance
wiring: everything except the neural networks themselves is the production code
path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# The service is a flat application, not an installed package, so tests import
# its modules the same way the CLI entry points do.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Settings  # noqa: E402
from database import LocalFileDatabase  # noqa: E402
from domain import BoundingBox, DetectedFace  # noqa: E402
from exceptions import MultipleFacesDetectedError, NoFaceDetectedError  # noqa: E402
from services.attendance_service import AttendanceService  # noqa: E402
from services.matcher import CosineSimilarityMatcher  # noqa: E402
from services.recognition_service import RecognitionService  # noqa: E402
from services.registration_service import RegistrationService  # noqa: E402
from utils.similarity import l2_normalize  # noqa: E402

_DIM = 512


class FakeDetector:
    """A detector whose output is scripted by the test.

    Satisfies :class:`services.detector.FaceDetectorProtocol` structurally, which
    is exactly the substitutability the Protocol exists to allow.
    """

    def __init__(self) -> None:
        self.faces: list[DetectedFace] = [make_face()]

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Return the scripted faces."""
        return list(self.faces)

    def detect_single(self, image: np.ndarray) -> DetectedFace:
        """Return the one scripted face, mirroring the real detector's rules."""
        faces = self.detect(image)
        if not faces:
            raise NoFaceDetectedError("No face was detected")
        if len(faces) > 1:
            raise MultipleFacesDetectedError(f"Detected {len(faces)} faces")
        return faces[0]

    def detect_primary(self, image: np.ndarray) -> DetectedFace:
        """Return the largest scripted face."""
        faces = self.detect(image)
        if not faces:
            raise NoFaceDetectedError("No face was detected")
        return faces[0]


class FakeEmbedder:
    """An embedder that derives a stable vector from the image's mean pixel.

    Two images with the same mean embed identically and two with different means
    embed nearly orthogonally, which gives the matcher a realistic
    "same person / different person" signal without any model weights.
    """

    def __init__(self) -> None:
        self.embedding_dim = _DIM

    @staticmethod
    def _seed_of(image: np.ndarray) -> int:
        return int(round(float(np.mean(image)))) % 1000

    def align(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Return a 112x112 crop carrying the same mean as the source image."""
        return np.full((112, 112, 3), int(round(float(np.mean(image)))), dtype=np.uint8)

    def embed(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Return the deterministic embedding for this image."""
        return self.embed_aligned(image)

    def embed_aligned(self, aligned_face: np.ndarray) -> np.ndarray:
        """Return a unit vector determined solely by the crop's mean pixel."""
        generator = np.random.default_rng(self._seed_of(aligned_face))
        return l2_normalize(generator.normal(size=_DIM).astype(np.float32))


def make_face(size: int = 200) -> DetectedFace:
    """Build a detected face for tests.

    Args:
        size: Box side length in pixels.

    Returns:
        A face with plausible landmarks and a high detector score.
    """
    keypoints = np.array(
        [[70, 80], [130, 80], [100, 110], [75, 140], [125, 140]], dtype=np.float32
    )
    return DetectedFace(
        bbox=BoundingBox(x1=50.0, y1=50.0, x2=50.0 + size, y2=50.0 + size),
        det_score=0.95,
        keypoints=keypoints,
    )


def make_image(value: int, size: int = 480) -> np.ndarray:
    """Build a synthetic image whose mean pixel is ``value``.

    The mean is what :class:`FakeEmbedder` keys on, so ``value`` acts as a
    stand-in for a person's identity.

    Args:
        value: Fill value in ``[0, 255]``; distinct values mean distinct people.
        size: Image side length.

    Returns:
        A uniform BGR image.
    """
    return np.full((size, size, 3), value, dtype=np.uint8)


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Settings pointed at a throwaway storage directory."""
    return Settings(
        storage_dir=tmp_path,
        embeddings_dir=tmp_path / "embeddings",
        attendance_dir=tmp_path / "attendance",
        logs_dir=tmp_path / "logs",
        models_dir=tmp_path / "models",
        registration_min_images=1,
        blur_threshold=0.0,  # synthetic images are flat; sharpness is meaningless
        log_to_file=False,
        warm_up_on_startup=False,  # never download real weights during tests
    )


@pytest.fixture()
def database(settings: Settings) -> LocalFileDatabase:
    """A local-file database backed by the temporary directory."""
    return LocalFileDatabase(settings)


@pytest.fixture()
def detector() -> FakeDetector:
    """The scripted detector."""
    return FakeDetector()


@pytest.fixture()
def embedder() -> FakeEmbedder:
    """The deterministic embedder."""
    return FakeEmbedder()


@pytest.fixture()
def matcher(
    settings: Settings, database: LocalFileDatabase
) -> CosineSimilarityMatcher:
    """A matcher over the temporary gallery."""
    return CosineSimilarityMatcher(settings, database.students, database.embeddings)


@pytest.fixture()
def attendance_service(
    settings: Settings, database: LocalFileDatabase
) -> AttendanceService:
    """An attendance service over the temporary log."""
    return AttendanceService(settings, database.attendance, database.students)


@pytest.fixture()
def registration_service(
    settings: Settings,
    detector: FakeDetector,
    embedder: FakeEmbedder,
    database: LocalFileDatabase,
    matcher: CosineSimilarityMatcher,
) -> RegistrationService:
    """A registration service wired to the fake models."""
    return RegistrationService(
        settings=settings,
        detector=detector,
        embedder=embedder,
        student_repository=database.students,
        embedding_repository=database.embeddings,
        matcher=matcher,
    )


@pytest.fixture()
def recognition_service(
    settings: Settings,
    detector: FakeDetector,
    embedder: FakeEmbedder,
    matcher: CosineSimilarityMatcher,
    attendance_service: AttendanceService,
) -> RecognitionService:
    """A recognition pipeline wired to the fake models."""
    return RecognitionService(
        settings=settings,
        detector=detector,
        embedder=embedder,
        matcher=matcher,
        attendance_service=attendance_service,
    )
