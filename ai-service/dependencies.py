"""Composition root: builds the object graph and wires dependencies together.

This is the **only** module that names concrete implementations. Every other
module depends on the abstract interfaces (``StudentRepository``,
``FaceDetectorProtocol``, …), so migrating from local files to MongoDB means
changing :meth:`ServiceContainer._build_database` here and nothing else.

The container is a process-wide singleton because the ONNX models cost seconds
to load and hundreds of megabytes to hold — one instance is shared across
requests, and the models are thread-safe for inference.
"""

from __future__ import annotations

import functools
import threading

from auth.repository import (
    MongoConnection,
    MongoResetTokenRepository,
    MongoUserRepository,
)
from auth.service import AuthService
from config import Settings, get_settings
from database import (
    AttendanceRepository,
    EmbeddingRepository,
    LocalFileDatabase,
    StudentRepository,
)
from logging_config import get_logger
from services.attendance_service import AttendanceService
from services.detector import InsightFaceDetector
from services.embedder import ArcFaceEmbedder
from services.engine import ModelPackManager
from services.matcher import CosineSimilarityMatcher
from services.recognition_service import RecognitionService
from services.registration_service import RegistrationService

logger = get_logger(__name__)


class ServiceContainer:
    """Owns every long-lived object the service needs.

    Args:
        settings: Configuration to build the graph from. Defaults to the
            process settings.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._settings.ensure_directories()
        self._lock = threading.Lock()

        logger.info(
            "Building service container (%s v%s)",
            self._settings.app_name,
            self._settings.app_version,
        )

        database = self._build_database()
        self._students: StudentRepository = database.students
        self._embeddings: EmbeddingRepository = database.embeddings
        self._attendance_repo: AttendanceRepository = database.attendance
        self._database = database

        self._model_pack = ModelPackManager(self._settings)
        self._detector = InsightFaceDetector(self._settings, self._model_pack)
        self._embedder = ArcFaceEmbedder(self._settings, self._model_pack)
        self._matcher = CosineSimilarityMatcher(
            self._settings, self._students, self._embeddings
        )

        self._attendance_service = AttendanceService(
            self._settings, self._attendance_repo, self._students
        )
        self._registration_service = RegistrationService(
            settings=self._settings,
            detector=self._detector,
            embedder=self._embedder,
            student_repository=self._students,
            embedding_repository=self._embeddings,
            matcher=self._matcher,
        )
        self._recognition_service = RecognitionService(
            settings=self._settings,
            detector=self._detector,
            embedder=self._embedder,
            matcher=self._matcher,
            attendance_service=self._attendance_service,
        )

        # Accounts live in MongoDB, separate from the face data. The connection
        # is lazy: MongoDB being down must not stop recognition from working.
        self._mongo = MongoConnection(self._settings)
        self._auth_service = AuthService(
            settings=self._settings,
            users=MongoUserRepository(self._mongo),
            reset_tokens=MongoResetTokenRepository(self._mongo),
        )
        logger.info("Service container ready")

    def _build_database(self) -> LocalFileDatabase:
        """Construct the storage backend.

        The seam for the planned MongoDB integration: implement the three
        repository interfaces from ``database.py`` against Motor/PyMongo and
        return that here (selected by a setting). Nothing upstream changes.
        """
        return LocalFileDatabase(self._settings)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def settings(self) -> Settings:
        """The configuration this container was built from."""
        return self._settings

    @property
    def database(self) -> LocalFileDatabase:
        """The storage backend facade."""
        return self._database

    @property
    def students(self) -> StudentRepository:
        """The student registry repository."""
        return self._students

    @property
    def auth_service(self) -> AuthService:
        """The account/authentication service."""
        return self._auth_service

    @property
    def mongo(self) -> MongoConnection:
        """The MongoDB connection backing the accounts."""
        return self._mongo

    @property
    def model_pack(self) -> ModelPackManager:
        """The pre-trained model pack manager."""
        return self._model_pack

    @property
    def detector(self) -> InsightFaceDetector:
        """The face detector."""
        return self._detector

    @property
    def embedder(self) -> ArcFaceEmbedder:
        """The ArcFace embedder."""
        return self._embedder

    @property
    def matcher(self) -> CosineSimilarityMatcher:
        """The gallery matcher."""
        return self._matcher

    @property
    def attendance_service(self) -> AttendanceService:
        """The attendance policy service."""
        return self._attendance_service

    @property
    def registration_service(self) -> RegistrationService:
        """The enrolment service."""
        return self._registration_service

    @property
    def recognition_service(self) -> RecognitionService:
        """The recognition pipeline."""
        return self._recognition_service

    @property
    def models_ready(self) -> bool:
        """Whether both models are loaded and ready to infer."""
        return self._detector.is_loaded and self._embedder.is_loaded

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def warm_up(self) -> None:
        """Load models and build the gallery index. Safe to call repeatedly."""
        with self._lock:
            self._recognition_service.warm_up()


@functools.lru_cache(maxsize=1)
def get_container() -> ServiceContainer:
    """Return the process-wide container, building it on first use.

    Usable both as a FastAPI dependency and directly from the CLI entry points.
    """
    return ServiceContainer()


def reset_container() -> None:
    """Drop the cached container so the next call rebuilds it.

    Intended for tests that need a container built against temporary storage
    paths; production code should never need this.
    """
    get_container.cache_clear()
    get_settings.cache_clear()


# ----------------------------------------------------------------------
# FastAPI dependency providers
# ----------------------------------------------------------------------
def get_recognition_service() -> RecognitionService:
    """FastAPI dependency: the recognition pipeline."""
    return get_container().recognition_service


def get_registration_service() -> RegistrationService:
    """FastAPI dependency: the enrolment service."""
    return get_container().registration_service


def get_attendance_service() -> AttendanceService:
    """FastAPI dependency: the attendance policy service."""
    return get_container().attendance_service


def get_student_repository() -> StudentRepository:
    """FastAPI dependency: the student registry."""
    return get_container().students
