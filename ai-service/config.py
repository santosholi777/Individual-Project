"""Central configuration for the DeepVisionAttend AI service.

All tunable behaviour is defined here and can be overridden through environment
variables (prefixed with ``DVA_``) or a ``.env`` file, which keeps the service
deployable across development, testing and production without code changes.

Example:
    ``DVA_RECOGNITION_THRESHOLD=0.5 DVA_CTX_ID=0 uvicorn app:app``
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR: Path = Path(__file__).resolve().parent

#: Placeholder JWT secret for local development.
#:
#: It is 64 characters because HS256 keys below 32 bytes are weak (RFC 7518
#: §3.2) and PyJWT rightly warns about them. It is *not* a secret — it is in the
#: repository — so :meth:`Settings.security_warnings` flags it, and /health
#: reports it, whenever it is still in use.
DEV_JWT_SECRET: str = "dev-only-insecure-secret-change-me-before-any-real-deployment"


class Settings(BaseSettings):
    """Strongly typed, validated application settings.

    Attributes are grouped by concern: service metadata, model configuration,
    detection/recognition thresholds, storage locations and camera options.
    """

    model_config = SettingsConfigDict(
        env_prefix="DVA_",
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # ------------------------------------------------------------------
    # Service metadata
    # ------------------------------------------------------------------
    app_name: str = "DeepVisionAttend AI Service"
    app_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_to_file: bool = True
    #: Load the models during start-up rather than on the first request. Turn
    #: off for a fast boot when the models are not needed (or in tests).
    warm_up_on_startup: bool = True

    #: Origins allowed to call this service (the React frontend / Node backend).
    #: Browsers treat "localhost" and "127.0.0.1" as different origins, so both
    #: spellings of each dev port are listed — otherwise the frontend works at
    #: one URL and is silently CORS-blocked at the other.
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ]
    )

    # ------------------------------------------------------------------
    # Model configuration (pre-trained InsightFace / ArcFace)
    # ------------------------------------------------------------------
    #: InsightFace model pack. ``buffalo_l`` = SCRFD-10G detector + ArcFace R50.
    model_pack: str = "buffalo_l"
    #: Filename of the ArcFace recognition network inside the model pack.
    recognition_model_file: str = "w600k_r50.onnx"
    #: Root directory the pre-trained weights are downloaded to / loaded from.
    models_dir: Path = BASE_DIR / "models"
    #: ONNX Runtime execution context: ``-1`` = CPU, ``>= 0`` = CUDA device id.
    ctx_id: int = -1
    #: Detector input resolution (width, height). Larger = slower but better recall.
    det_size: tuple[int, int] = (640, 640)
    #: Dimensionality of an ArcFace embedding.
    embedding_dim: int = 512

    # ------------------------------------------------------------------
    # Detection / recognition thresholds
    # ------------------------------------------------------------------
    #: Minimum detector confidence for a face to be considered valid.
    det_score_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    #: Minimum cosine similarity for a match to be accepted as an identity.
    recognition_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    #: Cosine similarity above which attendance may be marked automatically.
    attendance_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    #: Smallest acceptable face box (pixels) — filters distant/unusable faces.
    min_face_size: int = Field(default=50, ge=10)
    #: Laplacian variance below this value is treated as too blurry to enrol.
    blur_threshold: float = Field(default=45.0, ge=0.0)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    #: Number of frames captured per student by the webcam enrolment CLI.
    registration_captures: int = Field(default=5, ge=1)
    #: Minimum usable images required before a student can be enrolled.
    registration_min_images: int = Field(default=3, ge=1)
    #: Seconds between webcam captures during enrolment.
    registration_capture_interval: float = Field(default=1.0, ge=0.0)
    #: Maximum embeddings retained per student (oldest are discarded first).
    max_embeddings_per_student: int = Field(default=20, ge=1)

    # ------------------------------------------------------------------
    # Attendance policy
    # ------------------------------------------------------------------
    #: Default session label used when a caller does not supply one.
    default_session: str = "general"
    #: Minutes during which a repeat recognition is ignored as a duplicate.
    duplicate_cooldown_minutes: int = Field(default=0, ge=0)
    #: When true, a student may only be marked once per (date, session).
    once_per_session_per_day: bool = True

    # ------------------------------------------------------------------
    # Authentication (users live in MongoDB)
    # ------------------------------------------------------------------
    #: MongoDB connection string.
    mongodb_uri: str = "mongodb://127.0.0.1:27017"
    #: Database name. Kept separate from any other project on the same server.
    mongodb_db: str = "deepvisionattend"
    #: Seconds to wait for MongoDB before giving up — keeps a dead database from
    #: hanging a request for the driver's 30s default.
    mongodb_timeout_ms: int = Field(default=4000, ge=500)

    #: Secret used to sign JWTs. MUST be overridden outside development —
    #: anyone holding it can mint a valid token for any account. At least 32
    #: bytes, which is the minimum RFC 7518 recommends for HS256.
    #: Generate one with: python -c "import secrets; print(secrets.token_hex(32))"
    jwt_secret: str = Field(
        default=DEV_JWT_SECRET, min_length=32, description="JWT signing secret"
    )
    jwt_algorithm: str = "HS256"
    #: How long an access token stays valid (default 12 hours).
    jwt_expire_minutes: int = Field(default=720, ge=1)

    #: When true, the student/attendance endpoints require a valid token.
    require_auth: bool = True
    #: Minimum acceptable password length at signup.
    password_min_length: int = Field(default=8, ge=6)
    #: The first account created becomes an admin; later ones are lecturers.
    first_user_is_admin: bool = True

    #: How long a password-reset link stays valid.
    reset_token_expire_minutes: int = Field(default=30, ge=1)
    #: Development convenience: return the reset link in the API response
    #: instead of emailing it. MUST be false in production — it would let
    #: anyone reset any account by asking.
    expose_reset_link: bool = True
    #: Base URL the reset link points at (the React app).
    frontend_url: str = "http://localhost:5173"

    # ------------------------------------------------------------------
    # Storage (local filesystem for now; swappable for MongoDB later)
    # ------------------------------------------------------------------
    storage_dir: Path = BASE_DIR / "storage"
    embeddings_dir: Path = BASE_DIR / "storage" / "embeddings"
    attendance_dir: Path = BASE_DIR / "storage" / "attendance"
    logs_dir: Path = BASE_DIR / "storage" / "logs"

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    #: Frames discarded after opening the device to let auto-exposure settle.
    camera_warmup_frames: int = Field(default=10, ge=0)
    #: Run recognition on every Nth frame in the live CLI (1 = every frame).
    recognition_frame_interval: int = Field(default=3, ge=1)

    # ------------------------------------------------------------------
    # API limits
    # ------------------------------------------------------------------
    #: Reject uploads larger than this (bytes) before decoding them.
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, ge=1024)

    @field_validator("det_size", mode="before")
    @classmethod
    def _parse_det_size(cls, value: object) -> object:
        """Allow ``DVA_DET_SIZE=640,640`` in addition to a tuple/list."""
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
            if len(parts) != 2:
                raise ValueError("det_size must be given as 'width,height'")
            return tuple(int(part) for part in parts)
        return value

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: object) -> object:
        """Allow a comma-separated origin list from the environment."""
        if isinstance(value, str) and not value.strip().startswith("["):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @model_validator(mode="after")
    def _validate_thresholds(self) -> "Settings":
        """Attendance must never be marked below the recognition threshold."""
        if self.attendance_threshold < self.recognition_threshold:
            raise ValueError(
                "attendance_threshold must be >= recognition_threshold "
                f"(got {self.attendance_threshold} < {self.recognition_threshold})"
            )
        return self

    def security_warnings(self) -> list[str]:
        """Return the ways this configuration is unsafe for real students.

        Called at start-up and surfaced on ``/health``. The point is that an
        insecure deployment announces itself, rather than looking identical to a
        secure one.
        """
        warnings: list[str] = []

        if self.jwt_secret == DEV_JWT_SECRET:
            warnings.append(
                "DVA_JWT_SECRET is the built-in development value, which is "
                "public. Anyone could forge a login token. Generate one with: "
                'python -c "import secrets; print(secrets.token_hex(32))"'
            )
        if not self.require_auth:
            warnings.append(
                "DVA_REQUIRE_AUTH is false — every endpoint is open to anyone "
                "who can reach the port."
            )
        if self.expose_reset_link:
            warnings.append(
                "DVA_EXPOSE_RESET_LINK is true — the API hands password reset "
                "links straight back to the caller, so anyone could take over "
                "any account. Development only."
            )
        return warnings

    @property
    def vectors_dir(self) -> Path:
        """Directory holding one ``.npy`` embedding matrix per student."""
        return self.embeddings_dir / "vectors"

    @property
    def students_file(self) -> Path:
        """JSON file holding the student registry."""
        return self.embeddings_dir / "students.json"

    @property
    def attendance_file(self) -> Path:
        """JSON Lines file holding the append-only attendance log."""
        return self.attendance_dir / "attendance.jsonl"

    def ensure_directories(self) -> None:
        """Create every directory the service writes to.

        Called once during start-up so that the rest of the code can assume the
        storage layout exists.
        """
        for directory in (
            self.models_dir,
            self.storage_dir,
            self.embeddings_dir,
            self.vectors_dir,
            self.attendance_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    The result is cached so that configuration is parsed and validated exactly
    once, and so FastAPI can use this function directly as a dependency.
    """
    return Settings()
