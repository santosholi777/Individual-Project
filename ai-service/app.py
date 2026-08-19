"""FastAPI application — the HTTP face of the DeepVisionAttend AI service.

This layer only translates: HTTP in, domain calls out, JSON back. All the
intelligence lives in ``services/``, which is what allows the same pipeline to
be driven by the CLI tools with no duplicated logic.

Run it with::

    uvicorn app:app --host 0.0.0.0 --port 8000
    python app.py                      # equivalent, reads config.py

Interactive documentation is served at ``/docs`` (Swagger) and ``/redoc``.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated, AsyncIterator

import numpy as np
from fastapi import Depends, FastAPI, File, Form, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from auth.dependencies import AdminUser, CurrentUser
from auth.router import router as auth_router
from config import Settings, get_settings
from database import StudentRepository
from dependencies import (
    ServiceContainer,
    get_attendance_service,
    get_container,
    get_recognition_service,
    get_registration_service,
    get_student_repository,
)
from domain import utc_now
from exceptions import (
    DeepVisionAttendError,
    ImageDecodeError,
    StudentNotFoundError,
)
from logging_config import get_logger, setup_logging
from schemas import (
    AttendanceListResponse,
    AttendanceOutcomeSchema,
    AttendanceSummaryResponse,
    DeleteResponse,
    ErrorResponse,
    HealthResponse,
    MarkAttendanceRequest,
    RecognizeBase64Request,
    RecognizeResponse,
    RegisterBase64Request,
    RegisterResponse,
    StudentListResponse,
    StudentSchema,
)
from services.attendance_service import AttendanceService
from services.recognition_service import RecognitionService
from services.registration_service import RegistrationService
from utils.image_utils import decode_base64_image, decode_image_bytes

logger = get_logger(__name__)

_DESCRIPTION = """
Deep learning face recognition microservice for automated attendance.

Powered by **InsightFace**: a pre-trained SCRFD detector locates faces and a
pre-trained **ArcFace** (R50) network maps each aligned face to a 512-D
embedding. Identities are resolved by **cosine similarity** against the enrolled
gallery. No model is trained by this service — inference only.

**Privacy by design:** student photographs are never stored. Images are embedded
and discarded; only the resulting vectors persist, and an embedding cannot be
turned back into a photograph.

**Authentication:** every endpoint except `/health` and `/auth/*` requires a
bearer token. Sign up or sign in at `/auth/signup` or `/auth/login`, then send
`Authorization: Bearer <token>`. Deleting a student requires the admin role.
Accounts live in MongoDB; face data stays on the local filesystem.
"""

_TAGS_METADATA = [
    {"name": "Health", "description": "Liveness and readiness checks."},
    {
        "name": "Authentication",
        "description": (
            "Accounts, sign-in and password reset. Accounts are stored in "
            "MongoDB with bcrypt-hashed passwords; sessions are JWTs. Send the "
            "token as `Authorization: Bearer <token>`."
        ),
    },
    {"name": "Registration", "description": "Enrol students and manage their face data."},
    {"name": "Recognition", "description": "Identify students from an image."},
    {"name": "Attendance", "description": "Mark and query attendance records."},
]

# Injected dependencies. These must live at module scope: this module uses
# `from __future__ import annotations`, so FastAPI resolves endpoint type hints
# against module globals. Defined inside a function they would silently fail to
# resolve and be treated as query parameters.
SettingsDep = Annotated[Settings, Depends(get_settings)]
ContainerDep = Annotated[ServiceContainer, Depends(get_container)]
RegistrationDep = Annotated[RegistrationService, Depends(get_registration_service)]
RecognitionDep = Annotated[RecognitionService, Depends(get_recognition_service)]
AttendanceDep = Annotated[AttendanceService, Depends(get_attendance_service)]
StudentsDep = Annotated[StudentRepository, Depends(get_student_repository)]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load models and the gallery index before the first request.

    A failure here is logged but does not abort start-up: the service still
    serves ``/health``, which is what a container orchestrator needs in order to
    report *why* the service is unhealthy rather than crash-looping silently.
    """
    settings: Settings = app.state.settings
    setup_logging(
        level=settings.log_level,
        log_dir=settings.logs_dir,
        log_to_file=settings.log_to_file,
    )
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)

    # An insecure configuration should be impossible to deploy by accident.
    for warning in settings.security_warnings():
        logger.warning("SECURITY: %s", warning)

    if settings.warm_up_on_startup:
        try:
            get_container().warm_up()
        except Exception:  # noqa: BLE001 - report via /health instead of crashing
            logger.exception(
                "Model warm-up failed. The service is running but will report "
                "'degraded' until the models can be loaded."
            )
    else:
        logger.info("Start-up warm-up disabled; models load on first use")

    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application.

    A factory rather than a module-level instance so tests can construct an app
    against temporary settings.

    Args:
        settings: Optional configuration override.

    Returns:
        The configured application.
    """
    config = settings or get_settings()

    application = FastAPI(
        title=config.app_name,
        description=_DESCRIPTION,
        version=config.app_version,
        openapi_tags=_TAGS_METADATA,
        lifespan=lifespan,
        responses={
            422: {"model": ErrorResponse, "description": "Validation or face-quality error"},
            500: {"model": ErrorResponse, "description": "Internal error"},
        },
    )
    # Read back by lifespan, so a test app warms up against its own settings
    # rather than the process-wide ones.
    application.state.settings = config

    application.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_middleware(application)
    _register_exception_handlers(application)
    application.include_router(auth_router)
    _register_routes(application)
    return application


# ======================================================================
# Middleware and error handling
# ======================================================================
def _register_middleware(application: FastAPI) -> None:
    """Attach request logging and timing."""

    @application.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Log each request's outcome and attach an ``X-Process-Time`` header."""
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        response.headers["X-Process-Time"] = f"{elapsed_ms:.2f}ms"
        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


def _register_exception_handlers(application: FastAPI) -> None:
    """Map domain exceptions onto HTTP responses in one place.

    This is what lets services raise meaningful domain errors while staying
    completely free of HTTP concepts.
    """

    @application.exception_handler(DeepVisionAttendError)
    async def handle_domain_error(
        request: Request, exc: DeepVisionAttendError
    ) -> JSONResponse:
        """Render any domain error using its own status code and payload."""
        log = logger.error if exc.status_code >= 500 else logger.warning
        log("%s on %s: %s", type(exc).__name__, request.url.path, exc.message)
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @application.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        """Catch-all: log the traceback, return a generic message.

        Internal details stay in the logs rather than leaking to clients.
        """
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error_code="internal_error",
                message="An unexpected internal error occurred.",
                details={},
            ).model_dump(),
        )


# ======================================================================
# Helpers
# ======================================================================
async def _read_upload(file: UploadFile, settings: Settings) -> np.ndarray:
    """Read and decode an uploaded image, enforcing the size limit.

    Args:
        file: The uploaded file.
        settings: Supplies ``max_upload_bytes``.

    Returns:
        The decoded BGR image.

    Raises:
        ImageDecodeError: If the upload is too large or is not a valid image.
    """
    payload = await file.read()
    if len(payload) > settings.max_upload_bytes:
        raise ImageDecodeError(
            f"Image '{file.filename}' is larger than the "
            f"{settings.max_upload_bytes // (1024 * 1024)}MB limit",
            details={"filename": file.filename, "bytes": len(payload)},
        )
    return decode_image_bytes(payload)


# ======================================================================
# Routes
# ======================================================================
def _register_routes(application: FastAPI) -> None:
    """Attach every endpoint to the application."""

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------
    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Service health and readiness",
    )
    def health(settings: SettingsDep, container: ContainerDep) -> HealthResponse:
        """Report whether the models are loaded and the gallery is indexed.

        Deliberately public: a health check that needs credentials cannot tell a
        monitor the difference between "down" and "misconfigured".
        """
        database_up = container.mongo.ping()
        return HealthResponse(
            status="ok" if container.models_ready else "degraded",
            service=settings.app_name,
            version=settings.app_version,
            models_ready=container.models_ready,
            model_info=container.model_pack.describe(),
            index=container.matcher.describe(),
            storage={
                **container.database.stats(),
                "mongodb_connected": database_up,
                "mongodb_db": settings.mongodb_db,
            },
            auth={
                "required": settings.require_auth,
                "database_connected": database_up,
                # Surfaced so an insecure deployment is visible rather than
                # silently shipped.
                "security_warnings": settings.security_warnings(),
            },
            timestamp=utc_now(),
        )

    @application.get("/", tags=["Health"], summary="Service banner")
    def root(settings: SettingsDep) -> dict[str, str]:
        """Point callers at the interactive documentation."""
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health",
        }

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    @application.post(
        "/register",
        response_model=RegisterResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Registration"],
        summary="Register a student from uploaded face images",
        responses={
            409: {"model": ErrorResponse, "description": "Student already registered"},
            422: {"model": ErrorResponse, "description": "Too few usable face images"},
        },
    )
    async def register(
        user: CurrentUser,
        service: RegistrationDep,
        settings: SettingsDep,
        student_id: Annotated[str, Form(description="Unique student identifier")],
        name: Annotated[str, Form(description="Student display name")],
        files: Annotated[list[UploadFile], File(description="Face images, one face each")],
        overwrite: Annotated[bool, Form(description="Replace an existing enrolment")] = False,
    ) -> RegisterResponse:
        """Enrol a student by embedding their face images.

        Each image must contain exactly one clearly visible face. Images are
        embedded and then discarded — only the vectors are stored.
        """
        images = [await _read_upload(file, settings) for file in files]
        result = service.register(
            student_id=student_id, name=name, images=images, overwrite=overwrite
        )
        return RegisterResponse(
            success=True,
            student=StudentSchema(**result.student.to_dict()),
            accepted_images=result.accepted_images,
            rejected_images=result.rejected_images,
            rejections=result.rejections,
            total_embeddings=result.total_embeddings,
        )

    @application.post(
        "/register/base64",
        response_model=RegisterResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Registration"],
        summary="Register a student from base64 images (browser-friendly)",
    )
    def register_base64(
        payload: RegisterBase64Request, service: RegistrationDep, user: CurrentUser
    ) -> RegisterResponse:
        """Enrol a student from base64 frames.

        The JSON counterpart of ``POST /register``, for React clients capturing
        webcam frames with ``canvas.toDataURL()``.
        """
        images = [decode_base64_image(image) for image in payload.images]
        result = service.register(
            student_id=payload.student_id,
            name=payload.name,
            images=images,
            overwrite=payload.overwrite,
            metadata=payload.metadata,
        )
        return RegisterResponse(
            success=True,
            student=StudentSchema(**result.student.to_dict()),
            accepted_images=result.accepted_images,
            rejected_images=result.rejected_images,
            rejections=result.rejections,
            total_embeddings=result.total_embeddings,
        )

    # ------------------------------------------------------------------
    # Students
    # ------------------------------------------------------------------
    @application.get(
        "/students",
        response_model=StudentListResponse,
        tags=["Registration"],
        summary="List registered students",
    )
    def list_students(students: StudentsDep, user: CurrentUser) -> StudentListResponse:
        """Return every registered student and their enrolment metadata."""
        records = students.list_all()
        return StudentListResponse(
            count=len(records),
            students=[StudentSchema(**student.to_dict()) for student in records],
        )

    @application.get(
        "/students/{student_id}",
        response_model=StudentSchema,
        tags=["Registration"],
        summary="Fetch one student",
        responses={404: {"model": ErrorResponse, "description": "Student not found"}},
    )
    def get_student(
        student_id: str, students: StudentsDep, user: CurrentUser
    ) -> StudentSchema:
        """Return a single student by identifier."""
        student = students.get(student_id)
        if student is None:
            raise StudentNotFoundError(
                f"No student registered with id '{student_id}'",
                details={"student_id": student_id},
            )
        return StudentSchema(**student.to_dict())

    @application.delete(
        "/students/{student_id}",
        response_model=DeleteResponse,
        tags=["Registration"],
        summary="Delete a student and their face data",
        responses={404: {"model": ErrorResponse, "description": "Student not found"}},
    )
    def delete_student(
        student_id: str, service: RegistrationDep, admin: AdminUser
    ) -> DeleteResponse:
        """Remove a student and every embedding held for them.

        Supports withdrawal of consent: enrolment is as reversible as it is
        voluntary.
        """
        service.delete(student_id)
        return DeleteResponse(
            success=True,
            message=f"Student '{student_id}' and their face embeddings were deleted",
        )

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------
    @application.post(
        "/recognize",
        response_model=RecognizeResponse,
        tags=["Recognition"],
        summary="Recognise students in an uploaded image",
        responses={
            409: {"model": ErrorResponse, "description": "No students registered yet"},
            422: {"model": ErrorResponse, "description": "No usable face in the image"},
        },
    )
    async def recognize(
        user: CurrentUser,
        service: RecognitionDep,
        settings: SettingsDep,
        file: Annotated[UploadFile, File(description="Image containing one or more faces")],
        mark_attendance: Annotated[
            bool, Form(description="Also mark attendance for recognised students")
        ] = False,
        session: Annotated[str | None, Form(description="Session label")] = None,
        max_faces: Annotated[
            int | None, Form(description="Process at most this many faces, largest first")
        ] = None,
    ) -> RecognizeResponse:
        """Identify every face in an image and return each student and score.

        A face that matches nobody is still returned, with ``recognized: false``
        and its best candidates, so a near-miss can be told from a stranger.
        """
        image = await _read_upload(file, settings)
        report = (
            service.recognize_and_mark(image, session=session, max_faces=max_faces)
            if mark_attendance
            else service.recognize(image, max_faces=max_faces)
        )
        return _to_recognize_response(report)

    @application.post(
        "/recognize/base64",
        response_model=RecognizeResponse,
        tags=["Recognition"],
        summary="Recognise students in a base64 image (browser-friendly)",
    )
    def recognize_base64(
        payload: RecognizeBase64Request, service: RecognitionDep, user: CurrentUser
    ) -> RecognizeResponse:
        """The JSON counterpart of ``POST /recognize`` for React webcam frames."""
        image = decode_base64_image(payload.image)
        report = (
            service.recognize_and_mark(
                image, session=payload.session, max_faces=payload.max_faces
            )
            if payload.mark_attendance
            else service.recognize(image, max_faces=payload.max_faces)
        )
        return _to_recognize_response(report)

    # ------------------------------------------------------------------
    # Attendance
    # ------------------------------------------------------------------
    @application.post(
        "/attendance",
        response_model=AttendanceOutcomeSchema,
        tags=["Attendance"],
        summary="Mark attendance for a known student",
        responses={404: {"model": ErrorResponse, "description": "Student not found"}},
    )
    def mark_attendance(
        payload: MarkAttendanceRequest, service: AttendanceDep, user: CurrentUser
    ) -> AttendanceOutcomeSchema:
        """Mark attendance without an image, for manual entry by staff.

        Returns ``200`` in every non-error case: ``status`` reports whether the
        record was ``marked``, skipped as a ``duplicate``, or ``rejected`` for
        low confidence. Duplicates are an expected outcome, not a failure.
        """
        outcome = service.mark(
            student_id=payload.student_id,
            confidence=payload.confidence,
            session=payload.session,
            source=payload.source,
        )
        return AttendanceOutcomeSchema(**outcome.to_dict())

    @application.get(
        "/attendance",
        response_model=AttendanceListResponse,
        tags=["Attendance"],
        summary="Query attendance records",
    )
    def list_attendance(
        service: AttendanceDep,
        user: CurrentUser,
        student_id: Annotated[str | None, Query(description="Filter by student")] = None,
        date_from: Annotated[date | None, Query(description="Earliest date, inclusive")] = None,
        date_to: Annotated[date | None, Query(description="Latest date, inclusive")] = None,
        session: Annotated[str | None, Query(description="Filter by session label")] = None,
    ) -> AttendanceListResponse:
        """Return attendance records matching every supplied filter."""
        records = service.list_records(
            student_id=student_id,
            date_from=date_from,
            date_to=date_to,
            session=session,
        )
        return AttendanceListResponse(
            count=len(records),
            records=[record.to_dict() for record in records],  # type: ignore[misc]
        )

    @application.get(
        "/attendance/summary",
        response_model=AttendanceSummaryResponse,
        tags=["Attendance"],
        summary="Daily attendance summary",
    )
    def attendance_summary(
        service: AttendanceDep,
        user: CurrentUser,
        on_date: Annotated[
            date | None, Query(alias="date", description="Defaults to today (UTC)")
        ] = None,
    ) -> AttendanceSummaryResponse:
        """Return present/absent totals and the absentee list for one day."""
        return AttendanceSummaryResponse(**service.daily_summary(on_date))  # type: ignore[arg-type]


def _to_recognize_response(report) -> RecognizeResponse:  # type: ignore[no-untyped-def]
    """Convert a :class:`RecognitionReport` into its HTTP response model."""
    payload = report.to_dict()
    return RecognizeResponse(
        success=True,
        faces_detected=payload["faces_detected"],
        recognized_count=payload["recognized_count"],
        elapsed_ms=payload["elapsed_ms"],
        results=payload["results"],
        attendance=payload["attendance"],
    )


#: The ASGI application object uvicorn loads (``uvicorn app:app``).
app = create_app()


def main() -> None:
    """Run the service with uvicorn using the values from ``config.py``."""
    import uvicorn

    settings = get_settings()
    setup_logging(
        level=settings.log_level,
        log_dir=settings.logs_dir,
        log_to_file=settings.log_to_file,
    )
    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
