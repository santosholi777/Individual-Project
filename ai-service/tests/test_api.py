"""Tests for the HTTP contract.

The app is exercised through ``TestClient`` with the service layer wired to the
fake models via FastAPI's dependency overrides — the same seam the production
container plugs into.
"""

from __future__ import annotations

import base64
from typing import Iterator

import cv2
import pytest
from fastapi.testclient import TestClient

from app import create_app
from auth.dependencies import get_current_user, require_admin
from auth.models import User, UserRole
from config import Settings
from database import LocalFileDatabase
from dependencies import (
    get_attendance_service,
    get_container,
    get_recognition_service,
    get_registration_service,
    get_student_repository,
)
from services.attendance_service import AttendanceService
from services.matcher import CosineSimilarityMatcher
from services.recognition_service import RecognitionService
from services.registration_service import RegistrationService
from tests.conftest import make_image

_ALICE = 100
_BOB = 200
_STRANGER = 37


def _encode(value: int) -> bytes:
    """Encode a synthetic identity image as JPEG bytes."""
    success, buffer = cv2.imencode(".jpg", make_image(value))
    assert success
    return buffer.tobytes()


def _files(value: int, count: int = 3) -> list[tuple[str, tuple[str, bytes, str]]]:
    """Build a multipart file list for an upload."""
    return [
        ("files", (f"face_{index}.jpg", _encode(value), "image/jpeg"))
        for index in range(count)
    ]


class FakeMongo:
    """Stands in for the MongoDB connection in /health."""

    def ping(self) -> bool:
        """Report the database as reachable."""
        return True


class FakeModelPack:
    """Stands in for the model pack manager in /health."""

    def describe(self) -> dict[str, object]:
        """Minimal model description."""
        return {"pack": "buffalo_l", "device": "cpu"}


class FakeContainer:
    """The slice of the container that /health actually reads."""

    def __init__(self, matcher: object, database: LocalFileDatabase) -> None:
        self.models_ready = True
        self.model_pack = FakeModelPack()
        self.matcher = matcher
        self.database = database
        self.mongo = FakeMongo()


#: A signed-in admin, used to satisfy the route guards without MongoDB.
TEST_ADMIN = User(
    user_id="test-admin",
    email="admin@college.edu",
    name="Test Admin",
    password_hash="",
    role=UserRole.ADMIN,
)

#: A signed-in lecturer, for checking that admin-only routes are actually
#: admin-only.
TEST_LECTURER = User(
    user_id="test-lecturer",
    email="lecturer@college.edu",
    name="Test Lecturer",
    password_hash="",
    role=UserRole.LECTURER,
)


def _build_client(
    settings: Settings,
    database: LocalFileDatabase,
    registration_service: RegistrationService,
    recognition_service: RecognitionService,
    attendance_service: AttendanceService,
    matcher: object,
    user: User | None = TEST_ADMIN,
) -> TestClient:
    """Build a test client wired to the fake-model services.

    Args:
        user: The signed-in account the guards should resolve to. ``None``
            leaves the real guard in place, so the 401 path can be tested.
    """
    application = create_app(settings)
    application.dependency_overrides[get_registration_service] = lambda: registration_service
    application.dependency_overrides[get_recognition_service] = lambda: recognition_service
    application.dependency_overrides[get_attendance_service] = lambda: attendance_service
    application.dependency_overrides[get_student_repository] = lambda: database.students
    application.dependency_overrides[get_container] = lambda: FakeContainer(matcher, database)

    if user is not None:
        # Authentication itself is covered in test_auth.py; here it is stubbed
        # so these tests exercise the endpoints rather than the login flow.
        application.dependency_overrides[get_current_user] = lambda: user
        if user.is_admin:
            application.dependency_overrides[require_admin] = lambda: user

    # raise_server_exceptions=False lets the registered exception handlers turn
    # domain errors into responses, exactly as they do in production.
    return TestClient(application, raise_server_exceptions=False)


@pytest.fixture()
def client(
    settings: Settings,
    database: LocalFileDatabase,
    registration_service: RegistrationService,
    recognition_service: RecognitionService,
    attendance_service: AttendanceService,
    matcher: CosineSimilarityMatcher,
) -> Iterator[TestClient]:
    """A test client signed in as an admin."""
    with _build_client(
        settings,
        database,
        registration_service,
        recognition_service,
        attendance_service,
        matcher,
    ) as test_client:
        yield test_client


@pytest.fixture()
def anonymous_client(
    settings: Settings,
    database: LocalFileDatabase,
    registration_service: RegistrationService,
    recognition_service: RecognitionService,
    attendance_service: AttendanceService,
    matcher: CosineSimilarityMatcher,
) -> Iterator[TestClient]:
    """A test client with no credentials, for checking the guards bite."""
    with _build_client(
        settings,
        database,
        registration_service,
        recognition_service,
        attendance_service,
        matcher,
        user=None,
    ) as test_client:
        yield test_client


@pytest.fixture()
def lecturer_client(
    settings: Settings,
    database: LocalFileDatabase,
    registration_service: RegistrationService,
    recognition_service: RecognitionService,
    attendance_service: AttendanceService,
    matcher: CosineSimilarityMatcher,
) -> Iterator[TestClient]:
    """A test client signed in as a lecturer (not an admin)."""
    with _build_client(
        settings,
        database,
        registration_service,
        recognition_service,
        attendance_service,
        matcher,
        user=TEST_LECTURER,
    ) as test_client:
        yield test_client


@pytest.fixture()
def registered(client: TestClient) -> TestClient:
    """A client with Alice already enrolled."""
    response = client.post(
        "/register",
        data={"student_id": "CS2021001", "name": "Aditi Sharma"},
        files=_files(_ALICE),
    )
    assert response.status_code == 201
    return client


class TestEndpointsAreProtected:
    """The guards must actually bite.

    Authentication that can be skipped by omitting a header is decoration. Each
    protected route is checked without credentials.
    """

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("get", "/students"),
            ("get", "/students/CS2021001"),
            ("delete", "/students/CS2021001"),
            ("get", "/attendance"),
            ("get", "/attendance/summary"),
        ],
    )
    def test_read_routes_require_a_token(
        self, anonymous_client: TestClient, method: str, path: str
    ) -> None:
        """No token, no data."""
        response = getattr(anonymous_client, method)(path)

        assert response.status_code == 401
        assert response.json()["error_code"] == "not_authenticated"

    def test_register_requires_a_token(self, anonymous_client: TestClient) -> None:
        """Enrolling a face is not something an anonymous caller may do."""
        response = anonymous_client.post(
            "/register",
            data={"student_id": "X1", "name": "Nobody"},
            files=_files(_ALICE),
        )
        assert response.status_code == 401

    def test_recognize_requires_a_token(self, anonymous_client: TestClient) -> None:
        """Nor is running the model against a face."""
        response = anonymous_client.post(
            "/recognize", files={"file": ("p.jpg", _encode(_ALICE), "image/jpeg")}
        )
        assert response.status_code == 401

    def test_mark_attendance_requires_a_token(self, anonymous_client: TestClient) -> None:
        """Writing to the register certainly is not."""
        response = anonymous_client.post("/attendance", json={"student_id": "CS2021001"})
        assert response.status_code == 401

    def test_a_garbage_token_is_rejected(self, anonymous_client: TestClient) -> None:
        """A forged or corrupt token must not pass."""
        response = anonymous_client.get(
            "/students", headers={"Authorization": "Bearer not-a-real-token"}
        )

        assert response.status_code == 401
        assert response.json()["error_code"] == "invalid_token"

    def test_health_stays_public(self, anonymous_client: TestClient) -> None:
        """Monitoring must work without credentials."""
        assert anonymous_client.get("/health").status_code == 200


class TestRoleEnforcement:
    """Admin-only actions."""

    def test_lecturer_cannot_delete_a_student(
        self, lecturer_client: TestClient
    ) -> None:
        """Erasing biometric data is an administrator's decision."""
        lecturer_client.post(
            "/register",
            data={"student_id": "CS2021001", "name": "Aditi Sharma"},
            files=_files(_ALICE),
        )
        response = lecturer_client.delete("/students/CS2021001")

        assert response.status_code == 403
        assert response.json()["error_code"] == "forbidden"

    def test_lecturer_can_still_take_attendance(
        self, lecturer_client: TestClient
    ) -> None:
        """A lecturer's ordinary job is unaffected by the admin gate."""
        lecturer_client.post(
            "/register",
            data={"student_id": "CS2021001", "name": "Aditi Sharma"},
            files=_files(_ALICE),
        )
        response = lecturer_client.post(
            "/recognize",
            data={"mark_attendance": "true"},
            files={"file": ("p.jpg", _encode(_ALICE), "image/jpeg")},
        )
        assert response.status_code == 200

    def test_admin_can_delete_a_student(self, registered: TestClient) -> None:
        """The admin fixture is genuinely allowed through."""
        assert registered.delete("/students/CS2021001").status_code == 200


class TestRoot:
    """The service banner."""

    def test_points_at_the_docs(self, client: TestClient) -> None:
        """The root response tells a new integrator where to go."""
        payload = client.get("/").json()
        assert payload["docs"] == "/docs"

    def test_openapi_schema_is_generated(self, client: TestClient) -> None:
        """Every documented endpoint appears in the OpenAPI schema."""
        paths = client.get("/openapi.json").json()["paths"]

        for route in ("/register", "/recognize", "/attendance", "/students"):
            assert route in paths


class TestRegisterEndpoint:
    """``POST /register``."""

    def test_registers_a_student(self, client: TestClient) -> None:
        """A successful enrolment returns 201 and the stored student."""
        response = client.post(
            "/register",
            data={"student_id": "CS2021001", "name": "Aditi Sharma"},
            files=_files(_ALICE),
        )

        assert response.status_code == 201
        payload = response.json()
        assert payload["success"] is True
        assert payload["student"]["student_id"] == "CS2021001"
        assert payload["student"]["name"] == "Aditi Sharma"
        assert payload["total_embeddings"] == 3

    def test_duplicate_id_returns_409(self, registered: TestClient) -> None:
        """A conflicting enrolment is reported as a conflict."""
        response = registered.post(
            "/register",
            data={"student_id": "CS2021001", "name": "Someone Else"},
            files=_files(_ALICE),
        )

        assert response.status_code == 409
        assert response.json()["error_code"] == "student_already_exists"

    def test_overwrite_succeeds(self, registered: TestClient) -> None:
        """An explicit overwrite replaces the enrolment."""
        response = registered.post(
            "/register",
            data={"student_id": "CS2021001", "name": "Aditi S.", "overwrite": "true"},
            files=_files(_ALICE, count=2),
        )

        assert response.status_code == 201
        assert response.json()["student"]["name"] == "Aditi S."

    def test_invalid_student_id_returns_500_error_payload(
        self, client: TestClient
    ) -> None:
        """A path-traversal id is refused by the repository's validation."""
        response = client.post(
            "/register",
            data={"student_id": "../evil", "name": "Bad"},
            files=_files(_ALICE),
        )

        assert response.status_code == 500
        assert response.json()["error_code"] == "repository_error"

    def test_dark_images_return_422(self, client: TestClient) -> None:
        """Unusable captures are reported with a face-quality error."""
        response = client.post(
            "/register",
            data={"student_id": "S9", "name": "Dark"},
            files=_files(5),
        )

        assert response.status_code == 422
        assert response.json()["error_code"] == "registration_error"

    def test_corrupt_upload_returns_400(self, client: TestClient) -> None:
        """Bytes that are not an image are a client error."""
        response = client.post(
            "/register",
            data={"student_id": "S9", "name": "Bad"},
            files=[("files", ("x.jpg", b"not an image", "image/jpeg"))],
        )

        assert response.status_code == 400
        assert response.json()["error_code"] == "image_decode_error"

    def test_missing_fields_return_422(self, client: TestClient) -> None:
        """FastAPI's own validation covers absent form fields."""
        assert client.post("/register", files=_files(_ALICE)).status_code == 422


class TestRegisterBase64Endpoint:
    """``POST /register/base64`` — the browser-facing variant."""

    def test_accepts_a_data_uri(self, client: TestClient) -> None:
        """Exactly what canvas.toDataURL() produces is accepted."""
        encoded = base64.b64encode(_encode(_ALICE)).decode()
        response = client.post(
            "/register/base64",
            json={
                "student_id": "CS2021002",
                "name": "Rahul Verma",
                "images": [f"data:image/jpeg;base64,{encoded}"] * 3,
                "metadata": {"department": "CS"},
            },
        )

        assert response.status_code == 201
        assert response.json()["student"]["metadata"] == {"department": "CS"}

    def test_accepts_bare_base64(self, client: TestClient) -> None:
        """The data-URI prefix is optional."""
        encoded = base64.b64encode(_encode(_ALICE)).decode()
        response = client.post(
            "/register/base64",
            json={"student_id": "S5", "name": "Bare", "images": [encoded] * 3},
        )

        assert response.status_code == 201

    def test_invalid_base64_returns_400(self, client: TestClient) -> None:
        """Malformed base64 is a client error, not a crash."""
        response = client.post(
            "/register/base64",
            json={"student_id": "S6", "name": "Bad", "images": ["!!!not-base64!!!"]},
        )

        assert response.status_code == 400
        assert response.json()["error_code"] == "image_decode_error"


class TestStudentsEndpoint:
    """``GET /students`` and its detail routes."""

    def test_lists_registered_students(self, registered: TestClient) -> None:
        """The dashboard's student list."""
        payload = registered.get("/students").json()

        assert payload["count"] == 1
        assert payload["students"][0]["student_id"] == "CS2021001"
        assert payload["students"][0]["embedding_count"] == 3

    def test_empty_list_when_nobody_registered(self, client: TestClient) -> None:
        """A fresh install returns an empty list, not an error."""
        payload = client.get("/students").json()
        assert payload == {"count": 0, "students": []}

    def test_fetches_one_student(self, registered: TestClient) -> None:
        """Detail lookup by id."""
        payload = registered.get("/students/CS2021001").json()
        assert payload["name"] == "Aditi Sharma"

    def test_unknown_student_returns_404(self, client: TestClient) -> None:
        """An unknown id maps to 404."""
        response = client.get("/students/ghost")

        assert response.status_code == 404
        assert response.json()["error_code"] == "student_not_found"

    def test_deletes_a_student(self, registered: TestClient) -> None:
        """Consent withdrawal removes the student and their vectors."""
        assert registered.delete("/students/CS2021001").status_code == 200
        assert registered.get("/students").json()["count"] == 0

    def test_deleting_an_unknown_student_returns_404(self, client: TestClient) -> None:
        """Deleting somebody who was never enrolled is a 404."""
        assert client.delete("/students/ghost").status_code == 404


class TestRecognizeEndpoint:
    """``POST /recognize``."""

    def test_recognises_a_registered_student(self, registered: TestClient) -> None:
        """The core contract: id, name and confidence come back."""
        response = registered.post(
            "/recognize", files={"file": ("probe.jpg", _encode(_ALICE), "image/jpeg")}
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["faces_detected"] == 1
        assert payload["recognized_count"] == 1

        result = payload["results"][0]
        assert result["recognized"] is True
        assert result["student_id"] == "CS2021001"
        assert result["name"] == "Aditi Sharma"
        assert result["confidence"] > 0.9
        assert set(result["bbox"]) == {"x1", "y1", "x2", "y2"}

    def test_stranger_is_reported_as_unrecognised(self, registered: TestClient) -> None:
        """An unknown face returns 200 with recognized=false, not an error."""
        response = registered.post(
            "/recognize", files={"file": ("probe.jpg", _encode(_STRANGER), "image/jpeg")}
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["recognized_count"] == 0
        assert payload["results"][0]["recognized"] is False
        assert payload["results"][0]["student_id"] is None

    def test_recognise_before_any_registration_returns_409(
        self, client: TestClient
    ) -> None:
        """Recognition needs a gallery."""
        response = client.post(
            "/recognize", files={"file": ("probe.jpg", _encode(_ALICE), "image/jpeg")}
        )

        assert response.status_code == 409
        assert response.json()["error_code"] == "empty_gallery"

    def test_marks_attendance_when_asked(self, registered: TestClient) -> None:
        """The kiosk flow: recognise and record in one request."""
        response = registered.post(
            "/recognize",
            data={"mark_attendance": "true", "session": "lecture-1"},
            files={"file": ("probe.jpg", _encode(_ALICE), "image/jpeg")},
        )

        payload = response.json()
        assert payload["attendance"][0]["status"] == "marked"
        assert payload["attendance"][0]["record"]["session"] == "lecture-1"
        assert registered.get("/attendance").json()["count"] == 1

    def test_second_call_reports_a_duplicate(self, registered: TestClient) -> None:
        """Re-recognition does not create a second record."""
        for _ in range(2):
            response = registered.post(
                "/recognize",
                data={"mark_attendance": "true", "session": "lecture-1"},
                files={"file": ("probe.jpg", _encode(_ALICE), "image/jpeg")},
            )

        assert response.json()["attendance"][0]["status"] == "duplicate"
        assert registered.get("/attendance").json()["count"] == 1

    def test_base64_variant_recognises(self, registered: TestClient) -> None:
        """The JSON variant behaves identically."""
        encoded = base64.b64encode(_encode(_ALICE)).decode()
        response = registered.post(
            "/recognize/base64",
            json={"image": f"data:image/jpeg;base64,{encoded}", "mark_attendance": True},
        )

        assert response.status_code == 200
        assert response.json()["results"][0]["student_id"] == "CS2021001"

    def test_response_reports_processing_time(self, registered: TestClient) -> None:
        """Latency is part of the contract for the dashboard."""
        response = registered.post(
            "/recognize", files={"file": ("probe.jpg", _encode(_ALICE), "image/jpeg")}
        )

        assert response.json()["elapsed_ms"] >= 0
        assert "X-Process-Time" in response.headers


class TestAttendanceEndpoint:
    """``POST /attendance`` and ``GET /attendance``."""

    def test_marks_manually(self, registered: TestClient) -> None:
        """Staff can mark a student without an image."""
        response = registered.post(
            "/attendance",
            json={"student_id": "CS2021001", "session": "lecture-1", "source": "manual"},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "marked"
        assert payload["record"]["source"] == "manual"
        assert payload["record"]["confidence"] == 1.0

    def test_duplicate_returns_200_with_status(self, registered: TestClient) -> None:
        """A duplicate is an expected outcome, reported in the body."""
        registered.post("/attendance", json={"student_id": "CS2021001"})
        response = registered.post("/attendance", json={"student_id": "CS2021001"})

        assert response.status_code == 200
        assert response.json()["status"] == "duplicate"

    def test_unknown_student_returns_404(self, client: TestClient) -> None:
        """Attendance cannot be attributed to an unregistered id."""
        response = client.post("/attendance", json={"student_id": "ghost"})

        assert response.status_code == 404
        assert response.json()["error_code"] == "student_not_found"

    def test_lists_records(self, registered: TestClient) -> None:
        """The register, as the dashboard reads it."""
        registered.post("/attendance", json={"student_id": "CS2021001", "session": "lecture-1"})
        payload = registered.get("/attendance").json()

        assert payload["count"] == 1
        record = payload["records"][0]
        assert record["student_id"] == "CS2021001"
        assert record["name"] == "Aditi Sharma"
        assert record["session"] == "lecture-1"
        assert "timestamp" in record

    def test_filters_by_student(self, registered: TestClient) -> None:
        """Query filters narrow the register."""
        registered.post(
            "/register",
            data={"student_id": "CS2021002", "name": "Bob"},
            files=_files(_BOB),
        )
        registered.post("/attendance", json={"student_id": "CS2021001"})
        registered.post("/attendance", json={"student_id": "CS2021002"})

        payload = registered.get("/attendance", params={"student_id": "CS2021001"}).json()
        assert payload["count"] == 1

    def test_filters_by_session(self, registered: TestClient) -> None:
        """Sessions partition a day's records."""
        registered.post("/attendance", json={"student_id": "CS2021001", "session": "lecture-1"})
        registered.post("/attendance", json={"student_id": "CS2021001", "session": "lab-2"})

        assert registered.get("/attendance", params={"session": "lab-2"}).json()["count"] == 1

    def test_filters_by_date_range(self, registered: TestClient) -> None:
        """An old window excludes today's records."""
        registered.post("/attendance", json={"student_id": "CS2021001"})
        payload = registered.get(
            "/attendance", params={"date_from": "2020-01-01", "date_to": "2020-01-31"}
        ).json()

        assert payload["count"] == 0

    def test_invalid_date_returns_422(self, registered: TestClient) -> None:
        """Malformed query parameters are rejected by validation."""
        assert registered.get("/attendance", params={"date_from": "yesterday"}).status_code == 422

    def test_daily_summary(self, registered: TestClient) -> None:
        """Present/absent totals for the dashboard."""
        registered.post(
            "/register",
            data={"student_id": "CS2021002", "name": "Bob"},
            files=_files(_BOB),
        )
        registered.post("/attendance", json={"student_id": "CS2021001"})

        payload = registered.get("/attendance/summary").json()
        assert payload["total_students"] == 2
        assert payload["present"] == 1
        assert payload["absent"] == 1
        assert payload["attendance_rate"] == 50.0
        assert payload["absentees"][0]["student_id"] == "CS2021002"
