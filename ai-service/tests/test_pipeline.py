"""End-to-end tests of registration, matching and recognition.

The neural networks are replaced by the deterministic fakes from ``conftest``;
every other layer — quality gates, storage, index refresh, threshold logic,
attendance policy — is the real implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import Settings
from database import LocalFileDatabase
from domain import AttendanceStatus
from exceptions import (
    EmptyGalleryError,
    NoFaceDetectedError,
    RegistrationError,
    StudentAlreadyExistsError,
    StudentNotFoundError,
)
from services.matcher import CosineSimilarityMatcher
from services.recognition_service import RecognitionService
from services.registration_service import RegistrationService
from tests.conftest import FakeDetector, make_face, make_image

#: Distinct fill values stand in for distinct identities.
_ALICE = 100
_BOB = 200
_STRANGER = 37


class TestRegistration:
    """Behaviour of :class:`RegistrationService`."""

    def test_registers_a_student(
        self, registration_service: RegistrationService, database: LocalFileDatabase
    ) -> None:
        """Enrolment stores the student and one embedding per usable image."""
        result = registration_service.register(
            "CS2021001", "Alice", [make_image(_ALICE) for _ in range(3)]
        )

        assert result.accepted_images == 3
        assert result.total_embeddings == 3
        assert database.students.get("CS2021001") is not None
        assert database.embeddings.count("CS2021001") == 3

    def test_stores_no_images_only_embeddings(
        self, registration_service: RegistrationService, settings: Settings
    ) -> None:
        """The privacy guarantee: no photograph is written to disk."""
        registration_service.register("S1", "Alice", [make_image(_ALICE)])

        written = [path.suffix for path in settings.storage_dir.rglob("*") if path.is_file()]
        assert ".npy" in written
        assert not {".jpg", ".jpeg", ".png"} & set(written)

    def test_rejects_a_blank_name(
        self, registration_service: RegistrationService
    ) -> None:
        """A nameless enrolment would be useless on the register."""
        with pytest.raises(RegistrationError, match="name must not be empty"):
            registration_service.register("S1", "   ", [make_image(_ALICE)])

    def test_rejects_an_empty_image_list(
        self, registration_service: RegistrationService
    ) -> None:
        """There is nothing to embed."""
        with pytest.raises(RegistrationError, match="At least one face image"):
            registration_service.register("S1", "Alice", [])

    def test_duplicate_id_raises(
        self, registration_service: RegistrationService
    ) -> None:
        """Re-registering an id would silently overwrite biometric data."""
        registration_service.register("S1", "Alice", [make_image(_ALICE)])

        with pytest.raises(StudentAlreadyExistsError):
            registration_service.register("S1", "Alice Again", [make_image(_ALICE)])

    def test_overwrite_replaces_the_enrolment(
        self, registration_service: RegistrationService, database: LocalFileDatabase
    ) -> None:
        """An explicit overwrite is allowed and replaces the vectors."""
        registration_service.register("S1", "Alice", [make_image(_ALICE)] * 2)
        result = registration_service.register(
            "S1", "Alice Smith", [make_image(_ALICE)] * 3, overwrite=True
        )

        assert result.total_embeddings == 3
        assert database.students.get("S1").name == "Alice Smith"  # type: ignore[union-attr]

    def test_overwrite_preserves_the_original_created_at(
        self, registration_service: RegistrationService, database: LocalFileDatabase
    ) -> None:
        """Re-enrolment updates the record; it is not a brand new student."""
        first = registration_service.register("S1", "Alice", [make_image(_ALICE)])
        second = registration_service.register(
            "S1", "Alice", [make_image(_ALICE)], overwrite=True
        )

        assert second.student.created_at == first.student.created_at
        assert second.student.updated_at >= first.student.updated_at

    def test_too_few_usable_images_raises(
        self, settings: Settings, registration_service: RegistrationService
    ) -> None:
        """The minimum-image rule protects recognition quality."""
        settings.registration_min_images = 3

        with pytest.raises(RegistrationError, match="at least 3"):
            registration_service.register("S1", "Alice", [make_image(_ALICE)])

    def test_a_bad_frame_is_skipped_not_fatal(
        self,
        settings: Settings,
        registration_service: RegistrationService,
        detector: FakeDetector,
    ) -> None:
        """One unusable capture must not waste the whole enrolment session."""
        settings.registration_min_images = 1
        images = [make_image(_ALICE) for _ in range(3)]

        calls = {"n": 0}
        original = detector.detect_single

        def flaky(image: np.ndarray):  # type: ignore[no-untyped-def]
            """Fail on the second image only."""
            calls["n"] += 1
            if calls["n"] == 2:
                raise NoFaceDetectedError("No face was detected")
            return original(image)

        detector.detect_single = flaky  # type: ignore[assignment]
        result = registration_service.register("S1", "Alice", images)

        assert result.accepted_images == 2
        assert result.rejected_images == 1
        assert "Image 2" in result.rejections[0]

    def test_multiple_faces_is_rejected(
        self, registration_service: RegistrationService, detector: FakeDetector
    ) -> None:
        """Two faces make the enrolled identity ambiguous."""
        detector.faces = [make_face(200), make_face(150)]

        with pytest.raises(RegistrationError):
            registration_service.register("S1", "Alice", [make_image(_ALICE)])

    def test_add_images_extends_an_enrolment(
        self, registration_service: RegistrationService, database: LocalFileDatabase
    ) -> None:
        """Extra poses can be added later without re-enrolling."""
        registration_service.register("S1", "Alice", [make_image(_ALICE)])
        result = registration_service.add_images("S1", [make_image(_ALICE)] * 2)

        assert result.total_embeddings == 3
        assert database.embeddings.count("S1") == 3

    def test_add_images_to_an_unknown_student_raises(
        self, registration_service: RegistrationService
    ) -> None:
        """There is no implicit enrolment."""
        with pytest.raises(StudentNotFoundError):
            registration_service.add_images("ghost", [make_image(_ALICE)])

    def test_delete_removes_student_and_embeddings(
        self, registration_service: RegistrationService, database: LocalFileDatabase
    ) -> None:
        """Withdrawal of consent erases the biometric data too."""
        registration_service.register("S1", "Alice", [make_image(_ALICE)])
        registration_service.delete("S1")

        assert database.students.get("S1") is None
        assert database.embeddings.get("S1") is None


class TestQualityGate:
    """The enrolment quality checks."""

    def test_dark_image_is_rejected(
        self, registration_service: RegistrationService
    ) -> None:
        """An unlit face produces an unreliable embedding."""
        with pytest.raises(RegistrationError) as exc_info:
            registration_service.register("S1", "Alice", [make_image(5)])
        assert "too dark" in str(exc_info.value.details)

    def test_overexposed_image_is_rejected(
        self, registration_service: RegistrationService
    ) -> None:
        """A blown-out face carries almost no facial detail."""
        with pytest.raises(RegistrationError) as exc_info:
            registration_service.register("S1", "Alice", [make_image(250)])
        assert "over-exposed" in str(exc_info.value.details)

    def test_blurry_image_is_rejected(
        self, settings: Settings, registration_service: RegistrationService
    ) -> None:
        """A flat crop has zero Laplacian variance, i.e. no sharp detail."""
        settings.blur_threshold = 10.0

        with pytest.raises(RegistrationError) as exc_info:
            registration_service.register("S1", "Alice", [make_image(_ALICE)])
        assert "too blurry" in str(exc_info.value.details)


class TestMatcher:
    """Behaviour of :class:`CosineSimilarityMatcher`."""

    def test_empty_gallery_recognises_nobody(
        self, matcher: CosineSimilarityMatcher
    ) -> None:
        """With no one enrolled, every face is unknown."""
        result = matcher.match(np.zeros(512, dtype=np.float32), make_face())

        assert matcher.is_empty is True
        assert result.recognized is False

    def test_matches_the_enrolled_student(
        self,
        registration_service: RegistrationService,
        matcher: CosineSimilarityMatcher,
        embedder,  # noqa: ANN001 - fixture
    ) -> None:
        """The same identity matches with maximal similarity."""
        registration_service.register("S1", "Alice", [make_image(_ALICE)])
        probe = embedder.embed(make_image(_ALICE), make_face())

        result = matcher.match(probe, make_face())
        assert result.recognized is True
        assert result.student_id == "S1"
        assert result.confidence == pytest.approx(1.0, abs=1e-4)

    def test_a_stranger_is_not_recognised(
        self,
        registration_service: RegistrationService,
        matcher: CosineSimilarityMatcher,
        embedder,  # noqa: ANN001 - fixture
    ) -> None:
        """A different identity falls below the threshold."""
        registration_service.register("S1", "Alice", [make_image(_ALICE)])
        probe = embedder.embed(make_image(_STRANGER), make_face())

        result = matcher.match(probe, make_face())
        assert result.recognized is False
        assert result.student_id is None

    def test_picks_the_closest_of_several_students(
        self,
        registration_service: RegistrationService,
        matcher: CosineSimilarityMatcher,
        embedder,  # noqa: ANN001 - fixture
    ) -> None:
        """With a populated gallery, the right student wins."""
        registration_service.register("S1", "Alice", [make_image(_ALICE)])
        registration_service.register("S2", "Bob", [make_image(_BOB)])

        result = matcher.match(embedder.embed(make_image(_BOB), make_face()), make_face())
        assert result.student_id == "S2"
        assert result.name == "Bob"

    def test_reports_runner_up_candidates(
        self,
        registration_service: RegistrationService,
        matcher: CosineSimilarityMatcher,
        embedder,  # noqa: ANN001 - fixture
    ) -> None:
        """Diagnostics: the ranked alternatives come back with the result."""
        registration_service.register("S1", "Alice", [make_image(_ALICE)])
        registration_service.register("S2", "Bob", [make_image(_BOB)])

        result = matcher.match(embedder.embed(make_image(_ALICE), make_face()), make_face())
        assert len(result.candidates) == 2
        assert result.candidates[0].student_id == "S1"
        assert result.candidates[0].similarity >= result.candidates[1].similarity

    def test_margin_separates_best_from_runner_up(
        self,
        registration_service: RegistrationService,
        matcher: CosineSimilarityMatcher,
        embedder,  # noqa: ANN001 - fixture
    ) -> None:
        """A confident match is well clear of the next candidate."""
        registration_service.register("S1", "Alice", [make_image(_ALICE)])
        registration_service.register("S2", "Bob", [make_image(_BOB)])

        result = matcher.match(embedder.embed(make_image(_ALICE), make_face()), make_face())
        assert result.margin > 0.5

    def test_index_refreshes_after_registration(
        self, registration_service: RegistrationService, matcher: CosineSimilarityMatcher
    ) -> None:
        """A new student is recognisable without restarting the service."""
        assert matcher.is_empty is True
        registration_service.register("S1", "Alice", [make_image(_ALICE)] * 2)

        assert matcher.is_empty is False
        assert matcher.size == 2
        assert matcher.student_count == 1

    def test_index_refreshes_after_deletion(
        self, registration_service: RegistrationService, matcher: CosineSimilarityMatcher
    ) -> None:
        """A deleted student stops being matchable immediately."""
        registration_service.register("S1", "Alice", [make_image(_ALICE)])
        registration_service.delete("S1")

        assert matcher.is_empty is True

    def test_threshold_governs_recognition(
        self,
        settings: Settings,
        registration_service: RegistrationService,
        database: LocalFileDatabase,
        matcher: CosineSimilarityMatcher,
        embedder,  # noqa: ANN001 - fixture
    ) -> None:
        """The threshold alone decides identity: the same probe flips with it.

        This is the security-relevant knob. Dropping it to zero accepts a
        stranger the default configuration correctly rejects, which is why the
        default must never be lowered casually.
        """
        registration_service.register("S1", "Alice", [make_image(_ALICE)])
        probe = embedder.embed(make_image(_STRANGER), make_face())

        assert matcher.match(probe, make_face()).recognized is False

        permissive = Settings(
            **{
                **settings.model_dump(),
                "recognition_threshold": 0.0,
                "attendance_threshold": 0.0,
            }
        )
        lenient = CosineSimilarityMatcher(
            permissive, database.students, database.embeddings
        )
        assert lenient.match(probe, make_face()).recognized is True

    def test_thresholds_must_be_consistent(self, settings: Settings) -> None:
        """Attendance can never be marked below the recognition threshold."""
        with pytest.raises(ValueError, match="attendance_threshold must be >="):
            Settings(
                **{
                    **settings.model_dump(),
                    "recognition_threshold": 0.8,
                    "attendance_threshold": 0.5,
                }
            )

    def test_orphaned_embeddings_are_ignored(
        self, matcher: CosineSimilarityMatcher, database: LocalFileDatabase
    ) -> None:
        """Vectors with no registry entry must never produce a nameless match."""
        database.embeddings.save("orphan", np.ones((1, 512), dtype=np.float32))
        matcher.refresh()

        assert matcher.is_empty is True


class TestRecognitionPipeline:
    """Behaviour of :class:`RecognitionService`."""

    def test_recognises_a_registered_student(
        self,
        registration_service: RegistrationService,
        recognition_service: RecognitionService,
    ) -> None:
        """The full pipeline returns the student, name and confidence."""
        registration_service.register("CS2021001", "Alice", [make_image(_ALICE)] * 3)
        report = recognition_service.recognize(make_image(_ALICE))

        assert report.faces_detected == 1
        assert len(report.recognized) == 1
        best = report.best
        assert best is not None
        assert best.student_id == "CS2021001"
        assert best.name == "Alice"
        assert best.confidence > 0.9

    def test_empty_gallery_raises_by_default(
        self, recognition_service: RecognitionService
    ) -> None:
        """Recognising before anyone is enrolled is a caller error."""
        with pytest.raises(EmptyGalleryError):
            recognition_service.recognize(make_image(_ALICE))

    def test_empty_gallery_can_be_tolerated(
        self, recognition_service: RecognitionService
    ) -> None:
        """The live CLI keeps drawing 'Unknown' boxes instead of crashing."""
        report = recognition_service.recognize(make_image(_ALICE), require_gallery=False)

        assert report.faces_detected == 1
        assert report.recognized == []

    def test_no_face_yields_an_empty_report(
        self,
        registration_service: RegistrationService,
        recognition_service: RecognitionService,
        detector: FakeDetector,
    ) -> None:
        """An empty frame is normal for a live feed, not an error."""
        registration_service.register("S1", "Alice", [make_image(_ALICE)])
        detector.faces = []

        report = recognition_service.recognize(make_image(_ALICE))
        assert report.faces_detected == 0
        assert report.best is None

    def test_recognize_one_raises_when_no_face(
        self,
        registration_service: RegistrationService,
        recognition_service: RecognitionService,
        detector: FakeDetector,
    ) -> None:
        """Single-subject mode expects a face to be present."""
        registration_service.register("S1", "Alice", [make_image(_ALICE)])
        detector.faces = []

        with pytest.raises(NoFaceDetectedError):
            recognition_service.recognize_one(make_image(_ALICE))

    def test_max_faces_caps_processing(
        self,
        registration_service: RegistrationService,
        recognition_service: RecognitionService,
        detector: FakeDetector,
    ) -> None:
        """Bystanders can be excluded by processing only the largest face."""
        registration_service.register("S1", "Alice", [make_image(_ALICE)])
        detector.faces = [make_face(300), make_face(200), make_face(100)]

        assert recognition_service.recognize(make_image(_ALICE), max_faces=1).faces_detected == 1

    def test_recognize_and_mark_writes_attendance(
        self,
        registration_service: RegistrationService,
        recognition_service: RecognitionService,
        database: LocalFileDatabase,
    ) -> None:
        """The kiosk path: one call recognises and records."""
        registration_service.register("S1", "Alice", [make_image(_ALICE)])
        report = recognition_service.recognize_and_mark(
            make_image(_ALICE), session="lecture-1"
        )

        assert len(report.attendance) == 1
        assert report.attendance[0].status is AttendanceStatus.MARKED
        assert database.attendance.count() == 1

    def test_repeated_frames_mark_once(
        self,
        registration_service: RegistrationService,
        recognition_service: RecognitionService,
        database: LocalFileDatabase,
    ) -> None:
        """A camera sees the same student many times; the log gets one row."""
        registration_service.register("S1", "Alice", [make_image(_ALICE)])
        for _ in range(5):
            recognition_service.recognize_and_mark(make_image(_ALICE), session="lecture-1")

        assert database.attendance.count() == 1

    def test_a_stranger_is_not_marked(
        self,
        registration_service: RegistrationService,
        recognition_service: RecognitionService,
        database: LocalFileDatabase,
    ) -> None:
        """Proxy attendance: an unenrolled face must never be recorded."""
        registration_service.register("S1", "Alice", [make_image(_ALICE)])
        report = recognition_service.recognize_and_mark(make_image(_STRANGER))

        assert report.recognized == []
        assert database.attendance.count() == 0

    def test_report_serialises_for_the_api(
        self,
        registration_service: RegistrationService,
        recognition_service: RecognitionService,
    ) -> None:
        """The response payload carries the contract's required fields."""
        registration_service.register("S1", "Alice", [make_image(_ALICE)])
        payload = recognition_service.recognize(make_image(_ALICE)).to_dict()

        assert payload["faces_detected"] == 1
        result = payload["results"][0]
        assert {"student_id", "name", "confidence", "bbox", "recognized"} <= set(result)
        assert result["student_id"] == "S1"
