"""Runs an evaluation dataset through the real recognition pipeline.

Deliberately uses the production detector, embedder and matcher rather than a
parallel implementation: an evaluation that measures a different code path from
the deployed one measures nothing useful.

The gallery is built in memory. The evaluation never touches the service's real
student registry or attendance log — running it cannot disturb live data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import Settings
from evaluation.dataset import Dataset, Identity, ProbeImage
from evaluation.metrics import ProbeOutcome
from exceptions import DeepVisionAttendError
from logging_config import get_logger
from services.detector import FaceDetectorProtocol
from services.embedder import FaceEmbedderProtocol
from utils.image_utils import blur_score, load_image, resize_max_side
from utils.similarity import cosine_similarity_matrix, l2_normalize

logger = get_logger(__name__)


@dataclass(slots=True)
class Gallery:
    """The enrolled identities, as one stacked matrix for fast scoring."""

    matrix: np.ndarray
    owners: list[str]
    ids: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """Whether anything was successfully enrolled."""
        return self.matrix.shape[0] == 0

    def score(self, embedding: np.ndarray) -> dict[str, float]:
        """Score a probe against every identity.

        Scores are collapsed per identity by taking the **maximum** across that
        person's enrolment images — identical to the production matcher, so the
        measurement reflects what the deployed system does.
        """
        if self.is_empty:
            return {}

        similarities = cosine_similarity_matrix(embedding, self.matrix)
        best: dict[str, float] = {}
        for owner, score in zip(self.owners, similarities):
            value = float(score)
            if value > best.get(owner, -2.0):
                best[owner] = value
        return best


@dataclass(slots=True)
class EnrolmentReport:
    """What happened while building the gallery."""

    enrolled: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


class EvaluationRunner:
    """Builds a gallery from a dataset and scores every probe against it.

    Args:
        settings: Service configuration (thresholds, quality gates).
        detector: The production face detector.
        embedder: The production ArcFace embedder.
    """

    def __init__(
        self,
        settings: Settings,
        detector: FaceDetectorProtocol,
        embedder: FaceEmbedderProtocol,
    ) -> None:
        self._settings = settings
        self._detector = detector
        self._embedder = embedder

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    def _embed_file(self, path) -> tuple[np.ndarray | None, str | None]:
        """Embed the largest face in an image file.

        The largest face is used rather than requiring exactly one, so a
        bystander in a real classroom photo does not silently drop the sample.

        Returns:
            ``(embedding, failure_reason)`` — exactly one is non-None.
        """
        try:
            image = resize_max_side(load_image(path), max_side=1920)
            face = self._detector.detect_primary(image)
            aligned = self._embedder.align(image, face)
            return self._embedder.embed_aligned(aligned), None
        except DeepVisionAttendError as exc:
            return None, exc.message
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            logger.exception("Unexpected error embedding %s", path)
            return None, f"unexpected error: {exc}"

    # ------------------------------------------------------------------
    # Gallery
    # ------------------------------------------------------------------
    def build_gallery(self, identities: list[Identity]) -> tuple[Gallery, EnrolmentReport]:
        """Enrol every identity's enrolment images into an in-memory gallery.

        Args:
            identities: Identities to enrol.

        Returns:
            The gallery and a report of enrolment failures.
        """
        vectors: list[np.ndarray] = []
        owners: list[str] = []
        report = EnrolmentReport()

        for identity in identities:
            accepted = 0
            for path in identity.enrol_paths:
                embedding, failure = self._embed_file(path)
                if embedding is None:
                    report.failures.append(f"{identity.student_id}/{path.name}: {failure}")
                    continue
                vectors.append(embedding)
                owners.append(identity.student_id)
                accepted += 1

            if accepted:
                report.enrolled.append(identity.student_id)
            else:
                report.failures.append(
                    f"{identity.student_id}: no enrolment image produced an embedding"
                )

        matrix = (
            l2_normalize(np.vstack(vectors), axis=1)
            if vectors
            else np.zeros((0, self._settings.embedding_dim), dtype=np.float32)
        )
        logger.info(
            "Gallery built: %s embeddings across %s identities (%s failures)",
            matrix.shape[0],
            len(set(owners)),
            len(report.failures),
        )
        return Gallery(matrix=matrix, owners=owners, ids=sorted(set(owners))), report

    # ------------------------------------------------------------------
    # Probing
    # ------------------------------------------------------------------
    def score_probe(
        self, probe: ProbeImage, gallery: Gallery, is_unknown: bool = False
    ) -> ProbeOutcome:
        """Score one probe image against the gallery.

        Args:
            probe: The probe image and its ground truth.
            gallery: The enrolled gallery.
            is_unknown: True when the probe's person was deliberately not
                enrolled.

        Returns:
            The probe's outcome, including a failure reason if no face was
            usable.
        """
        outcome = ProbeOutcome(
            identity=probe.identity,
            condition=probe.condition,
            path=str(probe.path),
            predicted=None,
            best_score=0.0,
            genuine_score=None,
            is_unknown=is_unknown,
        )

        embedding, failure = self._embed_file(probe.path)
        if embedding is None:
            outcome.failure = failure
            return outcome

        scores = gallery.score(embedding)
        if not scores:
            outcome.failure = "gallery is empty"
            return outcome

        predicted = max(scores, key=lambda key: scores[key])
        outcome.predicted = predicted
        outcome.best_score = scores[predicted]

        if not is_unknown:
            outcome.genuine_score = scores.get(probe.identity)
        outcome.impostor_scores = [
            score
            for identity, score in scores.items()
            if is_unknown or identity != probe.identity
        ]
        return outcome

    def run(self, dataset: Dataset) -> tuple[list[ProbeOutcome], EnrolmentReport, Gallery]:
        """Build the gallery and score every probe in the dataset.

        Args:
            dataset: The loaded dataset.

        Returns:
            ``(outcomes, enrolment_report, gallery)``.

        Raises:
            DeepVisionAttendError: If nothing could be enrolled at all.
        """
        gallery, enrolment = self.build_gallery(dataset.identities)
        if gallery.is_empty:
            raise DeepVisionAttendError(
                "No enrolment image produced a usable face embedding, so there "
                "is nothing to evaluate against.",
                details={"failures": enrolment.failures[:10]},
            )

        outcomes: list[ProbeOutcome] = []
        total = dataset.probe_count + dataset.unknown_probe_count
        done = 0

        for identity in dataset.identities:
            for probe in identity.probes:
                outcomes.append(self.score_probe(probe, gallery, is_unknown=False))
                done += 1
                if done % 20 == 0:
                    logger.info("Scored %s/%s probes", done, total)

        for identity in dataset.unknowns:
            for probe in identity.probes:
                outcomes.append(self.score_probe(probe, gallery, is_unknown=True))
                done += 1

        failures = sum(1 for outcome in outcomes if outcome.failure is not None)
        logger.info("Scored %s probes (%s pipeline failures)", len(outcomes), failures)
        return outcomes, enrolment, gallery


def image_quality(path) -> float:
    """Sharpness of an image, for dataset triage.

    Exposed so a user can find out whether a poor result is the model's fault or
    the photograph's before drawing conclusions from it.
    """
    return blur_score(load_image(path))
