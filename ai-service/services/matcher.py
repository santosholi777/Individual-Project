"""Identity matching of face embeddings via cosine similarity.

Every registered embedding is stacked into one ``(N, 512)`` matrix so a probe
can be scored against the entire college in a single matrix multiplication,
rather than looping student by student. Scores are then grouped back per student
using the **maximum** similarity across that student's enrolled poses: a student
matches if the probe resembles *any* enrolled view of them, which is what makes
multi-image registration worthwhile.
"""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

import numpy as np

from config import Settings
from database import EmbeddingRepository, StudentRepository
from domain import DetectedFace, MatchCandidate, RecognitionResult
from logging_config import get_logger
from utils.similarity import cosine_similarity_matrix, l2_normalize

logger = get_logger(__name__)


@runtime_checkable
class FaceMatcherProtocol(Protocol):
    """The contract every matcher implementation must satisfy."""

    def match(self, embedding: np.ndarray, face: DetectedFace) -> RecognitionResult:
        """Match one probe embedding against the gallery."""
        ...

    def refresh(self) -> None:
        """Rebuild the in-memory index from the repositories."""
        ...


class CosineSimilarityMatcher:
    """Matches probe embeddings against an in-memory gallery index.

    The index is a cache of what the repositories hold. Any code path that
    changes stored embeddings must call :meth:`refresh` afterwards — the
    registration service does this, which is why a student can be recognised
    immediately after enrolling without a restart.

    Args:
        settings: Supplies the recognition threshold and embedding dimension.
        student_repository: Source of student names for match results.
        embedding_repository: Source of the enrolled embeddings.
    """

    def __init__(
        self,
        settings: Settings,
        student_repository: StudentRepository,
        embedding_repository: EmbeddingRepository,
    ) -> None:
        self._settings = settings
        self._students = student_repository
        self._embeddings = embedding_repository
        self._lock = threading.RLock()

        #: Stacked, L2-normalised gallery of shape ``(N, dim)``.
        self._matrix: np.ndarray = np.zeros(
            (0, settings.embedding_dim), dtype=np.float32
        )
        #: Student id for each row of :attr:`_matrix`.
        self._row_owners: list[str] = []
        #: Display name per student id.
        self._names: dict[str, str] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Rebuild the index from the repositories.

        Cheap for a college-sized gallery (a few thousand vectors), and always
        consistent, which is preferable to incremental updates that can drift
        out of sync with storage.
        """
        with self._lock:
            gallery = self._embeddings.load_all()
            names = {
                student.student_id: student.name for student in self._students.list_all()
            }

            vectors: list[np.ndarray] = []
            owners: list[str] = []

            for student_id, matrix in gallery.items():
                if student_id not in names:
                    # Embeddings without a registry entry are orphaned data;
                    # ignore them rather than reporting a nameless match.
                    logger.warning(
                        "Ignoring embeddings for unregistered student '%s'", student_id
                    )
                    continue
                if matrix.size == 0:
                    continue
                vectors.append(l2_normalize(matrix, axis=1))
                owners.extend([student_id] * matrix.shape[0])

            if vectors:
                self._matrix = np.vstack(vectors).astype(np.float32)
            else:
                self._matrix = np.zeros(
                    (0, self._settings.embedding_dim), dtype=np.float32
                )

            self._row_owners = owners
            self._names = names
            self._loaded = True
            logger.info(
                "Gallery index rebuilt: %s embedding(s) across %s student(s)",
                self._matrix.shape[0],
                len(set(owners)),
            )

    def _ensure_loaded(self) -> None:
        """Build the index on first use if it has not been built yet."""
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self.refresh()

    @property
    def is_empty(self) -> bool:
        """Whether the gallery holds no embeddings at all."""
        self._ensure_loaded()
        with self._lock:
            return self._matrix.shape[0] == 0

    @property
    def size(self) -> int:
        """Number of embeddings currently indexed."""
        self._ensure_loaded()
        with self._lock:
            return int(self._matrix.shape[0])

    @property
    def student_count(self) -> int:
        """Number of distinct students represented in the index."""
        self._ensure_loaded()
        with self._lock:
            return len(set(self._row_owners))

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------
    def match(
        self, embedding: np.ndarray, face: DetectedFace, top_k: int = 3
    ) -> RecognitionResult:
        """Match a probe embedding against the gallery.

        Args:
            embedding: The probe embedding, shape ``(dim,)``.
            face: The detected face the embedding came from, echoed back in the
                result so callers can draw the box.
            top_k: How many runner-up candidates to include for diagnostics.

        Returns:
            A result that is always populated: when nothing clears the
            threshold, ``recognized`` is ``False`` and the best candidates are
            still reported, which is what lets a low-confidence near-match be
            told apart from a total stranger.
        """
        self._ensure_loaded()

        with self._lock:
            matrix = self._matrix
            owners = list(self._row_owners)
            names = dict(self._names)

        if matrix.shape[0] == 0:
            logger.debug("Match attempted against an empty gallery")
            return RecognitionResult(face=face, recognized=False)

        similarities = cosine_similarity_matrix(embedding, matrix)

        # Collapse per-embedding scores into one score per student.
        best_per_student: dict[str, float] = {}
        for student_id, score in zip(owners, similarities):
            value = float(score)
            if value > best_per_student.get(student_id, -1.0):
                best_per_student[student_id] = value

        ranked = sorted(
            best_per_student.items(), key=lambda item: item[1], reverse=True
        )
        candidates = [
            MatchCandidate(
                student_id=student_id,
                name=names.get(student_id, "Unknown"),
                similarity=score,
            )
            for student_id, score in ranked[: max(top_k, 1)]
        ]

        best_id, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = float(best_score - runner_up)
        recognized = best_score >= self._settings.recognition_threshold

        if not recognized:
            logger.debug(
                "Best candidate %s scored %.3f, below threshold %.2f",
                best_id,
                best_score,
                self._settings.recognition_threshold,
            )
            return RecognitionResult(
                face=face,
                recognized=False,
                similarity=float(best_score),
                margin=margin,
                candidates=candidates,
            )

        return RecognitionResult(
            face=face,
            recognized=True,
            student_id=best_id,
            name=names.get(best_id, "Unknown"),
            similarity=float(best_score),
            margin=margin,
            candidates=candidates,
        )

    def match_many(
        self,
        embeddings: list[np.ndarray],
        faces: list[DetectedFace],
        top_k: int = 3,
    ) -> list[RecognitionResult]:
        """Match several probes, e.g. every face in a classroom frame.

        Args:
            embeddings: One probe embedding per face.
            faces: The detected faces, aligned with ``embeddings``.
            top_k: How many runner-up candidates to include per result.

        Returns:
            One result per input face, in the same order.

        Raises:
            ValueError: If the two lists have different lengths.
        """
        if len(embeddings) != len(faces):
            raise ValueError(
                f"Got {len(embeddings)} embeddings for {len(faces)} faces"
            )
        return [
            self.match(embedding, face, top_k=top_k)
            for embedding, face in zip(embeddings, faces)
        ]

    def describe(self) -> dict[str, object]:
        """Return index statistics for the ``/health`` endpoint."""
        self._ensure_loaded()
        with self._lock:
            return {
                "indexed_embeddings": int(self._matrix.shape[0]),
                "indexed_students": len(set(self._row_owners)),
                "recognition_threshold": self._settings.recognition_threshold,
            }
