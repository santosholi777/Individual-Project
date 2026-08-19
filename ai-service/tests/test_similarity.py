"""Tests for the cosine similarity maths underpinning identity matching."""

from __future__ import annotations

import numpy as np
import pytest

from utils.similarity import (
    cosine_similarity,
    cosine_similarity_matrix,
    l2_normalize,
    similarity_to_percentage,
)


class TestL2Normalize:
    """Behaviour of :func:`utils.similarity.l2_normalize`."""

    def test_produces_unit_vector(self) -> None:
        """A normalised vector has length 1."""
        result = l2_normalize(np.array([3.0, 4.0]))
        assert np.isclose(np.linalg.norm(result), 1.0)

    def test_normalizes_each_row_of_a_matrix(self) -> None:
        """Every row of a matrix is normalised independently."""
        result = l2_normalize(np.array([[3.0, 4.0], [1.0, 0.0]]), axis=1)
        assert np.allclose(np.linalg.norm(result, axis=1), [1.0, 1.0])

    def test_zero_vector_does_not_produce_nan(self) -> None:
        """A zero vector is returned as zeros rather than NaNs."""
        result = l2_normalize(np.zeros(4))
        assert not np.isnan(result).any()

    def test_preserves_direction(self) -> None:
        """Only magnitude changes; direction is untouched."""
        result = l2_normalize(np.array([2.0, 0.0, 0.0]))
        assert np.allclose(result, [1.0, 0.0, 0.0])


class TestCosineSimilarity:
    """Behaviour of :func:`utils.similarity.cosine_similarity`."""

    def test_identical_vectors_score_one(self) -> None:
        """A vector is maximally similar to itself."""
        vector = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(vector, vector) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self) -> None:
        """Perpendicular vectors share no direction."""
        assert cosine_similarity(
            np.array([1.0, 0.0]), np.array([0.0, 1.0])
        ) == pytest.approx(0.0)

    def test_opposite_vectors_score_minus_one(self) -> None:
        """Anti-parallel vectors score -1."""
        assert cosine_similarity(
            np.array([1.0, 0.0]), np.array([-1.0, 0.0])
        ) == pytest.approx(-1.0)

    def test_is_scale_invariant(self) -> None:
        """Magnitude is irrelevant: only the angle matters."""
        assert cosine_similarity(
            np.array([1.0, 2.0]), np.array([10.0, 20.0])
        ) == pytest.approx(1.0)

    def test_zero_vector_scores_zero_instead_of_dividing_by_zero(self) -> None:
        """A degenerate vector yields 0.0, not an exception."""
        assert cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0

    def test_mismatched_shapes_raise(self) -> None:
        """Comparing different dimensionalities is a programming error."""
        with pytest.raises(ValueError, match="Vector shapes differ"):
            cosine_similarity(np.zeros(3), np.zeros(4))


class TestCosineSimilarityMatrix:
    """Behaviour of :func:`utils.similarity.cosine_similarity_matrix`."""

    def test_scores_every_gallery_row(self) -> None:
        """One score is returned per gallery entry."""
        gallery = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        scores = cosine_similarity_matrix(np.array([1.0, 0.0]), gallery)

        assert scores.shape == (3,)
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.0)
        assert scores[2] == pytest.approx(0.7071, abs=1e-4)

    def test_agrees_with_the_pairwise_implementation(self) -> None:
        """The batched path must not drift from the single-pair one."""
        generator = np.random.default_rng(42)
        query = generator.normal(size=512)
        gallery = generator.normal(size=(10, 512))

        batched = cosine_similarity_matrix(query, gallery)
        pairwise = [cosine_similarity(query, row) for row in gallery]
        assert np.allclose(batched, pairwise, atol=1e-6)

    def test_empty_gallery_returns_no_scores(self) -> None:
        """Scoring against nobody yields an empty array, not an error."""
        assert cosine_similarity_matrix(np.zeros(512), np.zeros((0, 512))).shape == (0,)

    def test_dimension_mismatch_raises(self) -> None:
        """A gallery of the wrong width is a configuration error."""
        with pytest.raises(ValueError, match="Dimension mismatch"):
            cosine_similarity_matrix(np.zeros(512), np.zeros((3, 128)))

    def test_non_2d_gallery_raises(self) -> None:
        """The gallery must be a matrix."""
        with pytest.raises(ValueError, match="must be 2-D"):
            cosine_similarity_matrix(np.zeros(512), np.zeros(512))


class TestSimilarityToPercentage:
    """Behaviour of :func:`utils.similarity.similarity_to_percentage`."""

    def test_converts_to_percent(self) -> None:
        """0.8532 similarity reads as 85.32%."""
        assert similarity_to_percentage(0.8532) == 85.32

    def test_clips_negative_similarity_to_zero(self) -> None:
        """Negative similarity is meaningless on a dashboard."""
        assert similarity_to_percentage(-0.5) == 0.0

    def test_clips_above_one(self) -> None:
        """Floating-point overshoot never renders as >100%."""
        assert similarity_to_percentage(1.2) == 100.0
