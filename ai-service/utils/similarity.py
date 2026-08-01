"""Vector similarity helpers used by the face matcher.

ArcFace embeddings are compared with cosine similarity. Once vectors are
L2-normalised, cosine similarity reduces to a dot product, which lets the whole
gallery be scored with a single matrix multiplication.
"""

from __future__ import annotations

import numpy as np

#: Guards against division by zero when normalising a degenerate vector.
_EPSILON: float = 1e-10


def l2_normalize(vectors: np.ndarray, axis: int = -1) -> np.ndarray:
    """Scale vectors to unit length along ``axis``.

    Args:
        vectors: Array of shape ``(dim,)`` or ``(n, dim)``.
        axis: Axis along which the norm is computed.

    Returns:
        A float32 array of the same shape with unit-norm vectors. Zero vectors
        are returned unchanged rather than producing NaNs.
    """
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    return (array / np.maximum(norms, _EPSILON)).astype(np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return the cosine similarity between two 1-D vectors.

    Args:
        left: First vector, shape ``(dim,)``.
        right: Second vector, shape ``(dim,)``.

    Returns:
        Similarity in ``[-1, 1]``; ``1`` means identical direction.

    Raises:
        ValueError: If the vectors have different dimensionality.
    """
    a = np.asarray(left, dtype=np.float32).flatten()
    b = np.asarray(right, dtype=np.float32).flatten()
    if a.shape != b.shape:
        raise ValueError(f"Vector shapes differ: {a.shape} vs {b.shape}")

    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < _EPSILON:
        return 0.0
    return float(np.dot(a, b) / denominator)


def cosine_similarity_matrix(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Score one probe embedding against every gallery embedding at once.

    Both inputs are normalised defensively, so the caller may pass raw model
    output without a separate normalisation step.

    Args:
        query: Probe embedding, shape ``(dim,)``.
        gallery: Gallery matrix, shape ``(n, dim)``.

    Returns:
        Similarities of shape ``(n,)``, aligned with the gallery rows.

    Raises:
        ValueError: If the gallery is not 2-D or its dimensionality differs
            from the query's.
    """
    probe = l2_normalize(np.asarray(query, dtype=np.float32).flatten())
    matrix = np.asarray(gallery, dtype=np.float32)

    if matrix.ndim != 2:
        raise ValueError(f"gallery must be 2-D, got shape {matrix.shape}")
    if matrix.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if matrix.shape[1] != probe.shape[0]:
        raise ValueError(
            f"Dimension mismatch: query has {probe.shape[0]}, "
            f"gallery has {matrix.shape[1]}"
        )

    return (l2_normalize(matrix, axis=1) @ probe).astype(np.float32)


def similarity_to_percentage(similarity: float) -> float:
    """Convert a cosine similarity into a 0–100 score for display.

    Negative similarities are clipped to zero: for face embeddings they carry no
    useful "less than nothing" meaning and would only confuse a dashboard.

    Args:
        similarity: Cosine similarity in ``[-1, 1]``.

    Returns:
        Percentage in ``[0, 100]`` rounded to two decimals.
    """
    return round(float(min(max(similarity, 0.0), 1.0)) * 100.0, 2)
