"""Face embedding using the pre-trained ArcFace recogniser.

ArcFace (Deng et al., 2019) maps an aligned 112x112 face crop to a 512-D
vector trained with an additive angular margin loss, so that vectors of the same
identity point in nearly the same direction while different identities are
pushed apart. No training happens here: the network from the InsightFace pack is
used purely for inference.

Alignment lives in this module because it is part of the recogniser's input
contract — ArcFace expects a similarity-transformed crop derived from the five
facial landmarks, not an arbitrary bounding-box crop.
"""

from __future__ import annotations

import threading
from typing import Protocol, runtime_checkable

import numpy as np

from config import Settings
from domain import DetectedFace
from exceptions import InferenceError, ModelLoadError
from logging_config import get_logger
from services.engine import ModelPackManager
from utils.similarity import l2_normalize

logger = get_logger(__name__)

#: ArcFace's expected input resolution (width, height).
_ARCFACE_INPUT_SIZE: tuple[int, int] = (112, 112)


@runtime_checkable
class FaceEmbedderProtocol(Protocol):
    """The contract every embedding implementation must satisfy."""

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the vectors this embedder produces."""
        ...

    def align(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Return the aligned face crop the recogniser consumes."""
        ...

    def embed(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Return the L2-normalised embedding of one detected face."""
        ...

    def embed_aligned(self, aligned_face: np.ndarray) -> np.ndarray:
        """Return the embedding of an already-aligned face crop."""
        ...


class ArcFaceEmbedder:
    """ArcFace (R50) embedding model from the pre-trained InsightFace pack.

    Every embedding returned is L2-normalised, which makes cosine similarity a
    plain dot product downstream and keeps stored vectors directly comparable.

    Args:
        settings: Service configuration (execution context, expected dimension).
        model_pack: Manager that resolves and downloads the pre-trained weights.
    """

    def __init__(self, settings: Settings, model_pack: ModelPackManager) -> None:
        self._settings = settings
        self._model_pack = model_pack
        self._model: object | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load the ArcFace network. Idempotent and thread-safe.

        Raises:
            ModelLoadError: If InsightFace is missing, the weights cannot be
                found, or the ONNX session cannot be created.
        """
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            try:
                from insightface import model_zoo
            except ImportError as exc:  # pragma: no cover - environment issue
                raise ModelLoadError(
                    "The 'insightface' package is not installed. "
                    "Run: pip install -r requirements.txt",
                    details={"import_error": str(exc)},
                ) from exc

            model_path = self._model_pack.recognition_model_path()
            logger.info("Loading ArcFace recogniser from %s", model_path)

            try:
                model = model_zoo.get_model(
                    str(model_path), providers=self._model_pack.providers()
                )
                if model is None:
                    raise ModelLoadError(
                        f"InsightFace could not build a model from {model_path}",
                        details={"model_path": str(model_path)},
                    )
                model.prepare(ctx_id=self._settings.ctx_id)
            except ModelLoadError:
                raise
            except Exception as exc:  # noqa: BLE001 - upstream raises bare Exception
                raise ModelLoadError(
                    f"Failed to initialise the ArcFace recogniser: {exc}",
                    details={"model_path": str(model_path)},
                ) from exc

            self._model = model
            logger.info(
                "ArcFace recogniser ready (embedding_dim=%s)", self.embedding_dim
            )

    @property
    def is_loaded(self) -> bool:
        """Whether the recognition model has been loaded."""
        return self._model is not None

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embeddings produced (512 for ArcFace R50)."""
        return self._settings.embedding_dim

    @property
    def input_size(self) -> tuple[int, int]:
        """The aligned crop size the network expects, ``(width, height)``."""
        if self._model is not None:
            size = getattr(self._model, "input_size", None)
            if size is not None:
                return int(size[0]), int(size[1])
        return _ARCFACE_INPUT_SIZE

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def align(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Align a detected face into the canonical ArcFace crop.

        Applies a similarity transform that maps the face's five landmarks onto
        ArcFace's reference template, normalising rotation, scale and
        translation so that pose variation does not leak into the embedding.

        Args:
            image: The full BGR image the face was detected in.
            face: The detected face, including its landmarks.

        Returns:
            The aligned crop, shape ``(112, 112, 3)``.

        Raises:
            InferenceError: If alignment fails.
        """
        try:
            from insightface.utils import face_align
        except ImportError as exc:  # pragma: no cover - environment issue
            raise ModelLoadError(
                "The 'insightface' package is not installed. "
                "Run: pip install -r requirements.txt",
                details={"import_error": str(exc)},
            ) from exc

        try:
            return face_align.norm_crop(
                image, landmark=face.keypoints, image_size=self.input_size[0]
            )
        except Exception as exc:  # noqa: BLE001 - upstream raises bare Exception
            raise InferenceError(f"Face alignment failed: {exc}") from exc

    def embed(self, image: np.ndarray, face: DetectedFace) -> np.ndarray:
        """Align a face and return its L2-normalised ArcFace embedding.

        Args:
            image: The full BGR image the face was detected in.
            face: The detected face to embed.

        Returns:
            A unit-norm float32 vector of shape ``(embedding_dim,)``.

        Raises:
            ModelLoadError: If the model cannot be loaded.
            InferenceError: If alignment or inference fails, or the network
                returns an unexpected shape.
        """
        self.load()
        aligned = self.align(image, face)
        return self.embed_aligned(aligned)

    def embed_aligned(self, aligned_face: np.ndarray) -> np.ndarray:
        """Embed a crop that has already been aligned.

        Args:
            aligned_face: A ``(112, 112, 3)`` BGR crop from :meth:`align`.

        Returns:
            A unit-norm float32 vector of shape ``(embedding_dim,)``.

        Raises:
            InferenceError: If inference fails or the output shape is wrong.
        """
        self.load()

        try:
            features = self._model.get_feat(aligned_face)  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 - upstream raises bare Exception
            raise InferenceError(f"ArcFace inference failed: {exc}") from exc

        vector = np.asarray(features, dtype=np.float32).flatten()
        if vector.shape[0] != self.embedding_dim:
            raise InferenceError(
                "ArcFace returned an unexpected embedding size",
                details={
                    "expected": self.embedding_dim,
                    "received": int(vector.shape[0]),
                },
            )
        return l2_normalize(vector)

    def embed_many(
        self, image: np.ndarray, faces: list[DetectedFace]
    ) -> list[np.ndarray]:
        """Embed several faces from the same image.

        Args:
            image: The full BGR image the faces were detected in.
            faces: Faces to embed.

        Returns:
            One unit-norm vector per input face, in the same order.
        """
        return [self.embed(image, face) for face in faces]
