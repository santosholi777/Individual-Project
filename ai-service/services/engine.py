"""Management of the pre-trained InsightFace model pack.

This service performs **inference only** — no model is trained here. The
``buffalo_l`` pack (SCRFD-10G detector + ArcFace R50 recogniser, trained on
WebFace600K) is downloaded once on first run and cached under ``models/``.

Both the detector and the embedder resolve their weights through
:class:`ModelPackManager`, so the pack is downloaded once and the ONNX Runtime
execution providers are chosen in exactly one place.
"""

from __future__ import annotations

import threading
from pathlib import Path

from config import Settings
from exceptions import ModelLoadError
from logging_config import get_logger

logger = get_logger(__name__)


class ModelPackManager:
    """Ensures the pre-trained model pack is present and resolves its files.

    Args:
        settings: Service configuration supplying the pack name, cache directory
            and execution context.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pack_dir: Path | None = None
        self._lock = threading.Lock()

    @property
    def pack_name(self) -> str:
        """Name of the InsightFace model pack in use, e.g. ``buffalo_l``."""
        return self._settings.model_pack

    def ensure_pack(self) -> Path:
        """Download the model pack if needed and return its directory.

        The download runs at most once per process; concurrent callers block on
        a lock rather than racing to unzip the same archive.

        Returns:
            Directory containing the pack's ``.onnx`` files.

        Raises:
            ModelLoadError: If InsightFace is not installed, or the pack cannot
                be downloaded (typically no network on first run).
        """
        if self._pack_dir is not None:
            return self._pack_dir

        with self._lock:
            if self._pack_dir is not None:
                return self._pack_dir

            try:
                from insightface.utils import storage as insightface_storage
            except ImportError as exc:  # pragma: no cover - environment issue
                raise ModelLoadError(
                    "The 'insightface' package is not installed. "
                    "Run: pip install -r requirements.txt",
                    details={"import_error": str(exc)},
                ) from exc

            self._settings.models_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Ensuring pre-trained model pack '%s' is available under %s",
                self.pack_name,
                self._settings.models_dir,
            )

            try:
                # Downloads and unzips only when the directory is absent.
                pack_dir = insightface_storage.ensure_available(
                    "models", self.pack_name, root=str(self._settings.models_dir)
                )
            except Exception as exc:  # noqa: BLE001 - upstream raises bare Exception
                raise ModelLoadError(
                    f"Could not download the '{self.pack_name}' model pack. "
                    "The first run needs internet access to fetch the "
                    "pre-trained weights.",
                    details={"cause": str(exc), "pack": self.pack_name},
                ) from exc

            self._pack_dir = Path(pack_dir)
            logger.info("Model pack ready at %s", self._pack_dir)
            return self._pack_dir

    def recognition_model_path(self) -> Path:
        """Return the path to the ArcFace recognition network.

        Returns:
            Path to the recognition ``.onnx`` file inside the pack.

        Raises:
            ModelLoadError: If the file is missing from an otherwise valid pack.
        """
        pack_dir = self.ensure_pack()
        model_path = pack_dir / self._settings.recognition_model_file

        if not model_path.is_file():
            available = sorted(path.name for path in pack_dir.glob("*.onnx"))
            raise ModelLoadError(
                f"Recognition model '{self._settings.recognition_model_file}' was "
                f"not found in the '{self.pack_name}' pack.",
                details={"pack_dir": str(pack_dir), "available_models": available},
            )
        return model_path

    def providers(self) -> list[str]:
        """Return the ONNX Runtime execution providers, best first.

        A CUDA provider is requested when ``ctx_id >= 0``; ONNX Runtime falls
        back to CPU on its own if the GPU build is unavailable.

        Returns:
            Provider names in priority order.
        """
        if self._settings.ctx_id >= 0:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def describe(self) -> dict[str, object]:
        """Return a diagnostic summary for the ``/health`` endpoint."""
        return {
            "pack": self.pack_name,
            "pack_dir": str(self._pack_dir) if self._pack_dir else None,
            "downloaded": self._pack_dir is not None,
            "ctx_id": self._settings.ctx_id,
            "device": "cuda" if self._settings.ctx_id >= 0 else "cpu",
            "providers": self.providers(),
        }
