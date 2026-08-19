"""Webcam access for the registration and live recognition CLIs.

Wraps ``cv2.VideoCapture`` in a context manager so the device is always released
— including when the user aborts with Ctrl-C — and turns OpenCV's silent
``None``/``False`` failure modes into explicit :class:`exceptions.CameraError`s.
"""

from __future__ import annotations

import time
from types import TracebackType
from typing import Callable, Iterator

import cv2
import numpy as np

from exceptions import CameraError
from logging_config import get_logger

logger = get_logger(__name__)

#: Progress callback for :meth:`Camera.capture_burst`, called as
#: ``(index, total, frame)`` after every kept capture.
CaptureCallback = Callable[[int, int, np.ndarray], None]


class Camera:
    """A webcam device that yields BGR frames.

    Intended to be used as a context manager::

        with Camera(index=0, width=1280, height=720) as camera:
            for frame in camera.stream():
                ...

    Args:
        index: OpenCV device index (``0`` is the default built-in camera).
        width: Requested capture width in pixels.
        height: Requested capture height in pixels.
        warmup_frames: Frames read and discarded on open so that auto-exposure
            and white balance can settle before the first real capture.
        mirror: Horizontally flip frames, which makes a front-facing preview
            behave like a mirror for the person being enrolled.
    """

    def __init__(
        self,
        index: int = 0,
        width: int = 1280,
        height: int = 720,
        warmup_frames: int = 10,
        mirror: bool = True,
    ) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._warmup_frames = warmup_frames
        self._mirror = mirror
        self._capture: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def open(self) -> "Camera":
        """Open the device and let the sensor settle.

        Returns:
            This camera, opened and ready.

        Raises:
            CameraError: If the device cannot be opened or produces no frames
                (on macOS this usually means camera permission was denied).
        """
        if self._capture is not None:
            return self

        logger.info("Opening camera device %s", self._index)
        capture = cv2.VideoCapture(self._index)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Could not open camera device {self._index}. Check that it is "
                "connected, not in use by another application, and that this "
                "terminal has camera permission.",
                details={"camera_index": self._index},
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._capture = capture

        for _ in range(self._warmup_frames):
            capture.read()

        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info("Camera %s ready at %sx%s", self._index, actual_width, actual_height)
        return self

    def release(self) -> None:
        """Release the device. Safe to call when already closed."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("Camera %s released", self._index)

    def __enter__(self) -> "Camera":
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    @property
    def is_open(self) -> bool:
        """Whether the device is currently open."""
        return self._capture is not None and self._capture.isOpened()

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    def read(self) -> np.ndarray:
        """Read a single frame.

        Returns:
            A BGR frame, mirrored when ``mirror`` is enabled.

        Raises:
            CameraError: If the camera is closed or the read fails (device
                unplugged, stream ended).
        """
        if self._capture is None:
            raise CameraError("Camera is not open; call open() first")

        success, frame = self._capture.read()
        if not success or frame is None:
            raise CameraError(
                f"Failed to read a frame from camera {self._index}",
                details={"camera_index": self._index},
            )

        if self._mirror:
            frame = cv2.flip(frame, 1)
        return frame

    def stream(self) -> Iterator[np.ndarray]:
        """Yield frames continuously until the camera fails or is closed.

        Yields:
            Successive BGR frames.

        Raises:
            CameraError: Propagated from :meth:`read`.
        """
        while self.is_open:
            yield self.read()

    def capture_burst(
        self,
        count: int,
        interval: float = 1.0,
        on_capture: CaptureCallback | None = None,
    ) -> list[np.ndarray]:
        """Capture ``count`` frames spaced ``interval`` seconds apart.

        Used by the enrolment CLI to gather several poses of the same student.
        The stream keeps being read between captures so the returned frames are
        current rather than stale buffered ones.

        Args:
            count: Number of frames to keep.
            interval: Seconds to wait between kept frames.
            on_capture: Optional callback invoked as ``(index, total, frame)``
                after each capture — used to drive the on-screen preview.

        Returns:
            The captured frames, in order.

        Raises:
            CameraError: Propagated from :meth:`read`.
        """
        frames: list[np.ndarray] = []
        for position in range(count):
            if position > 0 and interval > 0:
                deadline = time.monotonic() + interval
                while time.monotonic() < deadline:
                    # Drain the buffer so the next kept frame is fresh.
                    self.read()

            frame = self.read()
            frames.append(frame)
            logger.debug("Captured frame %s/%s", position + 1, count)
            if on_capture is not None:
                on_capture(position + 1, count, frame)
        return frames
