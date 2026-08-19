"""Image decoding, quality assessment and annotation helpers.

Every image inside this service is a BGR ``numpy.ndarray`` — the layout OpenCV
and InsightFace both expect. Conversions from HTTP uploads, base64 payloads and
files all happen here so no other module has to think about encodings.
"""

from __future__ import annotations

import base64
import binascii
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from domain import BoundingBox
from exceptions import ImageDecodeError
from logging_config import get_logger

logger = get_logger(__name__)

#: File extensions accepted when enrolling students from a folder of images.
SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

_GREEN: Final[tuple[int, int, int]] = (0, 200, 0)
_RED: Final[tuple[int, int, int]] = (0, 0, 255)
_WHITE: Final[tuple[int, int, int]] = (255, 255, 255)


def decode_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes (JPEG/PNG/…) into a BGR array.

    Args:
        data: Encoded image bytes, typically an uploaded file's body.

    Returns:
        The decoded image with shape ``(h, w, 3)``.

    Raises:
        ImageDecodeError: If the payload is empty or is not a readable image.
    """
    if not data:
        raise ImageDecodeError("Received an empty image payload")

    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ImageDecodeError(
            "Could not decode the image. Supported formats: JPEG, PNG, BMP, WEBP",
            details={"bytes_received": len(data)},
        )
    return image


def decode_base64_image(payload: str) -> np.ndarray:
    """Decode a base64 image, with or without a data-URI prefix.

    Accepts the exact string a browser produces via ``canvas.toDataURL()``,
    which is how the React frontend will send webcam frames.

    Args:
        payload: Base64 text, optionally prefixed with ``data:image/...;base64,``.

    Returns:
        The decoded image with shape ``(h, w, 3)``.

    Raises:
        ImageDecodeError: If the payload is empty, not valid base64, or not a
            readable image.
    """
    if not payload or not payload.strip():
        raise ImageDecodeError("Received an empty base64 image payload")

    encoded = payload.strip()
    if encoded.startswith("data:"):
        _, _, encoded = encoded.partition(",")

    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ImageDecodeError(f"Invalid base64 image payload: {exc}") from exc

    return decode_image_bytes(raw)


def encode_image_to_base64(image: np.ndarray, extension: str = ".jpg") -> str:
    """Encode a BGR image into a base64 data URI.

    Args:
        image: BGR image array.
        extension: Target container, ``".jpg"`` or ``".png"``.

    Returns:
        A ``data:image/...;base64,...`` string ready to drop into an ``<img>``.

    Raises:
        ImageDecodeError: If OpenCV cannot encode the array.
    """
    success, buffer = cv2.imencode(extension, image)
    if not success:
        raise ImageDecodeError(f"Failed to encode image as '{extension}'")

    mime = "jpeg" if extension in {".jpg", ".jpeg"} else extension.lstrip(".")
    return f"data:image/{mime};base64,{base64.b64encode(buffer).decode('ascii')}"


def load_image(path: str | Path) -> np.ndarray:
    """Read an image from disk into a BGR array.

    Args:
        path: Path to the image file.

    Returns:
        The decoded image with shape ``(h, w, 3)``.

    Raises:
        ImageDecodeError: If the file is missing or unreadable as an image.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ImageDecodeError(f"Image file not found: {file_path}")

    # Read via bytes rather than cv2.imread so that non-ASCII paths work.
    try:
        data = file_path.read_bytes()
    except OSError as exc:
        raise ImageDecodeError(f"Could not read image file {file_path}: {exc}") from exc
    return decode_image_bytes(data)


def list_image_files(directory: str | Path) -> list[Path]:
    """List supported image files inside a directory, sorted by name.

    Args:
        directory: Folder to scan (non-recursive).

    Returns:
        Sorted paths of every supported image found.

    Raises:
        ImageDecodeError: If the directory does not exist.
    """
    folder = Path(directory)
    if not folder.is_dir():
        raise ImageDecodeError(f"Not a directory: {folder}")

    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def crop(image: np.ndarray, bbox: BoundingBox, margin: float = 0.0) -> np.ndarray:
    """Crop a bounding box out of an image, clamped to its bounds.

    Args:
        image: Source BGR image.
        bbox: Region to extract.
        margin: Fraction of the box size to add on each side, e.g. ``0.2``.

    Returns:
        The cropped region. May be empty if the box falls outside the image.
    """
    height, width = image.shape[:2]
    pad_x = bbox.width * margin
    pad_y = bbox.height * margin

    x1 = int(max(0, bbox.x1 - pad_x))
    y1 = int(max(0, bbox.y1 - pad_y))
    x2 = int(min(width, bbox.x2 + pad_x))
    y2 = int(min(height, bbox.y2 + pad_y))
    return image[y1:y2, x1:x2]


def blur_score(image: np.ndarray) -> float:
    """Estimate sharpness as the variance of the Laplacian.

    Higher is sharper. A motion-blurred or out-of-focus classroom capture
    typically scores well below 50, which is the basis of the enrolment quality
    gate (see :data:`config.Settings.blur_threshold`).

    Args:
        image: BGR or grayscale image.

    Returns:
        The Laplacian variance; ``0.0`` for an empty image.
    """
    if image.size == 0:
        return 0.0
    gray = to_grayscale(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness_score(image: np.ndarray) -> float:
    """Return mean pixel intensity in ``[0, 255]``.

    Used to reject captures that are too dark or blown out to enrol.

    Args:
        image: BGR or grayscale image.

    Returns:
        Mean intensity; ``0.0`` for an empty image.
    """
    if image.size == 0:
        return 0.0
    return float(np.mean(to_grayscale(image)))


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to grayscale, passing through 2-D input unchanged."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize_max_side(image: np.ndarray, max_side: int = 1920) -> np.ndarray:
    """Downscale an image so its longest side is at most ``max_side``.

    Large phone uploads are shrunk before detection to bound latency; images
    already within the limit are returned untouched (aspect ratio is preserved
    either way).

    Args:
        image: Source BGR image.
        max_side: Maximum allowed length of the longest side, in pixels.

    Returns:
        The original or a downscaled copy.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image

    scale = max_side / float(longest)
    new_size = (int(width * scale), int(height * scale))
    logger.debug("Resizing image from %sx%s to %sx%s", width, height, *new_size)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def draw_face_box(
    image: np.ndarray,
    bbox: BoundingBox,
    label: str,
    *,
    recognized: bool = True,
    confidence: float | None = None,
) -> np.ndarray:
    """Draw a labelled face box onto an image, in place.

    Green means recognised, red means unknown — the visual convention used by
    the live recognition CLI.

    Args:
        image: BGR image to annotate (modified in place).
        bbox: Box to draw.
        label: Text drawn above the box.
        recognized: Selects the box colour.
        confidence: Optional score appended to the label as a percentage.

    Returns:
        The same array, for convenient chaining.
    """
    colour = _GREEN if recognized else _RED
    x1, y1, x2, y2 = bbox.as_int_tuple()
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)

    text = label if confidence is None else f"{label} {confidence * 100:.1f}%"
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    )
    # Keep the caption on-screen when the face touches the top edge.
    top = max(y1 - text_height - baseline - 4, 0)
    cv2.rectangle(
        image, (x1, top), (x1 + text_width + 6, top + text_height + baseline + 4), colour, -1
    )
    cv2.putText(
        image,
        text,
        (x1 + 3, top + text_height + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        _WHITE,
        2,
        cv2.LINE_AA,
    )
    return image


def draw_hud(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw a translucent status panel in the top-left corner, in place.

    Args:
        image: BGR image to annotate (modified in place).
        lines: Text lines to display, top to bottom.

    Returns:
        The same array, for convenient chaining.
    """
    if not lines:
        return image

    panel_height = 10 + 22 * len(lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (430, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (10, 24 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            _WHITE,
            1,
            cv2.LINE_AA,
        )
    return image
