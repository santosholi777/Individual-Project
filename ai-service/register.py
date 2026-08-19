"""Command line student enrolment.

Captures face images from the webcam (or reads a folder of photos), embeds them
with the pre-trained ArcFace model and stores the vectors locally.

Usage::

    # Capture 5 frames from the webcam with a live preview
    python register.py --student-id CS2021001 --name "Aditi Sharma"

    # Enrol from existing photographs instead of the camera
    python register.py --student-id CS2021001 --name "Aditi Sharma" \\
        --from-folder ./photos/aditi

    # Replace an existing enrolment
    python register.py --student-id CS2021001 --name "Aditi Sharma" --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from config import get_settings
from dependencies import get_container
from exceptions import DeepVisionAttendError
from logging_config import get_logger, setup_logging
from utils.camera import Camera
from utils.image_utils import draw_hud, list_image_files, load_image

logger = get_logger(__name__)

_WINDOW = "DeepVisionAttend - Registration"


def build_parser() -> argparse.ArgumentParser:
    """Define the command line interface."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="register.py",
        description="Register a student's face with DeepVisionAttend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--student-id", required=True, help="Unique student identifier (roll number)"
    )
    parser.add_argument("--name", required=True, help="Student's full name")
    parser.add_argument(
        "--captures",
        type=int,
        default=settings.registration_captures,
        help="How many webcam frames to capture",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=settings.registration_capture_interval,
        help="Seconds between captures (change pose between them)",
    )
    parser.add_argument(
        "--camera-index", type=int, default=settings.camera_index, help="Webcam device index"
    )
    parser.add_argument(
        "--from-folder",
        type=Path,
        default=None,
        help="Read images from this folder instead of using the webcam",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the student's existing enrolment if one exists",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Capture without showing a preview window (for headless machines)",
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser


def capture_from_webcam(
    camera_index: int, captures: int, interval: float, show_preview: bool
) -> list[np.ndarray]:
    """Capture enrolment frames from the webcam.

    Shows a live preview with a countdown and flashes the frame border on each
    capture, so the student knows when to change pose.

    Args:
        camera_index: OpenCV device index.
        captures: Number of frames to keep.
        interval: Seconds between captures.
        show_preview: Whether to display the preview window.

    Returns:
        The captured BGR frames.

    Raises:
        CameraError: If the camera cannot be opened or read.
    """
    settings = get_settings()
    frames: list[np.ndarray] = []

    with Camera(
        index=camera_index,
        width=settings.camera_width,
        height=settings.camera_height,
        warmup_frames=settings.camera_warmup_frames,
        mirror=True,
    ) as camera:
        if show_preview:
            print("\nLook at the camera. Change your pose slightly between captures.")
            print("Press ESC at any time to abort.\n")
            _countdown(camera, seconds=3)

        def on_capture(index: int, total: int, frame: np.ndarray) -> None:
            """Show a confirmation flash after each captured frame."""
            print(f"  Captured {index}/{total}")
            if not show_preview:
                return
            flash = frame.copy()
            cv2.rectangle(
                flash, (0, 0), (flash.shape[1] - 1, flash.shape[0] - 1), (0, 255, 0), 12
            )
            draw_hud(flash, [f"Captured {index}/{total}", "Change your pose"])
            cv2.imshow(_WINDOW, flash)
            cv2.waitKey(220)

        frames = camera.capture_burst(
            count=captures, interval=interval, on_capture=on_capture
        )

    if show_preview:
        cv2.destroyAllWindows()
    return frames


def _countdown(camera: Camera, seconds: int) -> None:
    """Show a preview countdown before the first capture.

    Args:
        camera: An open camera.
        seconds: Countdown length.

    Raises:
        KeyboardInterrupt: If the user presses ESC to abort.
    """
    fps = 25
    for remaining in range(seconds, 0, -1):
        for _ in range(fps):
            frame = camera.read()
            draw_hud(
                frame,
                [
                    f"Starting in {remaining}...",
                    "Face the camera, keep your face centred",
                    "ESC to cancel",
                ],
            )
            cv2.imshow(_WINDOW, frame)
            if cv2.waitKey(1000 // fps) & 0xFF == 27:
                raise KeyboardInterrupt("Registration cancelled by the user")


def load_from_folder(folder: Path) -> list[np.ndarray]:
    """Load every supported image in a folder.

    Args:
        folder: Directory of face photographs.

    Returns:
        The decoded BGR images.

    Raises:
        SystemExit: If the folder holds no supported images.
    """
    paths = list_image_files(folder)
    if not paths:
        print(f"No supported images found in {folder}", file=sys.stderr)
        raise SystemExit(2)

    print(f"Loading {len(paths)} image(s) from {folder}")
    return [load_image(path) for path in paths]


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python register.py``.

    Args:
        argv: Command line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code: ``0`` success, ``1`` failure, ``130`` cancelled.
    """
    args = build_parser().parse_args(argv)
    settings = get_settings()
    setup_logging(level=args.log_level, log_dir=settings.logs_dir, log_to_file=settings.log_to_file)

    print("=" * 62)
    print("  DeepVisionAttend - Student Registration")
    print("=" * 62)
    print(f"  Student ID : {args.student_id}")
    print(f"  Name       : {args.name}")
    print(f"  Source     : {args.from_folder or f'webcam #{args.camera_index}'}")
    print("=" * 62)

    try:
        print("\nLoading the pre-trained face recognition models...")
        container = get_container()
        container.warm_up()

        images = (
            load_from_folder(args.from_folder)
            if args.from_folder
            else capture_from_webcam(
                camera_index=args.camera_index,
                captures=args.captures,
                interval=args.interval,
                show_preview=not args.no_preview,
            )
        )

        print(f"\nProcessing {len(images)} image(s)...")
        result = container.registration_service.register(
            student_id=args.student_id,
            name=args.name,
            images=images,
            overwrite=args.overwrite,
        )

        print("\n" + "=" * 62)
        print("  REGISTRATION SUCCESSFUL")
        print("=" * 62)
        print(f"  Student          : {result.student.name} ({result.student.student_id})")
        print(f"  Images accepted  : {result.accepted_images}")
        print(f"  Images rejected  : {result.rejected_images}")
        print(f"  Embeddings stored: {result.total_embeddings}")
        if result.rejections:
            print("\n  Rejected images:")
            for reason in result.rejections:
                print(f"    - {reason}")
        print("=" * 62)
        return 0

    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except DeepVisionAttendError as exc:
        print(f"\nRegistration failed: {exc.message}", file=sys.stderr)
        if exc.details:
            print(f"Details: {exc.details}", file=sys.stderr)
        logger.error("Registration failed: %s", exc.message)
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"\nUnexpected error: {exc}", file=sys.stderr)
        logger.exception("Unexpected error during registration")
        return 1
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
