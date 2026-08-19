"""Command line live face recognition.

Opens the webcam, recognises registered students in real time and (optionally)
marks their attendance. This is the demonstrable classroom kiosk mode.

Usage::

    # Live recognition with an on-screen overlay
    python recognize.py

    # Kiosk mode: recognise and mark attendance for a session
    python recognize.py --mark --session lecture-1

    # Recognise a single image file instead of the camera
    python recognize.py --image ./test/group_photo.jpg

Press ``q`` or ESC to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2

from config import get_settings
from dependencies import get_container
from domain import AttendanceStatus, RecognitionResult
from exceptions import DeepVisionAttendError, NoFaceDetectedError
from logging_config import get_logger, setup_logging
from utils.camera import Camera
from utils.image_utils import draw_face_box, draw_hud, load_image

logger = get_logger(__name__)

_WINDOW = "DeepVisionAttend - Live Recognition"


def build_parser() -> argparse.ArgumentParser:
    """Define the command line interface."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        prog="recognize.py",
        description="Run live face recognition and optionally mark attendance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mark", action="store_true", help="Mark attendance for recognised students"
    )
    parser.add_argument(
        "--session",
        default=settings.default_session,
        help="Session label recorded with each attendance entry",
    )
    parser.add_argument(
        "--camera-index", type=int, default=settings.camera_index, help="Webcam device index"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Recognise this image file and exit instead of opening the camera",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=settings.recognition_frame_interval,
        help="Run recognition on every Nth frame (higher = faster preview)",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=None,
        help="Process at most this many faces per frame, largest first",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Run without a window; log recognitions to the console only",
    )
    parser.add_argument(
        "--log-level",
        default=settings.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser


def recognize_image_file(path: Path, mark: bool, session: str) -> int:
    """Recognise every face in a single image file and print the results.

    Args:
        path: Image to process.
        mark: Whether to mark attendance for recognised students.
        session: Session label for attendance.

    Returns:
        A process exit code: ``0`` if at least one face was recognised.
    """
    container = get_container()
    container.warm_up()

    image = load_image(path)
    service = container.recognition_service
    report = (
        service.recognize_and_mark(image, session=session)
        if mark
        else service.recognize(image)
    )

    print(f"\nFaces detected: {report.faces_detected} ({report.elapsed_ms:.0f}ms)")
    for index, result in enumerate(report.results, start=1):
        if result.recognized:
            print(
                f"  {index}. {result.name} ({result.student_id}) "
                f"confidence={result.confidence:.3f} margin={result.margin:.3f}"
            )
        else:
            best = result.candidates[0] if result.candidates else None
            hint = (
                f" closest: {best.name} @ {best.similarity:.3f}" if best else ""
            )
            print(f"  {index}. Unknown{hint}")

    for outcome in report.attendance:
        if outcome.record is not None:
            print(f"  Attendance [{outcome.status.value}]: {outcome.record.name}")

    return 0 if report.recognized else 1


def run_live(args: argparse.Namespace) -> int:
    """Run the live webcam recognition loop.

    Recognition runs on every Nth frame while the preview stays at full frame
    rate: the last results are re-drawn on the frames in between, which keeps
    the video smooth on a CPU-only machine.

    Args:
        args: Parsed command line arguments.

    Returns:
        A process exit code.
    """
    settings = get_settings()
    container = get_container()

    print("Loading the pre-trained face recognition models...")
    container.warm_up()

    if container.matcher.is_empty:
        print(
            "\nNo students are registered yet. Run register.py first.", file=sys.stderr
        )
        return 2

    service = container.recognition_service
    attendance = container.attendance_service

    print(f"\nGallery: {container.matcher.student_count} student(s), "
          f"{container.matcher.size} embedding(s)")
    print(f"Attendance marking: {'ON (session=' + args.session + ')' if args.mark else 'OFF'}")
    print("Press 'q' or ESC to quit.\n")

    results: list[RecognitionResult] = []
    marked_this_run: set[str] = set()
    frame_times: deque[float] = deque(maxlen=30)
    frame_number = 0

    with Camera(
        index=args.camera_index,
        width=settings.camera_width,
        height=settings.camera_height,
        warmup_frames=settings.camera_warmup_frames,
        mirror=True,
    ) as camera:
        for frame in camera.stream():
            started = time.perf_counter()
            frame_number += 1

            if frame_number % max(args.frame_interval, 1) == 0:
                try:
                    report = service.recognize(
                        frame, max_faces=args.max_faces, require_gallery=False
                    )
                    results = report.results

                    if args.mark:
                        for outcome in attendance.mark_many(results, session=args.session):
                            if outcome.status is AttendanceStatus.MARKED and outcome.record:
                                marked_this_run.add(outcome.record.student_id)
                                print(
                                    f"  [MARKED] {outcome.record.name} "
                                    f"({outcome.record.student_id}) "
                                    f"confidence={outcome.record.confidence:.3f}"
                                )
                except NoFaceDetectedError:
                    results = []
                except DeepVisionAttendError as exc:
                    logger.warning("Recognition error on frame %s: %s", frame_number, exc.message)

            frame_times.append(time.perf_counter() - started)

            if args.no_preview:
                continue

            for result in results:
                draw_face_box(
                    frame,
                    result.face.bbox,
                    result.label,
                    recognized=result.recognized,
                    confidence=result.confidence if result.recognized else None,
                )

            fps = len(frame_times) / max(sum(frame_times), 1e-6)
            draw_hud(
                frame,
                [
                    f"FPS: {fps:4.1f}   Faces: {len(results)}",
                    f"Gallery: {container.matcher.student_count} students",
                    f"Session: {args.session}   Marked: {len(marked_this_run)}"
                    if args.mark
                    else "Attendance marking: OFF",
                    "Press 'q' or ESC to quit",
                ],
            )
            cv2.imshow(_WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    cv2.destroyAllWindows()
    if args.mark:
        print(f"\nSession complete. {len(marked_this_run)} student(s) marked present.")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python recognize.py``.

    Args:
        argv: Command line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code: ``0`` success, ``1`` failure, ``130`` cancelled.
    """
    args = build_parser().parse_args(argv)
    settings = get_settings()
    setup_logging(level=args.log_level, log_dir=settings.logs_dir, log_to_file=settings.log_to_file)

    print("=" * 62)
    print("  DeepVisionAttend - Face Recognition")
    print("=" * 62)

    try:
        if args.image is not None:
            return recognize_image_file(args.image, mark=args.mark, session=args.session)
        return run_live(args)
    except KeyboardInterrupt:
        print("\nStopped.", file=sys.stderr)
        return 130
    except DeepVisionAttendError as exc:
        print(f"\nError: {exc.message}", file=sys.stderr)
        logger.error("Recognition failed: %s", exc.message)
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"\nUnexpected error: {exc}", file=sys.stderr)
        logger.exception("Unexpected error during recognition")
        return 1
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
