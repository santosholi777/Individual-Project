"""Command line evaluation of the face recognition system.

Measures accuracy against a labelled dataset and writes a report you can put
straight into the project write-up.

Dataset layout — one folder per student, named by their id::

    dataset/
    ├── CS2021001/
    │   ├── 01.jpg  02.jpg  03.jpg     # first N enrol, rest become probes
    │   ├── dim_light/  *.jpg          # optional: condition sub-folders
    │   └── side_angle/ *.jpg
    ├── CS2021002/
    │   └── ...
    └── meta.json                      # optional: names and fairness groups

Usage::

    # Basic run
    python evaluate.py --dataset ./dataset

    # Full run: hold two people out as strangers, write a report and figures
    python evaluate.py --dataset ./dataset \\
        --holdout CS2021005,CS2021006 \\
        --report evaluation_report.md

    # Sweep the threshold to see the trade-off
    python evaluate.py --dataset ./dataset --threshold 0.6

The evaluation never touches the live student registry or attendance log — the
gallery is built in memory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import get_settings
from dependencies import get_container
from evaluation.dataset import load_dataset
from evaluation.figures import write_figures
from evaluation.metrics import (
    collect_scores,
    evaluate_threshold,
    recommend_threshold,
    slice_by,
    sweep_thresholds,
)
from evaluation.report import (
    EvaluationResults,
    print_console,
    write_json,
    write_markdown,
    write_scores_csv,
)
from evaluation.runner import EvaluationRunner
from exceptions import DeepVisionAttendError
from logging_config import get_logger, setup_logging

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Define the command line interface."""
    parser = argparse.ArgumentParser(
        prog="evaluate.py",
        description="Measure DeepVisionAttend's recognition accuracy on a labelled dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Dataset layout: one folder per student id, images inside. "
            "Optional condition sub-folders (dim_light/, side_angle/, mask/) "
            "let the report show where the system breaks."
        ),
    )
    parser.add_argument(
        "--dataset", type=Path, required=True, help="Dataset directory"
    )
    parser.add_argument(
        "--enrol-per-identity",
        type=int,
        default=3,
        help="Images to enrol per person; the rest become probes",
    )
    parser.add_argument(
        "--holdout",
        default="",
        help=(
            "Comma-separated student ids to exclude from the gallery and treat "
            "as strangers. This is what measures proxy-attendance resistance."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold to report against (default: the configured recognition threshold)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Write a Markdown report here (figures go alongside it)",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="Write the full results as JSON"
    )
    parser.add_argument(
        "--csv", type=Path, default=None, help="Write raw per-probe scores as CSV"
    )
    parser.add_argument(
        "--no-figures", action="store_true", help="Skip the SVG figures"
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (INFO shows per-image progress)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python evaluate.py``.

    Args:
        argv: Command line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code: ``0`` success, ``1`` failure, ``130`` cancelled.
    """
    args = build_parser().parse_args(argv)
    settings = get_settings()
    setup_logging(
        level=args.log_level, log_dir=settings.logs_dir, log_to_file=settings.log_to_file
    )

    threshold = (
        args.threshold if args.threshold is not None else settings.recognition_threshold
    )
    holdout = tuple(
        item.strip() for item in args.holdout.split(",") if item.strip()
    )

    print("=" * 68)
    print("  DeepVisionAttend — Evaluation")
    print("=" * 68)

    try:
        dataset = load_dataset(
            root=args.dataset,
            enrol_per_identity=args.enrol_per_identity,
            holdout=holdout,
        )
        summary = dataset.summary()
        print(f"  Identities   : {summary['identities']} enrolled, "
              f"{summary['unknown_identities']} held out")
        print(f"  Images       : {summary['enrol_images']} enrol, "
              f"{summary['probe_images']} probe, "
              f"{summary['unknown_probe_images']} stranger")
        print(f"  Conditions   : {', '.join(summary['conditions'])}")
        print(f"  Threshold    : {threshold}")

        print("\n  Loading the pre-trained models…")
        container = get_container()
        container.warm_up()

        print("  Embedding and scoring… (this takes a moment on CPU)\n")
        runner = EvaluationRunner(
            settings=settings, detector=container.detector, embedder=container.embedder
        )
        outcomes, enrolment, _ = runner.run(dataset)

        for failure in enrolment.failures:
            logger.warning("Enrolment failure: %s", failure)

        verification = collect_scores(outcomes)
        recommended_threshold = recommend_threshold(outcomes)
        groups = {
            identity.student_id: identity.group
            for identity in dataset.identities
            if identity.group
        }

        results = EvaluationResults(
            dataset=dataset,
            outcomes=outcomes,
            verification=verification,
            configured=evaluate_threshold(outcomes, threshold),
            recommended=evaluate_threshold(outcomes, recommended_threshold),
            sweep=sweep_thresholds(outcomes),
            by_condition=slice_by(outcomes, "condition", threshold),
            by_identity=slice_by(outcomes, "identity", threshold),
            by_group=slice_by(outcomes, "group", threshold, labels=groups) if groups else [],
            enrolment=enrolment,
            settings=settings,
        )

        print_console(results)

        written: list[Path] = []
        if args.report:
            write_markdown(args.report, results)
            written.append(args.report)
            if not args.no_figures:
                written.extend(
                    write_figures(
                        args.report.parent / "figures",
                        verification,
                        results.sweep,
                        threshold,
                    )
                )
        if args.json:
            write_json(args.json, results)
            written.append(args.json)
        if args.csv:
            write_scores_csv(args.csv, outcomes)
            written.append(args.csv)

        if written:
            print("  Written:")
            for path in written:
                print(f"    {path}")
            print()

        return 0

    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except DeepVisionAttendError as exc:
        print(f"\nEvaluation failed: {exc.message}", file=sys.stderr)
        if exc.details:
            print(f"Details: {exc.details}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - top-level CLI guard
        print(f"\nUnexpected error: {exc}", file=sys.stderr)
        logger.exception("Unexpected error during evaluation")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
