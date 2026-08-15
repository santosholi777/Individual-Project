"""Rendering evaluation results — console, Markdown, JSON and CSV.

The Markdown report is written to be pasted into the project write-up: it leads
with the numbers an examiner looks for, and it states its own limitations rather
than leaving the reader to find them.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from config import Settings
from evaluation.dataset import Dataset
from evaluation.metrics import (
    GroupResult,
    ProbeOutcome,
    ThresholdReport,
    VerificationMetrics,
)
from evaluation.runner import EnrolmentReport

#: Below this many probes, rates are too noisy to state as findings.
SMALL_SAMPLE = 30
#: Below this many identities per group, fairness comparison is not meaningful.
SMALL_GROUP = 5


@dataclass(slots=True)
class EvaluationResults:
    """Everything one evaluation run produced."""

    dataset: Dataset
    outcomes: list[ProbeOutcome]
    verification: VerificationMetrics
    configured: ThresholdReport
    recommended: ThresholdReport
    sweep: list[ThresholdReport]
    by_condition: list[GroupResult]
    by_identity: list[GroupResult]
    by_group: list[GroupResult]
    enrolment: EnrolmentReport
    settings: Settings

    @property
    def rank1(self) -> float:
        """Rank-1 identification accuracy, ignoring any threshold.

        "Is the right person the closest match?" — the model's ceiling, before
        the threshold rejects anything.
        """
        known = [o for o in self.outcomes if not o.is_unknown and o.failure is None]
        if not known:
            return 0.0
        return sum(1 for o in known if o.rank1_correct) / len(known)

    @property
    def pipeline_failures(self) -> int:
        """Probes where no usable face could be extracted at all."""
        return sum(1 for o in self.outcomes if o.failure is not None)

    def to_dict(self) -> dict[str, object]:
        """Serialise the whole run for the JSON report."""
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": {
                "pack": self.settings.model_pack,
                "recognition_model": self.settings.recognition_model_file,
                "det_size": list(self.settings.det_size),
                "device": "cuda" if self.settings.ctx_id >= 0 else "cpu",
            },
            "dataset": self.dataset.summary(),
            "rank1_accuracy": round(self.rank1, 4),
            "pipeline_failures": self.pipeline_failures,
            "verification": self.verification.to_dict(),
            "configured_threshold": self.configured.to_dict(),
            "recommended_threshold": self.recommended.to_dict(),
            "threshold_sweep": [report.to_dict() for report in self.sweep],
            "by_condition": [result.to_dict() for result in self.by_condition],
            "by_identity": [result.to_dict() for result in self.by_identity],
            "by_group": [result.to_dict() for result in self.by_group],
            "enrolment_failures": self.enrolment.failures,
        }


def _pct(value: float) -> str:
    """Format a 0–1 rate as a percentage string."""
    return f"{value * 100:.1f}%"


# ======================================================================
# Console
# ======================================================================
def print_console(results: EvaluationResults) -> None:
    """Print a human-readable summary to stdout."""
    line = "=" * 68
    verification = results.verification
    eer, eer_threshold = verification.eer

    print("\n" + line)
    print("  DeepVisionAttend — Evaluation Results")
    print(line)

    summary = results.dataset.summary()
    print(f"  Dataset            : {summary['root']}")
    print(
        f"  Identities         : {summary['identities']} enrolled"
        + (
            f", {summary['unknown_identities']} held out as strangers"
            if summary["unknown_identities"]
            else ""
        )
    )
    print(f"  Images             : {summary['enrol_images']} enrol, {summary['probe_images']} probe"
          + (f", {summary['unknown_probe_images']} stranger" if summary["unknown_probe_images"] else ""))
    if results.pipeline_failures:
        print(f"  Pipeline failures  : {results.pipeline_failures} (no face found)")

    print("\n  " + "-" * 64)
    print("  VERIFICATION (threshold-free: how separable are the classes?)")
    print("  " + "-" * 64)
    print(f"  Genuine mean       : {verification.genuine_mean:.4f}  (same person)")
    print(f"  Impostor mean      : {verification.impostor_mean:.4f}  (different people)")
    print(f"  Separation gap     : {verification.separation:+.4f}  "
          f"(worst genuine - best impostor)")
    print(f"  d-prime            : {verification.d_prime:.2f}")
    print(f"  ROC AUC            : {verification.auc:.5f}")
    print(f"  Equal error rate   : {_pct(eer)} at threshold {eer_threshold:.3f}")
    tar1, thr1 = verification.tar_at_far(0.01)
    print(f"  TAR @ FAR=1%       : {_pct(tar1)} at threshold {thr1:.3f}")

    print("\n  " + "-" * 64)
    print("  IDENTIFICATION (what the attendance system actually does)")
    print("  " + "-" * 64)
    print(f"  Rank-1 accuracy    : {_pct(results.rank1)}  (closest match is correct)")

    for title, report in (
        (f"At the configured threshold ({results.configured.threshold:.2f})", results.configured),
        (f"At the recommended threshold ({results.recommended.threshold:.2f})", results.recommended),
    ):
        print(f"\n  {title}:")
        print(f"    Marked correctly : {report.correct:>4} / {report.total}  ({_pct(report.correct_rate)})")
        print(f"    WRONG student    : {report.wrong_id:>4} / {report.total}  ({_pct(report.wrong_id_rate)})")
        print(f"    Not marked       : {report.rejected:>4} / {report.total}  ({_pct(report.rejection_rate)})")
        if report.unknown_total:
            print(f"    Strangers accepted: {report.unknown_accepted:>3} / {report.unknown_total}  "
                  f"({_pct(report.unknown_accept_rate)})")

    if results.by_condition and len(results.by_condition) > 1:
        print("\n  " + "-" * 64)
        print("  BY CONDITION (worst first)")
        print("  " + "-" * 64)
        print(f"  {'CONDITION':<22} {'PROBES':>7} {'CORRECT':>9} {'WRONG':>7} {'MEAN SIM':>9}")
        for result in results.by_condition:
            print(f"  {result.label[:21]:<22} {result.total:>7} "
                  f"{_pct(result.correct_rate):>9} {result.wrong_id:>7} "
                  f"{result.mean_genuine:>9.3f}")

    if results.by_group and len(results.by_group) > 1:
        print("\n  " + "-" * 64)
        print("  BY GROUP (fairness — read the caveat in the report)")
        print("  " + "-" * 64)
        print(f"  {'GROUP':<22} {'PROBES':>7} {'CORRECT':>9} {'MEAN SIM':>9}")
        for result in results.by_group:
            print(f"  {result.label[:21]:<22} {result.total:>7} "
                  f"{_pct(result.correct_rate):>9} {result.mean_genuine:>9.3f}")

    worst = [r for r in results.by_identity if r.correct_rate < 1.0][:5]
    if worst:
        print("\n  " + "-" * 64)
        print("  IDENTITIES WITH ERRORS")
        print("  " + "-" * 64)
        for result in worst:
            print(f"  {result.label[:21]:<22} {result.total:>7} probes  "
                  f"{_pct(result.correct_rate):>7} correct  {result.wrong_id} wrong")

    print("\n" + line)
    _print_verdict(results)
    print(line + "\n")


def _print_verdict(results: EvaluationResults) -> None:
    """Print the honest headline, including whether the sample supports it."""
    total = results.configured.total
    verification = results.verification

    if total < SMALL_SAMPLE:
        print(f"  NOTE: only {total} probes. These rates are indicative, not")
        print("  conclusive — one error moves them by several points. Collect")
        print(f"  at least {SMALL_SAMPLE} probes before quoting them as findings.")
        return

    if results.configured.wrong_id_rate > 0:
        print(f"  WARNING: {results.configured.wrong_id} probe(s) were attributed to the")
        print("  WRONG student. Raise the threshold — a wrong name on the")
        print("  register is worse than asking a student to retry.")
    elif verification.separation > 0:
        print("  Genuine and impostor scores do not overlap: the threshold sits")
        print(f"  inside a gap of {verification.separation:.3f}, so its exact value is not")
        print("  delicate on this dataset.")
    else:
        print("  Genuine and impostor scores overlap. Every threshold trades a")
        print("  wrong ID against a rejection; see the trade-off figure.")


# ======================================================================
# Markdown
# ======================================================================
def render_markdown(results: EvaluationResults) -> str:
    """Render the full report as Markdown for the project write-up."""
    verification = results.verification
    eer, eer_threshold = verification.eer
    tar1, thr1 = verification.tar_at_far(0.01)
    summary = results.dataset.summary()
    configured = results.configured
    recommended = results.recommended
    total = configured.total

    lines: list[str] = [
        "# DeepVisionAttend — Evaluation Report",
        "",
        f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*",
        "",
        "## 1. Method",
        "",
        "Each identity's enrolment images were embedded into a gallery using the "
        "pre-trained **ArcFace (R50)** recogniser from the InsightFace "
        "`buffalo_l` pack, with **SCRFD** for detection. Every probe image was "
        "then embedded and compared against the gallery by cosine similarity. "
        "A student's score is the **maximum** similarity across their enrolment "
        "images, which is exactly how the deployed matcher behaves.",
        "",
        "No model was trained or fine-tuned: this measures a pre-trained model "
        "applied to this dataset.",
        "",
        "| Setting | Value |",
        "|---|---|",
        f"| Model pack | `{results.settings.model_pack}` |",
        f"| Recogniser | `{results.settings.recognition_model_file}` |",
        f"| Detector input | {results.settings.det_size[0]}×{results.settings.det_size[1]} |",
        f"| Device | {'CUDA' if results.settings.ctx_id >= 0 else 'CPU'} |",
        f"| Configured threshold | {results.settings.recognition_threshold} |",
        "",
        "### Dataset",
        "",
        "| | Count |",
        "|---|---|",
        f"| Enrolled identities | {summary['identities']} |",
        f"| Enrolment images | {summary['enrol_images']} |",
        f"| Probe images | {summary['probe_images']} |",
        f"| Held-out strangers | {summary['unknown_identities']} |",
        f"| Stranger probe images | {summary['unknown_probe_images']} |",
        f"| Conditions tested | {', '.join(summary['conditions']) or 'none'} |",
        f"| Pipeline failures | {results.pipeline_failures} |",
        "",
    ]

    if total < SMALL_SAMPLE:
        lines += [
            "> **Sample-size caveat.** This run used "
            f"{total} probe images across {summary['identities']} people. That is "
            "too few to support a precise accuracy claim — a single error moves "
            "every rate by several percentage points. Treat the numbers below as "
            "indicative, and collect more data before quoting them as findings.",
            "",
        ]

    lines += [
        "## 2. Headline results",
        "",
        "| Metric | Value | What it means |",
        "|---|---|---|",
        f"| **Rank-1 accuracy** | **{_pct(results.rank1)}** | The closest match in the gallery is the right person |",
        f"| ROC AUC | {verification.auc:.4f} | Probability a genuine pair outscores an impostor pair |",
        f"| Equal error rate | {_pct(eer)} | Where false accepts and false rejects balance (at threshold {eer_threshold:.3f}) |",
        f"| TAR @ FAR=1% | {_pct(tar1)} | Students recognised if at most 1% impostor acceptance is allowed |",
        f"| Separation gap | {verification.separation:+.3f} | Worst genuine score minus best impostor score |",
        f"| d′ | {verification.d_prime:.2f} | Class separation in pooled standard deviations |",
        "",
        "### Score distributions",
        "",
        "| | Mean | Std | Extreme |",
        "|---|---|---|---|",
        f"| Same person (genuine) | {verification.genuine_mean:.4f} | "
        f"{verification.to_dict()['genuine_std']} | min {verification.to_dict()['genuine_min']} |",
        f"| Different people (impostor) | {verification.impostor_mean:.4f} | "
        f"{verification.to_dict()['impostor_std']} | max {verification.to_dict()['impostor_max']} |",
        "",
        "![Score distribution](figures/score_distribution.svg)",
        "",
    ]

    if verification.separation > 0:
        lines += [
            f"The two classes **do not overlap** — the lowest genuine score sits "
            f"{verification.separation:.3f} above the highest impostor score. Any "
            "threshold inside that gap separates them perfectly on this dataset, "
            "so the exact value is not delicate.",
            "",
        ]
    else:
        lines += [
            "The two classes **overlap**: no threshold separates them perfectly on "
            "this dataset. Every choice trades a wrong identification against a "
            "rejection — see the trade-off below.",
            "",
        ]

    lines += [
        "## 3. Operational outcomes",
        "",
        "Rank-1 accuracy ignores the threshold. In deployment the threshold "
        "decides, and its three outcomes are not equally serious:",
        "",
        "* **Correct** — the right student is marked.",
        "* **Wrong ID** — a *different* enrolled student is marked. The worst "
        "case: it is silent, and it is proxy attendance by accident.",
        "* **Rejected** — nobody is marked; the student looks again. Annoying, "
        "not harmful.",
        "",
        f"| Outcome | At configured ({configured.threshold:.2f}) | At recommended ({recommended.threshold:.2f}) |",
        "|---|---|---|",
        f"| Marked correctly | {configured.correct}/{configured.total} ({_pct(configured.correct_rate)}) | "
        f"{recommended.correct}/{recommended.total} ({_pct(recommended.correct_rate)}) |",
        f"| **Wrong student** | {configured.wrong_id} ({_pct(configured.wrong_id_rate)}) | "
        f"{recommended.wrong_id} ({_pct(recommended.wrong_id_rate)}) |",
        f"| Not marked | {configured.rejected} ({_pct(configured.rejection_rate)}) | "
        f"{recommended.rejected} ({_pct(recommended.rejection_rate)}) |",
    ]

    if configured.unknown_total:
        lines.append(
            f"| Strangers accepted | {configured.unknown_accepted}/{configured.unknown_total} "
            f"({_pct(configured.unknown_accept_rate)}) | {recommended.unknown_accepted}/"
            f"{recommended.unknown_total} ({_pct(recommended.unknown_accept_rate)}) |"
        )

    lines += [
        "",
        "![Threshold trade-off](figures/threshold_tradeoff.svg)",
        "",
    ]

    if configured.unknown_total:
        lines += [
            "### Proxy-attendance resistance",
            "",
            f"{configured.unknown_total} probe images from "
            f"{summary['unknown_identities']} people who were **never enrolled** "
            "were presented to the system. Correct behaviour is to reject all of "
            "them.",
            "",
            f"**{configured.unknown_accepted} of {configured.unknown_total}** were "
            f"wrongly accepted as an enrolled student "
            f"({_pct(configured.unknown_accept_rate)} false-accept rate).",
            "",
        ]

    if results.by_condition and len(results.by_condition) > 1:
        lines += [
            "## 4. Performance by condition",
            "",
            "Worst-performing first — this is where the system breaks.",
            "",
            "| Condition | Probes | Correct | Wrong ID | Rejected | Mean genuine score |",
            "|---|---|---|---|---|---|",
        ]
        for result in results.by_condition:
            flag = " ⚠️" if result.total < 10 else ""
            lines.append(
                f"| {result.label}{flag} | {result.total} | {_pct(result.correct_rate)} | "
                f"{result.wrong_id} | {result.rejected} | {result.mean_genuine:.3f} |"
            )
        lines += [
            "",
            "⚠️ = fewer than 10 probes; that row is indicative only.",
            "",
        ]

    if results.by_group and len(results.by_group) > 1:
        lines += [
            "## 5. Fairness",
            "",
            "Accuracy broken down by the group labels supplied in `meta.json`.",
            "",
            "| Group | Probes | Correct | Wrong ID | Mean genuine score |",
            "|---|---|---|---|---|",
        ]
        for result in results.by_group:
            lines.append(
                f"| {result.label} | {result.total} | {_pct(result.correct_rate)} | "
                f"{result.wrong_id} | {result.mean_genuine:.3f} |"
            )

        spread = (
            max(r.correct_rate for r in results.by_group)
            - min(r.correct_rate for r in results.by_group)
        )
        lines += [
            "",
            f"Spread between the best and worst group: **{_pct(spread)}**.",
            "",
            "> **How much weight this carries.** A difference between groups here "
            "is not evidence of bias on its own, and the absence of one is not "
            "evidence of fairness. With this few people per group, the "
            "confidence interval on each rate is far wider than the gap between "
            "them, and group labels are self-assigned rather than a recognised "
            "protocol. Published audits (Buolamwini & Gebru, 2018; Raji & "
            "Buolamwini, 2019) use thousands of subjects sampled deliberately "
            "across skin tone and gender. This table is a sanity check that the "
            "question was asked, not an audit — report it as such.",
            "",
        ]

    lines += [
        "## 6. Limitations",
        "",
        f"1. **Sample size.** {summary['identities']} people and "
        f"{summary['probe_images']} probes. Small samples produce wide error bars.",
        "2. **The gallery is small.** False accepts get more likely as the "
        "gallery grows: more enrolled people means more chances for a stranger "
        "to resemble one of them. Results here will not transfer unchanged to a "
        "college-sized register.",
        "3. **The images are not a random sample** of a real classroom — they "
        "were collected deliberately, which tends to flatter the result.",
        "4. **No model was trained**, so nothing here measures the model's "
        "quality in general; it measures a pre-trained model on this data.",
    ]

    if not summary["conditions"] or summary["conditions"] == ["default"]:
        lines.append(
            "5. **Only one condition was tested.** Classroom reality includes dim "
            "light, side angles, motion blur, masks and occlusion. Without those, "
            "this measures the easy case — the exact criticism this project makes "
            "of prior work."
        )
    if not summary["unknown_identities"]:
        lines.append(
            "6. **No strangers were tested.** Without held-out identities, the "
            "false-accept rate against unenrolled people is unmeasured — and "
            "that is the number that decides whether proxy attendance is "
            "possible. Re-run with `--holdout`."
        )

    lines += [
        "",
        "## 7. Reproducing this",
        "",
        "```bash",
        f"python evaluate.py --dataset {summary['root']} \\",
        f"    --enrol-per-identity {len(results.dataset.identities[0].enrol_paths) if results.dataset.identities else 3} \\",
        "    --report evaluation_report.md",
        "```",
        "",
    ]
    return "\n".join(lines)


# ======================================================================
# Files
# ======================================================================
def write_json(path: Path, results: EvaluationResults) -> None:
    """Write the full results as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results.to_dict(), indent=2), encoding="utf-8")


def write_scores_csv(path: Path, outcomes: list[ProbeOutcome]) -> None:
    """Write the raw per-probe scores, so the numbers can be re-analysed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "path",
                "true_identity",
                "condition",
                "is_unknown",
                "predicted",
                "best_score",
                "genuine_score",
                "max_impostor_score",
                "rank1_correct",
                "failure",
            ]
        )
        for outcome in outcomes:
            writer.writerow(
                [
                    outcome.path,
                    outcome.identity,
                    outcome.condition,
                    outcome.is_unknown,
                    outcome.predicted or "",
                    f"{outcome.best_score:.6f}",
                    f"{outcome.genuine_score:.6f}" if outcome.genuine_score is not None else "",
                    f"{max(outcome.impostor_scores):.6f}" if outcome.impostor_scores else "",
                    outcome.rank1_correct,
                    outcome.failure or "",
                ]
            )


def write_markdown(path: Path, results: EvaluationResults) -> None:
    """Write the Markdown report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(results), encoding="utf-8")
