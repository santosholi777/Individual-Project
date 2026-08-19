"""Biometric evaluation metrics.

Two questions are answered separately, because they are genuinely different:

**Verification (1:1)** — "are these two faces the same person?" Measured over
every genuine and impostor score pair, independent of any threshold: ROC AUC,
EER, TAR at a fixed FAR. This describes the *model*.

**Open-set identification (1:N)** — "who is this, and is it anyone we know?"
This is what the attendance system actually does, and it needs a threshold. Its
three outcomes are not equally bad, which is the whole point:

* **Correct** — the right student is marked.
* **Wrong ID** — a *different* enrolled student is marked. The worst outcome:
  silent, and it is proxy attendance by accident.
* **Rejected** — nobody is marked; the student retries. Annoying, not harmful.

A stranger accepted as an enrolled student is reported separately as the
false-accept rate over unknown probes — the direct measure of whether the system
can be fooled by someone who never registered.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(slots=True)
class ProbeOutcome:
    """The full result of matching one probe against the gallery."""

    identity: str
    condition: str
    path: str
    #: Best-matching identity in the gallery, or None if the gallery was empty.
    predicted: str | None
    #: Similarity of the best match.
    best_score: float
    #: Similarity to the probe's own true identity (None for unknown probes).
    genuine_score: float | None
    #: Similarities to every other identity.
    impostor_scores: list[float] = field(default_factory=list)
    #: True when this probe came from a deliberately unenrolled person.
    is_unknown: bool = False
    #: Set when the face pipeline could not produce an embedding at all.
    failure: str | None = None

    @property
    def rank1_correct(self) -> bool:
        """Whether the closest gallery identity is the right one."""
        return not self.is_unknown and self.predicted == self.identity


@dataclass(slots=True)
class ThresholdReport:
    """Operational rates at one specific threshold."""

    threshold: float
    correct: int
    wrong_id: int
    rejected: int
    #: Unknown probes wrongly accepted as an enrolled student.
    unknown_accepted: int
    unknown_total: int

    @property
    def total(self) -> int:
        """Number of known probes evaluated at this threshold."""
        return self.correct + self.wrong_id + self.rejected

    @property
    def correct_rate(self) -> float:
        """Share of known probes marked correctly."""
        return self.correct / self.total if self.total else 0.0

    @property
    def wrong_id_rate(self) -> float:
        """Share of known probes attributed to the wrong student."""
        return self.wrong_id / self.total if self.total else 0.0

    @property
    def rejection_rate(self) -> float:
        """Share of known probes not marked at all."""
        return self.rejected / self.total if self.total else 0.0

    @property
    def unknown_accept_rate(self) -> float:
        """Share of strangers wrongly accepted as somebody."""
        return self.unknown_accepted / self.unknown_total if self.unknown_total else 0.0

    def to_dict(self) -> dict[str, object]:
        """Serialise for the JSON report."""
        return {
            "threshold": round(self.threshold, 4),
            "correct": self.correct,
            "wrong_id": self.wrong_id,
            "rejected": self.rejected,
            "correct_rate": round(self.correct_rate, 4),
            "wrong_id_rate": round(self.wrong_id_rate, 4),
            "rejection_rate": round(self.rejection_rate, 4),
            "unknown_total": self.unknown_total,
            "unknown_accepted": self.unknown_accepted,
            "unknown_accept_rate": round(self.unknown_accept_rate, 4),
        }


@dataclass(slots=True)
class VerificationMetrics:
    """Threshold-free description of how separable the two score classes are."""

    genuine: np.ndarray
    impostor: np.ndarray

    @property
    def genuine_mean(self) -> float:
        """Mean similarity between images of the same person."""
        return float(np.mean(self.genuine)) if self.genuine.size else 0.0

    @property
    def impostor_mean(self) -> float:
        """Mean similarity between images of different people."""
        return float(np.mean(self.impostor)) if self.impostor.size else 0.0

    @property
    def separation(self) -> float:
        """Gap between the worst genuine score and the best impostor score.

        Positive means the two classes do not overlap at all: some threshold
        separates them perfectly, and the exact value is not delicate.
        Negative means they overlap and every threshold trades one error for
        the other.
        """
        if not self.genuine.size or not self.impostor.size:
            return 0.0
        return float(np.min(self.genuine) - np.max(self.impostor))

    @property
    def d_prime(self) -> float:
        """Sensitivity index: class separation in pooled standard deviations.

        Scale-free, so it can be compared across datasets. Above ~3 is a
        comfortably separable system.
        """
        if self.genuine.size < 2 or self.impostor.size < 2:
            return 0.0
        pooled = np.sqrt((np.var(self.genuine) + np.var(self.impostor)) / 2)
        if pooled < 1e-9:
            return float("inf")
        return float((self.genuine_mean - self.impostor_mean) / pooled)

    def tar_at(self, threshold: float) -> float:
        """True accept rate: genuine pairs correctly accepted."""
        if not self.genuine.size:
            return 0.0
        return float(np.mean(self.genuine >= threshold))

    def far_at(self, threshold: float) -> float:
        """False accept rate: impostor pairs wrongly accepted."""
        if not self.impostor.size:
            return 0.0
        return float(np.mean(self.impostor >= threshold))

    def roc(self, steps: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sweep the threshold and return ``(thresholds, far, tar)``."""
        thresholds = np.linspace(-1.0, 1.0, steps)
        far = np.array([self.far_at(t) for t in thresholds])
        tar = np.array([self.tar_at(t) for t in thresholds])
        return thresholds, far, tar

    @property
    def auc(self) -> float:
        """Area under the ROC curve.

        Computed directly as the probability that a random genuine pair scores
        above a random impostor pair — exact, and free of the binning error a
        curve-integration would introduce.
        """
        if not self.genuine.size or not self.impostor.size:
            return 0.0
        comparisons = self.genuine[:, None] > self.impostor[None, :]
        ties = self.genuine[:, None] == self.impostor[None, :]
        return float(np.mean(comparisons) + 0.5 * np.mean(ties))

    @property
    def eer(self) -> tuple[float, float]:
        """Equal error rate and the threshold where it occurs.

        Returns:
            ``(eer, threshold)`` — the point where wrongly-accepted impostors
            and wrongly-rejected genuines balance.
        """
        thresholds, far, tar = self.roc()
        frr = 1.0 - tar
        index = int(np.argmin(np.abs(far - frr)))
        return float((far[index] + frr[index]) / 2), float(thresholds[index])

    def tar_at_far(self, target_far: float) -> tuple[float, float]:
        """Highest TAR achievable without exceeding ``target_far``.

        The operating point a deployment actually picks: "how many students are
        recognised if we allow at most this much impostor acceptance?"

        Returns:
            ``(tar, threshold)``. TAR is 0 if no threshold meets the target.
        """
        thresholds, far, tar = self.roc()
        acceptable = far <= target_far
        if not acceptable.any():
            return 0.0, 1.0
        index = int(np.argmax(np.where(acceptable, tar, -1.0)))
        return float(tar[index]), float(thresholds[index])

    def to_dict(self) -> dict[str, object]:
        """Serialise for the JSON report."""
        eer, eer_threshold = self.eer
        tar_1, thr_1 = self.tar_at_far(0.01)
        tar_01, thr_01 = self.tar_at_far(0.001)
        return {
            "genuine_pairs": int(self.genuine.size),
            "impostor_pairs": int(self.impostor.size),
            "genuine_mean": round(self.genuine_mean, 4),
            "genuine_min": round(float(np.min(self.genuine)), 4) if self.genuine.size else None,
            "genuine_std": round(float(np.std(self.genuine)), 4) if self.genuine.size else None,
            "impostor_mean": round(self.impostor_mean, 4),
            "impostor_max": round(float(np.max(self.impostor)), 4) if self.impostor.size else None,
            "impostor_std": round(float(np.std(self.impostor)), 4) if self.impostor.size else None,
            "separation": round(self.separation, 4),
            "d_prime": round(self.d_prime, 3),
            "auc": round(self.auc, 5),
            "eer": round(eer, 5),
            "eer_threshold": round(eer_threshold, 4),
            "tar_at_far_1pct": round(tar_1, 4),
            "threshold_at_far_1pct": round(thr_1, 4),
            "tar_at_far_0.1pct": round(tar_01, 4),
            "threshold_at_far_0.1pct": round(thr_01, 4),
        }


@dataclass(slots=True)
class GroupResult:
    """Rates for one slice of the probes (a condition, an identity, a group)."""

    label: str
    total: int
    correct: int
    wrong_id: int
    rejected: int
    mean_genuine: float

    @property
    def correct_rate(self) -> float:
        """Share of this slice's probes marked correctly."""
        return self.correct / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, object]:
        """Serialise for the JSON report."""
        return {
            "label": self.label,
            "total": self.total,
            "correct": self.correct,
            "wrong_id": self.wrong_id,
            "rejected": self.rejected,
            "correct_rate": round(self.correct_rate, 4),
            "mean_genuine": round(self.mean_genuine, 4),
        }


def collect_scores(outcomes: list[ProbeOutcome]) -> VerificationMetrics:
    """Pull the genuine and impostor score arrays out of the probe outcomes.

    Unknown probes contribute impostor scores only — by definition they have no
    genuine pair in the gallery.
    """
    genuine: list[float] = []
    impostor: list[float] = []

    for outcome in outcomes:
        if outcome.failure is not None:
            continue
        if outcome.genuine_score is not None:
            genuine.append(outcome.genuine_score)
        impostor.extend(outcome.impostor_scores)

    return VerificationMetrics(
        genuine=np.asarray(genuine, dtype=np.float32),
        impostor=np.asarray(impostor, dtype=np.float32),
    )


def evaluate_threshold(outcomes: list[ProbeOutcome], threshold: float) -> ThresholdReport:
    """Score the operational outcomes at one threshold.

    A probe the pipeline failed on (no face found) counts as ``rejected``:
    from the student's point of view they were not marked, and pretending that
    failure did not happen would flatter the result.
    """
    correct = wrong_id = rejected = 0
    unknown_accepted = unknown_total = 0

    for outcome in outcomes:
        if outcome.is_unknown:
            unknown_total += 1
            if outcome.failure is None and outcome.best_score >= threshold:
                unknown_accepted += 1
            continue

        if outcome.failure is not None or outcome.best_score < threshold:
            rejected += 1
        elif outcome.predicted == outcome.identity:
            correct += 1
        else:
            wrong_id += 1

    return ThresholdReport(
        threshold=threshold,
        correct=correct,
        wrong_id=wrong_id,
        rejected=rejected,
        unknown_accepted=unknown_accepted,
        unknown_total=unknown_total,
    )


def sweep_thresholds(
    outcomes: list[ProbeOutcome], steps: int = 41
) -> list[ThresholdReport]:
    """Evaluate a range of thresholds, for the trade-off table and figure."""
    return [
        evaluate_threshold(outcomes, float(threshold))
        for threshold in np.linspace(0.0, 1.0, steps)
    ]


def recommend_threshold(outcomes: list[ProbeOutcome], steps: int = 201) -> float:
    """Suggest a threshold from the data.

    The penalty encodes the project's own risk position: a wrong name on the
    register is far worse than asking a student to look at the camera again, so
    a wrong ID and an accepted stranger cost an order of magnitude more than a
    rejection.

    When the classes separate cleanly, many thresholds tie on penalty. Picking
    an extreme of that plateau would sit exactly on the lowest genuine score
    observed — perfect on this sample and brittle on the next face, which is
    overfitting to the test set. The **midpoint of the widest tied run** is
    returned instead: the max-margin choice, furthest from both error classes.

    Returns:
        The recommended threshold.
    """
    thresholds = np.linspace(0.0, 1.0, steps)
    penalties = []
    for threshold in thresholds:
        report = evaluate_threshold(outcomes, float(threshold))
        penalties.append(
            10.0 * report.wrong_id
            + 10.0 * report.unknown_accepted
            + 1.0 * report.rejected
        )

    best = min(penalties)
    # Find the longest contiguous run of thresholds achieving the best penalty.
    best_start = best_length = current_start = current_length = 0
    for index, penalty in enumerate(penalties):
        if penalty == best:
            if current_length == 0:
                current_start = index
            current_length += 1
            if current_length > best_length:
                best_length, best_start = current_length, current_start
        else:
            current_length = 0

    return float(thresholds[best_start + best_length // 2])


def slice_by(
    outcomes: list[ProbeOutcome],
    key: str,
    threshold: float,
    labels: dict[str, str] | None = None,
) -> list[GroupResult]:
    """Break the results down by condition, identity or group.

    Args:
        outcomes: Probe outcomes for enrolled identities.
        key: ``"condition"``, ``"identity"``, or ``"group"``.
        threshold: Threshold to apply when classifying each outcome.
        labels: For ``key="group"``, a mapping of identity → group label.

    Returns:
        One result per slice, ordered worst-performing first — the interesting
        end of the table.
    """
    buckets: dict[str, list[ProbeOutcome]] = {}

    for outcome in outcomes:
        if outcome.is_unknown:
            continue
        if key == "condition":
            label = outcome.condition
        elif key == "identity":
            label = outcome.identity
        elif key == "group":
            label = (labels or {}).get(outcome.identity) or "(unlabelled)"
        else:
            raise ValueError(f"Unknown slice key: {key}")
        buckets.setdefault(label, []).append(outcome)

    results: list[GroupResult] = []
    for label, items in buckets.items():
        report = evaluate_threshold(items, threshold)
        genuine = [
            item.genuine_score for item in items if item.genuine_score is not None
        ]
        results.append(
            GroupResult(
                label=label,
                total=report.total,
                correct=report.correct,
                wrong_id=report.wrong_id,
                rejected=report.rejected,
                mean_genuine=float(np.mean(genuine)) if genuine else 0.0,
            )
        )

    results.sort(key=lambda result: (result.correct_rate, result.label))
    return results
