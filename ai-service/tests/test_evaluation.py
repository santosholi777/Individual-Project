"""Tests for the evaluation harness.

These matter more than most: a metric that miscomputes does not crash, it just
puts a confidently wrong number into the project report. Each test therefore
checks a value that can be derived by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.dataset import DatasetError, load_dataset
from evaluation.figures import score_distribution_svg, threshold_sweep_svg
from evaluation.metrics import (
    ProbeOutcome,
    VerificationMetrics,
    collect_scores,
    evaluate_threshold,
    recommend_threshold,
    slice_by,
    sweep_thresholds,
)


def make_outcome(
    identity: str = "S1",
    predicted: str | None = "S1",
    best: float = 0.9,
    genuine: float | None = 0.9,
    impostors: tuple[float, ...] = (0.1,),
    condition: str = "frontal",
    is_unknown: bool = False,
    failure: str | None = None,
) -> ProbeOutcome:
    """Build a probe outcome for tests."""
    return ProbeOutcome(
        identity=identity,
        condition=condition,
        path=f"{identity}/x.jpg",
        predicted=predicted,
        best_score=best,
        genuine_score=genuine,
        impostor_scores=list(impostors),
        is_unknown=is_unknown,
        failure=failure,
    )


class TestVerificationMetrics:
    """The threshold-free separability measures."""

    def test_perfectly_separated_classes_score_auc_1(self) -> None:
        """Every genuine above every impostor is a perfect ranking."""
        metrics = VerificationMetrics(
            genuine=np.array([0.8, 0.9, 1.0], dtype=np.float32),
            impostor=np.array([0.0, 0.1, 0.2], dtype=np.float32),
        )
        assert metrics.auc == pytest.approx(1.0)

    def test_reversed_classes_score_auc_0(self) -> None:
        """AUC is directional: a fully inverted ranking scores 0."""
        metrics = VerificationMetrics(
            genuine=np.array([0.0, 0.1], dtype=np.float32),
            impostor=np.array([0.8, 0.9], dtype=np.float32),
        )
        assert metrics.auc == pytest.approx(0.0)

    def test_identical_classes_score_auc_half(self) -> None:
        """Indistinguishable classes are a coin flip; ties count as half."""
        values = np.array([0.5, 0.5], dtype=np.float32)
        metrics = VerificationMetrics(genuine=values, impostor=values.copy())
        assert metrics.auc == pytest.approx(0.5)

    def test_separation_is_positive_when_classes_do_not_overlap(self) -> None:
        """Worst genuine minus best impostor, the gap a threshold sits in."""
        metrics = VerificationMetrics(
            genuine=np.array([0.8, 0.95], dtype=np.float32),
            impostor=np.array([0.1, 0.3], dtype=np.float32),
        )
        assert metrics.separation == pytest.approx(0.5, abs=1e-6)

    def test_separation_is_negative_when_classes_overlap(self) -> None:
        """A negative gap is the signal that no threshold is perfect."""
        metrics = VerificationMetrics(
            genuine=np.array([0.4, 0.9], dtype=np.float32),
            impostor=np.array([0.1, 0.6], dtype=np.float32),
        )
        assert metrics.separation == pytest.approx(-0.2, abs=1e-6)

    def test_eer_is_zero_for_separable_classes(self) -> None:
        """With a gap, some threshold makes both error rates zero."""
        metrics = VerificationMetrics(
            genuine=np.array([0.8, 0.9, 1.0], dtype=np.float32),
            impostor=np.array([0.0, 0.1], dtype=np.float32),
        )
        eer, threshold = metrics.eer
        assert eer == pytest.approx(0.0, abs=1e-6)
        assert 0.1 < threshold < 0.8

    def test_tar_and_far_at_a_threshold(self) -> None:
        """Rates are simple shares at or above the threshold."""
        metrics = VerificationMetrics(
            genuine=np.array([0.4, 0.6, 0.8, 1.0], dtype=np.float32),
            impostor=np.array([0.0, 0.2, 0.6, 0.9], dtype=np.float32),
        )
        assert metrics.tar_at(0.5) == pytest.approx(0.75)
        assert metrics.far_at(0.5) == pytest.approx(0.5)

    def test_tar_at_far_respects_the_budget(self) -> None:
        """The operating point must not exceed the allowed impostor rate."""
        metrics = VerificationMetrics(
            genuine=np.array([0.8, 0.9], dtype=np.float32),
            impostor=np.array([0.1, 0.2], dtype=np.float32),
        )
        tar, threshold = metrics.tar_at_far(0.0)
        assert tar == pytest.approx(1.0)
        assert metrics.far_at(threshold) == 0.0

    def test_empty_input_does_not_crash(self) -> None:
        """A dataset that produced no pairs reports zeros, not exceptions."""
        metrics = VerificationMetrics(
            genuine=np.array([], dtype=np.float32),
            impostor=np.array([], dtype=np.float32),
        )
        assert metrics.auc == 0.0
        assert metrics.separation == 0.0
        assert metrics.d_prime == 0.0


class TestCollectScores:
    """Pulling score arrays out of probe outcomes."""

    def test_splits_genuine_from_impostor(self) -> None:
        """Each outcome contributes one genuine and its impostor scores."""
        metrics = collect_scores(
            [
                make_outcome(genuine=0.9, impostors=(0.1, 0.2)),
                make_outcome(genuine=0.8, impostors=(0.3,)),
            ]
        )
        assert sorted(metrics.genuine.tolist()) == pytest.approx([0.8, 0.9])
        assert sorted(metrics.impostor.tolist()) == pytest.approx([0.1, 0.2, 0.3])

    def test_unknown_probes_contribute_no_genuine_score(self) -> None:
        """A stranger has no true identity in the gallery, by definition."""
        metrics = collect_scores(
            [make_outcome(genuine=None, impostors=(0.2, 0.3), is_unknown=True)]
        )
        assert metrics.genuine.size == 0
        assert metrics.impostor.size == 2

    def test_failed_probes_are_excluded_from_scores(self) -> None:
        """A probe with no face has no score to contribute."""
        metrics = collect_scores([make_outcome(failure="no face detected")])
        assert metrics.genuine.size == 0


class TestEvaluateThreshold:
    """The three operational outcomes."""

    def test_confident_correct_match_is_marked(self) -> None:
        """The happy path."""
        report = evaluate_threshold([make_outcome(best=0.9, predicted="S1")], 0.45)
        assert (report.correct, report.wrong_id, report.rejected) == (1, 0, 0)

    def test_confident_wrong_match_is_a_wrong_id(self) -> None:
        """Above threshold but the wrong person — the dangerous outcome."""
        report = evaluate_threshold(
            [make_outcome(identity="S1", predicted="S2", best=0.9)], 0.45
        )
        assert (report.correct, report.wrong_id, report.rejected) == (0, 1, 0)
        assert report.wrong_id_rate == pytest.approx(1.0)

    def test_low_confidence_is_rejected(self) -> None:
        """Below threshold, nobody is marked."""
        report = evaluate_threshold([make_outcome(best=0.2)], 0.45)
        assert (report.correct, report.wrong_id, report.rejected) == (0, 0, 1)

    def test_score_exactly_at_threshold_is_accepted(self) -> None:
        """The comparison is >=, matching the production matcher."""
        report = evaluate_threshold([make_outcome(best=0.45)], 0.45)
        assert report.correct == 1

    def test_pipeline_failure_counts_as_rejected(self) -> None:
        """No face found means the student was not marked — not a free pass."""
        report = evaluate_threshold([make_outcome(failure="no face")], 0.45)
        assert report.rejected == 1

    def test_unknown_probes_are_tracked_separately(self) -> None:
        """Strangers are scored against their own denominator."""
        report = evaluate_threshold(
            [
                make_outcome(best=0.9, is_unknown=True, genuine=None),
                make_outcome(best=0.1, is_unknown=True, genuine=None),
            ],
            0.45,
        )
        assert report.unknown_total == 2
        assert report.unknown_accepted == 1
        assert report.unknown_accept_rate == pytest.approx(0.5)
        # Strangers must not pollute the known-probe rates.
        assert report.total == 0

    def test_rates_sum_to_one(self) -> None:
        """The three outcomes are exhaustive and mutually exclusive."""
        report = evaluate_threshold(
            [
                make_outcome(best=0.9, predicted="S1"),
                make_outcome(identity="S1", predicted="S2", best=0.9),
                make_outcome(best=0.1),
            ],
            0.45,
        )
        total = report.correct_rate + report.wrong_id_rate + report.rejection_rate
        assert total == pytest.approx(1.0)


class TestRecommendThreshold:
    """The suggested operating point."""

    def test_lands_inside_the_gap_for_separable_data(self) -> None:
        """With a clean gap, the recommendation sits between the classes."""
        outcomes = [
            make_outcome(identity="S1", predicted="S1", best=0.9, genuine=0.9, impostors=(0.1,)),
            make_outcome(identity="S2", predicted="S2", best=0.85, genuine=0.85, impostors=(0.05,)),
        ]
        assert 0.1 < recommend_threshold(outcomes) < 0.85

    def test_does_not_cling_to_the_worst_genuine_score(self) -> None:
        """The midpoint rule exists to avoid overfitting to this sample.

        A threshold sitting exactly on the lowest genuine score scores perfectly
        here and rejects the next slightly-worse face.
        """
        outcomes = [
            make_outcome(identity="S1", predicted="S1", best=0.90, genuine=0.90, impostors=(0.0,)),
            make_outcome(identity="S2", predicted="S2", best=0.92, genuine=0.92, impostors=(0.0,)),
        ]
        assert recommend_threshold(outcomes) < 0.85

    def test_rises_to_exclude_strangers(self) -> None:
        """A stranger scoring 0.5 must push the recommendation above it."""
        outcomes = [
            make_outcome(identity="S1", predicted="S1", best=0.95, genuine=0.95, impostors=(0.1,)),
            make_outcome(predicted="S1", best=0.5, genuine=None, is_unknown=True, impostors=(0.5,)),
        ]
        assert recommend_threshold(outcomes) > 0.5


class TestSlicing:
    """Breakdowns by condition, identity and group."""

    def test_groups_by_condition(self) -> None:
        """Each condition gets its own rates."""
        outcomes = [
            make_outcome(condition="dim", best=0.1),
            make_outcome(condition="dim", best=0.1),
            make_outcome(condition="frontal", best=0.9),
        ]
        results = slice_by(outcomes, "condition", 0.45)

        by_label = {result.label: result for result in results}
        assert by_label["dim"].correct_rate == 0.0
        assert by_label["frontal"].correct_rate == 1.0

    def test_worst_slice_is_listed_first(self) -> None:
        """The interesting end of the table comes first."""
        outcomes = [
            make_outcome(condition="good", best=0.9),
            make_outcome(condition="bad", best=0.1),
        ]
        assert slice_by(outcomes, "condition", 0.45)[0].label == "bad"

    def test_groups_by_fairness_label(self) -> None:
        """Group labels come from the caller, never inferred."""
        outcomes = [
            make_outcome(identity="S1", predicted="S1", best=0.9),
            make_outcome(identity="S2", predicted="S2", best=0.9),
        ]
        results = slice_by(
            outcomes, "group", 0.45, labels={"S1": "group-a", "S2": "group-b"}
        )
        assert {result.label for result in results} == {"group-a", "group-b"}

    def test_unlabelled_identities_are_bucketed_explicitly(self) -> None:
        """Missing labels are visible rather than silently dropped."""
        results = slice_by(
            [make_outcome(identity="S9", predicted="S9")], "group", 0.45, labels={}
        )
        assert results[0].label == "(unlabelled)"

    def test_unknown_probes_are_never_sliced(self) -> None:
        """Strangers have no true identity, so they belong in no slice."""
        outcomes = [make_outcome(is_unknown=True, genuine=None)]
        assert slice_by(outcomes, "condition", 0.45) == []

    def test_unknown_slice_key_raises(self) -> None:
        """A typo in the key is a programming error, not a silent empty table."""
        with pytest.raises(ValueError, match="Unknown slice key"):
            slice_by([make_outcome()], "nonsense", 0.45)


class TestSweep:
    """The threshold sweep behind the trade-off figure."""

    def test_correct_rate_falls_as_the_threshold_rises(self) -> None:
        """Stricter thresholds mark fewer people; the curve is monotonic."""
        outcomes = [make_outcome(best=score, genuine=score) for score in (0.3, 0.6, 0.9)]
        sweep = sweep_thresholds(outcomes, steps=11)
        rates = [report.correct_rate for report in sweep]

        assert rates[0] == pytest.approx(1.0)
        assert rates[-1] == pytest.approx(0.0)
        assert all(a >= b for a, b in zip(rates, rates[1:]))


class TestFigures:
    """The SVG report figures."""

    @staticmethod
    def _metrics() -> VerificationMetrics:
        return VerificationMetrics(
            genuine=np.array([0.8, 0.9, 0.95], dtype=np.float32),
            impostor=np.array([0.0, 0.1, 0.2], dtype=np.float32),
        )

    def test_distribution_svg_is_well_formed_xml(self) -> None:
        """A malformed SVG renders as a broken image with no other warning."""
        from xml.etree import ElementTree

        ElementTree.fromstring(score_distribution_svg(self._metrics(), 0.45))

    def test_sweep_svg_is_well_formed_xml(self) -> None:
        """Same guarantee for the trade-off figure."""
        from xml.etree import ElementTree

        outcomes = [make_outcome(best=0.9, genuine=0.9)]
        ElementTree.fromstring(threshold_sweep_svg(sweep_thresholds(outcomes), 0.45))

    def test_font_family_quotes_are_escaped(self) -> None:
        """The exact bug that once shipped an unparseable figure."""
        svg = score_distribution_svg(self._metrics(), 0.45)
        assert '"Segoe UI"' not in svg
        assert "&quot;Segoe UI&quot;" in svg

    def test_distribution_svg_reports_the_threshold(self) -> None:
        """The figure must state the decision boundary it draws."""
        assert "threshold 0.45" in score_distribution_svg(self._metrics(), 0.45)


class TestDatasetLoading:
    """Reading a dataset off disk."""

    @staticmethod
    def _build(root, identities: dict[str, int]) -> None:
        """Create a dataset tree of empty (unreadable) placeholder images."""
        for student_id, count in identities.items():
            folder = root / student_id
            folder.mkdir(parents=True)
            for index in range(count):
                (folder / f"{index:02d}.jpg").write_bytes(b"placeholder")

    def test_splits_enrolment_from_probes(self, tmp_path) -> None:
        """The first N images enrol; the remainder probe."""
        self._build(tmp_path, {"S1": 5})
        dataset = load_dataset(tmp_path, enrol_per_identity=3)

        assert len(dataset.identities) == 1
        assert len(dataset.identities[0].enrol_paths) == 3
        assert len(dataset.identities[0].probes) == 2

    def test_condition_subfolders_tag_their_probes(self, tmp_path) -> None:
        """Sub-folder names become condition labels."""
        self._build(tmp_path, {"S1": 4})
        dim = tmp_path / "S1" / "dim_light"
        dim.mkdir()
        (dim / "a.jpg").write_bytes(b"placeholder")

        dataset = load_dataset(tmp_path, enrol_per_identity=3)
        assert "dim_light" in dataset.conditions
        assert any(probe.condition == "dim_light" for probe in dataset.identities[0].probes)

    def test_explicit_enrol_folder_is_respected(self, tmp_path) -> None:
        """An enrol/ folder overrides the count-based split."""
        self._build(tmp_path, {"S1": 2})
        enrol = tmp_path / "S1" / "enrol"
        enrol.mkdir()
        for index in range(3):
            (enrol / f"e{index}.jpg").write_bytes(b"placeholder")

        dataset = load_dataset(tmp_path, enrol_per_identity=99)
        assert len(dataset.identities[0].enrol_paths) == 3
        assert len(dataset.identities[0].probes) == 2

    def test_holdout_identities_become_strangers(self, tmp_path) -> None:
        """A held-out person is never enrolled; all their images are probes."""
        self._build(tmp_path, {"S1": 4, "S2": 4})
        dataset = load_dataset(tmp_path, enrol_per_identity=3, holdout=("S2",))

        assert [identity.student_id for identity in dataset.identities] == ["S1"]
        assert [identity.student_id for identity in dataset.unknowns] == ["S2"]
        assert dataset.unknown_probe_count == 4

    def test_meta_json_supplies_names_and_groups(self, tmp_path) -> None:
        """Display names and fairness labels come from meta.json."""
        self._build(tmp_path, {"S1": 4})
        (tmp_path / "meta.json").write_text(
            '{"S1": {"name": "Aditi Sharma", "group": "female"}}'
        )
        dataset = load_dataset(tmp_path, enrol_per_identity=3)

        assert dataset.identities[0].name == "Aditi Sharma"
        assert dataset.identities[0].group == "female"
        assert dataset.groups == ["female"]

    def test_identity_without_probes_is_skipped(self, tmp_path) -> None:
        """Too few images to split means the identity cannot be evaluated."""
        self._build(tmp_path, {"S1": 3, "S2": 5})
        dataset = load_dataset(tmp_path, enrol_per_identity=3)

        assert [identity.student_id for identity in dataset.identities] == ["S2"]

    def test_missing_directory_raises(self, tmp_path) -> None:
        """A wrong path should say so plainly."""
        with pytest.raises(DatasetError, match="not found"):
            load_dataset(tmp_path / "nope")

    def test_empty_dataset_explains_the_requirement(self, tmp_path) -> None:
        """The error tells the user how many images per person are needed."""
        self._build(tmp_path, {"S1": 1})
        with pytest.raises(DatasetError, match="at least"):
            load_dataset(tmp_path, enrol_per_identity=3)
