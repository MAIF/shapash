"""Unit tests for confident-learning label-noise detection (pure numpy, no model).

Probabilities are constructed by hand so every expected count is derivable on paper: the tests assert
the algorithm's *intermediate* quantities (thresholds, confident joint, calibrated joint) as well as
the ranked output, since a plausible-looking issue list can hide a wrong noise matrix and vice versa.
"""

import logging
import unittest

import numpy as np
import pandas as pd

from shapash.compute.diagnostics.label_noise import (
    SCORE_METHODS,
    LabelNoiseReport,
    _calibrate,
    _class_thresholds,
    _confident_joint,
    _prune_by_noise_rate,
    _quality_scores,
    detect_label_issues,
    has_usable_probabilities,
)

_NAMES = ["joy", "surprise", "anger"]


def _confident_probs(true_classes, n_classes=3, peak=0.90):
    """One row per entry: ``peak`` on the true class, the rest spread evenly."""
    rest = (1.0 - peak) / (n_classes - 1)
    probs = np.full((len(true_classes), n_classes), rest)
    probs[np.arange(len(true_classes)), true_classes] = peak
    return probs


def _planted_corpus(n=300, n_flips=15, seed=0):
    """A confident model on ``n`` samples, with ``n_flips`` joy samples mislabelled as surprise."""
    rng = np.random.default_rng(seed)
    true = rng.integers(0, 3, n)
    probs = _confident_probs(true)
    given = true.copy()
    flipped = np.flatnonzero(true == 0)[:n_flips]
    given[flipped] = 1
    labels = [_NAMES[i] for i in given]
    texts = [f"text-{i}" for i in range(n)]
    return probs, labels, texts, set(flipped.tolist())


class TestHasUsableProbabilities(unittest.TestCase):
    def test_none_is_unusable(self):
        self.assertFalse(has_usable_probabilities(None))

    def test_legacy_single_probability_column_is_unusable(self):
        # The raw-pipeline path reports only the winning class's confidence; the losing classes'
        # probabilities are exactly what confident learning needs.
        self.assertFalse(has_usable_probabilities(pd.DataFrame({"probability": [0.9, 0.8]})))

    def test_per_class_columns_are_usable(self):
        self.assertTrue(has_usable_probabilities(pd.DataFrame({"joy": [0.9, 0.2], "surprise": [0.1, 0.8]})))


class TestThresholds(unittest.TestCase):
    def test_threshold_is_the_mean_self_confidence_of_each_class(self):
        probs = np.array([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7]])
        given = np.array([0, 0, 1])
        thresholds = _class_thresholds(probs, given, 2)
        np.testing.assert_allclose(thresholds, [(0.8 + 0.6) / 2, 0.7])

    def test_class_without_labels_gets_an_infinite_threshold_and_warns(self):
        probs = np.array([[0.5, 0.3, 0.2], [0.4, 0.4, 0.2]])
        given = np.array([0, 0])
        with self.assertLogs("shapash.compute.diagnostics.label_noise", level=logging.WARNING) as logs:
            thresholds = _class_thresholds(probs, given, 3)
        self.assertEqual(thresholds[1], np.inf)
        self.assertEqual(thresholds[2], np.inf)
        self.assertIn("[1, 2]", "".join(logs.output))

    def test_an_unlabelled_class_is_never_suggested(self):
        # Class 2 has no labelled sample, yet the model scores it highest on both rows. Without the
        # inf threshold it would be "suggested" on evidence the corpus says nothing about.
        probs = np.array([[0.2, 0.1, 0.7], [0.15, 0.05, 0.80]])
        labels = ["joy", "joy"]
        report = detect_label_issues(probs, labels, ["a", "b"], _NAMES)
        self.assertEqual([i.suggested_label for i in report.issues], [])


class TestConfidentJoint(unittest.TestCase):
    def test_counts_given_against_confidently_predicted(self):
        # t = [0.5, 0.5]. Row 0: only class 0 clears -> C[0,0]. Row 1: only class 1 clears, given 0
        # -> C[0,1]. Row 2: only class 1 clears, given 1 -> C[1,1].
        probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7]])
        given = np.array([0, 0, 1])
        counts = _confident_joint(probs, given, np.array([0.5, 0.5]))
        np.testing.assert_array_equal(counts, [[1, 1], [0, 1]])

    def test_a_sample_clearing_no_threshold_abstains(self):
        # This is what separates the confident joint from a plain confusion matrix: the uncertain row
        # is dropped rather than forced onto its argmax.
        probs = np.array([[0.55, 0.45]])
        counts = _confident_joint(probs, np.array([0]), np.array([0.9, 0.9]))
        np.testing.assert_array_equal(counts, np.zeros((2, 2), dtype=int))

    def test_identical_self_confidences_still_clear_their_threshold(self):
        # A threshold is the mean of the very values compared against it, so exact `>=` loses to
        # float rounding when a class's samples share a confidence — the class would abstain entirely.
        true = np.array([0, 0, 1, 1, 2, 2])
        probs = _confident_probs(true)
        probs = probs / probs.sum(axis=1, keepdims=True)
        thresholds = _class_thresholds(probs, true, 3)
        counts = _confident_joint(probs, true, thresholds)
        np.testing.assert_array_equal(np.diag(counts), [2, 2, 2])


class TestCalibrate(unittest.TestCase):
    def test_rows_are_rescaled_to_the_observed_class_counts_and_sum_to_one(self):
        # Class 0 has 10 labelled samples but only 5 confident votes; calibration restores the
        # marginal so the joint reads as a fraction of the corpus.
        counts = np.array([[4, 1], [0, 5]])
        given = np.array([0] * 10 + [1] * 10)
        joint = _calibrate(counts, given, 2)
        self.assertAlmostEqual(joint.sum(), 1.0)
        np.testing.assert_allclose(joint.sum(axis=1), [0.5, 0.5])
        self.assertAlmostEqual(joint[0, 1], (1 / 5) * 10 / 20)

    def test_all_zero_counts_stay_zero_instead_of_dividing_by_zero(self):
        joint = _calibrate(np.zeros((2, 2), dtype=int), np.array([0, 1]), 2)
        np.testing.assert_array_equal(joint, np.zeros((2, 2)))


class TestQualityScores(unittest.TestCase):
    def test_self_confidence_is_the_given_labels_probability(self):
        probs = np.array([[0.7, 0.3], [0.1, 0.9]])
        np.testing.assert_allclose(_quality_scores(probs, np.array([0, 0]), "self_confidence"), [0.7, 0.1])

    def test_normalized_margin_subtracts_the_best_competitor(self):
        probs = np.array([[0.7, 0.3], [0.1, 0.9]])
        np.testing.assert_allclose(_quality_scores(probs, np.array([0, 0]), "normalized_margin"), [0.4, -0.8])

    def test_the_two_methods_can_disagree_on_ordering(self):
        # Row 0 is merely uncertain; row 1 has a confident alternative. Self-confidence prefers row 1
        # as the worse label, the margin agrees more strongly — the ranking is not interchangeable.
        probs = np.array([[0.34, 0.33, 0.33], [0.30, 0.65, 0.05]])
        given = np.array([0, 0])
        conf = _quality_scores(probs, given, "self_confidence")
        margin = _quality_scores(probs, given, "normalized_margin")
        self.assertLess(conf[1], conf[0])
        self.assertLess(margin[1], margin[0])
        self.assertGreater(margin[0], 0.0)  # nothing beats the given label
        self.assertLess(margin[1], 0.0)  # something does

    def test_every_documented_method_is_accepted(self):
        probs = np.array([[0.7, 0.3], [0.1, 0.9]])
        for method in SCORE_METHODS:
            self.assertEqual(_quality_scores(probs, np.array([0, 0]), method).shape, (2,))


class TestPruneByNoiseRate(unittest.TestCase):
    def test_never_suggests_a_class_the_model_ranks_below_the_given_label(self):
        # The joint asks for 2 errors in cell (0, 1), but only one sample actually favours class 1.
        # Taking the second would "suggest" a class the model scores lower than the label already on
        # the row.
        probs = np.array([[0.2, 0.8], [0.9, 0.1]])
        joint = np.array([[0.0, 1.0], [0.0, 0.0]])
        selected = _prune_by_noise_rate(probs, np.array([0, 0]), joint, 2)
        self.assertEqual(selected, {0: 1})

    def test_a_cell_with_no_positive_margin_selects_nothing(self):
        # The joint asks for an error in cell (0, 1), but the model prefers the given label on every
        # class-0 sample, so there is no defensible row to flag.
        probs = np.array([[0.9, 0.1], [0.8, 0.2]])
        joint = np.array([[0.0, 1.0], [0.0, 0.0]])
        self.assertEqual(_prune_by_noise_rate(probs, np.array([0, 0]), joint, 2), {})

    def test_a_sample_claimed_by_two_cells_keeps_the_larger_margin(self):
        probs = np.array([[0.1, 0.3, 0.6]])
        joint = np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        selected = _prune_by_noise_rate(probs, np.array([0]), joint, 3)
        self.assertEqual(selected, {0: 2})  # margin 0.5 beats 0.2


class TestDetectLabelIssues(unittest.TestCase):
    def test_recovers_planted_mislabels_exactly(self):
        probs, labels, texts, planted = _planted_corpus()
        report = detect_label_issues(probs, labels, texts, _NAMES, top_n=100)
        self.assertEqual(report.n_issues, len(planted))
        self.assertEqual({i.index for i in report.issues}, planted)
        for issue in report.issues:
            self.assertEqual(issue.given_label, "surprise")
            self.assertEqual(issue.suggested_label, "joy")
            self.assertGreater(issue.suggested_prob, issue.given_prob)

    def test_estimates_the_noise_matrix_and_rate(self):
        probs, labels, texts, planted = _planted_corpus(n=300, n_flips=15)
        report = detect_label_issues(probs, labels, texts, _NAMES, top_n=100)
        self.assertAlmostEqual(report.noise_matrix.sum(), 1.0)
        self.assertAlmostEqual(report.noise_rate, len(planted) / 300, places=6)
        # The contamination is surprise-labelled rows that are really joy, and nothing else.
        self.assertAlmostEqual(report.noise_matrix[1, 0], len(planted) / 300, places=6)
        off_diagonal = report.noise_matrix.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        off_diagonal[1, 0] = 0.0
        np.testing.assert_allclose(off_diagonal, 0.0, atol=1e-12)

    def test_a_clean_corpus_yields_no_issues_and_a_diagonal_matrix(self):
        rng = np.random.default_rng(3)
        true = rng.integers(0, 3, 120)
        probs = _confident_probs(true)
        report = detect_label_issues(probs, [_NAMES[i] for i in true], [f"t{i}" for i in range(120)], _NAMES)
        self.assertEqual(report.issues, [])
        self.assertEqual(report.n_issues, 0)
        self.assertAlmostEqual(report.noise_rate, 0.0)
        np.testing.assert_allclose(report.noise_matrix.sum(axis=1), np.bincount(true) / 120)

    def test_issues_are_ranked_worst_first(self):
        probs, labels, texts, _ = _planted_corpus()
        report = detect_label_issues(probs, labels, texts, _NAMES, top_n=100, score="normalized_margin")
        scores = [i.score for i in report.issues]
        self.assertEqual(scores, sorted(scores))

    def test_top_n_truncates_the_list_but_not_the_count(self):
        probs, labels, texts, planted = _planted_corpus()
        report = detect_label_issues(probs, labels, texts, _NAMES, top_n=4)
        self.assertEqual(len(report.issues), 4)
        self.assertEqual(report.n_issues, len(planted))
        # The kept ones are the worst-scoring of the full set.
        full = detect_label_issues(probs, labels, texts, _NAMES, top_n=100)
        self.assertEqual([i.index for i in report.issues], [i.index for i in full.issues[:4]])

    def test_top_n_zero_returns_no_issues_but_still_reports_the_count(self):
        probs, labels, texts, planted = _planted_corpus()
        report = detect_label_issues(probs, labels, texts, _NAMES, top_n=0)
        self.assertEqual(report.issues, [])
        self.assertEqual(report.n_issues, len(planted))

    def test_issues_carry_their_text_and_position(self):
        probs, labels, texts, _ = _planted_corpus()
        issue = detect_label_issues(probs, labels, texts, _NAMES, top_n=1).issues[0]
        self.assertEqual(issue.text, texts[issue.index])
        self.assertIsNone(issue.probe)  # the second opinion is the caller's, not the algorithm's

    def test_report_is_a_frozen_dataclass_with_the_documented_fields(self):
        probs, labels, texts, _ = _planted_corpus(n=30, n_flips=2)
        report = detect_label_issues(probs, labels, texts, _NAMES)
        self.assertIsInstance(report, LabelNoiseReport)
        self.assertEqual(report.label_names, _NAMES)
        self.assertEqual(report.n_samples, 30)
        self.assertEqual(report.thresholds.shape, (3,))
        with self.assertRaises(Exception):
            report.issues = []  # type: ignore[misc]

    def test_warns_when_probability_rows_do_not_sum_to_one(self):
        probs = np.array([[0.2, 0.2], [0.9, 0.05]])
        with self.assertLogs("shapash.compute.diagnostics.label_noise", level=logging.WARNING) as logs:
            detect_label_issues(probs, ["joy", "joy"], ["a", "b"], ["joy", "surprise"])
        self.assertIn("do not sum to 1", "".join(logs.output))


class TestValidation(unittest.TestCase):
    def setUp(self):
        self.probs = np.array([[0.9, 0.1], [0.2, 0.8]])
        self.labels = ["joy", "surprise"]
        self.texts = ["a", "b"]
        self.names = ["joy", "surprise"]

    def test_rejects_non_2d_probabilities(self):
        with self.assertRaisesRegex(ValueError, "must be 2-D"):
            detect_label_issues(np.array([0.9, 0.1]), self.labels, self.texts, self.names)

    def test_rejects_fewer_than_two_classes(self):
        with self.assertRaisesRegex(ValueError, "at least 2 classes"):
            detect_label_issues(np.array([[1.0], [1.0]]), self.labels, self.texts, ["joy"])

    def test_rejects_label_names_of_the_wrong_length(self):
        with self.assertRaisesRegex(ValueError, "label_names has 3 entries"):
            detect_label_issues(self.probs, self.labels, self.texts, _NAMES)

    def test_rejects_mismatched_labels_length(self):
        with self.assertRaisesRegex(ValueError, "labels has 1 entries"):
            detect_label_issues(self.probs, ["joy"], self.texts, self.names)

    def test_rejects_mismatched_texts_length(self):
        with self.assertRaisesRegex(ValueError, "texts has 1 entries"):
            detect_label_issues(self.probs, self.labels, ["a"], self.names)

    def test_rejects_a_label_absent_from_label_names(self):
        with self.assertRaisesRegex(ValueError, r"\['anger'\]"):
            detect_label_issues(self.probs, ["joy", "anger"], self.texts, self.names)

    def test_rejects_an_unknown_score_method(self):
        with self.assertRaisesRegex(ValueError, "Unknown score method"):
            detect_label_issues(self.probs, self.labels, self.texts, self.names, score="entropy")


if __name__ == "__main__":
    unittest.main()
