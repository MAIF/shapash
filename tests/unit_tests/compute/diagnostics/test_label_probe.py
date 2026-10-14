"""Unit tests for the model-independent label probe."""

import logging
import unittest

from shapash.compute.diagnostics.label_probe import MIN_PER_CLASS, LabelProbe, ProbeVerdict

# A corpus with two lexically separable classes, so a bag-of-words probe is reliable on it.
POSITIVE = [
    "this film is wonderful and moving",
    "a wonderful delightful movie",
    "delightful and heartwarming story",
    "i loved this wonderful picture",
    "heartwarming and delightful throughout",
    "a moving and wonderful tale",
]
NEGATIVE = [
    "this film is dreadful and boring",
    "a dreadful tedious movie",
    "tedious and boring story",
    "i hated this dreadful picture",
    "boring and tedious throughout",
    "a tedious and dreadful tale",
]
TEXTS = POSITIVE + NEGATIVE
LABELS = ["pos"] * len(POSITIVE) + ["neg"] * len(NEGATIVE)


class TestConstruction(unittest.TestCase):
    """Input validation and corpus reporting."""

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError) as ctx:
            LabelProbe(TEXTS, LABELS[:-1])
        self.assertIn("must match", str(ctx.exception))

    def test_single_class_corpus_raises(self):
        with self.assertRaises(ValueError) as ctx:
            LabelProbe(POSITIVE, ["pos"] * len(POSITIVE))
        self.assertIn("at least 2 classes", str(ctx.exception))

    def test_classes_reports_the_reference_labels(self):
        self.assertEqual(LabelProbe(TEXTS, LABELS).classes, ["neg", "pos"])

    def test_thin_class_warns_but_still_builds(self):
        texts = [*POSITIVE, "a dreadful tedious movie"]
        labels = ["pos"] * len(POSITIVE) + ["neg"]
        with self.assertLogs("shapash.compute.diagnostics.label_probe", level=logging.WARNING) as logs:
            probe = LabelProbe(texts, labels)
        self.assertIn("neg", logs.output[0])
        self.assertIn(str(MIN_PER_CLASS), logs.output[0])
        self.assertEqual(probe.classes, ["neg", "pos"])

    def test_values_are_coerced_to_str(self):
        probe = LabelProbe(TEXTS, [1] * 6 + [0] * 6)
        self.assertEqual(probe.classes, ["0", "1"])


class TestVerdicts(unittest.TestCase):
    """The per-row second opinion."""

    def setUp(self):
        self.probe = LabelProbe(TEXTS, LABELS)

    def test_backs_a_label_the_corpus_supports(self):
        [verdict] = self.probe.verdicts(["a wonderful and heartwarming film"], ["pos"])
        self.assertIsInstance(verdict, ProbeVerdict)
        self.assertTrue(verdict.backs_given)
        self.assertEqual(verdict.top_label, "pos")
        self.assertGreater(verdict.given_prob, 0.5)

    def test_rejects_a_label_the_corpus_contradicts(self):
        # The text is plainly negative but carries the positive label: the probe should side against it.
        [verdict] = self.probe.verdicts(["a dreadful boring tedious film"], ["pos"])
        self.assertFalse(verdict.backs_given)
        self.assertEqual(verdict.top_label, "neg")
        self.assertLess(verdict.given_prob, 0.5)

    def test_given_prob_is_the_probability_of_the_given_label_not_the_top_one(self):
        text = "a dreadful boring tedious film"
        [as_pos] = self.probe.verdicts([text], ["pos"])
        [as_neg] = self.probe.verdicts([text], ["neg"])
        # Same text, same model: only which column is read changes, and the two must sum to 1.
        self.assertAlmostEqual(as_pos.given_prob + as_neg.given_prob, 1.0, places=6)
        self.assertEqual(as_pos.top_label, as_neg.top_label)

    def test_label_absent_from_the_reference_corpus_scores_zero(self):
        [verdict] = self.probe.verdicts(["a wonderful film"], ["surprise"])
        self.assertEqual(verdict.given_prob, 0.0)
        self.assertFalse(verdict.backs_given)
        self.assertIn(verdict.top_label, {"pos", "neg"})

    def test_batch_preserves_input_order(self):
        verdicts = self.probe.verdicts(
            ["a wonderful heartwarming film", "a dreadful tedious film"], ["pos", "neg"]
        )
        self.assertEqual([v.top_label for v in verdicts], ["pos", "neg"])
        self.assertTrue(all(v.backs_given for v in verdicts))

    def test_empty_input_returns_empty_without_fitting(self):
        probe = LabelProbe(TEXTS, LABELS)
        self.assertEqual(probe.verdicts([], []), [])
        self.assertIsNone(probe._pipeline)

    def test_mismatched_batch_lengths_raise(self):
        with self.assertRaises(ValueError) as ctx:
            self.probe.verdicts(["a", "b"], ["pos"])
        self.assertIn("must match", str(ctx.exception))

    def test_verdict_is_frozen(self):
        [verdict] = self.probe.verdicts(["a wonderful film"], ["pos"])
        with self.assertRaises(Exception):
            verdict.given_prob = 0.0  # type: ignore[misc]


class TestFit(unittest.TestCase):
    """Fitting is lazy and happens once."""

    def test_fit_is_idempotent_and_reuses_the_pipeline(self):
        probe = LabelProbe(TEXTS, LABELS)
        self.assertIsNone(probe._pipeline)
        probe.fit()
        first = probe._pipeline
        probe.fit()
        self.assertIs(probe._pipeline, first)

    def test_verdicts_fits_on_demand(self):
        probe = LabelProbe(TEXTS, LABELS)
        probe.verdicts(["a wonderful film"], ["pos"])
        self.assertIsNotNone(probe._pipeline)

    def test_probe_is_independent_of_any_model(self):
        # The whole point: nothing about the audited model reaches the probe. Constructing and
        # scoring must work with no model, backend or explainer in sight.
        probe = LabelProbe(TEXTS, LABELS)
        [verdict] = probe.verdicts(["wonderful delightful"], ["pos"])
        self.assertTrue(verdict.backs_given)


if __name__ == "__main__":
    unittest.main()
