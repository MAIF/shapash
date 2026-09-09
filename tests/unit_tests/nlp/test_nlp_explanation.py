"""Unit tests for ``NlpExplanation`` — the ``explain()`` return value and its persistence.

Covers the round-trip (save -> load) across the shape matrix that matters: 1-D (binary/
regression) vs 2-D (multi-class) contribution arrays, with/without a baseline, with/without
ground truth and probabilities, with/without a bundled scatter projection, and a sample with
zero tokens (the edge case that broke a naive "infer counts from the tidy tables" design —
see ``_n_classes``/``_frames_to_contributions`` in the module under test).
"""

import json
import tempfile
import unittest
import unittest.mock
import zipfile
import dataclasses
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from shapash.explainer.nlp_explanation import NlpExplanation


def _make_explanation(values_ndim: int, with_base: bool, with_true: bool, with_prob: bool) -> NlpExplanation:
    texts = pd.Series(["hello world", "i am happy today", "ok"], index=[10, 11, 12])
    label_names = ["neg", "pos"] if values_ndim == 2 else None

    if values_ndim == 2:
        values = [
            np.array([[1.0, -1.0], [2.0, -2.0]]),
            np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
            np.zeros((0, 2)),  # the third sample has zero tokens
        ]
        base_values = np.array([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]) if with_base else None
    else:
        values = [np.array([1.0, 2.0]), np.array([0.1, 0.3, 0.5]), np.zeros((0,))]
        base_values = np.array([0.1, 0.2, 0.3]) if with_base else None

    y_pred = pd.Series(["pos", "neg", "pos"], index=texts.index, name="prediction")
    y_prob = pd.DataFrame({"neg": [0.1, 0.8, 0.3], "pos": [0.9, 0.2, 0.7]}, index=texts.index) if with_prob else None
    y_true = pd.Series(["pos", "pos", "pos"], index=texts.index, name="ground_truth") if with_true else None

    return NlpExplanation(
        texts=texts,
        token_strings=[["hello", "world"], ["i", "am", "happy"], []],
        values=values,
        base_values=base_values,
        y_pred=y_pred,
        y_prob=y_prob,
        y_true=y_true,
        label_names=label_names,
        folds_case=True,
        backend_name="nlp_shap",
        is_additive=True,
        reference_kind="none",
        output_space="probability",
    )


class TestNlpExplanationRoundTrip(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

    def _round_trip(self, expl: NlpExplanation, scatter_xy=None) -> tuple[NlpExplanation, np.ndarray | None]:
        path = self.tmp_path / "explanation.zip"
        expl.save(path, scatter_xy=scatter_xy)
        return NlpExplanation.load(path)

    def _assert_round_trips(self, expl: NlpExplanation, scatter_xy=None):
        loaded, loaded_scatter = self._round_trip(expl, scatter_xy=scatter_xy)

        for original, restored in zip(expl.values, loaded.values, strict=True):
            np.testing.assert_allclose(original, restored)
        self.assertEqual(expl.token_strings, loaded.token_strings)

        if expl.base_values is None:
            self.assertIsNone(loaded.base_values)
        else:
            np.testing.assert_allclose(expl.base_values, loaded.base_values)

        pd.testing.assert_series_equal(expl.texts, loaded.texts, check_names=False)
        pd.testing.assert_series_equal(expl.y_pred, loaded.y_pred, check_names=False)

        if expl.y_true is None:
            self.assertIsNone(loaded.y_true)
        else:
            pd.testing.assert_series_equal(expl.y_true, loaded.y_true, check_names=False)

        if expl.y_prob is None:
            self.assertIsNone(loaded.y_prob)
        else:
            pd.testing.assert_frame_equal(expl.y_prob, loaded.y_prob)

        self.assertEqual(loaded.backend_name, expl.backend_name)
        self.assertEqual(loaded.is_additive, expl.is_additive)
        self.assertEqual(loaded.reference_kind, expl.reference_kind)
        self.assertEqual(loaded.output_space, expl.output_space)
        self.assertEqual(loaded.label_names, expl.label_names)
        self.assertEqual(loaded.folds_case, expl.folds_case)

        if scatter_xy is None:
            self.assertIsNone(loaded_scatter)
        else:
            np.testing.assert_allclose(loaded_scatter, scatter_xy)

    def test_multiclass_with_base_ground_truth_probabilities_and_scatter(self):
        expl = _make_explanation(values_ndim=2, with_base=True, with_true=True, with_prob=True)
        scatter = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        self._assert_round_trips(expl, scatter_xy=scatter)

    def test_binary_without_ground_truth_or_probabilities(self):
        expl = _make_explanation(values_ndim=1, with_base=True, with_true=False, with_prob=False)
        self._assert_round_trips(expl)

    def test_multiclass_without_base_values(self):
        expl = _make_explanation(values_ndim=2, with_base=False, with_true=True, with_prob=True)
        self._assert_round_trips(expl)

    def test_binary_without_base_values_ground_truth_or_probabilities(self):
        expl = _make_explanation(values_ndim=1, with_base=False, with_true=False, with_prob=False)
        self._assert_round_trips(expl)

    def test_meta_json_is_plain_text_readable_without_shapash(self):
        expl = _make_explanation(values_ndim=2, with_base=True, with_true=True, with_prob=True)
        path = self.tmp_path / "explanation.zip"
        expl.save(path)
        with zipfile.ZipFile(path) as zf:
            meta = json.loads(zf.read("meta.json"))
        self.assertEqual(meta["backend_name"], "nlp_shap")
        self.assertTrue(meta["is_additive"])
        self.assertEqual(meta["reference_kind"], "none")
        self.assertEqual(meta["output_space"], "probability")
        self.assertEqual(meta["label_names"], ["neg", "pos"])
        self.assertEqual(meta["n_samples"], 3)
        self.assertIn("shapash_version", meta)
        self.assertIn("created_at", meta)

    def test_unsupported_format_version_raises_actionable_error(self):
        expl = _make_explanation(values_ndim=2, with_base=True, with_true=True, with_prob=True)
        path = self.tmp_path / "explanation.zip"
        expl.save(path)

        with zipfile.ZipFile(path) as zin:
            meta = json.loads(zin.read("meta.json"))
            meta["format_version"] = 999
            members = {item.filename: zin.read(item.filename) for item in zin.infolist()}
        members["meta.json"] = json.dumps(meta).encode()

        bad_path = self.tmp_path / "bad_version.zip"
        with zipfile.ZipFile(bad_path, "w") as zout:
            for name, data in members.items():
                zout.writestr(name, data)

        with self.assertRaises(ValueError) as ctx:
            NlpExplanation.load(bad_path)
        self.assertIn("format_version", str(ctx.exception))
        self.assertIn("999", str(ctx.exception))

    def _resave_without_output_space(self, expl: NlpExplanation) -> Path:
        """A file as an earlier shapash (pre-``output_space``) would have written it."""
        path = self.tmp_path / "legacy.zip"
        expl.save(path)
        with zipfile.ZipFile(path) as zin:
            meta = json.loads(zin.read("meta.json"))
            del meta["output_space"]
            members = {item.filename: zin.read(item.filename) for item in zin.infolist()}
        members["meta.json"] = json.dumps(meta).encode()
        legacy_path = self.tmp_path / "legacy_no_output_space.zip"
        with zipfile.ZipFile(legacy_path, "w") as zout:
            for name, data in members.items():
                zout.writestr(name, data)
        return legacy_path

    def test_legacy_file_without_output_space_defaults_by_backend(self):
        # nlp_captum_lig has always explained raw logits, everything else probabilities — a single
        # global default would mislabel one of the two. See docs/architecture/explanation-space.md §5.3.
        shap_expl = _make_explanation(values_ndim=2, with_base=True, with_true=True, with_prob=True)
        lig_expl = replace(shap_expl, backend_name="nlp_captum_lig")

        loaded_shap, _ = NlpExplanation.load(self._resave_without_output_space(shap_expl))
        loaded_lig, _ = NlpExplanation.load(self._resave_without_output_space(lig_expl))

        self.assertEqual(loaded_shap.output_space, "probability")
        self.assertEqual(loaded_lig.output_space, "logit")


if __name__ == "__main__":
    unittest.main()


class TestNlpExplanationDescriptors(unittest.TestCase):
    """Batch descriptors the webapp reads straight off the artifact.

    These four used to live on a separate ``NlpView`` wrapper that held no state of its own —
    every display choice already lives in the webapp's ``dcc.Store``s, so the wrapper only
    forwarded attribute reads and the app routed around it to reach the explanation anyway.
    They are derivations of the data, so they belong with the data.
    """

    def test_counts_and_ground_truth_flag(self):
        explanation = _make_explanation(values_ndim=2, with_base=True, with_true=True, with_prob=True)
        self.assertEqual(explanation.n_samples, 3)
        self.assertEqual(explanation.n_classes, 2)
        self.assertTrue(explanation.has_ground_truth)

    def test_has_ground_truth_is_false_without_y_true(self):
        explanation = _make_explanation(values_ndim=2, with_base=True, with_true=False, with_prob=True)
        self.assertFalse(explanation.has_ground_truth)

    def test_n_classes_is_one_for_binary_1d_values(self):
        explanation = _make_explanation(values_ndim=1, with_base=True, with_true=True, with_prob=True)
        self.assertEqual(explanation.n_classes, 1)

    def test_label_to_idx_follows_label_names_order(self):
        explanation = _make_explanation(values_ndim=2, with_base=True, with_true=True, with_prob=True)
        self.assertEqual(explanation.label_to_idx, {"neg": 0, "pos": 1})

    def test_label_to_idx_falls_back_to_column_indices_without_names(self):
        explanation = replace(
            _make_explanation(values_ndim=2, with_base=True, with_true=True, with_prob=True), label_names=None
        )
        self.assertEqual(explanation.label_to_idx, {"0": 0, "1": 1})

    def test_n_classes_survives_an_all_empty_batch(self):
        """The old wrapper read ``values[0].shape[1]``, which raised here; ``_n_classes`` does not."""
        explanation = replace(
            _make_explanation(values_ndim=2, with_base=False, with_true=False, with_prob=False),
            token_strings=[[], [], []],
            values=[np.zeros((0, 2)) for _ in range(3)],
        )
        self.assertEqual(explanation.n_classes, 2)  # recovered from label_names


class TestFieldPartition(unittest.TestCase):
    """The caller/computed split that makes memoizing on text content alone sound."""

    def test_every_field_is_classified(self):
        # Mirrors the import-time guard at the bottom of nlp_explanation.py, so the reason for
        # the failure is legible from the test suite and not only from a broken import.
        names = {f.name for f in dataclasses.fields(NlpExplanation)}
        classified = NlpExplanation._CALLER_FIELDS | NlpExplanation._COMPUTED_FIELDS
        self.assertEqual(names, classified)

    def test_the_two_halves_do_not_overlap(self):
        self.assertEqual(NlpExplanation._CALLER_FIELDS & NlpExplanation._COMPUTED_FIELDS, frozenset())

    def test_relabelled_refuses_a_caller_field_it_does_not_produce(self):
        # The import-time guard only establishes that a field has been *classified*. This is the
        # second half: a field declared caller-owned but not produced in relabelled() would be
        # inherited from the cached run, which is the exact bug the method exists to prevent.
        exp = _make_explanation(values_ndim=2, with_base=True, with_true=True, with_prob=True)
        with unittest.mock.patch.object(
            NlpExplanation, "_CALLER_FIELDS", NlpExplanation._CALLER_FIELDS | {"label_names"}
        ):
            with self.assertRaises(ValueError) as ctx:
                exp.relabelled(exp.texts.set_axis(["a", "b", "c"]))
        self.assertIn("label_names", str(ctx.exception))
        self.assertIn("not produced here", str(ctx.exception))

    def test_relabelled_moves_every_caller_field_and_no_computed_one(self):
        exp = _make_explanation(values_ndim=2, with_base=True, with_true=True, with_prob=True)
        texts = exp.texts.set_axis(["a", "b", "c"])
        y_true = pd.Series(["neg", "pos", "neg"], index=["a", "b", "c"])

        out = exp.relabelled(texts, y_true=y_true)

        for name in NlpExplanation._CALLER_FIELDS:
            field = getattr(out, name)
            if field is not None:
                self.assertTrue(field.index.equals(texts.index), f"{name} kept the old index")
        self.assertIs(out.y_true, y_true)
        for name in NlpExplanation._COMPUTED_FIELDS:
            self.assertIs(getattr(out, name), getattr(exp, name), f"{name} was not carried through")


class TestImmutability(unittest.TestCase):
    """The artifact is immutable in depth, and ``replace`` is the supported way to vary it.

    These are not style assertions. The whole display-state design rests on "what the webapp and
    the plotter render is what ``explain()`` computed", and two of the fields below are read as
    correctness guards (:attr:`is_additive` gates the waterfall, ``y_true`` gates the confusion
    matrix) — a guard a caller can switch off by assignment is not a guard.
    """

    def setUp(self):
        self.explanation = _make_explanation(values_ndim=2, with_base=True, with_true=True, with_prob=True)

    def test_rebinding_a_field_is_refused(self):
        for field_name, value in (
            ("is_additive", False),  # the waterfall guard
            ("backend_name", "something-else"),
            ("label_names", None),
            ("y_true", None),  # the confusion-matrix guard
        ):
            with self.subTest(field=field_name), self.assertRaises(dataclasses.FrozenInstanceError):
                setattr(self.explanation, field_name, value)

    def test_contribution_arrays_are_sealed_against_in_place_writes(self):
        with self.assertRaises(ValueError):
            self.explanation.values[0][0, 0] = 999.0
        with self.assertRaises(ValueError):
            self.explanation.base_values[0, 0] = 999.0

    def test_replace_is_the_supported_way_to_vary_a_field(self):
        derived = replace(self.explanation, label_names=["a", "b"])
        self.assertEqual(derived.label_names, ["a", "b"])
        self.assertEqual(self.explanation.label_names, ["neg", "pos"])  # original untouched
        # replace() re-runs __post_init__, so the derived artifact is sealed too.
        with self.assertRaises(ValueError):
            derived.values[0][0, 0] = 1.0

    def test_the_regex_constant_is_not_a_field(self):
        """A ClassVar, so it stays out of ``__init__``, ``replace()`` and the saved payload."""
        self.assertNotIn("_SPECIAL_RE", {f.name for f in dataclasses.fields(NlpExplanation)})

    def test_identity_equality_replaces_an__eq__that_could_only_raise(self):
        other = _make_explanation(values_ndim=2, with_base=True, with_true=True, with_prob=True)
        self.assertEqual(self.explanation, self.explanation)
        self.assertNotEqual(self.explanation, other)
        self.assertEqual(len({self.explanation, other}), 2)  # hashable again, by identity


class TestSharedIndexInvariant(unittest.TestCase):
    """``texts``, ``y_pred``, ``y_prob`` and ``y_true`` must carry one index.

    Enforced in ``__post_init__`` rather than at the call sites because ``replace`` — the mandated
    way to vary a frozen artifact — says nothing about the fields it is not handed. An artifact
    whose halves are indexed differently raises nowhere on its own: pandas alignment yields
    all-NaN and :meth:`confusion_matrix` zips positionally, reporting plausible wrong counts.
    """

    def setUp(self):
        self.explanation = _make_explanation(values_ndim=2, with_base=True, with_true=True, with_prob=True)

    def test_replacing_texts_alone_is_refused(self):
        """The shape of the real bug: patch one index-bearing field, leave the rest stale."""
        shifted = pd.Series(self.explanation.texts.to_numpy(), index=[77, 88, 99])
        with self.assertRaises(ValueError) as ctx:
            replace(self.explanation, texts=shifted)
        self.assertIn("y_pred", str(ctx.exception))

    def test_each_labelled_field_is_checked(self):
        for field_name in ("y_pred", "y_prob", "y_true"):
            original = getattr(self.explanation, field_name)
            with self.subTest(field=field_name), self.assertRaises(ValueError) as ctx:
                replace(self.explanation, **{field_name: original.set_axis([77, 88, 99])})
            self.assertIn(field_name, str(ctx.exception))

    def test_replacing_every_index_bearing_field_together_is_allowed(self):
        index = pd.Index([77, 88, 99])
        derived = replace(
            self.explanation,
            texts=self.explanation.texts.set_axis(index),
            y_pred=self.explanation.y_pred.set_axis(index),
            y_prob=self.explanation.y_prob.set_axis(index),
            y_true=self.explanation.y_true.set_axis(index),
        )
        self.assertTrue(derived.y_pred.index.equals(index))

    def test_absent_optional_fields_are_skipped_not_faulted(self):
        bare = _make_explanation(values_ndim=2, with_base=False, with_true=False, with_prob=False)
        self.assertIsNone(bare.y_true)
        self.assertIsNone(bare.y_prob)

    def test_a_saved_artifact_reloads_with_one_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.zip"
            self.explanation.save(path)
            restored, _ = NlpExplanation.load(path)
        self.assertTrue(restored.y_pred.index.equals(restored.texts.index))
        self.assertTrue(restored.y_true.index.equals(restored.texts.index))
