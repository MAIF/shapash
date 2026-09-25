"""Unit tests for ``NlpExplanation`` — the ``explain()`` return value and its persistence.

Covers the round-trip (save -> load) across the shape matrix that matters: 1-D (binary/
regression) vs 2-D (multi-class) contribution arrays, with/without a baseline, with/without
ground truth and probabilities, and a sample with
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

from shapash.explainer.nlp_explanation import (
    WORD_AGGREGATIONS,
    NlpExplanation,
    aggregate_word_contributions,
    rank_word_samples,
    word_contributions_by_sample,
)
from shapash.plots.plot_word_importance import plot_word_importance


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

    def _round_trip(self, expl: NlpExplanation) -> NlpExplanation:
        path = self.tmp_path / "explanation.zip"
        expl.save(path)
        return NlpExplanation.load(path)

    def _assert_round_trips(self, expl: NlpExplanation):
        loaded = self._round_trip(expl)

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

        # Survives the round trip because the texts do: it is derived, never stored-and-restored.
        self.assertEqual(loaded.corpus_id, expl.corpus_id)

    def test_multiclass_with_base_ground_truth_and_probabilities(self):
        expl = _make_explanation(values_ndim=2, with_base=True, with_true=True, with_prob=True)
        self._assert_round_trips(expl)

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

        loaded_shap = NlpExplanation.load(self._resave_without_output_space(shap_expl))
        loaded_lig = NlpExplanation.load(self._resave_without_output_space(lig_expl))

        self.assertEqual(loaded_shap.output_space, "probability")
        self.assertEqual(loaded_lig.output_space, "logit")


class TestCorpusIdentity(unittest.TestCase):
    """``corpus_id`` — the texts-only digest that pairs an explanation with its embeddings.

    It has to depend on the texts and on *nothing else*, because that is what makes two artifacts
    over one dataset recognisable as such: a SHAP and a Captum explanation of the same batch, an
    ``Embedding`` of it, a projection of that embedding.
    """

    def test_same_texts_agree_across_backend_and_model(self):
        shap_expl = _make_explanation(values_ndim=2, with_base=True, with_true=True, with_prob=True)
        lig_expl = replace(shap_expl, backend_name="nlp_captum_lig", output_space="logit", model_id="other/model")
        self.assertEqual(shap_expl.corpus_id, lig_expl.corpus_id)

    def test_different_texts_disagree(self):
        expl = _make_explanation(values_ndim=2, with_base=True, with_true=True, with_prob=True)
        other = replace(expl, texts=pd.Series(["hello world", "i am happy today", "KO"], index=expl.texts.index))
        self.assertNotEqual(expl.corpus_id, other.corpus_id)

    def test_order_is_part_of_the_identity(self):
        # Row i of a projection means "the projection of texts[i]", so a reordered corpus is a
        # different corpus even though it holds the same strings.
        expl = _make_explanation(values_ndim=2, with_base=True, with_true=True, with_prob=True)
        shuffled = replace(expl, texts=pd.Series(expl.texts.tolist()[::-1], index=expl.texts.index))
        self.assertNotEqual(expl.corpus_id, shuffled.corpus_id)

    def test_reindexing_does_not_change_it(self):
        # relabelled() rebinds texts to the caller's index without touching the strings; an id that
        # moved here would break the pairing for every explanation served out of the explain cache.
        expl = _make_explanation(values_ndim=2, with_base=True, with_true=True, with_prob=True)
        reindexed = pd.Series(expl.texts.tolist(), index=[0, 1, 2])
        derived = expl.relabelled(texts=reindexed, y_true=None)
        self.assertEqual(derived.corpus_id, expl.corpus_id)


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

    def test_repr_is_a_one_line_summary_not_the_dataclass_default(self):
        # The generated dataclass repr would print every text, token and contribution array in
        # full — a notebook cell that just names the variable must not dump the whole batch.
        explanation = _make_explanation(values_ndim=2, with_base=True, with_true=True, with_prob=True)
        repr_str = repr(explanation)
        self.assertEqual(
            repr_str, "NlpExplanation(n_samples=3, n_classes=2, backend='nlp_shap', has_ground_truth=True)"
        )

    def test_repr_reports_has_ground_truth_false_when_absent(self):
        explanation = _make_explanation(values_ndim=2, with_base=True, with_true=False, with_prob=True)
        self.assertIn("has_ground_truth=False", repr(explanation))


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
            restored = NlpExplanation.load(path)
        self.assertTrue(restored.y_pred.index.equals(restored.texts.index))
        self.assertTrue(restored.y_true.index.equals(restored.texts.index))


# ---------------------------------------------------------------------------
# NlpExplanation.word_importance / resolve_lowercase / vocabulary / word_counts
# / word_occurrences / aggregate_word_contributions / rank_word_samples /
# word_contributions_by_sample
# ---------------------------------------------------------------------------

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
N_CLASSES = len(LABEL_NAMES)


def _make_word_token_data() -> tuple[list[list[str]], list[np.ndarray], np.ndarray]:
    """Synthetic per-token contribution data for 3 samples, 6 classes."""
    rng = np.random.default_rng(42)
    token_strings = [
        ["", "i", "feel", "so", "happy", "today", ""],
        ["", "this", "is", "terrible", "and", "sad", ""],
        ["", "what", "a", "wonderful", "day", ""],
    ]
    values = [rng.uniform(-0.4, 0.4, size=(len(t), N_CLASSES)).astype(np.float32) for t in token_strings]
    base_values = rng.uniform(-0.1, 0.1, size=(3, N_CLASSES)).astype(np.float32)
    return token_strings, values, base_values


def _minimal_word_explanation(
    token_strings: list[list[str]],
    values: list[np.ndarray],
    base_values: np.ndarray | None,
    folds_case: bool | None = None,
) -> NlpExplanation:
    """Bare ``NlpExplanation`` for word-importance/case-folding tests — only the token data matters."""
    n = len(token_strings)
    texts = pd.Series([""] * n, index=pd.RangeIndex(n))
    return NlpExplanation(
        texts=texts,
        token_strings=token_strings,
        values=values,
        base_values=base_values,
        y_pred=pd.Series([""] * n, index=texts.index, name="prediction"),
        y_prob=None,
        y_true=None,
        label_names=None,
        folds_case=folds_case,
        backend_name="nlp_shap",
        is_additive=True,
        reference_kind="none",
        output_space="probability",
    )


class TestWordImportance(unittest.TestCase):
    def setUp(self):
        token_strings, values, base_values = _make_word_token_data()
        self.explanation = _minimal_word_explanation(token_strings, values, base_values)

    def test_len(self):
        self.assertEqual(len(self.explanation), 3)

    def test_word_importance_returns_series(self):
        imp = self.explanation.word_importance(label_idx=1)
        self.assertIsInstance(imp, pd.Series)

    def test_word_importance_filters_special_tokens(self):
        imp = self.explanation.word_importance(label_idx=0, filter_special=True)
        self.assertNotIn("", imp.index)
        self.assertNotIn(" ", imp.index)

    def test_word_importance_keeps_special_when_disabled(self):
        imp = self.explanation.word_importance(label_idx=0, filter_special=False)
        # Empty strings (BOS/EOS) should now be present
        self.assertIn("", imp.index)

    def test_word_importance_hides_punctuation_by_default(self):
        explanation = _minimal_word_explanation(
            token_strings=[["great", "!", "!", "movie", ".", ","]],
            values=[np.array([[3.0], [9.0], [9.0], [2.0], [8.0], [7.0]])],
            base_values=np.zeros((1, 1)),
        )
        imp = explanation.word_importance(label_idx=0, n_top=20)
        self.assertEqual(sorted(imp.index), ["great", "movie"])

    def test_word_importance_keeps_punctuation_when_asked(self):
        explanation = _minimal_word_explanation(
            token_strings=[["great", "!", "!"]],
            values=[np.array([[3.0], [9.0], [9.0]])],
            base_values=np.zeros((1, 1)),
        )
        imp = explanation.word_importance(label_idx=0, n_top=20, filter_punctuation=False)
        self.assertIn("!", imp.index)
        # Punctuation kept means it can outrank real words, which is why it is hidden by default.
        self.assertEqual(imp.index[0], "!")

    def test_word_importance_keeps_words_containing_punctuation(self):
        # Only *pure* punctuation units are dropped — a hyphenated or apostrophised word stays.
        explanation = _minimal_word_explanation(
            token_strings=[["state-of-the-art", "don't", "-"]],
            values=[np.array([[3.0], [2.0], [9.0]])],
            base_values=np.zeros((1, 1)),
        )
        imp = explanation.word_importance(label_idx=0, n_top=20)
        self.assertEqual(sorted(imp.index), ["don't", "state-of-the-art"])

    def test_resolve_lowercase_follows_the_model(self):
        for folds_case, expected in ((True, True), (False, False), (None, True)):
            with self.subTest(folds_case=folds_case):
                explanation = replace(self.explanation, folds_case=folds_case)
                self.assertEqual(explanation.resolve_lowercase(), expected)

    def test_resolve_lowercase_explicit_argument_wins(self):
        self.assertFalse(replace(self.explanation, folds_case=True).resolve_lowercase(False))
        self.assertTrue(replace(self.explanation, folds_case=False).resolve_lowercase(True))

    def test_word_importance_keeps_case_on_a_cased_model(self):
        # A cased tokenizer encodes AWFUL and awful to *different* ids, so they are different
        # inputs with genuinely different attributions — merging them would hide that.
        explanation = _minimal_word_explanation(
            token_strings=[["AWFUL", "awful"]],
            values=[np.array([[-9.0], [-3.0]])],
            base_values=np.zeros((1, 1)),
            folds_case=False,
        )
        imp = explanation.word_importance(label_idx=0, n_top=20)
        self.assertEqual(sorted(imp.index), ["AWFUL", "awful"])

    def test_word_importance_merges_case_on_an_uncased_model(self):
        # An uncased tokenizer maps both spellings to one input id — the model cannot tell them
        # apart, so they must not appear as two rows.
        explanation = _minimal_word_explanation(
            token_strings=[["AWFUL", "awful"]],
            values=[np.array([[-9.0], [-3.0]])],
            base_values=np.zeros((1, 1)),
            folds_case=True,
        )
        imp = explanation.word_importance(label_idx=0, n_top=20)
        self.assertEqual(list(imp.index), ["awful"])
        self.assertAlmostEqual(imp["awful"], -6.0)

    def test_word_importance_merges_case_variants_by_default(self):
        # One row per word, averaged over every casing — not three rows of one occurrence each.
        explanation = _minimal_word_explanation(
            token_strings=[["AWFUL", "Awful", "awful", "good"]],
            values=[np.array([[-9.0], [-3.0], [-3.0], [1.0]])],
            base_values=np.zeros((1, 1)),
        )
        imp = explanation.word_importance(label_idx=0, n_top=20)
        self.assertEqual(sorted(imp.index), ["awful", "good"])
        self.assertAlmostEqual(imp["awful"], -5.0)

    def test_word_importance_case_fragmentation_lets_rare_variants_outrank(self):
        # The reason lowercasing is the default: a single capitalised occurrence keeps its own
        # extreme value instead of being averaged into the 3 common ones, and tops the ranking.
        explanation = _minimal_word_explanation(
            token_strings=[["TERRIBLE", *["terrible"] * 9, "dull"]],
            values=[np.array([[-9.0], *[[-0.1]] * 9, [-2.0]])],
            base_values=np.zeros((1, 1)),
        )
        cased = explanation.word_importance(label_idx=0, n_top=20, lowercase=False)
        self.assertEqual(cased.index[0], "TERRIBLE")
        folded = explanation.word_importance(label_idx=0, n_top=20)
        self.assertEqual(folded.index[0], "dull")

    def test_word_importance_keeps_case_when_disabled(self):
        explanation = _minimal_word_explanation(
            token_strings=[["AWFUL", "awful"]],
            values=[np.array([[-9.0], [-3.0]])],
            base_values=np.zeros((1, 1)),
        )
        imp = explanation.word_importance(label_idx=0, n_top=20, lowercase=False)
        self.assertEqual(sorted(imp.index), ["AWFUL", "awful"])

    def test_word_importance_exclusion_is_case_insensitive_when_folding(self):
        # The webapp's dropdown offers lowercase entries; excluding one must drop every casing.
        explanation = _minimal_word_explanation(
            token_strings=[["AWFUL", "awful", "good"]],
            values=[np.array([[-9.0], [-3.0], [1.0]])],
            base_values=np.zeros((1, 1)),
        )
        imp = explanation.word_importance(label_idx=0, n_top=20, exclude_words={"AwFuL"})
        self.assertEqual(list(imp.index), ["good"])

    def test_word_importance_respects_n_top(self):
        imp = self.explanation.word_importance(label_idx=1, n_top=3)
        self.assertLessEqual(len(imp), 3)

    def test_word_importance_sorted_by_absolute_value(self):
        imp = self.explanation.word_importance(label_idx=2, n_top=20)
        abs_vals = imp.abs().tolist()
        self.assertEqual(abs_vals, sorted(abs_vals, reverse=True))

    def test_word_importance_aggregates_repeated_words(self):
        # "feel" only appears once; check it has a single contribution value
        imp = self.explanation.word_importance(label_idx=1, n_top=20, filter_special=True)
        self.assertIn("feel", imp.index)
        # Should be a scalar (mean of one occurrence)
        self.assertIsInstance(imp["feel"], float)

    def test_word_importance_all_labels(self):
        for idx in range(N_CLASSES):
            imp = self.explanation.word_importance(label_idx=idx)
            self.assertIsInstance(imp, pd.Series)
            self.assertGreater(len(imp), 0)

    def test_word_importance_filter_sign_positive(self):
        imp = self.explanation.word_importance(label_idx=0, filter_sign="positive")
        if len(imp) > 0:
            self.assertTrue((imp > 0).all(), "positive filter should return only positive values")

    def test_word_importance_filter_sign_negative(self):
        imp = self.explanation.word_importance(label_idx=0, filter_sign="negative")
        if len(imp) > 0:
            self.assertTrue((imp < 0).all(), "negative filter should return only negative values")

    def test_word_importance_exclude_words(self):
        imp_full = self.explanation.word_importance(label_idx=1, filter_special=True)
        if len(imp_full) == 0:
            return
        word_to_exclude = imp_full.index[0]
        imp_filtered = self.explanation.word_importance(
            label_idx=1, filter_special=True, exclude_words={word_to_exclude}
        )
        self.assertNotIn(word_to_exclude, imp_filtered.index)

    def test_word_importance_exclude_words_empty_set(self):
        imp_no_exclude = self.explanation.word_importance(label_idx=1, exclude_words=set())
        imp_none_exclude = self.explanation.word_importance(label_idx=1, exclude_words=None)
        pd.testing.assert_series_equal(imp_no_exclude, imp_none_exclude)

    def test_word_importance_sample_indices(self):
        imp = self.explanation.word_importance(label_idx=0, sample_indices=[0], n_top=50)
        self.assertGreater(len(imp), 0)
        # "terrible" only exists in sample 1 — must not appear in the sample-0 subset
        self.assertNotIn("terrible", imp.index)


# ---------------------------------------------------------------------------
# Single-word profile: word_occurrences / vocabulary / aggregators
# ---------------------------------------------------------------------------


def _word_profile_explanation(folds_case=True):
    """Three samples, two classes, with 'happy' appearing in three of them (twice in one)."""
    token_strings = [
        ["[CLS]", "so", "happy", "today", "!"],
        ["Happy", "and", "happy", "again"],
        ["not", "happy", "at", "all"],
        ["nothing", "here"],
    ]
    values = [
        np.array([[0.0, 0.0], [0.1, -0.1], [0.4, -0.4], [0.05, -0.05], [0.0, 0.0]]),
        np.array([[0.2, -0.2], [0.0, 0.0], [0.1, -0.1], [0.0, 0.0]]),
        np.array([[-0.1, 0.1], [-0.6, 0.6], [0.0, 0.0], [0.0, 0.0]]),
        np.array([[0.0, 0.0], [0.0, 0.0]]),
    ]
    texts = pd.Series(["so happy today !", "Happy and happy again", "not happy at all", "nothing here"])
    return NlpExplanation(
        texts=texts,
        token_strings=token_strings,
        values=values,
        base_values=None,
        y_pred=pd.Series(["pos", "pos", "neg", "neg"], index=texts.index, name="prediction"),
        y_prob=None,
        y_true=pd.Series(["pos", "neg", "neg", "neg"], index=texts.index, name="ground_truth"),
        label_names=["pos", "neg"],
        folds_case=folds_case,
        backend_name="nlp_shap",
        is_additive=True,
        reference_kind="none",
        output_space="probability",
    )


class TestVocabulary(unittest.TestCase):
    def test_filters_special_and_punctuation_and_folds(self):
        vocab = _word_profile_explanation().vocabulary()
        self.assertNotIn("[CLS]", vocab)
        self.assertNotIn("!", vocab)
        # Folded: "Happy" and "happy" collapse to one entry.
        self.assertIn("happy", vocab)
        self.assertNotIn("Happy", vocab)

    def test_keeps_case_when_model_does_not_fold(self):
        vocab = _word_profile_explanation(folds_case=False).vocabulary()
        self.assertIn("Happy", vocab)
        self.assertIn("happy", vocab)

    def test_sorted_and_unique(self):
        vocab = _word_profile_explanation().vocabulary()
        self.assertEqual(vocab, sorted(set(vocab)))

    def test_offers_exactly_what_word_importance_ranks(self):
        # The contract the pickers rely on: every unit offered has occurrences to show.
        explanation = _word_profile_explanation()
        for word in explanation.vocabulary():
            self.assertFalse(explanation.word_occurrences(word).empty, word)

    def test_can_keep_punctuation_and_special(self):
        vocab = _word_profile_explanation().vocabulary(filter_special=False, filter_punctuation=False)
        self.assertIn("[cls]", vocab)
        self.assertIn("!", vocab)


class TestWordImportanceRanking(unittest.TestCase):
    """``rank_by`` + ``min_occurrences``: ordering criteria that never touch the reported sign."""

    def setUp(self):
        # "rare" appears once with a large pull; "common" appears four times with a small one.
        # Under |mean| rare wins; under |sum| common wins (4 x 0.3 = 1.2 > 0.9).
        token_strings = [
            ["rare", "common"],
            ["common"],
            ["common", "common"],
            ["mild"],
        ]
        values = [
            np.array([[0.9], [0.3]]),
            np.array([[0.3]]),
            np.array([[0.3], [0.3]]),
            np.array([[-0.5]]),
        ]
        self.explanation = _minimal_word_explanation(token_strings, values, None)

    def test_mean_is_the_default_and_unchanged(self):
        imp = self.explanation.word_importance(label_idx=0)
        self.assertEqual(imp.index[0], "rare")
        self.assertAlmostEqual(imp["common"], 0.3)

    def test_sum_reranks_by_total_mass(self):
        imp = self.explanation.word_importance(label_idx=0, rank_by="sum")
        self.assertEqual(imp.index[0], "common")
        self.assertAlmostEqual(imp["common"], 1.2)

    def test_series_name_records_the_statistic(self):
        self.assertEqual(self.explanation.word_importance(label_idx=0).name, "mean")
        self.assertEqual(self.explanation.word_importance(label_idx=0, rank_by="sum").name, "sum")

    def test_ranking_is_absolute_but_values_stay_signed(self):
        # "mild" is negative and must survive ranking, keeping its sign for the sign filter and
        # for the renderer's red/blue colouring.
        for rank_by in ("mean", "sum"):
            imp = self.explanation.word_importance(label_idx=0, rank_by=rank_by)
            self.assertLess(imp["mild"], 0)

    def test_sign_filter_works_in_both_modes(self):
        for rank_by in ("mean", "sum"):
            pos = self.explanation.word_importance(label_idx=0, rank_by=rank_by, filter_sign="positive")
            neg = self.explanation.word_importance(label_idx=0, rank_by=rank_by, filter_sign="negative")
            self.assertNotIn("mild", pos.index)
            self.assertEqual(list(neg.index), ["mild"])

    def test_min_occurrences_drops_rare_words(self):
        imp = self.explanation.word_importance(label_idx=0, min_occurrences=2)
        self.assertEqual(list(imp.index), ["common"])

    def test_min_occurrences_of_one_is_no_filter(self):
        self.assertEqual(
            set(self.explanation.word_importance(label_idx=0, min_occurrences=1).index),
            {"rare", "common", "mild"},
        )

    def test_min_occurrences_counts_within_the_selection(self):
        # "common" occurs 4x across the batch but only once in sample 1, so a floor of 2 must
        # exclude it there — otherwise a threshold would mean something different per scope.
        imp = self.explanation.word_importance(label_idx=0, sample_indices=[1], min_occurrences=2)
        self.assertTrue(imp.empty)
        self.assertEqual(
            list(self.explanation.word_importance(label_idx=0, sample_indices=[2], min_occurrences=2).index),
            ["common"],
        )

    def test_impossible_floor_returns_an_empty_series_not_an_error(self):
        imp = self.explanation.word_importance(label_idx=0, min_occurrences=999)
        self.assertTrue(imp.empty)
        self.assertEqual(imp.dtype, np.float64)

    def test_unknown_rank_by_raises(self):
        with self.assertRaisesRegex(ValueError, "must be 'mean' or 'sum'"):
            self.explanation.word_importance(label_idx=0, rank_by="median")

    def test_plotter_labels_the_axis_for_the_statistic(self):
        self.assertEqual(
            self.explanation.plot.word_importance(label_idx=0, rank_by="sum").layout.xaxis.title.text,
            "Total SHAP contribution",
        )
        self.assertEqual(
            self.explanation.plot.word_importance(label_idx=0).layout.xaxis.title.text,
            "Mean SHAP contribution",
        )

    def test_plotter_hover_counts_match_the_scope_it_ranked(self):
        # The plotter forwards only the keying arguments to word_counts, so a count on a bar must
        # be the count over the *same* samples the aggregate was taken over.
        fig = self.explanation.plot.word_importance(label_idx=0, sample_indices=[2])
        counts = self.explanation.word_counts(sample_indices=[2])["n_occurrences"]
        drawn = list(fig.data[0].y)
        self.assertEqual([int(c[0]) for c in fig.data[0].customdata], [int(counts[w]) for w in drawn])


class TestWordImportanceAcrossClasses(unittest.TestCase):
    """``label_idx=None``: one ranking spanning every class, on magnitudes."""

    @staticmethod
    def _three_class():
        # "alpha" is decisive for class c only; "beta" is mild and spread. Three classes so max
        # and mean over classes are actually distinguishable (with two they coincide).
        texts = pd.Series(["alpha beta", "alpha beta"])
        rows = np.array([[-0.3, -0.3, 0.6], [0.1, -0.2, 0.1]])
        return NlpExplanation(
            texts=texts,
            token_strings=[["alpha", "beta"], ["alpha", "beta"]],
            values=[rows, rows],
            base_values=None,
            y_pred=pd.Series(["a", "a"], index=texts.index, name="prediction"),
            y_prob=None,
            y_true=None,
            label_names=["a", "b", "c"],
            folds_case=True,
            backend_name="nlp_shap",
            is_additive=True,
            reference_kind="none",
            output_space="probability",
        )

    def test_value_is_the_strongest_classs_magnitude(self):
        # Not the mean over classes, which would read 0.4 and 0.133 — smaller than any real
        # contribution and matching nothing the single-class views show.
        self.assertEqual(self._three_class().word_importance(label_idx=None).to_dict(), {"alpha": 0.6, "beta": 0.2})

    def test_it_matches_what_the_driving_class_shows(self):
        exp = self._three_class()
        self.assertEqual(exp.word_importance(label_idx=None)["alpha"], abs(exp.word_importance(label_idx=2)["alpha"]))

    def test_signed_averaging_across_classes_would_be_zero(self):
        # The reason the collapse discards sign: an explainer of a normalised output cancels
        # across classes, so a signed cross-class mean is a chart of zeros.
        exp = self._three_class()
        per_class = np.array([exp.word_importance(label_idx=i)["alpha"] for i in range(3)])
        self.assertAlmostEqual(float(per_class.mean()), 0.0)
        self.assertGreater(exp.word_importance(label_idx=None)["alpha"], 0.5)

    def test_single_output_model_collapses_to_the_absolute_value(self):
        # A 1-D contribution array has no class axis to reduce over; it still must not raise.
        texts = pd.Series(["alpha beta"])
        exp = NlpExplanation(
            texts=texts,
            token_strings=[["alpha", "beta"]],
            values=[np.array([0.5, -0.2])],
            base_values=None,
            y_pred=pd.Series(["a"], index=texts.index, name="prediction"),
            y_prob=None,
            y_true=None,
            label_names=None,
            folds_case=True,
            backend_name="nlp_shap",
            is_additive=True,
            reference_kind="none",
            output_space="probability",
        )
        self.assertEqual(exp.word_importance(label_idx=None).to_dict(), {"alpha": 0.5, "beta": 0.2})

    def test_every_value_is_a_magnitude(self):
        self.assertTrue((_word_profile_explanation().word_importance(label_idx=None) >= 0).all())

    def test_name_records_the_statistic(self):
        exp = _word_profile_explanation()
        self.assertEqual(exp.word_importance(label_idx=None).name, "mean_across_classes")
        self.assertEqual(exp.word_importance(label_idx=None, rank_by="sum").name, "sum_across_classes")

    def test_a_negative_sign_filter_has_nothing_to_select(self):
        self.assertTrue(_word_profile_explanation().word_importance(label_idx=None, filter_sign="negative").empty)

    def test_the_frequency_floor_still_applies(self):
        exp = _word_profile_explanation()
        self.assertIn("happy", exp.word_importance(label_idx=None, min_occurrences=4).index)
        self.assertNotIn("today", exp.word_importance(label_idx=None, min_occurrences=2).index)

    def test_plotter_titles_and_labels_the_cross_class_view(self):
        fig = _word_profile_explanation().plot.word_importance(label_idx=None)
        self.assertEqual(fig.layout.title.text, "Word importance — all classes")
        self.assertEqual(fig.layout.xaxis.title.text, "Largest |mean SHAP| across classes")


class TestWordImportanceReadability(unittest.TestCase):
    """The chart must name every bar it draws, whatever K is."""

    @staticmethod
    def _imp(n):
        return pd.Series(
            [(-1) ** i * (n - i) / n for i in range(n)],
            index=[f"word{i}" for i in range(n)],
        )

    def test_every_word_gets_a_tick(self):
        # dtick=1 on a categorical axis is what stops plotly thinning labels on a long ranking.
        fig = plot_word_importance(self._imp(50))
        self.assertEqual(fig.layout.yaxis.dtick, 1)
        self.assertEqual(len(set(fig.data[0].y)), 50)

    def test_height_grows_with_the_word_count(self):
        heights = [plot_word_importance(self._imp(n)).layout.height for n in (5, 20, 50)]
        self.assertEqual(heights, sorted(heights))
        # Every word keeps room to render, rather than the chart being squeezed to a fixed panel.
        for n, h in zip((5, 20, 50), heights):
            self.assertGreaterEqual(h / n, 24)

    def test_explicit_height_still_wins(self):
        self.assertEqual(plot_word_importance(self._imp(50), height=300).layout.height, 300)

    def test_value_labels_are_signed_and_on_by_default(self):
        fig = plot_word_importance(pd.Series([0.9, -0.5], index=["a", "b"]))
        self.assertEqual(list(fig.data[0].text), ["-0.500", "+0.900"])  # reversed for drawing
        self.assertEqual(fig.data[0].textposition, "outside")
        # Outside text sits past the bar end and would be cut off at the plot edge without this.
        self.assertFalse(fig.data[0].cliponaxis)

    def test_value_labels_get_headroom(self):
        fig = plot_word_importance(pd.Series([1.0, -0.5], index=["a", "b"]))
        lo, hi = fig.layout.xaxis.range
        self.assertLess(lo, -0.5)
        self.assertGreater(hi, 1.0)

    def test_value_labels_can_be_turned_off(self):
        fig = plot_word_importance(pd.Series([0.9], index=["a"]), show_values=False)
        self.assertIsNone(fig.data[0].text)
        # No outside text means no need to reserve headroom; plotly autoscales.
        self.assertIsNone(fig.layout.xaxis.range)

    def test_degenerate_inputs_autoscale_rather_than_collapse(self):
        # An all-zero (or empty) series has no span to pad, and a [0, 0] range would be invalid.
        self.assertIsNone(plot_word_importance(pd.Series(dtype=float)).layout.xaxis.range)
        self.assertIsNone(plot_word_importance(pd.Series([0.0, 0.0], index=["a", "b"])).layout.xaxis.range)

    def test_tick_font_is_set(self):
        self.assertEqual(plot_word_importance(self._imp(3)).layout.yaxis.tickfont.size, 12)


class TestWordImportanceHover(unittest.TestCase):
    """The hover names the statistic and, when supplied, how many occurrences it aggregated."""

    imp = pd.Series([0.9, -0.5], index=["a", "b"])

    def test_hover_names_the_statistic_on_the_axis(self):
        # The axis title scrolls off a long chart; the hover is where the reader can still check
        # whether they are looking at a mean or a total.
        fig = plot_word_importance(self.imp, x_title="Total SHAP contribution")
        self.assertIn("Total SHAP contribution: %{x:.4f}", fig.data[0].hovertemplate)

    def test_counts_are_reversed_with_the_bars(self):
        fig = plot_word_importance(self.imp, counts={"a": 12, "b": 3})
        self.assertEqual([int(c[0]) for c in fig.data[0].customdata], [3, 12])
        self.assertIn("Occurrences:", fig.data[0].hovertemplate)

    def test_counts_accept_a_series(self):
        fig = plot_word_importance(self.imp, counts=pd.Series({"a": 12, "b": 3}))
        self.assertEqual([int(c[0]) for c in fig.data[0].customdata], [3, 12])

    def test_without_counts_the_hover_is_unchanged(self):
        fig = plot_word_importance(self.imp)
        self.assertIsNone(fig.data[0].customdata)
        self.assertNotIn("Occurrences", fig.data[0].hovertemplate)

    def test_a_partial_mapping_is_dropped_rather_than_shown_as_zero(self):
        # A missing word can only mean the caller counted under different filters, which would
        # misreport every bar — not just the uncovered one.
        fig = plot_word_importance(self.imp, counts={"a": 12})
        self.assertIsNone(fig.data[0].customdata)
        self.assertNotIn("Occurrences", fig.data[0].hovertemplate)


class TestWordCounts(unittest.TestCase):
    """The frequency table behind the word picker's order, labels and threshold."""

    def setUp(self):
        self.explanation = _word_profile_explanation()

    def test_occurrences_and_samples_differ_for_a_repeated_word(self):
        counts = self.explanation.word_counts()
        # "happy" occurs 4x (twice in one sample) across 3 samples.
        self.assertEqual(counts.loc["happy", "n_occurrences"], 4)
        self.assertEqual(counts.loc["happy", "n_samples"], 3)

    def test_sorted_by_frequency_then_alphabetically(self):
        counts = self.explanation.word_counts()
        self.assertEqual(counts.index[0], "happy")
        ties = counts[counts["n_occurrences"] == 1].index.tolist()
        self.assertEqual(ties, sorted(ties))

    def test_applies_the_same_filters_as_word_importance(self):
        counts = self.explanation.word_counts()
        self.assertNotIn("[CLS]", counts.index)
        self.assertNotIn("!", counts.index)
        self.assertNotIn("Happy", counts.index)  # folded into "happy"

    def test_index_is_exactly_the_vocabulary(self):
        self.assertEqual(sorted(self.explanation.word_counts().index), self.explanation.vocabulary())

    def test_scoped_to_sample_indices(self):
        counts = self.explanation.word_counts(sample_indices=[1])
        self.assertEqual(counts.loc["happy", "n_occurrences"], 2)
        self.assertEqual(counts.loc["happy", "n_samples"], 1)

    def test_counts_are_the_denominator_word_importance_filters_on(self):
        # The contract the min-occurrence control depends on: the count shown beside a word is the
        # count its aggregate was computed over.
        counts = self.explanation.word_counts()
        for word, n in counts["n_occurrences"].items():
            kept = self.explanation.word_importance(label_idx=0, n_top=999, min_occurrences=int(n))
            self.assertIn(word, kept.index, word)
            dropped = self.explanation.word_importance(label_idx=0, n_top=999, min_occurrences=int(n) + 1)
            self.assertNotIn(word, dropped.index, word)

    def test_empty_batch_returns_an_empty_frame(self):
        empty = _minimal_word_explanation([[]], [np.zeros((0, 2))], None)
        self.assertTrue(empty.word_counts().empty)
        self.assertEqual(empty.vocabulary(), [])


class TestWordOccurrences(unittest.TestCase):
    def setUp(self):
        self.explanation = _word_profile_explanation()

    def test_one_row_per_occurrence_and_class(self):
        occ = self.explanation.word_occurrences("happy")
        # 4 occurrences (one in sample 0, two in sample 1, one in sample 2) x 2 classes.
        self.assertEqual(len(occ), 8)
        self.assertEqual(list(occ.columns), ["sample", "token_pos", "token", "class_idx", "contribution"])

    def test_case_folding_follows_the_model(self):
        self.assertEqual(len(self.explanation.word_occurrences("HAPPY")), 8)
        cased = _word_profile_explanation(folds_case=False)
        self.assertEqual(len(cased.word_occurrences("happy")), 6)
        self.assertEqual(len(cased.word_occurrences("Happy")), 2)

    def test_explicit_lowercase_overrides(self):
        cased = _word_profile_explanation(folds_case=False)
        self.assertEqual(len(cased.word_occurrences("happy", lowercase=True)), 8)

    def test_token_keeps_original_casing(self):
        occ = self.explanation.word_occurrences("happy")
        self.assertIn("Happy", set(occ["token"]))

    def test_sample_indices_scope(self):
        occ = self.explanation.word_occurrences("happy", sample_indices=[2])
        self.assertEqual(set(occ["sample"]), {2})
        self.assertEqual(len(occ), 2)

    def test_missing_word_returns_typed_empty_frame(self):
        occ = self.explanation.word_occurrences("absent")
        self.assertTrue(occ.empty)
        self.assertEqual(occ["contribution"].dtype, np.float64)
        # An empty frame must still support the numeric work the callers do on it.
        self.assertTrue(aggregate_word_contributions(occ, "mean_abs").empty)
        self.assertTrue(rank_word_samples(occ, class_idx=0).empty)

    def test_binary_1d_values(self):
        explanation = _minimal_word_explanation(
            [["good", "day"], ["good"]],
            [np.array([0.5, 0.1]), np.array([-0.3])],
            None,
        )
        occ = explanation.word_occurrences("good")
        self.assertEqual(set(occ["class_idx"]), {0})
        self.assertEqual(sorted(occ["contribution"].round(3)), [-0.3, 0.5])


class TestAggregateWordContributions(unittest.TestCase):
    def setUp(self):
        self.occ = _word_profile_explanation().word_occurrences("happy")

    def test_mean_is_signed(self):
        stats = aggregate_word_contributions(self.occ, "mean")
        # class 0: (0.4 + 0.2 + 0.1 - 0.6) / 4
        self.assertAlmostEqual(stats[0], 0.025)
        self.assertAlmostEqual(stats[1], -0.025)

    def test_sum_scales_with_frequency(self):
        stats = aggregate_word_contributions(self.occ, "sum")
        self.assertAlmostEqual(stats[0], 0.1)

    def test_abs_forms_expose_the_two_way_word(self):
        signed = aggregate_word_contributions(self.occ, "mean")
        magnitude = aggregate_word_contributions(self.occ, "mean_abs")
        # The whole point of the abs forms: a word averaging to ~0 is not a weak word.
        self.assertLess(abs(signed[0]), 0.05)
        self.assertAlmostEqual(magnitude[0], 0.325)
        self.assertAlmostEqual(aggregate_word_contributions(self.occ, "sum_abs")[0], 1.3)

    def test_index_is_every_class_in_order(self):
        self.assertEqual(list(aggregate_word_contributions(self.occ, "sum").index), [0, 1])

    def test_unknown_agg_raises(self):
        with self.assertRaisesRegex(ValueError, "not one of"):
            aggregate_word_contributions(self.occ, "median")

    def test_all_documented_ops_run(self):
        for agg in WORD_AGGREGATIONS:
            self.assertEqual(len(aggregate_word_contributions(self.occ, agg)), 2)


class TestRankWordSamples(unittest.TestCase):
    def setUp(self):
        self.occ = _word_profile_explanation().word_occurrences("happy")

    def test_occurrences_are_summed_within_a_sample(self):
        ranked = rank_word_samples(self.occ, class_idx=0, order="most")
        row = ranked[ranked["sample"] == 1].iloc[0]
        self.assertAlmostEqual(row["contribution"], 0.3)  # 0.2 + 0.1
        self.assertEqual(row["n_occurrences"], 2)
        # One row per sample, never one per occurrence.
        self.assertEqual(len(ranked), ranked["sample"].nunique())

    def test_most_ranks_positive_first(self):
        ranked = rank_word_samples(self.occ, class_idx=0, order="most")
        self.assertEqual(ranked["sample"].tolist(), [0, 1, 2])

    def test_least_ranks_negative_first(self):
        ranked = rank_word_samples(self.occ, class_idx=0, order="least")
        self.assertEqual(ranked["sample"].iloc[0], 2)

    def test_strongest_ignores_sign(self):
        ranked = rank_word_samples(self.occ, class_idx=0, order="strongest")
        self.assertEqual(ranked["sample"].iloc[0], 2)  # |-0.6| is the largest

    def test_class_idx_selects_the_column(self):
        pos = rank_word_samples(self.occ, class_idx=0, order="most")["contribution"].tolist()
        neg = rank_word_samples(self.occ, class_idx=1, order="most")["contribution"].tolist()
        self.assertNotEqual(pos, neg)

    def test_n_top_truncates(self):
        self.assertEqual(len(rank_word_samples(self.occ, class_idx=0, n_top=1)), 1)

    def test_unknown_order_raises(self):
        with self.assertRaisesRegex(ValueError, "must be one of"):
            rank_word_samples(self.occ, class_idx=0, order="alphabetical")

    def test_absent_class_returns_typed_empty(self):
        ranked = rank_word_samples(self.occ, class_idx=99)
        self.assertTrue(ranked.empty)
        self.assertEqual(list(ranked.columns), ["sample", "contribution", "n_occurrences"])


class TestWordContributionsBySample(unittest.TestCase):
    def setUp(self):
        self.explanation = _word_profile_explanation()

    def test_sums_one_words_occurrences_per_sample(self):
        # class 0: sample0 has one "happy" (0.4), sample1 has two ("Happy" 0.2 + "happy" 0.1 = 0.3),
        # sample2 has one (-0.6), sample3 has none.
        result = word_contributions_by_sample(self.explanation, ["happy"], label_idx=0)
        np.testing.assert_allclose(result, [0.4, 0.3, -0.6, 0.0])

    def test_sums_across_words_within_a_sample(self):
        # "today" only occurs in sample0 (0.05, class 0), added on top of "happy"'s own 0.4.
        result = word_contributions_by_sample(self.explanation, ["happy", "today"], label_idx=0)
        np.testing.assert_allclose(result, [0.45, 0.3, -0.6, 0.0])

    def test_label_idx_selects_the_class(self):
        class0 = word_contributions_by_sample(self.explanation, ["happy"], label_idx=0)
        class1 = word_contributions_by_sample(self.explanation, ["happy"], label_idx=1)
        np.testing.assert_allclose(class1, -class0)

    def test_absent_word_returns_all_zeros(self):
        result = word_contributions_by_sample(self.explanation, ["nonexistent"], label_idx=0)
        np.testing.assert_allclose(result, np.zeros(len(self.explanation)))

    def test_result_length_matches_the_batch(self):
        result = word_contributions_by_sample(self.explanation, ["happy"], label_idx=0)
        self.assertEqual(len(result), len(self.explanation))
