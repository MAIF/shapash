"""Unit tests for ``NlpExplainer`` — engine wiring, generator discovery, ``explain()`` caching
(in-memory and on-disk), and label-noise detection. A real NLP model is not required — synthetic
data is used throughout so the suite runs in CI without transformers/datasets.
"""

import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from shapash.backend.nlp_backend import NlpBackend, NlpContributions
from shapash.backend.nlp_lime_backend import NlpLimeBackend
from shapash.compute.generators import AblationFlipGenerator, HotFlipGenerator
from shapash.explainer.nlp_explainer import NlpExplainer
from shapash.explainer.nlp_explanation import NlpExplanation
from shapash.model.base import SupportsEmbeddings, SupportsGradients, SupportsTokenization, TextModel

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
N_CLASSES = len(LABEL_NAMES)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_token_data() -> tuple[list[list[str]], list[np.ndarray], np.ndarray]:
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


def _make_texts() -> pd.Series:
    return pd.Series(
        ["i feel so happy today", "this is terrible and sad", "what a wonderful day"],
        index=pd.RangeIndex(3),
    )


def _make_explanation(
    texts: pd.Series | None = None,
    y_pred: pd.Series | None = None,
    y_prob: pd.DataFrame | None = None,
    y_true: pd.Series | None = None,
    label_names: list[str] | None = LABEL_NAMES,
    backend_name: str = "nlp_shap",
    is_additive: bool = True,
    reference_kind: str = "none",
    output_space: str = "probability",
) -> NlpExplanation:
    """Synthetic ``NlpExplanation`` for 3 samples, 6 classes — the ``explain()`` return value."""
    texts = _make_texts() if texts is None else texts
    token_strings, values, base_values = _make_token_data()
    return NlpExplanation(
        texts=texts,
        token_strings=token_strings,
        values=values,
        base_values=base_values,
        y_pred=(
            pd.Series(["joy", "sadness", "joy"], index=texts.index, name="prediction") if y_pred is None else y_pred
        ),
        y_prob=y_prob,
        y_true=y_true,
        label_names=label_names,
        folds_case=None,
        backend_name=backend_name,
        is_additive=is_additive,
        reference_kind=reference_kind,
        output_space=output_space,
    )


def _make_explainer() -> NlpExplainer:
    """Bare ``NlpExplainer`` (engine role only), bypassing ``__init__`` to avoid ``shap.Explainer(None)``.

    Per ``fit``/``explain``, the explainer never holds compiled results — tests that need a computed
    batch build an :class:`NlpExplanation` directly with :func:`_make_explanation` instead.
    """
    xpl = object.__new__(NlpExplainer)
    xpl.model = None
    xpl.label_names = LABEL_NAMES
    xpl.backend = None
    return xpl


# ---------------------------------------------------------------------------
# NlpExplainer (no real model)
# ---------------------------------------------------------------------------


class TestNlpExplainer(unittest.TestCase):
    def setUp(self):
        self.xpl = _make_explainer()
        self.explanation = _make_explanation()

    def test_plot_tokens_returns_figure(self):
        fig = self.explanation.plot.tokens(row=0, label_idx=1)
        self.assertIsInstance(fig, go.Figure)

    def test_folds_case_is_none_without_a_tokenizing_model(self):
        # A bare classifier_fn has no tokenizer to ask; None is the honest answer, not a guess.
        self.assertIsNone(self.xpl._folds_case())

    def test_folds_case_reads_the_model_capability(self):
        class _Uncased(TextModel, SupportsTokenization):
            def predict(self, texts):
                return np.tile([0.5, 0.5], (len(texts), 1))

            def tokenize(self, text):
                return text.lower().split()

            def detokenize(self, tokens):
                return " ".join(tokens)

        self.xpl._text_model = _Uncased(label_names=["neg", "pos"])
        self.assertTrue(self.xpl._folds_case())

    def test_plot_tokens_all_samples(self):
        for row in range(3):
            fig = self.explanation.plot.tokens(row=row, label_idx=0)
            self.assertIsInstance(fig, go.Figure)

    def test_plot_tokens_all_labels(self):
        for label_idx in range(N_CLASSES):
            fig = self.explanation.plot.tokens(row=0, label_idx=label_idx)
            self.assertIsInstance(fig, go.Figure)

    def test_plot_tokens_max_tokens(self):
        fig = self.explanation.plot.tokens(row=0, label_idx=1, max_tokens=3)
        self.assertIsInstance(fig, go.Figure)
        self.assertLessEqual(len(fig.data[0].x), 3)

    def test_y_pred_stored(self):
        self.assertIsNotNone(self.explanation.y_pred)
        self.assertEqual(len(self.explanation.y_pred), 3)

    def test_label_names_propagated(self):
        self.assertEqual(self.explanation.label_names, LABEL_NAMES)

    def test_y_true_is_none_by_default(self):
        self.assertIsNone(self.explanation.y_true)


# ---------------------------------------------------------------------------
# NlpExplainer — counterfactual generator discovery / selection (captum-free)
# ---------------------------------------------------------------------------


class _FullCapModel(TextModel, SupportsTokenization, SupportsEmbeddings, SupportsGradients):
    """A model exposing every capability — both HotFlip and AblationFlip are compatible."""

    def __init__(self):
        super().__init__(label_names=["neg", "pos"])

    def predict(self, texts):
        return np.tile([0.4, 0.6], (len(texts), 1))

    def tokenize(self, text):
        return text.split()

    def detokenize(self, tokens):
        return " ".join(tokens)

    def get_embedding_table(self):
        return (["a"], np.zeros((1, 2)))

    def embed(self, texts):
        return np.zeros((len(texts), 2))

    def token_gradients(self, text, target_class):
        toks = text.split()
        return toks, np.zeros((len(toks), 2))

    @property
    def shap_callable(self):
        return self.predict


class _TokenizeOnlyModel(TextModel, SupportsTokenization):
    """Tokenizable but gradient-free — only AblationFlip is compatible."""

    def __init__(self):
        super().__init__(label_names=["neg", "pos"])

    def predict(self, texts):
        return np.tile([0.4, 0.6], (len(texts), 1))

    def tokenize(self, text):
        return text.split()

    def detokenize(self, tokens):
        return " ".join(tokens)

    @property
    def shap_callable(self):
        return self.predict


class TestNlpExplainerGenerators(unittest.TestCase):
    """Generator auto-discovery drives the webapp's method selector (no captum needed here)."""

    def test_full_capability_model_offers_both_methods(self):
        xpl = NlpExplainer(_FullCapModel(), backend=object())
        self.assertEqual(
            xpl.available_cf_generators(),
            [("hotflip", "HotFlip"), ("ablation_flip", "Ablation")],
        )
        # The preferred (first-discovered) generator stays the active default.
        self.assertIsInstance(xpl.cf_generator, HotFlipGenerator)
        self.assertEqual(set(xpl.cf_generators), {"hotflip", "ablation_flip"})

    def test_tokenize_only_model_offers_ablation_only(self):
        xpl = NlpExplainer(_TokenizeOnlyModel(), backend=object())
        self.assertEqual(xpl.available_cf_generators(), [("ablation_flip", "Ablation")])
        self.assertIsInstance(xpl.cf_generator, AblationFlipGenerator)

    def test_explicit_generator_used_verbatim_no_extras(self):
        model = _FullCapModel()
        gen = AblationFlipGenerator(model)
        xpl = NlpExplainer(model, backend=object(), cf_generator=gen)
        # An explicit choice is not augmented with the other compatible built-ins.
        self.assertEqual(xpl.available_cf_generators(), [("ablation_flip", "Ablation")])
        self.assertIs(xpl.cf_generator, gen)

    def test_cf_config_spec_selected_by_generator(self):
        xpl = NlpExplainer(_FullCapModel(), backend=object())
        self.assertIn("max_flips", xpl.cf_config_spec("hotflip"))
        self.assertIn("max_ablations", xpl.cf_config_spec("ablation_flip"))
        # No argument → the active generator's spec.
        self.assertIn("max_flips", xpl.cf_config_spec())

    def test_unknown_generator_raises(self):
        xpl = NlpExplainer(_FullCapModel(), backend=object())
        with self.assertRaises(KeyError):
            xpl.cf_config_spec("does_not_exist")
        with self.assertRaises(KeyError):
            xpl.generate_counterfactuals("hi there", generator="does_not_exist")

    def test_no_generators_without_text_model(self):
        # A plain callable is neither a TextModel nor a pipeline → no generators, empty selector.
        xpl = NlpExplainer(
            lambda texts: np.tile([0.5, 0.5], (len(texts), 1)), label_names=["neg", "pos"], backend=object()
        )
        self.assertEqual(xpl.available_cf_generators(), [])
        self.assertEqual(xpl.cf_config_spec(), {})
        self.assertIsNone(xpl.cf_generator)


# ---------------------------------------------------------------------------
# NlpExplainer with NlpLimeBackend
# ---------------------------------------------------------------------------

# Keep num_samples tiny so LIME tests run fast in CI.
_LIME_COMPUTE_ARGS = {"num_samples": 50, "num_features": 5}
_SAMPLE_TEXTS = ["i feel so happy today", "this is terrible and sad"]


def _fake_classifier(texts: list[str]) -> np.ndarray:
    """Deterministic fake classifier returning (n_texts, N_CLASSES) probabilities."""
    rng = np.random.default_rng(0)
    probs = rng.random((len(texts), N_CLASSES)).astype(np.float32)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def _make_lime_backend() -> NlpLimeBackend:
    return NlpLimeBackend(
        _fake_classifier,
        label_names=LABEL_NAMES,
        explainer_compute_args=_LIME_COMPUTE_ARGS,
    )


class TestNlpExplainerWithLimeBackend(unittest.TestCase):
    def _make_explainer_lime(self) -> NlpExplainer:
        """NlpExplainer backed by NlpLimeBackend."""
        xpl = _make_explainer()
        xpl.backend = _make_lime_backend()
        return xpl

    def test_backend_is_lime_instance(self):
        xpl = self._make_explainer_lime()
        self.assertIsInstance(xpl.backend, NlpLimeBackend)

    def test_plot_tokens_works_for_a_lime_explanation(self):
        # LIME is non-additive, so .plot.waterfall refuses; .plot.tokens stays available.
        explanation = replace(_make_explanation(), backend_name="nlp_lime", is_additive=False)
        self.assertIsInstance(explanation.plot.tokens(row=0, label_idx=0), go.Figure)
        with self.assertRaises(ValueError):
            explanation.plot.waterfall(row=0, label_idx=0)

    def test_explain_sets_contributions(self):
        backend = _make_lime_backend()
        xpl = NlpExplainer(_fake_classifier, label_names=LABEL_NAMES, backend=backend)
        fake_pred_df = pd.DataFrame(
            {"prediction": ["joy"] * len(_SAMPLE_TEXTS)},
            index=pd.RangeIndex(len(_SAMPLE_TEXTS)),
        )
        with patch.object(xpl, "_predict", return_value=fake_pred_df):
            explanation = xpl.explain(_SAMPLE_TEXTS)
        self.assertEqual(len(explanation), len(_SAMPLE_TEXTS))
        self.assertEqual(explanation.label_names, LABEL_NAMES)

    def test_explain_caching_skips_rerun(self):
        backend = _make_lime_backend()
        xpl = NlpExplainer(_fake_classifier, label_names=LABEL_NAMES, backend=backend)
        fake_pred_df = pd.DataFrame(
            {"prediction": ["joy"] * len(_SAMPLE_TEXTS)},
            index=pd.RangeIndex(len(_SAMPLE_TEXTS)),
        )
        with patch.object(xpl, "_predict", return_value=fake_pred_df) as mocked_predict:
            xpl.explain(_SAMPLE_TEXTS)
            xpl.explain(_SAMPLE_TEXTS)  # same data — must hit in-memory cache
        self.assertEqual(mocked_predict.call_count, 1, "memoization must skip the second _predict call")


# ---------------------------------------------------------------------------
# compile() cache key
# ---------------------------------------------------------------------------


class _KeyModel(TextModel):
    """Prediction-only model with a caller-chosen ``model_id``, so keys can be varied deliberately."""

    def __init__(self, ident="model-a"):
        super().__init__(label_names=LABEL_NAMES)
        self._ident = ident

    @property
    def model_id(self):
        return self._ident

    def predict(self, texts):
        probs = np.full((len(texts), N_CLASSES), 1.0 / N_CLASSES)
        return probs


class _MarkerBackend(NlpBackend):
    """Returns a constant contribution equal to ``marker``, and counts its explainer runs.

    The constant makes it visible *whose* result a cache served: a value of 1.0 can only have come
    from the backend built with ``marker=1.0``.
    """

    name = "marker_backend"
    reference_kind = "none"
    is_additive = True
    output_space = "probability"

    def __init__(self, marker=1.0, **kwargs):
        super().__init__(model=None, label_names=LABEL_NAMES, **kwargs)
        self.marker = marker
        self.calls = 0

    def run_explainer(self, x):
        self.calls += 1
        texts = list(x)
        return NlpContributions(
            token_strings=[["a", "b"] for _ in texts],
            values=[np.full((2, N_CLASSES), self.marker) for _ in texts],
            base_values=np.zeros((len(texts), N_CLASSES)),
        )


class _OtherMarkerBackend(_MarkerBackend):
    """Same behaviour under a different registered ``name`` — a distinct attribution method."""

    name = "other_marker_backend"


def _explainer(model=None, backend=None, label_names=LABEL_NAMES):
    return NlpExplainer(
        model or _KeyModel(),
        label_names=label_names,
        backend=backend or _MarkerBackend(),
    )


class TestExplainCacheKey(unittest.TestCase):
    """``explain`` results depend on *(texts, model, backend)* — so all three must be in the key.

    Keying on the texts alone means swapping the model or the attribution backend and pointing at the
    same ``cache_dir`` silently reloads the previous run's contributions, and even without a
    ``cache_dir`` the in-memory guard turns a re-``explain`` into a no-op.
    """

    def test_identical_inputs_still_hit_the_in_memory_cache(self):
        backend = _MarkerBackend()
        xpl = _explainer(backend=backend)
        xpl.explain(_SAMPLE_TEXTS)
        xpl.explain(_SAMPLE_TEXTS)
        self.assertEqual(backend.calls, 1, "memoization must survive the richer key")

    def test_the_memo_cannot_be_poisoned_through_a_returned_artifact(self):
        """``explain`` returns a ``replace()`` of the memoized artifact — a *shallow* copy.

        The two artifacts therefore share their contribution arrays, so before the arrays were
        sealed an in-place edit of the first return value silently rewrote the cache, and every
        later ``explain`` of the same texts served the corrupted numbers as a fresh result. The
        seal turns that into an error at the point of the edit.
        """
        backend = _MarkerBackend(marker=1.0)
        xpl = _explainer(backend=backend)
        first = xpl.explain(_SAMPLE_TEXTS)

        with self.assertRaises(ValueError):
            first.values[0][0] = -42.0

        second = xpl.explain(_SAMPLE_TEXTS)
        self.assertEqual(backend.calls, 1, "still a memo hit — the seal must not defeat caching")
        np.testing.assert_allclose(second.values[0], 1.0)

    def test_a_memo_hit_is_relabelled_onto_the_caller_index(self):
        """The memo is keyed by text *content*, so the same texts from a differently-indexed
        source is a hit — and the cached predictions still carry the first caller's index.
        ``replace`` says nothing about the fields it is not handed, so every index-bearing field
        has to be re-labelled, not just ``texts``.
        """
        backend = _MarkerBackend()
        xpl = _explainer(backend=backend)
        xpl.explain(pd.Series(_SAMPLE_TEXTS))  # first call: default RangeIndex

        index = pd.Index([77 + i for i in range(len(_SAMPLE_TEXTS))])
        second = xpl.explain(pd.Series(_SAMPLE_TEXTS, index=index))

        self.assertEqual(backend.calls, 1, "re-labelling must not cost a recompute")
        for field_name in ("texts", "y_pred", "y_prob"):
            with self.subTest(field=field_name):
                self.assertTrue(getattr(second, field_name).index.equals(index))

    def test_a_y_series_that_does_not_match_x_is_refused(self):
        """Reindexing it silently would pair labels positionally while looking like alignment."""
        xpl = _explainer(backend=_MarkerBackend())
        texts = pd.Series(_SAMPLE_TEXTS, index=[10 + i for i in range(len(_SAMPLE_TEXTS))])
        misaligned = pd.Series(["joy"] * len(_SAMPLE_TEXTS), index=range(len(_SAMPLE_TEXTS)))
        with self.assertRaises(ValueError) as ctx:
            xpl.explain(texts, y=misaligned)
        self.assertIn("indexed differently", str(ctx.exception))

    def test_a_y_series_aligned_to_x_is_kept_and_a_list_is_indexed_positionally(self):
        xpl = _explainer(backend=_MarkerBackend())
        index = pd.Index([10 + i for i in range(len(_SAMPLE_TEXTS))])
        texts = pd.Series(_SAMPLE_TEXTS, index=index)
        labels = ["joy"] * len(_SAMPLE_TEXTS)
        for tag, y in (("series", pd.Series(labels, index=index)), ("list", labels)):
            with self.subTest(y=tag):
                self.assertTrue(xpl.explain(texts, y=y).y_true.index.equals(index))

    def test_swapping_the_backend_recomputes(self):
        xpl = _explainer(backend=_MarkerBackend(marker=1.0))
        xpl.explain(_SAMPLE_TEXTS)
        replacement = _OtherMarkerBackend(marker=2.0)
        xpl.backend = replacement
        explanation = xpl.explain(_SAMPLE_TEXTS)
        self.assertEqual(replacement.calls, 1, "in-memory guard served a stale, other-backend result")
        np.testing.assert_allclose(explanation.values[0], 2.0)

    def test_backend_compute_args_are_part_of_the_key(self):
        xpl = _explainer(backend=_MarkerBackend(explainer_compute_args={"n_steps": 50}))
        first = xpl._compute_key(_SAMPLE_TEXTS)
        xpl.backend = _MarkerBackend(explainer_compute_args={"n_steps": 200})
        self.assertNotEqual(first, xpl._compute_key(_SAMPLE_TEXTS))

    def test_compute_args_key_is_order_insensitive(self):
        # Same settings written in a different order are the same configuration.
        a = _explainer(backend=_MarkerBackend(explainer_compute_args={"x": 1, "y": 2}))
        b = _explainer(backend=_MarkerBackend(explainer_compute_args={"y": 2, "x": 1}))
        self.assertEqual(a._compute_key(_SAMPLE_TEXTS), b._compute_key(_SAMPLE_TEXTS))

    def test_model_identity_is_part_of_the_key(self):
        a = _explainer(model=_KeyModel("model-a"))
        b = _explainer(model=_KeyModel("model-b"))
        self.assertNotEqual(a._compute_key(_SAMPLE_TEXTS), b._compute_key(_SAMPLE_TEXTS))

    def test_label_names_are_part_of_the_key(self):
        # label_names fixes the column order of y_prob, so it changes the cached payload.
        a = _explainer(label_names=LABEL_NAMES)
        b = _explainer(label_names=list(reversed(LABEL_NAMES)))
        self.assertNotEqual(a._compute_key(_SAMPLE_TEXTS), b._compute_key(_SAMPLE_TEXTS))

    def test_key_is_collision_safe_across_text_boundaries(self):
        # Without a separator between texts these two corpora hash identically.
        xpl = _explainer()
        self.assertNotEqual(xpl._compute_key(["ab", "c"]), xpl._compute_key(["a", "bc"]))

    def test_key_is_order_sensitive(self):
        xpl = _explainer()
        self.assertNotEqual(xpl._compute_key(["a", "b"]), xpl._compute_key(["b", "a"]))


def _read_meta(path) -> dict:
    with zipfile.ZipFile(path) as zf:
        return json.loads(zf.read("meta.json"))


def _drop_meta_key(path, key) -> None:
    """Rewrite a saved ``.xpl`` in place without one ``meta.json`` key — how an older layout is faked."""
    with zipfile.ZipFile(path) as zin:
        members = {item.filename: zin.read(item.filename) for item in zin.infolist()}
    meta = json.loads(members["meta.json"])
    del meta[key]
    members["meta.json"] = json.dumps(meta).encode()
    with zipfile.ZipFile(path, "w") as zout:
        for name, data in members.items():
            zout.writestr(name, data)


class TestExplainDiskCacheIsolation(unittest.TestCase):
    """The disk cache is the dangerous case: a stale entry survives the process that wrote it."""

    def test_two_backends_share_a_cache_dir_without_collision(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            first = _MarkerBackend(marker=1.0)
            _explainer(backend=first).explain(_SAMPLE_TEXTS, cache_dir=cache_dir)

            second = _OtherMarkerBackend(marker=2.0)
            xpl = _explainer(backend=second)
            explanation = xpl.explain(_SAMPLE_TEXTS, cache_dir=cache_dir)

            self.assertEqual(second.calls, 1, "loaded the other backend's cached contributions")
            np.testing.assert_allclose(explanation.values[0], 2.0)

    def test_two_models_share_a_cache_dir_without_collision(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            _explainer(model=_KeyModel("model-a"), backend=_MarkerBackend(marker=1.0)).explain(
                _SAMPLE_TEXTS, cache_dir=cache_dir
            )
            backend_b = _MarkerBackend(marker=2.0)
            xpl = _explainer(model=_KeyModel("model-b"), backend=backend_b)
            explanation = xpl.explain(_SAMPLE_TEXTS, cache_dir=cache_dir)

            self.assertEqual(backend_b.calls, 1, "loaded the other model's cached contributions")
            np.testing.assert_allclose(explanation.values[0], 2.0)

    def test_same_model_and_backend_reload_from_disk(self):
        # The cache must still *work* — a fresh instance skips the expensive run.
        with tempfile.TemporaryDirectory() as cache_dir:
            _explainer(backend=_MarkerBackend(marker=3.0)).explain(_SAMPLE_TEXTS, cache_dir=cache_dir)
            reloaded_backend = _MarkerBackend(marker=3.0)
            xpl = _explainer(backend=reloaded_backend)
            explanation = xpl.explain(_SAMPLE_TEXTS, cache_dir=cache_dir)
            self.assertEqual(reloaded_backend.calls, 0, "disk cache did not hit for identical inputs")
            np.testing.assert_allclose(explanation.values[0], 3.0)

    def test_cache_path_points_at_the_file_explain_writes(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            xpl = _explainer()
            path = xpl.cache_path(_SAMPLE_TEXTS, cache_dir)
            self.assertFalse(path.exists())
            xpl.explain(_SAMPLE_TEXTS, cache_dir=cache_dir)
            self.assertTrue(path.exists(), "cache_path disagrees with where explain() wrote")

    def test_stale_layout_is_a_miss_not_a_crash(self):
        # The cache key covers texts/model/backend but not the shapash version, so a layout change
        # leaves entries the new reader cannot parse. The entry must be recomputed and overwritten.
        with tempfile.TemporaryDirectory() as cache_dir:
            _explainer(backend=_MarkerBackend(marker=4.0)).explain(_SAMPLE_TEXTS, cache_dir=cache_dir)
            path = _explainer().cache_path(_SAMPLE_TEXTS, cache_dir)
            _drop_meta_key(path, "values_ndim")

            backend = _MarkerBackend(marker=4.0)
            explanation = _explainer(backend=backend).explain(_SAMPLE_TEXTS, cache_dir=cache_dir)

            self.assertEqual(backend.calls, 1, "a rejected cache entry must fall through to a recompute")
            np.testing.assert_allclose(explanation.values[0], 4.0)
            self.assertIn("values_ndim", _read_meta(path), "stale entry was not replaced")

    def test_corrupt_cache_entry_is_a_miss_not_a_crash(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            _explainer(backend=_MarkerBackend(marker=5.0)).explain(_SAMPLE_TEXTS, cache_dir=cache_dir)
            path = _explainer().cache_path(_SAMPLE_TEXTS, cache_dir)
            path.write_bytes(b"not a zip file")

            backend = _MarkerBackend(marker=5.0)
            explanation = _explainer(backend=backend).explain(_SAMPLE_TEXTS, cache_dir=cache_dir)

            self.assertEqual(backend.calls, 1)
            np.testing.assert_allclose(explanation.values[0], 5.0)

    def test_explicit_load_still_raises_on_a_stale_file(self):
        # The miss-not-crash rule is scoped to the cache. A caller who names a file gets an error
        # rather than a silent substitution.
        with tempfile.TemporaryDirectory() as cache_dir:
            _explainer().explain(_SAMPLE_TEXTS, cache_dir=cache_dir)
            path = _explainer().cache_path(_SAMPLE_TEXTS, cache_dir)
            _drop_meta_key(path, "values_ndim")
            with self.assertRaises(KeyError):
                NlpExplanation.load(path)

    def test_clear_cache_forces_a_recompute(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            backend = _MarkerBackend()
            xpl = _explainer(backend=backend)
            xpl.explain(_SAMPLE_TEXTS, cache_dir=cache_dir)
            xpl.clear_cache(_SAMPLE_TEXTS, cache_dir)
            self.assertFalse(xpl.cache_path(_SAMPLE_TEXTS, cache_dir).exists())
            xpl.explain(_SAMPLE_TEXTS, cache_dir=cache_dir)
            self.assertEqual(backend.calls, 2, "clear_cache must defeat the in-memory guard too")

    def test_clear_cache_leaves_other_backends_entries_intact(self):
        with tempfile.TemporaryDirectory() as cache_dir:
            keeper = _MarkerBackend(marker=1.0)
            _explainer(backend=keeper).explain(_SAMPLE_TEXTS, cache_dir=cache_dir)
            keeper_path = _explainer(backend=_MarkerBackend(marker=1.0)).cache_path(_SAMPLE_TEXTS, cache_dir)

            other = _explainer(backend=_OtherMarkerBackend(marker=2.0))
            other.explain(_SAMPLE_TEXTS, cache_dir=cache_dir)
            other.clear_cache(_SAMPLE_TEXTS, cache_dir)

            self.assertTrue(keeper_path.exists(), "clearing one backend dropped another's cache")


class TestDetectLabelNoise(unittest.TestCase):
    """The explainer's confident-learning surface, over an ``NlpExplanation`` (no model, no backend)."""

    def _with_labels(self, y_true=None, y_prob=None) -> NlpExplanation:
        """A batch carrying ground truth and per-class probabilities.

        Sample 1 is labelled ``joy`` while the model confidently says ``sadness`` — the planted
        error. Both classes carry at least one label, without which ``sadness`` would have no
        estimable threshold and could never be suggested (covered separately in the compute tests).
        """
        confident = ["joy", "sadness", "sadness"]
        probs = np.full((3, N_CLASSES), 0.02)
        for row, name in enumerate(confident):
            probs[row, LABEL_NAMES.index(name)] = 0.90
        probs = probs / probs.sum(axis=1, keepdims=True)
        y_pred = pd.Series(confident, index=pd.RangeIndex(3), name="prediction")
        resolved_y_prob = pd.DataFrame(probs, index=pd.RangeIndex(3), columns=LABEL_NAMES) if y_prob is None else y_prob
        resolved_y_true = (
            pd.Series(["joy", "joy", "sadness"], index=pd.RangeIndex(3), name="ground_truth")
            if y_true is None
            else y_true
        )
        return _make_explanation(y_pred=y_pred, y_prob=resolved_y_prob, y_true=resolved_y_true)

    # ── capability flag ────────────────────────────────────────────────
    def test_available_with_ground_truth_and_per_class_probabilities(self):
        self.assertTrue(_make_explainer().can_detect_label_noise(self._with_labels()))

    def test_unavailable_without_ground_truth(self):
        explanation = replace(self._with_labels(), y_true=None)
        self.assertFalse(_make_explainer().can_detect_label_noise(explanation))

    def test_unavailable_with_only_the_winning_class_probability(self):
        # The raw-pipeline path emits a single "probability" column; the losing classes' scores are
        # precisely what confident learning needs.
        legacy = pd.DataFrame({"probability": [0.9, 0.9, 0.9]}, index=pd.RangeIndex(3))
        self.assertFalse(_make_explainer().can_detect_label_noise(self._with_labels(y_prob=legacy)))

    def test_unavailable_on_an_explanation_with_no_probabilities(self):
        self.assertFalse(_make_explainer().can_detect_label_noise(_make_explanation()))

    def test_available_without_a_model(self):
        # Unlike the other capability flags this needs no live model.
        xpl = _make_explainer()
        self.assertIsNone(xpl.model)
        self.assertTrue(xpl.can_detect_label_noise(self._with_labels()))

    # ── detection ──────────────────────────────────────────────────────
    def test_raises_when_the_prerequisites_are_missing(self):
        with self.assertRaisesRegex(RuntimeError, "ground-truth labels"):
            _make_explainer().detect_label_noise(_make_explanation())

    def test_flags_the_planted_mislabel(self):
        report = _make_explainer().detect_label_noise(self._with_labels())
        self.assertEqual([i.index for i in report.issues], [1])
        issue = report.issues[0]
        self.assertEqual(issue.given_label, "joy")
        self.assertEqual(issue.suggested_label, "sadness")
        self.assertEqual(issue.text, "this is terrible and sad")

    def test_label_names_come_from_the_probability_columns(self):
        report = _make_explainer().detect_label_noise(self._with_labels())
        self.assertEqual(report.label_names, LABEL_NAMES)
        self.assertEqual(report.noise_matrix.shape, (N_CLASSES, N_CLASSES))
        self.assertEqual(report.n_samples, 3)

    def test_respects_top_n_and_score(self):
        report = _make_explainer().detect_label_noise(self._with_labels(), top_n=0, score="normalized_margin")
        self.assertEqual(report.issues, [])
        self.assertEqual(report.n_issues, 1)

    # ── memoisation ────────────────────────────────────────────────────
    def test_repeats_are_served_from_the_memo(self):
        xpl = _make_explainer()
        explanation = self._with_labels()
        first = xpl.detect_label_noise(explanation, top_n=5)
        self.assertIs(xpl.detect_label_noise(explanation, top_n=5), first)

    def test_different_arguments_recompute(self):
        xpl = _make_explainer()
        explanation = self._with_labels()
        self.assertIsNot(xpl.detect_label_noise(explanation, top_n=5), xpl.detect_label_noise(explanation, top_n=4))

    # ── independent probe ──────────────────────────────────────────────
    def _probe_corpus(self):
        """A reference corpus separable by words the audited fixture's texts also use."""
        texts = [
            "this is wonderful and joyful",
            "wonderful joyful and bright",
            "a joyful wonderful day",
            "bright and wonderful joy",
            "this is terrible and sad",
            "terrible sad and bleak",
            "a sad terrible day",
            "bleak and terrible sadness",
        ]
        labels = ["joy"] * 4 + ["sadness"] * 4
        return texts, labels

    def test_no_probe_when_no_reference_corpus_is_bound(self):
        xpl = _make_explainer()
        self.assertFalse(xpl.can_probe_labels())
        self.assertIsNone(xpl.detect_label_noise(self._with_labels()).issues[0].probe)

    def test_probe_verdict_is_attached_when_a_labelled_corpus_is_bound(self):
        xpl = _make_explainer()
        xpl.reference_ = self._probe_corpus()
        self.assertTrue(xpl.can_probe_labels())
        issue = xpl.detect_label_noise(self._with_labels()).issues[0]
        self.assertIsNotNone(issue.probe)
        self.assertIn(issue.probe.top_label, {"joy", "sadness"})
        self.assertEqual(issue.probe.backs_given, issue.probe.top_label == issue.given_label)

    def test_probe_corroborates_a_genuine_label_error(self):
        # The flagged row is "this is terrible and sad" carrying the label "joy". The reference
        # corpus puts that vocabulary firmly in "sadness", so the probe rejects the given label too
        # — the two-signals-agree case, which is the one worth relabelling.
        xpl = _make_explainer()
        xpl.reference_ = self._probe_corpus()
        issue = xpl.detect_label_noise(self._with_labels()).issues[0]
        self.assertEqual((issue.given_label, issue.text), ("joy", "this is terrible and sad"))
        self.assertFalse(issue.probe.backs_given)
        self.assertEqual(issue.probe.top_label, "sadness")
        self.assertLess(issue.probe.given_prob, 0.5)

    def test_probe_backs_the_label_when_the_corpus_sides_with_it(self):
        # The mirror case, and the reason the column exists: same flagged row, but a corpus that
        # calls this vocabulary "joy". The probe now defends the label, marking the row as the
        # audited model's error rather than the corpus's.
        xpl = _make_explainer()
        texts, _ = self._probe_corpus()
        xpl.reference_ = (texts, ["sadness"] * 4 + ["joy"] * 4)
        issue = xpl.detect_label_noise(self._with_labels()).issues[0]
        self.assertEqual(issue.given_label, "joy")
        self.assertTrue(issue.probe.backs_given)
        self.assertGreater(issue.probe.given_prob, 0.5)

    def test_probe_is_skipped_when_not_requested(self):
        xpl = _make_explainer()
        xpl.reference_ = self._probe_corpus()
        self.assertIsNone(xpl.detect_label_noise(self._with_labels(), probe=False).issues[0].probe)

    def test_probe_is_fit_once_and_reused_across_calls(self):
        xpl = _make_explainer()
        xpl.reference_ = self._probe_corpus()
        explanation = self._with_labels()
        xpl.detect_label_noise(explanation, top_n=5)
        first = xpl._label_probe
        self.assertIsNotNone(first)
        xpl.detect_label_noise(explanation, top_n=4)  # different args -> recompute, but the probe is kept
        self.assertIs(xpl._label_probe, first)

    def test_probe_needs_no_model_or_retriever(self):
        # The point of fitting on plain text: a prediction-only pipeline that cannot embed (so
        # can_find_similar() is False) still gets the second opinion.
        xpl = _make_explainer()
        xpl.reference_ = self._probe_corpus()
        xpl._retriever = None
        xpl._text_model = None
        self.assertFalse(xpl.can_find_similar())
        self.assertTrue(xpl.can_probe_labels())
        self.assertIsNotNone(xpl.detect_label_noise(self._with_labels()).issues[0].probe)


if __name__ == "__main__":
    unittest.main()
