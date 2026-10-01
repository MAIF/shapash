"""Unit tests for ``Embedding`` (the vectors artifact and its ``.npz`` persistence) and for
``projection_coords``, the one check that pairs a projection with an explanation."""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from shapash.compute.embeddings import Embedding, projection_coords
from shapash.explainer.nlp_explainer import NlpExplainer
from shapash.explainer.nlp_explanation import NlpExplanation
from shapash.model.base import SupportsEmbeddings, SupportsTokenization, TextModel


class _Scaler:
    """Deterministic stand-in for a reducer: keeps two columns, scaled."""

    def __init__(self, scale=1.0):
        self.scale = scale

    def fit_transform(self, vectors, **kwargs):
        self.kwargs = kwargs
        return np.asarray(vectors)[:, :2] * self.scale


def _embedding(n=4, d=6, **overrides) -> Embedding:
    fields = {
        "vectors": np.arange(n * d, dtype="float32").reshape(n, d),
        "model_id": "fake:v1",
        "space": "decision",
        "corpus_id": "c0ffee",
    }
    fields.update(overrides)
    return Embedding(**fields)


class TestConstruction(unittest.TestCase):
    def test_reports_its_shape(self):
        emb = _embedding(n=4, d=6)
        self.assertEqual((emb.n_samples, emb.n_components), (4, 6))

    def test_vectors_are_sealed_against_in_place_writes(self):
        emb = _embedding()
        with self.assertRaises(ValueError):
            emb.vectors[0, 0] = 999.0

    def test_repr_says_what_it_is_at_a_glance(self):
        self.assertIn("n_samples=4, n_components=6, space='decision'", repr(_embedding()))
        self.assertIn("reducer='_scaler'", repr(_embedding().project(_Scaler())))

    def test_repr_omits_identity_strings(self):
        # model_id and corpus_id are cache-lookup fingerprints, not something to read at a glance.
        repr_str = repr(_embedding())
        self.assertNotIn("fake:v1", repr_str)
        self.assertNotIn("c0ffee", repr_str)

    def test_a_1d_array_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            Embedding(np.zeros(5), model_id="m", space="s", corpus_id="c")
        self.assertIn("2-D", str(ctx.exception))


class TestProjection(unittest.TestCase):
    def test_project_inherits_provenance_and_records_the_reducer(self):
        emb = _embedding()
        projected = emb.project(_Scaler(scale=2.0))

        self.assertEqual(
            (projected.corpus_id, projected.model_id, projected.space), (emb.corpus_id, emb.model_id, emb.space)
        )
        self.assertEqual(projected.reducer_tag, "_scaler")
        np.testing.assert_allclose(projected.vectors, emb.vectors[:, :2] * 2.0)

    def test_default_reducer_is_pca_to_two_components(self):
        projected = _embedding(n=5, d=8).project()
        self.assertEqual((projected.n_components, projected.reducer_tag), (2, "pca"))

    def test_fit_transform_kwargs_are_forwarded(self):
        reducer = _Scaler()
        _embedding().project(reducer, init="pca")
        self.assertEqual(reducer.kwargs, {"init": "pca"})

    def test_a_reducer_that_drops_rows_is_refused(self):
        # Row i must stay the projection of text i, or every point is labelled from the wrong sample.
        class _DropsRows:
            def fit_transform(self, vectors, **kwargs):
                return np.asarray(vectors)[:-1, :2]

        with self.assertRaises(ValueError) as ctx:
            _embedding().project(_DropsRows())
        self.assertIn("one row per text", str(ctx.exception))


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)

    def test_round_trip_preserves_vectors_and_provenance(self):
        emb = _embedding().project(_Scaler(scale=3.0))
        path = self.tmp_path / "run.emb"
        emb.save(path)
        loaded = Embedding.load(path)

        np.testing.assert_allclose(loaded.vectors, emb.vectors)
        self.assertEqual(
            (loaded.model_id, loaded.space, loaded.corpus_id, loaded.reducer_tag),
            (emb.model_id, emb.space, emb.corpus_id, emb.reducer_tag),
        )

    def test_raw_embeddings_round_trip_with_a_null_reducer_tag(self):
        path = self.tmp_path / "raw.emb"
        _embedding().save(path)
        self.assertIsNone(Embedding.load(path).reducer_tag)

    def test_the_path_is_written_verbatim(self):
        # np.savez appends ".npz" to a path given by name that lacks it.
        path = self.tmp_path / "run.emb"
        _embedding().save(path)
        self.assertEqual([p.name for p in self.tmp_path.iterdir()], ["run.emb"])

    def test_the_container_is_a_zip_of_npy_members_with_json_meta(self):
        path = self.tmp_path / "run.emb"
        _embedding().save(path)
        with zipfile.ZipFile(path) as zf:
            self.assertEqual(sorted(zf.namelist()), ["meta.npy", "vectors.npy"])
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"].item()))
        self.assertEqual(meta["corpus_id"], "c0ffee")

    def test_load_refuses_pickled_payloads(self):
        path = self.tmp_path / "pickled.emb"
        with path.open("wb") as fh:
            np.savez(fh, vectors=np.array([{"not": "an array"}], dtype=object), meta="{}")
        with self.assertRaises(ValueError):
            Embedding.load(path)


class TestProjectionCoords(unittest.TestCase):
    explanation = SimpleNamespace(n_samples=4, corpus_id="c0ffee")

    def test_an_embedding_of_the_same_texts_returns_its_vectors(self):
        projection = _embedding().project(_Scaler())
        np.testing.assert_array_equal(projection_coords(projection, self.explanation), projection.vectors)

    def test_an_embedding_of_different_texts_is_refused(self):
        projection = _embedding(corpus_id="other").project(_Scaler())
        with self.assertRaises(ValueError) as ctx:
            projection_coords(projection, self.explanation)
        self.assertIn("different texts", str(ctx.exception))

    def test_an_unreduced_embedding_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            projection_coords(_embedding(), self.explanation)
        self.assertIn("project", str(ctx.exception))

    def test_a_bare_array_is_accepted_on_shape_alone(self):
        coords = projection_coords(np.zeros((4, 2)), self.explanation)
        self.assertEqual(coords.shape, (4, 2))
        with self.assertRaises(ValueError):
            projection_coords(np.zeros((3, 2)), self.explanation)


# ---------------------------------------------------------------------------
# NlpExplainer.compute_projection / compute_embeddings / fit — reducer wiring
# and embedding precompute at fit time
# ---------------------------------------------------------------------------

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]


class _ProjectableModel(TextModel, SupportsEmbeddings):
    """Embeds each text to a deterministic 3-D point; counts calls so cache hits are observable."""

    def __init__(self, space="decision"):
        super().__init__(label_names=LABEL_NAMES[:2])
        self.space = space
        self.calls = 0

    def resolve_space(self, space=None):
        return space if space is not None else self.space

    def predict(self, texts):
        return np.tile([0.4, 0.6], (len(texts), 1))

    def get_embedding_table(self):
        return (["a"], np.zeros((1, 3)))

    def embed(self, texts, space=None):
        self.calls += 1
        return np.array([[float(len(t)), float(t.count("a")), 1.0] for t in texts])

    @property
    def shap_callable(self):
        return self.predict


class _CountingReducer:
    """A deterministic reducer that counts its fits and records its call-time kwargs."""

    def __init__(self, scale=1.0):
        self.scale = scale
        self.fits = 0
        self.last_kwargs = None

    def fit_transform(self, x, **kwargs):
        self.fits += 1
        self.last_kwargs = kwargs
        return np.asarray(x)[:, :2] * self.scale


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


def _projection_explanation(texts: pd.Series) -> NlpExplanation:
    """A minimal ``NlpExplanation`` carrying only ``texts`` — all ``compute_projection`` needs."""
    n = len(texts)
    return NlpExplanation(
        texts=texts,
        token_strings=[[] for _ in range(n)],
        values=[np.zeros((0, 2)) for _ in range(n)],
        base_values=np.zeros((n, 2)),
        y_pred=pd.Series(["pos"] * n, index=texts.index, name="prediction"),
        y_prob=None,
        y_true=None,
        label_names=None,
        folds_case=None,
        backend_name="test",
        is_additive=True,
        reference_kind="none",
        output_space="probability",
    )


class TestComputeProjection(unittest.TestCase):
    """The library owns the space + the embedding cache; the caller injects only the reducer."""

    def setUp(self):
        self.model = _ProjectableModel()
        self.xpl = NlpExplainer(self.model, backend=object())
        self.explanation = _projection_explanation(pd.Series(["alpha", "beta banana", "gamma"]))

    def test_returns_two_columns_aligned_with_the_texts(self):
        projection = self.xpl.compute_projection(self.explanation)
        self.assertEqual(projection.vectors.shape, (3, 2))
        self.assertEqual(projection.n_samples, self.explanation.n_samples)

    def test_defaults_to_pca_without_any_extra_dependency(self):
        """The default reducer must be something a core install already has — sklearn's PCA."""
        projection = self.xpl.compute_projection(self.explanation)
        self.assertEqual(projection.n_components, 2)
        self.assertEqual(self.model.calls, 1)

    def test_injected_reducer_is_used(self):
        reducer = _CountingReducer(scale=2.0)
        projection = self.xpl.compute_projection(self.explanation, reducer=reducer)
        self.assertEqual(reducer.fits, 1)
        np.testing.assert_allclose(projection.vectors[0], [10.0, 4.0])  # "alpha": len 5, 2 a's, doubled

    def test_projection_carries_the_provenance_needed_to_pair_it_back(self):
        # The point of returning an Embedding rather than a bare array: the coordinates say which
        # corpus, model and space they belong to, so a mismatch can be caught instead of drawn.
        projection = self.xpl.compute_projection(self.explanation)
        self.assertEqual(projection.corpus_id, self.explanation.corpus_id)
        self.assertEqual(projection.model_id, self.model.model_id)
        self.assertEqual(projection.space, self.model.resolve_space())
        self.assertEqual(projection.reducer_tag, "pca")

    def test_embeddings_are_raw_and_reusable_for_several_projections(self):
        # The flow the split exists for: embed once (needs the model), reduce many times (does not).
        embedding = self.xpl.compute_embeddings(self.explanation)
        self.assertEqual(self.model.calls, 1)
        self.assertIsNone(embedding.reducer_tag)
        self.assertEqual(embedding.corpus_id, self.explanation.corpus_id)

        first = embedding.project(_CountingReducer(scale=1.0))
        second = embedding.project(_CountingReducer(scale=3.0))
        self.assertEqual(self.model.calls, 1, "reducing must not re-embed")
        np.testing.assert_allclose(second.vectors, first.vectors * 3.0)

    def test_raises_for_a_model_that_cannot_embed(self):
        """A prediction-only model gets a clear error pointing at the escape hatch, not an AttributeError."""
        xpl = NlpExplainer(_TokenizeOnlyModel(), backend=object())
        explanation = _projection_explanation(pd.Series(["alpha", "beta"]))
        with self.assertRaises(TypeError):
            xpl.compute_projection(explanation)

    def test_embeddings_cached_across_instances_while_the_reducer_reruns(self):
        # Only the expensive half is cached; a layout worth keeping is saved explicitly.
        with tempfile.TemporaryDirectory() as d:
            reducer = _CountingReducer()
            self.xpl.compute_projection(self.explanation, reducer=reducer, cache_dir=d)

            fresh_model = _ProjectableModel()
            fresh = NlpExplainer(fresh_model, backend=object())
            fresh.compute_projection(self.explanation, reducer=reducer, cache_dir=d)
            self.assertEqual(fresh_model.calls, 0)
            self.assertEqual(reducer.fits, 2)

    def test_model_space_takes_part_in_the_key(self):
        """Moving the model's space must re-embed, not reload the other space's vectors."""
        with tempfile.TemporaryDirectory() as d:
            self.xpl.compute_projection(self.explanation, reducer=_CountingReducer(), cache_dir=d)
            self.model.space = "pooled"
            self.model.calls = 0
            self.xpl.compute_projection(self.explanation, reducer=_CountingReducer(), cache_dir=d)
            self.assertEqual(self.model.calls, 1)  # re-embedded under the new space

    def test_recompute_forces_a_fresh_fit(self):
        with tempfile.TemporaryDirectory() as d:
            reducer = _CountingReducer()
            self.xpl.compute_projection(self.explanation, reducer=reducer, cache_dir=d)
            self.model.calls = 0
            self.xpl.compute_projection(self.explanation, reducer=reducer, cache_dir=d, recompute=True)
            self.assertEqual(self.model.calls, 1)
            self.assertEqual(reducer.fits, 2)

    def test_fit_transform_kwargs_are_forwarded(self):
        reducer = _CountingReducer()
        self.xpl.compute_projection(self.explanation, reducer=reducer, init="pca", verbose=False)
        self.assertEqual(reducer.last_kwargs, {"init": "pca", "verbose": False})


class _UnembeddableModel(_ProjectableModel):
    """Embeds nothing: stands in for a corpus the model chokes on (OOM, bad encoding)."""

    def embed(self, texts, space=None):
        raise RuntimeError("cannot embed this corpus")


class TestFitPrecompute(unittest.TestCase):
    CORPUS = ["a happy line", "a sad line", "another happy one"]
    LABELS = ["joy", "sadness", "joy"]

    def test_precompute_embeds_the_bank_at_fit(self):
        model = _ProjectableModel()
        xpl = NlpExplainer(model, backend=object()).fit(self.CORPUS, y=self.LABELS)
        self.assertEqual(model.calls, 1)  # the corpus, embedded once, inside fit
        xpl.find_similar("a happy line")
        self.assertEqual(model.calls, 2)  # the query only — the bank was already there

    def test_precompute_false_defers_to_first_query(self):
        model = _ProjectableModel()
        xpl = NlpExplainer(model, backend=object()).fit(self.CORPUS, y=self.LABELS, precompute=False)
        self.assertEqual(model.calls, 0)
        xpl.find_similar("a happy line")
        self.assertEqual(model.calls, 2)  # bank + query, both charged to the first click

    def test_find_similar_threshold_filters_by_score_and_reports_total(self):
        model = _ProjectableModel()
        xpl = NlpExplainer(model, backend=object()).fit(self.CORPUS, y=self.LABELS)
        neighbors, total = xpl.find_similar_threshold("a happy line", threshold=-1.0, limit=1)
        self.assertLessEqual(len(neighbors), 1)
        self.assertGreaterEqual(total, len(neighbors))

    def test_find_similar_threshold_requires_a_retriever(self):
        xpl = NlpExplainer(_TokenizeOnlyModel(), backend=object()).fit(self.CORPUS, y=self.LABELS)
        with self.assertRaises(RuntimeError):
            xpl.find_similar_threshold("a happy line")

    def test_precompute_is_a_noop_when_no_retriever_was_built(self):
        xpl = NlpExplainer(_TokenizeOnlyModel(), backend=object()).fit(self.CORPUS, y=self.LABELS)
        self.assertFalse(xpl.can_find_similar())  # model cannot embed
        self.assertTrue(xpl.can_probe_labels())  # ... but the model-free probe still fit

    def test_a_bank_failure_still_leaves_the_model_free_probe_usable(self):
        xpl = NlpExplainer(_UnembeddableModel(), backend=object())
        with self.assertRaises(RuntimeError):
            xpl.fit(self.CORPUS, y=self.LABELS)
        # reference_/classes_ are assigned before the bank is built, so the half-fitted object keeps
        # the feature that never needed the model.
        self.assertEqual(xpl.reference_, (self.CORPUS, self.LABELS))
        self.assertEqual(xpl.classes_, LABEL_NAMES[:2])  # from the model, not derived from y
        self.assertTrue(xpl.can_probe_labels())


if __name__ == "__main__":
    unittest.main()
