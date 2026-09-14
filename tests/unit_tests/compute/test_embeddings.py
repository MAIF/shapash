"""Unit tests for ``Embedding`` (the vectors artifact and its ``.npz`` persistence) and for
``projection_coords``, the one check that pairs a projection with an explanation."""

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from shapash.compute.embeddings import Embedding, projection_coords


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


if __name__ == "__main__":
    unittest.main()
