"""Unit tests for the similar-example retriever (no torch/transformers required).

A tiny fake model returns fixed vectors per text, so the retrieval math (cosine, top-k ordering), label
plumbing, corpus validation and the ``.npy`` bank cache are exercised deterministically without a real
transformer. Retrieval always goes through ``embed`` in the *model's* configured space — the retriever
has no space of its own — and that space takes part in the cache key of the
:class:`~shapash.compute.embedding_store.EmbeddingStore` the bank is drawn from.
"""

import unittest

import numpy as np

from shapash.compute.retrieval.similar_examples import (
    Neighbor,
    SimilarExampleRetriever,
    _l2_normalize,
)
from shapash.model.base import (
    SupportsEmbeddings,
    TextModel,
    has_capabilities,
)

# Fixed 2-D vectors: each text maps to a point; cosine ranks by angle from the origin.
_VECTORS = {
    "happy joy": [1.0, 0.0],
    "joyful glad": [0.9, 0.1],
    "sad down": [0.0, 1.0],
    "miserable": [0.1, 0.9],
    "neutral": [1.0, 1.0],
}


def _vecs(texts):
    return np.array([_VECTORS.get(t, [0.0, 0.0]) for t in texts], dtype=float)


class FakeEmbeddingModel(TextModel, SupportsEmbeddings):
    """Model returning a fixed vector per known text from ``embed``.

    ``calls`` counts encode calls (so the bank-reuse/cache tests can assert the bank is not rebuilt) and
    ``spaces`` records the ``space`` argument each call received. ``embedding_space``, ``resolve_space``
    and ``model_id`` mirror a real adapter's surface so the retriever's cache key behaves as it does in
    production.
    """

    def __init__(self, model_id="FakeEmbeddingModel:v1", embedding_space="decision"):
        super().__init__(label_names=["neg", "pos"])
        self.calls = 0
        self.spaces = []
        self._model_id = model_id
        self.embedding_space = embedding_space

    def resolve_space(self, space=None):
        return space if space is not None else self.embedding_space

    @property
    def model_id(self):
        return self._model_id

    def predict(self, texts):
        return np.tile([0.5, 0.5], (len(texts), 1))

    def embed(self, texts, space=None):
        self.calls += 1
        self.spaces.append(space)
        return _vecs(texts)

    def get_embedding_table(self):
        return ["<pad>"], np.zeros((1, 2), dtype=float)  # unused by retrieval; satisfies the ABC


class PredictOnlyModel(TextModel):
    """No embedding capability."""

    def __init__(self):
        super().__init__(label_names=["neg", "pos"])

    def predict(self, texts):
        return np.tile([0.5, 0.5], (len(texts), 1))


class TestRetriever(unittest.TestCase):
    def setUp(self):
        self.model = FakeEmbeddingModel()
        self.texts = ["happy joy", "joyful glad", "sad down", "miserable"]
        self.labels = ["pos", "pos", "neg", "neg"]
        self.retriever = SimilarExampleRetriever(self.model, self.texts, self.labels)

    def test_embeds_in_the_models_space_without_overriding_it(self):
        # The retriever never passes a space of its own: embed() is called bare, so whatever the model
        # is configured with (the same space the scatter projects) is what neighbours are ranked in.
        # The cache key still names that space, resolved via the model.
        self.assertEqual(self.retriever.size, 4)
        self.retriever.query("happy joy", top_k=1)
        self.assertEqual(set(self.model.spaces), {None})
        self.assertIn("decision", self.retriever.store.space_key)

    def test_cache_key_follows_the_models_space(self):
        # Moving the model to another space moves the retriever with it — one setting drives both.
        model = FakeEmbeddingModel(embedding_space="pooled")
        retriever = SimilarExampleRetriever(model, self.texts, self.labels)
        retriever.query("happy joy", top_k=1)
        self.assertEqual(set(model.spaces), {None})
        self.assertIn("pooled", retriever.store.space_key)

    def test_a_space_change_rebuilds_the_bank(self):
        """A bank held across a live space switch would be cosine-compared against queries from the
        new space — two unrelated spaces, yielding a plausible number rather than an error."""
        self.retriever.query("happy joy", top_k=1)
        self.model.calls = 0
        self.model.embedding_space = "pooled"
        self.retriever.query("happy joy", top_k=1)
        self.assertEqual(self.model.calls, 2)  # bank rebuilt in the new space, plus the query

    def test_the_bank_is_built_once_while_the_space_holds(self):
        """Guard on the fix above: the bank is the expensive, amortizable half and must stay so."""
        for _ in range(3):
            self.retriever.query("happy joy", top_k=1)
        self.assertEqual(self.model.calls, 4)  # one bank + three queries

    def test_query_ranks_by_cosine_similarity(self):
        neighbors = self.retriever.query("happy joy", top_k=2)
        self.assertEqual([n.text for n in neighbors], ["happy joy", "joyful glad"])
        # Cosine to itself is 1.0; scores are sorted descending.
        self.assertAlmostEqual(neighbors[0].score, 1.0, places=6)
        self.assertGreaterEqual(neighbors[0].score, neighbors[1].score)

    def test_query_returns_labels_and_neighbor_type(self):
        neighbors = self.retriever.query("miserable", top_k=1)
        self.assertIsInstance(neighbors[0], Neighbor)
        self.assertEqual(neighbors[0].text, "miserable")
        self.assertEqual(neighbors[0].label, "neg")

    def test_top_k_clamped_to_corpus_size(self):
        neighbors = self.retriever.query("neutral", top_k=99)
        self.assertEqual(len(neighbors), 4)


class TestQueryThreshold(unittest.TestCase):
    """The threshold path: not a fixed count, ranked descending, with an uncapped total."""

    def setUp(self):
        self.model = FakeEmbeddingModel()
        self.texts = ["happy joy", "joyful glad", "sad down", "miserable"]
        self.labels = ["pos", "pos", "neg", "neg"]
        self.retriever = SimilarExampleRetriever(self.model, self.texts, self.labels)

    def test_returns_only_matches_above_threshold_descending(self):
        # cosine("happy joy", "happy joy") = 1.0, cosine("happy joy", "joyful glad") ~= 0.994,
        # the other two are near-orthogonal — well below any threshold used here.
        neighbors, total = self.retriever.query_threshold("happy joy", threshold=0.99)
        self.assertEqual([n.text for n in neighbors], ["happy joy", "joyful glad"])
        self.assertEqual(total, 2)
        self.assertGreaterEqual(neighbors[0].score, neighbors[1].score)

    def test_threshold_is_strict_exclusive(self):
        # A threshold equal to the best possible score (self-similarity = 1.0) admits nothing.
        neighbors, total = self.retriever.query_threshold("happy joy", threshold=1.0)
        self.assertEqual(neighbors, [])
        self.assertEqual(total, 0)

    def test_limit_caps_returned_neighbors_but_not_the_total(self):
        neighbors, total = self.retriever.query_threshold("happy joy", threshold=0.99, limit=1)
        self.assertEqual([n.text for n in neighbors], ["happy joy"])
        self.assertEqual(total, 2)  # still 2 cleared the threshold, only 1 is returned

    def test_no_limit_returns_every_match(self):
        neighbors, total = self.retriever.query_threshold("happy joy", threshold=-1.0, limit=None)
        self.assertEqual(len(neighbors), self.retriever.size)
        self.assertEqual(total, self.retriever.size)

    def test_labels_ride_along(self):
        neighbors, _total = self.retriever.query_threshold("happy joy", threshold=0.99)
        self.assertEqual([n.label for n in neighbors], ["pos", "pos"])

    def test_bank_built_once_and_reused(self):
        self.retriever.query("happy joy", top_k=1)
        calls_after_first = self.model.calls
        self.retriever.query("sad down", top_k=1)
        # Two queries => one bank build + two query embeds = 3 calls total (bank not rebuilt).
        self.assertEqual(self.model.calls, calls_after_first + 1)

    def test_labels_optional(self):
        retriever = SimilarExampleRetriever(self.model, self.texts, reference_labels=None)
        neighbors = retriever.query("happy joy", top_k=1)
        self.assertIsNone(neighbors[0].label)

    def test_mismatched_labels_length_raises(self):
        with self.assertRaises(ValueError):
            SimilarExampleRetriever(self.model, self.texts, reference_labels=["pos"])

    def test_incompatible_model_raises(self):
        model = PredictOnlyModel()
        self.assertFalse(has_capabilities(model, SupportsEmbeddings))
        with self.assertRaises(TypeError):  # retrieval needs embed()
            SimilarExampleRetriever(model, self.texts, self.labels)

    def test_bank_cached_to_disk_and_reloaded(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            r1 = SimilarExampleRetriever(self.model, self.texts, self.labels, cache_dir=d)
            r1.build()
            cache_file = r1.store.path
            self.assertTrue(cache_file.exists())

            # A fresh retriever over the same corpus/space/model loads the bank from disk.
            fresh_model = FakeEmbeddingModel()
            r2 = SimilarExampleRetriever(fresh_model, self.texts, self.labels, cache_dir=d)
            neighbors = r2.query("happy joy", top_k=1)
            self.assertEqual(fresh_model.calls, 1)  # only the query embed, bank came from disk
            self.assertEqual(neighbors[0].text, "happy joy")

    def test_different_models_do_not_share_cache(self):
        """Two models over the same corpus must not collide — the bug when the key was the class name."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            a = FakeEmbeddingModel(model_id="checkpoint-a")
            b = FakeEmbeddingModel(model_id="checkpoint-b")
            r_a = SimilarExampleRetriever(a, self.texts, self.labels, cache_dir=d)
            r_b = SimilarExampleRetriever(b, self.texts, self.labels, cache_dir=d)
            self.assertNotEqual(r_a.store.key, r_b.store.key)
            r_a.build()
            r_b.build()
            self.assertGreater(b.calls, 0)  # b built its own bank rather than loading a's

    def test_different_spaces_do_not_share_cache(self):
        """Same model, different representation space => different bank."""
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            m_dec = FakeEmbeddingModel(embedding_space="decision")
            m_pool = FakeEmbeddingModel(embedding_space="pooled")
            r_dec = SimilarExampleRetriever(m_dec, self.texts, self.labels, cache_dir=d)
            r_pool = SimilarExampleRetriever(m_pool, self.texts, self.labels, cache_dir=d)
            self.assertNotEqual(r_dec.store.key, r_pool.store.key)


class TestQueryMany(unittest.TestCase):
    """The batch path must agree with the single-text one, and pay one embed call for the lot."""

    def setUp(self):
        self.model = FakeEmbeddingModel()
        self.texts = ["happy joy", "joyful glad", "sad down", "miserable"]
        self.labels = ["pos", "pos", "neg", "neg"]
        self.retriever = SimilarExampleRetriever(self.model, self.texts, self.labels)

    def test_matches_query_row_for_row(self):
        queries = ["happy joy", "miserable"]
        batched = self.retriever.query_many(queries, top_k=3)
        one_at_a_time = [self.retriever.query(q, top_k=3) for q in queries]
        self.assertEqual(batched, one_at_a_time)

    def test_embeds_the_whole_batch_in_one_call(self):
        self.retriever.build()  # pay the bank up front so only query encoding is counted
        before = self.model.calls
        self.retriever.query_many(["happy joy", "miserable", "neutral"], top_k=2)
        self.assertEqual(self.model.calls - before, 1)

    def test_empty_input_returns_empty_without_embedding(self):
        before = self.model.calls
        self.assertEqual(self.retriever.query_many([], top_k=3), [])
        self.assertEqual(self.model.calls, before)

    def test_single_text_batch(self):
        [neighbors] = self.retriever.query_many(["happy joy"], top_k=2)
        self.assertEqual([n.text for n in neighbors], ["happy joy", "joyful glad"])

    def test_top_k_is_clamped_to_the_corpus_size(self):
        [neighbors] = self.retriever.query_many(["happy joy"], top_k=99)
        self.assertEqual(len(neighbors), self.retriever.size)

    def test_exclude_self_drops_the_query_from_its_own_results(self):
        # The case that matters: the queries are themselves corpus members, so without this every
        # row's nearest neighbour is itself.
        [neighbors] = self.retriever.query_many(["happy joy"], top_k=2, exclude_self=True)
        self.assertEqual([n.text for n in neighbors], ["joyful glad", "miserable"])

    def test_exclude_self_removes_every_copy_of_a_duplicated_text(self):
        retriever = SimilarExampleRetriever(
            FakeEmbeddingModel(), ["happy joy", "happy joy", "sad down"], ["pos", "pos", "neg"]
        )
        [neighbors] = retriever.query_many(["happy joy"], top_k=3, exclude_self=True)
        self.assertEqual([n.text for n in neighbors], ["sad down"])

    def test_exclude_self_leaves_texts_that_merely_look_alike(self):
        # Matching on text equality, not a similarity cutoff: "joyful glad" is a near-duplicate in
        # embedding space and must survive.
        [neighbors] = self.retriever.query_many(["happy joy"], top_k=1, exclude_self=True)
        self.assertEqual(neighbors[0].text, "joyful glad")

    def test_labels_ride_along_as_in_the_single_query_path(self):
        [neighbors] = self.retriever.query_many(["sad down"], top_k=2)
        self.assertTrue(all(isinstance(n, Neighbor) for n in neighbors))
        self.assertEqual([n.label for n in neighbors], ["neg", "neg"])

    def test_unlabelled_corpus_yields_none_labels(self):
        retriever = SimilarExampleRetriever(FakeEmbeddingModel(), self.texts)
        [neighbors] = retriever.query_many(["happy joy"], top_k=2)
        self.assertEqual([n.label for n in neighbors], [None, None])


class TestL2Normalize(unittest.TestCase):
    def test_unit_norm_rows(self):
        out = _l2_normalize(np.array([[3.0, 4.0], [1.0, 0.0]]))
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), [1.0, 1.0])

    def test_zero_row_stays_zero(self):
        out = _l2_normalize(np.array([[0.0, 0.0], [0.0, 5.0]]))
        np.testing.assert_allclose(out[0], [0.0, 0.0])
        np.testing.assert_allclose(out[1], [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
