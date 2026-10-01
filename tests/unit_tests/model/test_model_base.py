"""Unit tests for the model capability layer (no torch/transformers required).

Uses a tiny deterministic fake model implementing the capability mixins, so the interface and the
``has_capabilities`` discovery helper are exercised without a real transformer.
"""

import unittest

import numpy as np

from shapash.model.base import (
    SupportsEmbeddings,
    SupportsGradients,
    SupportsTokenization,
    TextModel,
    has_capabilities,
    is_word_token,
)


class PredictOnlyModel(TextModel):
    """Minimal model implementing only ``predict``."""

    def __init__(self):
        super().__init__(label_names=["neg", "pos"])

    def predict(self, texts):
        return np.tile([0.4, 0.6], (len(texts), 1))


class FullModel(TextModel, SupportsTokenization, SupportsEmbeddings, SupportsGradients):
    """Model implementing every capability."""

    def __init__(self):
        super().__init__(label_names=["neg", "pos"])

    def predict(self, texts):
        return np.tile([0.3, 0.7], (len(texts), 1))

    def tokenize(self, text):
        return text.split()

    def detokenize(self, tokens):
        return " ".join(tokens)

    def get_embedding_table(self):
        return ["a", "b"], np.eye(2)

    def embed(self, texts):
        return np.zeros((len(texts), 2))

    def token_gradients(self, text, target_class):
        toks = text.split()
        return toks, np.ones((len(toks), 2))


class TestCapabilities(unittest.TestCase):
    def test_predict_only_lacks_optional_capabilities(self):
        m = PredictOnlyModel()
        self.assertIsInstance(m, TextModel)
        self.assertFalse(has_capabilities(m, SupportsGradients))
        self.assertFalse(has_capabilities(m, SupportsEmbeddings))
        self.assertFalse(has_capabilities(m, SupportsTokenization))

    def test_full_model_has_all_capabilities(self):
        m = FullModel()
        self.assertTrue(has_capabilities(m, SupportsTokenization))
        self.assertTrue(has_capabilities(m, SupportsEmbeddings))
        self.assertTrue(has_capabilities(m, SupportsGradients))
        self.assertTrue(has_capabilities(m, SupportsGradients, SupportsEmbeddings))

    def test_predict_shape_and_n_classes(self):
        m = FullModel()
        probs = m.predict(["x", "y", "z"])
        self.assertEqual(probs.shape, (3, 2))
        self.assertEqual(m.n_classes, 2)

    def test_n_classes_none_without_labels(self):
        m = FullModel()
        m.label_names = None
        self.assertIsNone(m.n_classes)


class _SchemeModel(TextModel, SupportsTokenization):
    """Tokenizes by whitespace, then applies a subword-marker convention to each word.

    Stands in for the three families the real adapters meet: WordPiece-style continuation marking
    (no prefix), byte-level BPE (``Ġ``) and SentencePiece (``▁``).
    """

    def __init__(self, marker=""):
        super().__init__(label_names=["neg", "pos"])
        self.marker = marker
        self.tokenize_calls = 0

    def predict(self, texts):
        return np.tile([0.5, 0.5], (len(texts), 1))

    def tokenize(self, text):
        self.tokenize_calls += 1
        return [f"{self.marker}{w}" for w in text.split()]

    def detokenize(self, tokens):
        return " ".join(t.removeprefix(self.marker) for t in tokens)


class TestWordStartMarker(unittest.TestCase):
    """The marker is probed from what the tokenizer emits, not from a class name or model list."""

    def test_detects_no_marker_for_continuation_scheme(self):
        self.assertIsNone(_SchemeModel().word_start_marker())

    def test_detects_byte_bpe_marker(self):
        self.assertEqual(_SchemeModel(marker="Ġ").word_start_marker(), "Ġ")

    def test_detects_sentencepiece_marker(self):
        self.assertEqual(_SchemeModel(marker="▁").word_start_marker(), "▁")

    def test_probe_runs_once_and_is_memoized(self):
        model = _SchemeModel(marker="▁")
        for _ in range(5):
            model.word_start_marker()
        self.assertEqual(model.tokenize_calls, 1)

    def test_unprobeable_tokenizer_degrades_to_no_marker(self):
        # A tokenizer that cannot run the probe must not break word-hood entirely.
        class _Broken(_SchemeModel):
            def tokenize(self, text):
                raise RuntimeError("tokenizer unavailable")

        model = _Broken()
        self.assertIsNone(model.word_start_marker())
        self.assertTrue(model.is_substitutable("happy"))


class _CasingModel(_SchemeModel):
    """Whitespace tokenizer that optionally case-folds, as an uncased checkpoint's normalizer does."""

    def __init__(self, lowercase=False):
        super().__init__()
        self.lowercase = lowercase

    def tokenize(self, text):
        return super().tokenize(text.lower() if self.lowercase else text)


class TestFoldsCase(unittest.TestCase):
    """Casing is probed from what the tokenizer emits — no real family records it the same way.

    RoBERTa and CamemBERTv2 both report ``normalizer=None`` while a BERT-family uncased checkpoint
    hides it in ``BertNormalizer(lowercase=True)``, so reading attributes would misclassify them.
    """

    def test_detects_uncased_tokenizer(self):
        self.assertTrue(_CasingModel(lowercase=True).folds_case())

    def test_detects_cased_tokenizer(self):
        self.assertFalse(_CasingModel(lowercase=False).folds_case())

    def test_probe_runs_once_and_is_memoized(self):
        model = _CasingModel(lowercase=True)
        for _ in range(5):
            model.folds_case()
        # Two calls for the single probe (lower + upper), then memoized.
        self.assertEqual(model.tokenize_calls, 2)

    def test_unprobeable_tokenizer_is_treated_as_cased(self):
        # Guessing "uncased" would merge units of a model we know nothing about; keep them apart.
        class _Broken(_CasingModel):
            def tokenize(self, text):
                raise RuntimeError("tokenizer unavailable")

        self.assertFalse(_Broken().folds_case())


class TestIsSubstitutable(unittest.TestCase):
    """Word-hood must follow the model's own scheme — a single ``isalpha`` rule is wrong for two of three.

    ``"▁good"`` is not alphabetic and ``"Ġ"`` (U+0120) *is*, so a bare ``isalpha`` check rejects every
    SentencePiece content token and accepts every mid-word byte-BPE fragment.
    """

    def test_continuation_scheme_accepts_bare_words(self):
        m = _SchemeModel()
        self.assertTrue(m.is_substitutable("happy"))
        self.assertFalse(m.is_substitutable("##ing"))  # WordPiece continuation
        self.assertFalse(m.is_substitutable("[CLS]"))
        self.assertFalse(m.is_substitutable("1b"))
        self.assertFalse(m.is_substitutable("!"))

    def test_sentencepiece_scheme_accepts_marked_words(self):
        m = _SchemeModel(marker="▁")
        self.assertTrue(m.is_substitutable("▁good"))  # the regression: was False for every token
        self.assertFalse(m.is_substitutable("good"))  # bare -> mid-word piece under this scheme
        self.assertFalse(m.is_substitutable("▁"))  # marker alone is not a word
        self.assertFalse(m.is_substitutable("▁123"))
        self.assertFalse(m.is_substitutable("<s>"))

    def test_byte_bpe_scheme_accepts_marked_words(self):
        m = _SchemeModel(marker="Ġ")
        self.assertTrue(m.is_substitutable("Ġgood"))
        self.assertFalse(m.is_substitutable("good"))  # the other regression: was True (mid-word piece)
        self.assertFalse(m.is_substitutable("Ġ"))
        self.assertFalse(m.is_substitutable("Ċ"))

    def test_is_word_token_is_the_marker_free_default(self):
        self.assertTrue(is_word_token("happy"))
        self.assertFalse(is_word_token("##ing"))
        self.assertFalse(is_word_token("▁good"))  # why it cannot be used alone


class TestIsSubstitutableAt(unittest.TestCase):
    """A multi-piece word's *head* is a word token but not a substitutable *position*.

    ``"grouchy"`` tokenizes to ``["gr", "##ou", "##chy"]``; ``"gr"`` is bare and alphabetic, so every
    token-level test accepts it, and substituting there rebuilt ``"superouchy"``. Only a check that can
    see the following token rejects it.
    """

    def test_continuation_scheme_rejects_multi_piece_head(self):
        m = _SchemeModel()
        tokens = ["i", "am", "gr", "##ou", "##chy", "today"]
        self.assertTrue(m.is_substitutable(tokens[2]))  # token-level says yes...
        self.assertFalse(m.is_substitutable_at(tokens, 2))  # ...position-level says no
        self.assertEqual([i for i in range(len(tokens)) if m.is_substitutable_at(tokens, i)], [0, 1, 5])

    def test_marked_scheme_rejects_multi_piece_head(self):
        m = _SchemeModel(marker="▁")
        tokens = ["▁i", "▁am", "▁grou", "chy", "▁today"]
        self.assertTrue(m.is_substitutable(tokens[2]))
        self.assertFalse(m.is_substitutable_at(tokens, 2))
        self.assertEqual([i for i in range(len(tokens)) if m.is_substitutable_at(tokens, i)], [0, 1, 4])

    def test_marked_scheme_keeps_word_before_punctuation(self):
        """Punctuation is unmarked but starts no word — the word before a comma stays perturbable.

        Treating every unmarked token as a continuation would silently make the last word of every
        sentence, and any word before a comma, unsubstitutable.
        """
        m = _SchemeModel(marker="Ġ")
        tokens = ["Ġrude", ",", "Ġawful", "."]
        self.assertTrue(m.is_substitutable_at(tokens, 0))
        self.assertTrue(m.is_substitutable_at(tokens, 2))

    def test_final_position_has_no_successor(self):
        m = _SchemeModel()
        self.assertTrue(m.is_substitutable_at(["hello", "world"], 1))

    def test_marked_scheme_accepts_the_unmarked_opening_word(self):
        """byte-BPE leaves the first token of a text bare, so the marker test misreads it.

        ``"the waiter was rude"`` tokenizes to ``["the", "Ġwaiter", ...]``; judging ``"the"`` by the
        marker made the opening word of every input unperturbable on the whole RoBERTa/GPT-2 family.
        """
        m = _SchemeModel(marker="Ġ")
        tokens = ["the", "Ġwaiter", "Ġwas", "Ġrude"]
        self.assertFalse(m.is_substitutable(tokens[0]))  # token-level still reads it as mid-word
        self.assertTrue(m.is_substitutable_at(tokens, 0))  # position 0 knows better

    def test_opening_word_still_rejected_when_multi_piece(self):
        """The index-0 clause relaxes the *marker* rule, not the whole-word rule.

        ``"rude waiters"`` tokenizes to ``["r", "ude", "Ġwait", "ers"]`` — ``"r"`` opens the text but
        still only heads a multi-piece word, so substituting there would rebuild ``"Xude"``.
        """
        m = _SchemeModel(marker="Ġ")
        self.assertFalse(m.is_substitutable_at(["r", "ude", "Ġwait", "ers"], 0))

    def test_unmarked_token_is_only_forgiven_at_index_zero(self):
        m = _SchemeModel(marker="Ġ")
        # Same bare token, mid-text: still a mid-word piece, still rejected.
        self.assertFalse(m.is_substitutable_at(["Ġthe", "the", "Ġrude"], 1))

    def test_continues_word_follows_the_scheme(self):
        self.assertTrue(_SchemeModel().continues_word("##ou"))
        self.assertFalse(_SchemeModel().continues_word("and"))
        # Marked schemes invert it: bare = mid-word, marked = new word, punctuation = neither.
        self.assertTrue(_SchemeModel(marker="▁").continues_word("chy"))
        self.assertFalse(_SchemeModel(marker="▁").continues_word("▁and"))
        self.assertFalse(_SchemeModel(marker="▁").continues_word(","))


if __name__ == "__main__":
    unittest.main()
