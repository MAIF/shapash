"""Unit tests for the SHAP backend's token-id masking path.

The fake model's logits are a sum over its tokens (an additive game), so every exact Shapley/Owen
value is known in closed form: a feature is worth the sum of ``W[id] - W[mask]`` over its tokens.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("shap")
pytestmark = pytest.mark.nlp

from shapash.backend import NlpShapBackend  # noqa: E402
from shapash.backend.nlp_shap_backend import (  # noqa: E402
    _clustering,
    _encode,
    _supports_token_ids,
    _TokenIdMasker,
)
from shapash.model.base import SupportsCaptumIG, SupportsLogits, TextModel  # noqa: E402

CLS, SEP, MASK = 1, 2, 3
TEXT = "un café 😍\n\nok"
# (id, offsets, word id): a byte-level split "é" (two tokens on one character), a zero-width
# word-start marker before the emoji, and a newline that is a word of its own.
PLAN = [
    (CLS, (0, 0), None),
    (10, (0, 2), 0),  # un
    (11, (3, 6), 1),  # caf
    (12, (6, 7), 1),  # é, first byte
    (13, (6, 7), 1),  # é, second byte
    (14, (8, 8), 2),  # ▁, zero width
    (15, (8, 9), 2),  # 😍
    (16, (9, 11), 3),  # Ċ
    (17, (11, 13), 4),  # ok
    (SEP, (0, 0), None),
]
ONLY_SPECIALS = "   "


class _Tokenizer:
    def convert_ids_to_tokens(self, ids):
        return [f"t{i}" for i in ids]


class AdditiveModel(TextModel, SupportsCaptumIG, SupportsLogits):
    """Logits = Σ W[token id], over a fixed encoding plan per text."""

    def __init__(self, fast=True):
        super().__init__(label_names=["neg", "pos"])
        self.fast = fast
        self.tokenizer = _Tokenizer()
        self.weights = np.random.default_rng(0).normal(size=(20, 2))
        self._embedding = torch.nn.Embedding(20, 2)
        self._embedding.weight.data = torch.tensor(self.weights, dtype=torch.float32)

    def _plan(self, text):
        return PLAN if text == TEXT else [(CLS, (0, 0), None), (SEP, (0, 0), None)]

    def predict(self, texts):
        return torch.softmax(torch.tensor(self.predict_logits(texts)), dim=-1).numpy()

    def predict_logits(self, texts):
        return np.stack([self.weights[[i for i, _, _ in self._plan(t)]].sum(axis=0) for t in texts])

    @property
    def embedding_layer(self):
        return self._embedding

    def encode(self, text):
        ids = torch.tensor([[i for i, _, _ in self._plan(text)]])
        return ids, torch.ones_like(ids), [f"t{i}" for i in ids[0].tolist()]

    def reference_ids(self, input_ids):
        return torch.tensor([[i if i in (CLS, SEP) else MASK for i in input_ids[0].tolist()]])

    @property
    def baseline_token(self):
        return "[MASK]"

    def logits(self, input_ids, attention_mask):
        return self._embedding(input_ids).sum(dim=1)

    def word_alignment(self, text):
        if not self.fast:
            return None
        plan = self._plan(text)
        positions: dict[int, list[int]] = {}
        for p, (_, _, w) in enumerate(plan):
            if w is not None:
                positions.setdefault(w, []).append(p)
        specials = [p for p, (_, _, w) in enumerate(plan) if w is None]
        return [f"w{w}" for w in positions], list(positions.values()), specials

    def token_offsets(self, text):
        return [o for _, o, _ in self._plan(text)] if self.fast else None


def _token_value(model, token_id):
    return model.weights[token_id] - model.weights[MASK]


class TestEncoding:
    def test_features_words_and_reference(self):
        encoding = _encode(AdditiveModel(), TEXT)
        # The features are the layout's pieces: "é"'s two tokens are one, the marker joins the emoji.
        assert encoding.layout.words == [[[1]], [[2], [3, 4]], [[5, 6]], [[7]], [[8]]]
        assert encoding.layout.n_pieces == 6
        # Specials keep their ids in the reference; every feature position is masked.
        assert encoding.reference_ids.tolist() == [CLS] + [MASK] * 8 + [SEP]
        assert encoding.feature_of_token.tolist() == [-1, 0, 1, 2, 2, 3, 3, 4, 5, -1]

    def test_mismatched_offsets_are_refused(self):
        model = AdditiveModel()
        model.token_offsets = lambda text: [(0, 1)]
        with pytest.raises(ValueError, match="no word alignment or offsets"):
            _encode(model, TEXT)

    def test_an_alignment_past_the_encoding_is_refused(self):
        model = AdditiveModel()
        model.word_alignment = lambda text: (["w"], [[99]], [])
        with pytest.raises(ValueError, match="index past"):
            _encode(model, TEXT)


class TestMasker:
    def test_nothing_hidden_is_the_encoding_everything_hidden_the_reference(self):
        masker = _TokenIdMasker(AdditiveModel())
        encoding = masker.encoding(TEXT)
        (kept,) = masker(np.ones(6, dtype=bool), TEXT)
        (hidden,) = masker(np.zeros(6, dtype=bool), TEXT)
        assert kept.tolist() == [encoding.input_ids.tolist()]
        assert hidden.tolist() == [encoding.reference_ids.tolist()]

    def test_hiding_a_feature_touches_only_its_tokens(self):
        masker = _TokenIdMasker(AdditiveModel())
        mask = np.ones(6, dtype=bool)
        mask[2] = False  # "é"
        (out,) = masker(mask, TEXT)
        expected = [p[0] for p in PLAN]
        expected[3] = expected[4] = MASK
        assert out.tolist() == [expected]

    def test_shapes_names_and_mask_token(self):
        masker = _TokenIdMasker(AdditiveModel())
        assert masker.shape(TEXT) == (1, 6)
        assert masker.mask_shapes(TEXT) == [(6,)]
        assert masker.feature_names(TEXT) == [["t10", "t11", "t12t13", "t14t15", "t16", "t17"]]
        assert masker.mask_token == "[MASK]"

    def test_every_word_is_one_branch_of_the_hierarchy(self):
        layout = _encode(AdditiveModel(), TEXT).layout
        tree = _TokenIdMasker(AdditiveModel()).clustering(TEXT)
        n = layout.n_pieces
        assert tree.shape == (n - 1, 4)
        leaves = {i: {i} for i in range(n)}
        for k, (a, b, height, size) in enumerate(tree):
            leaves[n + k] = leaves[int(a)] | leaves[int(b)]
            assert size == len(leaves[n + k]) and 0 < height <= 1
        first = 0
        for word in layout.words:
            assert set(range(first, first + len(word))) in leaves.values()
            first += len(word)

    def test_a_single_feature_has_an_empty_hierarchy(self):
        layout = _encode(AdditiveModel(), TEXT).layout
        single = type(layout)(TEXT, [[[1]]], [(0, 2)])
        assert _clustering(single).shape == (0, 4)


class TestSupportsTokenIds:
    def test_a_fast_captum_model_is_supported(self):
        assert _supports_token_ids(AdditiveModel())

    def test_a_slow_tokenizer_or_a_failing_probe_is_not(self):
        assert not _supports_token_ids(AdditiveModel(fast=False))
        broken = AdditiveModel()
        broken.word_alignment = lambda text: 1 / 0
        assert not _supports_token_ids(broken)

    def test_a_model_without_ids_is_not(self):
        assert not _supports_token_ids(object())


class TestShapBackendOnTokenIds:
    def test_values_are_the_exact_additive_contributions(self):
        model = AdditiveModel()
        backend = NlpShapBackend(model, label_names=model.label_names, output_space="logit")
        assert backend.masking == "token_ids"
        assert backend.baseline_token == "[MASK]"
        out = backend.run_explainer([TEXT])
        assert out.token_strings == [["un", "café", "😍", "ok"]]
        assert out.token_spans == [[((0, 2),), ((3, 7),), ((8, 9),), ((11, 13),)]]
        expected = [
            _token_value(model, 10),
            _token_value(model, 11) + _token_value(model, 12) + _token_value(model, 13),
            _token_value(model, 14) + _token_value(model, 15),
            _token_value(model, 17),
        ]
        np.testing.assert_allclose(out.values[0], expected, atol=1e-5)
        # Additive against the logits of the text itself, not of a rebuilt string.
        np.testing.assert_allclose(
            out.base_values[0] + out.values[0].sum(axis=0), model.predict_logits([TEXT])[0], atol=1e-5
        )

    def test_probability_space_is_additive_against_predict(self):
        model = AdditiveModel()
        out = NlpShapBackend(model, label_names=model.label_names).run_explainer([TEXT])
        np.testing.assert_allclose(out.base_values[0] + out.values[0].sum(axis=0), model.predict([TEXT])[0], atol=1e-5)

    def test_a_text_with_nothing_to_hide_has_no_words(self):
        model = AdditiveModel()
        out = NlpShapBackend(model, label_names=model.label_names, output_space="logit").run_explainer([ONLY_SPECIALS])
        assert out.token_strings == [[]] and out.values[0].shape == (0, 2)
        np.testing.assert_allclose(out.base_values[0], model.predict_logits([ONLY_SPECIALS])[0], atol=1e-5)

    def test_an_explicit_masker_or_a_slow_tokenizer_takes_the_string_path(self):
        import shap

        class _Tok:
            mask_token = "[MASK]"

            def __call__(self, s, return_offsets_mapping=False):
                return {"input_ids": [0], "offset_mapping": [(0, 0)]}

        model = AdditiveModel()
        explicit = NlpShapBackend(
            model, label_names=model.label_names, output_space="logit", masker=shap.maskers.Text()
        )
        assert explicit.masking == "text"
        slow = AdditiveModel(fast=False)
        slow.tokenizer = None
        with pytest.raises(TypeError, match="tokenizer to mask text with"):
            # The string path needs a real tokenizer; reaching that error proves the fallback.
            NlpShapBackend(slow, label_names=slow.label_names, output_space="logit")


class TestNlpExplainerDefault:
    def test_the_default_backend_masks_token_ids_and_the_key_says_so(self):
        import shap

        from shapash.explainer.nlp_explainer import NlpExplainer

        model = AdditiveModel()
        default = NlpExplainer(model)
        assert default.backend.masking == "token_ids"
        strings = NlpExplainer(
            model,
            backend=NlpShapBackend(model, label_names=model.label_names, masker=shap.maskers.Text()),
        )
        assert strings.backend.masking == "text"
        assert default._compute_key([TEXT]) != strings._compute_key([TEXT])
