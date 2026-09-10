"""HotFlip counterfactual generator — a port of Google PAIR LIT's ``hotflip.py``.

HotFlip finds a *minimal* set of token substitutions that changes the model's prediction, using a
single backward pass to estimate each token's impact:

1. Rank content tokens by the L2 norm of their input-embedding gradient (largest first), keeping only
   positions holding a **complete word** (:meth:`~shapash.model.base.SupportsTokenization.is_substitutable_at`).
2. For each position, shortlist the top-K **word** candidates by the first-order estimate
   (``embedding_matrix @ token_gradient``, most-negative first), then **re-score that shortlist
   against the model** and keep the one that most reduces the original-class probability. The linear
   estimate alone is an unreliable proxy — its single best pick often does not flip at all — so
   model verification is what surfaces genuine substitutions. Sub-word continuations, the *heads* of
   multi-piece words, and non-word tokens are all excluded so rebuilt text stays well-formed.
3. Hand the chosen substitutions to the shared minimal search
   (:meth:`~shapash.compute.generators.base.CounterfactualGenerator.search_minimal`), which tries flip
   combinations of increasing size (1..``max_flips``) in batched ``predict`` calls and keeps the
   minimal ones that flip.

Requires a model exposing :class:`~shapash.model.base.SupportsGradients` and
:class:`~shapash.model.base.SupportsEmbeddings`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np

from shapash.compute.generators.base import (
    Counterfactual,
    CounterfactualGenerator,
    Field,
    IntField,
    TokenListField,
)
from shapash.compute.generators.cf_utils import display_form
from shapash.model.base import (
    SupportsEmbeddings,
    SupportsGradients,
    SupportsTokenization,
    TextModel,
    has_capabilities,
)

_MAX_FLIPPABLE_TOKENS = 10
_CANDIDATES_PER_POSITION = 50


class _GradientModel(Protocol):
    """The exact capability surface HotFlip uses (predict + gradients + embeddings + tokenization)."""

    label_names: list[str] | None

    def predict(self, texts: list[str]) -> np.ndarray: ...

    def token_gradients(self, text: str, target_class: int) -> tuple[list[str], np.ndarray]: ...

    def get_embedding_table(self) -> tuple[list[str], np.ndarray]: ...

    def detokenize(self, tokens: list[str]) -> str: ...

    def is_substitutable(self, token: str) -> bool: ...

    def is_substitutable_at(self, tokens: Sequence[str], index: int) -> bool: ...


class HotFlipGenerator(CounterfactualGenerator):
    """Gradient-based token-substitution counterfactuals (LIT HotFlip)."""

    name = "hotflip"
    display_name = "HotFlip"

    def config_spec(self) -> dict[str, Field]:
        """Expose ``num_examples``, ``max_flips`` and ``tokens_to_ignore`` as tunable knobs."""
        return {
            "num_examples": IntField(label="Max counterfactuals", default=5, minimum=1, maximum=20),
            "max_flips": IntField(label="Max token flips", default=3, minimum=1, maximum=5),
            "tokens_to_ignore": TokenListField(label="Tokens to ignore", default=[]),
        }

    @classmethod
    def is_compatible(cls, model: TextModel) -> bool:
        """Compatible only with models exposing gradients, an embedding table *and* tokenization.

        ``SupportsTokenization`` is required because rebuilding a flipped sentence goes through
        ``detokenize``, choosing which vocab entries are usable replacements goes through
        ``is_substitutable``, and choosing which *positions* may be flipped goes through
        ``is_substitutable_at`` — all three live on that capability. It was previously used without
        being declared, which happened to work only because every gradient-capable adapter also
        tokenizes.
        """
        return has_capabilities(model, SupportsGradients, SupportsEmbeddings, SupportsTokenization)

    def generate(self, text: str, target_label: str | None = None, config: dict | None = None) -> list[Counterfactual]:
        """Generate minimal HotFlip counterfactuals for ``text`` (see module docstring)."""
        cfg = self.resolve_config(config)
        num_examples = int(cfg["num_examples"])
        max_flips = int(cfg["max_flips"])
        ignore = {t.strip().lower() for t in cfg["tokens_to_ignore"]}

        model = cast(_GradientModel, self.model)  # is_compatible guarantees these capabilities
        label_names = model.label_names or []

        orig_probs = model.predict([text])[0]
        orig_class = int(np.argmax(orig_probs))
        target_class = label_names.index(target_label) if (target_label and target_label in label_names) else None

        tokens, grads = model.token_gradients(text, orig_class)
        if not tokens:
            return []

        # Rank content tokens by gradient L2 norm; keep the top-K flippable positions. Positions are
        # filtered by word-hood as well as candidates: substituting a whole word into a *mid-word*
        # position rebuilds malformed text, which is what the module docstring already promises to
        # avoid — it was previously enforced on candidates only.
        #
        # ``is_substitutable_at`` (not ``is_substitutable``) because the *head* of a multi-piece word is
        # itself a bare word token: ``"grouchy"`` tokenizes to ``["gr", "##ou", "##chy"]``, and flipping
        # position ``"gr"`` rebuilt ``"superouchy"``. Only the position form can see the continuation
        # that follows.
        grad_l2 = np.sum(grads * grads, axis=-1)
        ranked = np.argsort(-grad_l2).tolist()
        flippable = [
            p for p in ranked if model.is_substitutable_at(tokens, p) and display_form(model, tokens[p]) not in ignore
        ][:_MAX_FLIPPABLE_TOKENS]

        vocab, matrix = model.get_embedding_table()
        replacement: dict[int, str] = {}
        for pos in flippable:
            # Shortlist the top-K word candidates by the first-order estimate (most-negative first).
            # Candidate word-hood is judged by the model (``is_substitutable``) because the vocab
            # carries the tokenizer's own markers — under SentencePiece/byte-BPE every usable word is
            # ``▁word``/``Ġword``, which a bare ``isalpha`` check would reject, emptying every shortlist.
            scores = matrix @ grads[pos]
            shortlist: list[str] = []
            for cand in np.argsort(scores):
                cand_tok = vocab[int(cand)]
                if cand_tok != tokens[pos] and model.is_substitutable(cand_tok):
                    shortlist.append(cand_tok)
                    if len(shortlist) >= _CANDIDATES_PER_POSITION:
                        break
            if not shortlist:
                continue
            # ...then re-score the shortlist against the model and keep the one that most reduces the
            # original-class probability (the linear estimate's top pick is an unreliable proxy).
            cand_texts = [model.detokenize([*tokens[:pos], tok, *tokens[pos + 1 :]]) for tok in shortlist]
            cand_orig_probs = model.predict(cand_texts)[:, orig_class]
            replacement[pos] = shortlist[int(np.argmin(cand_orig_probs))]

        flippable = [p for p in flippable if p in replacement]
        return self.search_minimal(
            text,
            tokens,
            flippable,
            replacement.__getitem__,
            max_size=max_flips,
            num_examples=num_examples,
            orig_probs=orig_probs,
            target_class=target_class,
        )
