"""AblationFlip counterfactual generator — forward-pass-only token removal.

The perturbation-based sibling of :class:`~shapash.compute.generators.hotflip.HotFlipGenerator`. Where
HotFlip *substitutes* tokens using embedding gradients, AblationFlip *removes* tokens it scores as most
supportive of the current prediction, until the prediction flips:

1. Score each content token leave-one-out (:meth:`_ablation_scores`): starting from the full text,
   drop one token at a time and measure the drop in the original class's probability. Positive score
   = the token supports the original class (removing it hurts that class most).
2. Hand the highest-scoring positions to the shared minimal search
   (:meth:`~shapash.compute.generators.base.CounterfactualGenerator.search_minimal`), expressing a
   removal as an *empty* replacement, and keep the minimal removal sets that flip the prediction.

It needs only forward passes (``predict``) plus tokenization — no gradients, no torch, no captum — so
unlike HotFlip it also works on prediction-only models such as ``HFPipelineModel``. The scoring step is
kept separate from the search so each is testable on its own.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapash.compute.generators.base import (
    Counterfactual,
    CounterfactualGenerator,
    Field,
    IntField,
    TokenListField,
)
from shapash.compute.generators.cf_utils import display_form
from shapash.model.base import SupportsTokenization, TextModel, has_capabilities

if TYPE_CHECKING:
    from collections.abc import Sequence


class AblationFlipGenerator(CounterfactualGenerator):
    """Perturbation-based token-removal counterfactuals (Captum ``FeatureAblation``)."""

    name = "ablation_flip"
    display_name = "Ablation"

    def config_spec(self) -> dict[str, Field]:
        """Expose ``num_examples``, ``max_ablations`` and ``tokens_to_ignore`` as tunable knobs."""
        return {
            "num_examples": IntField(label="Max counterfactuals", default=5, minimum=1, maximum=20),
            "max_ablations": IntField(label="Max token removals", default=3, minimum=1, maximum=5),
            "tokens_to_ignore": TokenListField(label="Tokens to ignore", default=[]),
        }

    @classmethod
    def is_compatible(cls, model: TextModel) -> bool:
        """Compatible with any tokenizable model — only forward passes and tokenization are needed."""
        return has_capabilities(model, SupportsTokenization)

    def generate(self, text: str, target_label: str | None = None, config: dict | None = None) -> list[Counterfactual]:
        """Generate minimal token-removal counterfactuals for ``text`` (see module docstring)."""
        cfg = self.resolve_config(config)
        num_examples = int(cfg["num_examples"])
        max_ablations = int(cfg["max_ablations"])
        ignore = {t.strip().lower() for t in cfg["tokens_to_ignore"]}

        model = self.model
        if not isinstance(model, SupportsTokenization):  # is_compatible guarantees this; guard direct use
            raise TypeError("AblationFlipGenerator requires a model implementing SupportsTokenization.")
        label_names = model.label_names or []

        orig_probs = model.predict([text])[0]
        orig_class = int(np.argmax(orig_probs))
        target_class = label_names.index(target_label) if (target_label and target_label in label_names) else None

        tokens = model.tokenize(text)
        # Removable positions are decided by the *model's* tokenization scheme (``is_substitutable_at``),
        # not by a string rule here: under SentencePiece/byte-BPE every content token carries a
        # ``▁``/``Ġ`` word-start marker, which a bare ``isalpha`` check rejects wholesale. ``ignore``
        # holds user-typed words, so it is matched against each token's display form for the same
        # reason — ``"great"`` must match the token ``"▁great"``.
        #
        # The *position* form matters here as much as in HotFlip: dropping the head of a multi-piece
        # word strands its continuations, turning ``["gr", "##ou", "##chy"]`` into ``"ouchy"`` rather
        # than removing ``"grouchy"``.
        content_positions = [
            i
            for i, t in enumerate(tokens)
            if model.is_substitutable_at(tokens, i) and display_form(model, t) not in ignore
        ]
        if not content_positions:
            return []

        scores = self._ablation_scores(tokens, content_positions, orig_class)
        # Keep the most supportive positions (largest drop when removed) as removal candidates.
        ranked = [content_positions[j] for j in np.argsort(-scores)][: self.max_candidate_positions]
        return self.search_minimal(
            text,
            tokens,
            ranked,
            lambda _position: "",  # an ablation is a substitution by nothing
            max_size=max_ablations,
            num_examples=num_examples,
            orig_probs=orig_probs,
            target_class=target_class,
        )

    def _ablation_scores(self, tokens: list[str], content_positions: Sequence[int], orig_class: int) -> np.ndarray:
        """Score each content token by the drop in ``orig_class`` probability when it is removed.

        Leave-one-out over the content positions: score the full text once, then once per position with
        that token dropped, and take the difference. Returns one score per entry of
        ``content_positions`` (aligned by order); a positive score means the token supports
        ``orig_class``.

        This is what Captum's ``FeatureAblation`` computes over a token-presence mask, done directly
        instead. The single ``predict`` call lets the model batch the perturbations on its own device,
        where ``FeatureAblation`` issued one forward pass per token — the dominant cost of a
        counterfactual run on CPU. It also drops torch and captum from this path entirely, so
        AblationFlip now needs nothing beyond the model's own ``predict``.
        """
        model = self.model
        if not isinstance(model, SupportsTokenization):  # is_compatible guarantees this; guard direct use
            raise TypeError("AblationFlipGenerator requires a model implementing SupportsTokenization.")

        texts = [model.detokenize(list(tokens))]
        texts += [model.detokenize([t for i, t in enumerate(tokens) if i != p]) for p in content_positions]
        probs = model.predict(texts)[:, orig_class]
        return probs[0] - probs[1:]
