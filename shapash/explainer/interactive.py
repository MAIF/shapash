"""Interactive (live) compute contract for what-if tooling.

A compiled explanation result is a **read-only** view over past output — it deliberately cannot run
the model on new inputs. Interactive what-if (re-predicting text the user edits, generating
counterfactuals on the fly) needs the opposite: live access to the model, explanation backend and
generator.

``InteractiveEngine`` is that live counterpart, kept separate from the read-only view (mirroring LIT
keeping ``Model`` separate from data). Webapp components that mutate/regenerate inputs *require* an
engine and self-disable when one is absent (e.g. an explainer restored from a snapshot, which holds
no model). ``NlpExplainer`` implements this Protocol.

The capability flags (``can_edit`` / ``can_counterfactual``) let components check exactly what the
bound engine supports, the same way generators check model capabilities.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from shapash.backend.nlp_backend import NlpContributions
from shapash.compute.diagnostics.label_noise import LabelNoiseReport
from shapash.compute.generators.base import Counterfactual, Field
from shapash.compute.retrieval.similar_examples import Neighbor
from shapash.explainer.nlp_explanation import NlpExplanation


@runtime_checkable
class InteractiveEngine(Protocol):
    """Live compute surface for interactive text what-if components."""

    def can_edit(self) -> bool:
        """Whether live re-prediction / re-explanation of edited text is available."""
        ...

    def can_counterfactual(self) -> bool:
        """Whether automated counterfactual generation is available."""
        ...

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        """Return ``(predicted_label, {label: probability})`` for a single text."""
        ...

    def explain_text(self, text: str) -> tuple[NlpContributions, str, dict[str, float]]:
        """Re-explain a single (possibly edited) text.

        Returns
        -------
        contributions : NlpContributions
            Token-level contributions for the single text (one sample).
        label : str
            Predicted label.
        probabilities : dict[str, float]
            Per-class probabilities.
        """
        ...

    def available_cf_generators(self) -> list[tuple[str, str]]:
        """Return ``(name, display_name)`` for each selectable counterfactual generator.

        Empty when no generator is bound. The webapp renders a method selector from this list.
        """
        ...

    def generate_counterfactuals(
        self, text: str, config: dict | None = None, generator: str | None = None
    ) -> list[Counterfactual]:
        """Generate counterfactuals for ``text`` using the selected (or active) generator."""
        ...

    def cf_config_spec(self, generator: str | None = None) -> dict[str, Field]:
        """Return the selected (or active) generator's config spec (empty when no generator)."""
        ...

    def can_find_similar(self) -> bool:
        """Whether similar-example retrieval over a reference corpus is available."""
        ...

    def find_similar(self, text: str, top_k: int = 5) -> list[Neighbor]:
        """Return the ``top_k`` reference examples most similar to ``text``, most-similar first."""
        ...

    def find_similar_threshold(self, text: str, threshold: float = 0.95, limit: int = 50) -> tuple[list[Neighbor], int]:
        """Return reference examples scoring above ``threshold`` (not a fixed count), capped at ``limit``.

        Returns ``(neighbors, total)`` where ``total`` is the count that cleared ``threshold`` before
        capping to ``limit``.
        """
        ...

    def can_detect_label_noise(self, explanation: NlpExplanation) -> bool:
        """Whether ``explanation`` carries the ground truth + per-class probabilities this needs."""
        ...

    def can_probe_labels(self) -> bool:
        """Whether a labelled reference corpus is bound to second-guess flagged labels."""
        ...

    def detect_label_noise(
        self, explanation: NlpExplanation, top_n: int = 50, score: str = "self_confidence", probe: bool = True
    ) -> LabelNoiseReport:
        """Rank probably-mislabelled samples in ``explanation`` by confident learning."""
        ...
