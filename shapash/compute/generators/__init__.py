"""Counterfactual / what-if generators — the generative axis of the compute layer.

Mirrors Google PAIR LIT's ``Generator`` component: given one input, produce synthetic variants
that change the model's prediction. Generators declare their model-capability requirements via
``is_compatible`` and their tunable knobs via ``config_spec`` (auto-rendered by the webapp).

Ships :class:`HotFlipGenerator` (gradient-based token substitution) and :class:`AblationFlipGenerator`
(forward-pass-only token removal, via Captum ``FeatureAblation``) — both using the same ABC.
"""

from __future__ import annotations

from shapash.compute.generators.ablation_flip import AblationFlipGenerator
from shapash.compute.generators.base import (
    Counterfactual,
    CounterfactualGenerator,
    Field,
    IntField,
    TokenListField,
)
from shapash.compute.generators.hotflip import HotFlipGenerator

__all__ = [
    "AblationFlipGenerator",
    "Counterfactual",
    "CounterfactualGenerator",
    "Field",
    "IntField",
    "TokenListField",
    "HotFlipGenerator",
]
