"""Counterfactual utility functions (ports of LIT ``cf_utils``).

Small, pure helpers shared by generators: detecting a prediction flip and measuring how far a
counterfactual moved the original class probability.

Token word-hood used to live here too, as a bare ``token.isalpha()``. It now lives in the model layer
(:func:`~shapash.model.base.is_word_token` and the scheme-aware
:meth:`~shapash.model.base.SupportsTokenization.is_substitutable`), because the answer depends on the
tokenization scheme and only the model knows which one it uses.
"""

from __future__ import annotations

import numpy as np


def is_prediction_flip(orig_probs: np.ndarray, cf_probs: np.ndarray, target_class: int | None = None) -> bool:
    """Return ``True`` when the counterfactual changes the predicted class.

    Parameters
    ----------
    orig_probs : np.ndarray, shape (n_classes,)
        Class probabilities for the original text.
    cf_probs : np.ndarray, shape (n_classes,)
        Class probabilities for the counterfactual text.
    target_class : int or None
        When given, success requires the argmax to become exactly ``target_class``; otherwise any
        change of argmax counts.

    Returns
    -------
    bool
        Whether the prediction flipped (towards ``target_class`` if specified).
    """
    orig_class = int(np.argmax(orig_probs))
    cf_class = int(np.argmax(cf_probs))
    if target_class is not None:
        return cf_class == target_class and cf_class != orig_class
    return cf_class != orig_class


def prediction_difference(orig_probs: np.ndarray, cf_probs: np.ndarray, class_idx: int) -> float:
    """Return how much ``class_idx``'s probability dropped from original to counterfactual."""
    return float(orig_probs[class_idx] - cf_probs[class_idx])


def format_edit(old_token: str, new_token: str) -> str:
    """Render one token edit for display: a substitution as ``old→new``, a removal as ``−old``.

    An empty ``new_token`` means the position was dropped rather than replaced (how removal-based
    generators such as AblationFlip express themselves in ``Counterfactual.substitutions``) — spelled
    out here so a removal doesn't read as "replaced with nothing" but as what it is, a drop.
    """
    return f"{old_token}→{new_token}" if new_token else f"−{old_token}"


def display_form(model, token: str) -> str:
    """Return the user-facing spelling of ``token`` — markers stripped, lowercased.

    Generators match tokens against user-typed word lists (``tokens_to_ignore``, typed into the
    webapp), which hold plain words — while a token may carry a ``▁``/``Ġ`` word-start marker. Comparing
    the two directly means ``"great"`` never matches the token ``"▁great"``, so the ignore list silently
    does nothing on SentencePiece and byte-BPE models. Round-tripping through the model's own
    ``detokenize`` yields the display spelling for any tokenization scheme, with no marker table here.

    Parameters
    ----------
    model : SupportsTokenization
        The model whose tokenizer produced ``token``.
    token : str
        A single token string.

    Returns
    -------
    str
        The token as a user would write it.
    """
    return model.detokenize([token]).strip().lower()
