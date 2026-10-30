"""Serializable "current datapoint" payload shared across the NLP webapp panels.

A single ``dcc.Store(id="current-datapoint")`` is the app's *primary selection* (LIT's term): the one
text every per-instance panel points at. It is written from two sources — a selected dataset row
(precomputed contributions) or the data editor's live ``explain_text`` on a scratch text — and read by
the sentence-highlight, waterfall, and counterfactual panels. Keeping both writers and all readers on a
single JSON-serialisable shape is what lets "select a row" and "edit + predict a new text" flow through
the same downstream code.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from shapash.explainer.nlp_explanation import select_label_column


def pack_datapoint(
    *,
    text: str,
    orig_idx: int | None,
    tokens: list[str],
    values: Any,
    base_values: Any,
    label: str | None = None,
) -> dict:
    """Serialise one sample into the ``current-datapoint`` store shape.

    Parameters
    ----------
    text : str
        The raw text of the datapoint.
    orig_idx : int or None
        Row index in the compiled batch, or ``None`` for a scratch (editor) datapoint that has no
        precomputed row.
    tokens : list of str
        Token strings for the sample.
    values : array-like
        Per-token SHAP values, shape ``(n_tokens,)`` or ``(n_tokens, n_classes)``. Stored whole so the
        detail panels can re-slice by the active class without re-selecting.
    base_values : array-like or None
        Per-class base value(s) for the sample, or ``None``.
    label : str or None, optional
        Predicted label, when known.

    Returns
    -------
    dict
        JSON-serialisable payload with keys ``text``, ``orig_idx``, ``tokens``, ``values``,
        ``base_values``, ``label``.
    """
    values_arr = np.asarray(values, dtype=float)
    base_list = None if base_values is None else np.asarray(base_values, dtype=float).reshape(-1).tolist()
    return {
        "text": text,
        "orig_idx": orig_idx,
        "tokens": list(tokens),
        "values": values_arr.tolist(),
        "base_values": base_list,
        "label": label,
    }


def datapoint_from_contributions(text: str, contributions: Any, label: str | None = None) -> dict:
    """Pack a scratch datapoint from a freshly computed single-text explanation.

    Shared by the data editor (Predict) and the counterfactual panel (Apply): both call
    ``engine.explain_text`` and turn its ``NlpContributions`` (batch of one) into the store shape.

    Parameters
    ----------
    text : str
        The explained text.
    contributions : NlpContributions
        The result of ``engine.explain_text(text)`` — a one-sample batch.
    label : str or None, optional
        Predicted label, when known.

    Returns
    -------
    dict
        A :func:`pack_datapoint` payload with ``orig_idx=None`` (scratch).
    """
    base_values = contributions.base_values
    return pack_datapoint(
        text=text,
        orig_idx=None,
        tokens=contributions.token_strings[0],
        values=contributions.values[0],
        base_values=(base_values[0] if base_values is not None else None),
        label=label,
    )


def unpack_datapoint(dp: dict, label_idx: int) -> tuple[list[str], np.ndarray, float | None, str | None]:
    """Slice a stored datapoint down to one class for rendering.

    Parameters
    ----------
    dp : dict
        A payload produced by :func:`pack_datapoint`.
    label_idx : int
        Index of the class to render.

    Returns
    -------
    tokens : list of str
    values : numpy.ndarray
        1-D per-token values for ``label_idx``.
    base_value : float or None
        Base value for ``label_idx`` (falls back to the first entry when there is a single scalar).
    label : str or None
        The stored predicted label, if any.
    """
    tokens = dp["tokens"]
    values = np.asarray(dp["values"], dtype=float)
    vals = select_label_column(values, label_idx)
    base_list = dp.get("base_values")
    base_value: float | None = None
    if base_list:
        base_value = float(base_list[label_idx]) if label_idx < len(base_list) else float(base_list[0])
    return tokens, vals, base_value, dp.get("label")
