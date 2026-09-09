"""``NlpPlotter`` — rendering helpers bound to one :class:`NlpExplanation`.

Reached as ``explanation.plot``. It holds a reference to the artifact and nothing else: every
call slices that artifact, hands the slice to a pure function in :mod:`shapash.plots`, and
returns the figure. No display state is stored anywhere — pass different arguments and you get
a different figure from the *same* untouched explanation.

Why a separate object rather than methods on :class:`NlpExplanation` itself: the explanation is
the persistence boundary (``save``/``load`` with a ``format_version``), so keeping rendering off
it keeps the dataclass a dataclass. It also mirrors the tabular side, where ``SmartExplainer``
exposes :class:`~shapash.explainer.smart_plotter.SmartPlotter` as ``.plot``.

The webapp deliberately does *not* go through here: its panels render ``dcc.Store`` datapoints
via :func:`~shapash.webapp.nlp_components.datapoint.unpack_datapoint`, because a what-if
datapoint has no row in the artifact at all. This accessor serves notebook use, including a
snapshot reloaded with :meth:`NlpExplanation.load`, which has no model and no backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from dash import html
from plotly import graph_objs as go

from shapash.explainer.nlp_explanation import (
    WORD_AGGREGATIONS,
    aggregate_word_contributions,
    select_label_column,
)
from shapash.plots.plot_confusion_matrix import plot_confusion_matrix
from shapash.plots.plot_sentence_highlight import plot_sentence_highlight
from shapash.plots.plot_token_highlight import plot_token_highlight
from shapash.plots.plot_waterfall import plot_waterfall
from shapash.plots.plot_word_importance import plot_word_importance, word_importance_axis_title
from shapash.plots.plot_word_profile import plot_word_profile
from shapash.webapp.utils.dash_to_html import DashHtmlPreview

if TYPE_CHECKING:
    from shapash.explainer.nlp_explanation import NlpExplanation

# The arguments that decide how word units are *keyed*, and so how many times each one occurs.
# Shared by word_importance and word_counts; forwarding exactly these keeps a hover count equal to
# the count its bar's aggregate was computed over.
_COUNT_KWARGS = ("lowercase", "filter_special", "filter_punctuation", "sample_indices")


class NlpPlotter:
    """Plots over one :class:`~shapash.explainer.nlp_explanation.NlpExplanation`.

    Not instantiated directly — reach it as ``explanation.plot``.

    Parameters
    ----------
    explanation : NlpExplanation
        The artifact to render. Held by reference and never written to.

    Examples
    --------
    >>> explanation, _ = NlpExplanation.load("run.zip")  # no model needed
    >>> explanation.plot.waterfall(row=0, label_idx=1).show()
    """

    def __init__(self, explanation: NlpExplanation) -> None:
        self._exp = explanation

    def __repr__(self) -> str:
        exp = self._exp
        return f"<NlpPlotter over {len(exp)} sample(s), backend={exp.backend_name!r}>"

    # ── internals ───────────────────────────────────────────────────────────────────────
    def _check_label_idx(self, label_idx: int) -> int:
        """Fail with the class list rather than a bare numpy ``IndexError``."""
        n_classes = self._exp.n_classes
        if not -n_classes <= label_idx < n_classes:
            names = self._exp.label_names
            known = f" Classes: {names}." if names else ""
            raise IndexError(f"label_idx={label_idx} is out of range for {n_classes} output column(s).{known}")
        return label_idx % n_classes

    def _label_name(self, label_idx: int) -> str | None:
        names = self._exp.label_names
        return names[label_idx] if names is not None and label_idx < len(names) else None

    def _slice(self, row: int, label_idx: int) -> tuple[list[str], np.ndarray, float | None]:
        """One sample, one class — the shape every per-instance plot function takes.

        Returns ``(tokens, 1-D values, base_value)``. ``base_value`` is ``None`` when the
        backend had no reference at all (``reference_kind == "none"``).
        """
        exp = self._exp
        if not -len(exp) <= row < len(exp):
            raise IndexError(f"row={row} is out of range for {len(exp)} sample(s).")

        tokens = exp.token_strings[row]
        values = select_label_column(exp.values[row], label_idx)

        base = exp.base_values
        base_value: float | None = None
        if base is not None:
            base_value = float(base[row, label_idx]) if base.ndim == 2 else float(base[row])

        return list(tokens), np.asarray(values, dtype=float), base_value

    # ── per-instance plots ──────────────────────────────────────────────────────────────
    def tokens(
        self,
        row: int = 0,
        label_idx: int = 0,
        max_tokens: int | None = None,
        title: str | None = None,
        width: int = 900,
        height: int | None = None,
    ) -> go.Figure:
        """Bar chart of one sample's token contributions for one class.

        Parameters
        ----------
        row : int
            Positional index of the sample (``0`` is the first row, negatives count from the
            end). This is a position, not a label from ``texts.index``.
        label_idx : int
            Index of the class to display, in ``label_names`` order.
        max_tokens : int, optional
            Keep only the ``max_tokens`` highest-magnitude tokens, in sentence order.
        title : str, optional
            Overrides the default, which names the class when the model exposes class names.
        width, height : int, optional
            Figure size in pixels.

        Returns
        -------
        plotly.graph_objs.Figure
        """
        label_idx = self._check_label_idx(label_idx)
        toks, values, _ = self._slice(row, label_idx)
        name = self._label_name(label_idx)
        if title is None:
            title = f"Token contributions — {name}" if name else "Token contributions"
        return plot_token_highlight(
            tokens=toks, values=values, title=title, max_tokens=max_tokens, width=width, height=height
        )

    def waterfall(
        self,
        row: int = 0,
        label_idx: int = 0,
        min_pct: float = 0.10,
        filter_special: bool = True,
        title: str | None = None,
        width: int | None = None,
    ) -> go.Figure:
        """Waterfall decomposing one prediction from the baseline into token contributions.

        Parameters
        ----------
        row : int
            Positional index of the sample (see :meth:`tokens`).
        label_idx : int
            Index of the class to display, in ``label_names`` order.
        min_pct : float
            Fraction of the largest absolute contribution below which tokens collapse into a
            single "other" bar. Range [0, 1].
        filter_special : bool
            Drop ``[CLS]``/``[SEP]``-style and ``##subword`` tokens.
        title : str, optional
            Overrides the default, which names the class when the model exposes class names.
        width : int, optional
            Figure width in pixels.

        Returns
        -------
        plotly.graph_objs.Figure

        Raises
        ------
        ValueError
            If the producing backend is not additive. A waterfall's running total only means
            something when the contributions sum to the prediction, which is exactly what
            :attr:`~shapash.explainer.nlp_explanation.NlpExplanation.is_additive` records — so
            this refuses rather than drawing a chart whose arithmetic is meaningless.
        """
        exp = self._exp
        if not exp.is_additive:
            raise ValueError(
                f"A waterfall decomposes a prediction into contributions that sum to it, but the "
                f"{exp.backend_name!r} backend is not additive, so the running total would be "
                f"meaningless. Use .plot.tokens() or .plot.sentence() instead."
            )
        label_idx = self._check_label_idx(label_idx)
        toks, values, base_value = self._slice(row, label_idx)
        name = self._label_name(label_idx)
        if title is None:
            title = f"Token contributions — {name}" if name else "Token contributions"
        return plot_waterfall(
            tokens=toks,
            values=values,
            base_value=base_value,
            min_pct=min_pct,
            filter_special=filter_special,
            title=title,
            width=width,
        )

    def sentence(self, row: int = 0, label_idx: int = 0, notebook: bool = False) -> html.Div | DashHtmlPreview:
        """One sample rendered inline, each token background-shaded by its contribution.

        Parameters
        ----------
        row : int
            Positional index of the sample (see :meth:`tokens`).
        label_idx : int
            Index of the class to display, in ``label_names`` order.
        notebook : bool
            ``False`` (default) returns the raw Dash component — directly usable as ``children``
            in your own Dash layout, the same way :meth:`tokens`/:meth:`waterfall` return a
            ``go.Figure`` directly usable in ``dcc.Graph(figure=...)``. ``True`` wraps it in a
            :class:`~shapash.webapp.utils.dash_to_html.DashHtmlPreview`, so leaving the call as
            the last expression in a notebook cell (or ``display()``-ing it) renders it as static
            HTML — a Dash component's own ``str()``/``repr()`` is Python source, not markup, so
            ``IPython.display.HTML(...)`` on the raw component would just show that source as text.

        Returns
        -------
        dash.html.Div or DashHtmlPreview
            A Dash component (``notebook=False``), or that same component wrapped for notebook
            display (``notebook=True`` — the original is still reachable as ``.component``).
        """
        label_idx = self._check_label_idx(label_idx)
        toks, values, base_value = self._slice(row, label_idx)
        div = plot_sentence_highlight(tokens=toks, values=values, base_value=base_value)
        return DashHtmlPreview(div) if notebook else div

    # ── batch-level plots ───────────────────────────────────────────────────────────────
    def word_importance(
        self,
        label_idx: int | None = 0,
        n_top: int = 20,
        title: str | None = None,
        width: int = 900,
        height: int | None = None,
        **kwargs: Any,
    ) -> go.Figure:
        """Corpus-level word importance for one class, or across all of them.

        Parameters
        ----------
        label_idx : int or None
            Index of the class to aggregate, in ``label_names`` order. ``None`` collapses across
            classes into ``max_c |statistic|`` — "how hard does this word push the model at all",
            regardless of where it pushes. The bars are then magnitudes, so they are all one
            colour and ``filter_sign`` has nothing to select on; the per-class breakdown of a word
            found this way is :meth:`word_profile`.
        n_top : int
            Number of words to show, ranked by ``|mean contribution|``.
        title : str, optional
            Overrides the default, which names the class when the model exposes class names.
        width, height : int, optional
            Figure size in pixels.
        **kwargs
            Forwarded to
            :meth:`~shapash.explainer.nlp_explanation.NlpExplanation.word_importance` —
            ``filter_special``, ``filter_punctuation``, ``lowercase``, ``filter_sign``,
            ``exclude_words``, ``sample_indices``, ``rank_by``, ``min_occurrences``.

        Returns
        -------
        plotly.graph_objs.Figure

        Notes
        -----
        The default ranking is by ``|mean|`` with no frequency floor, which on a real corpus is
        dominated by words seen once. Pass ``min_occurrences=3`` (or ``rank_by="sum"``, which
        weights by frequency inherently) for a ranking that reflects stable model behaviour.
        """
        across_classes = label_idx is None
        if label_idx is not None:
            label_idx = self._check_label_idx(label_idx)
        word_imp = self._exp.word_importance(label_idx=label_idx, n_top=n_top, **kwargs)
        name = self._label_name(label_idx) if label_idx is not None else None
        if title is None:
            if across_classes:
                title = "Word importance — all classes"
            else:
                title = f"Word importance — {name}" if name else "Word importance"
        # The axis names whichever statistic word_importance actually returned (it stamps its
        # rank_by onto the Series name), so a total is never read as an average — and, across
        # classes, so a magnitude is never read as a signed contribution.
        x_title = word_importance_axis_title(str(word_imp.name))
        # Hover counts, computed under the same filters/scope as the ranking so a bar's count is
        # the one its aggregate was taken over. Only the keying arguments are forwarded — the rest
        # (filter_sign, exclude_words, rank_by...) select and order words, they do not change how
        # many times a word occurs.
        count_kwargs = {k: v for k, v in kwargs.items() if k in _COUNT_KWARGS}
        counts = self._exp.word_counts(**count_kwargs)
        return plot_word_importance(
            word_imp,
            title=title,
            x_title=x_title,
            width=width,
            height=height,
            counts=counts["n_occurrences"],
        )

    def word_profile(
        self,
        word: str,
        agg: str = "mean",
        lowercase: bool | None = None,
        sample_indices: list[int] | None = None,
        title: str | None = None,
        width: int | None = 640,
        height: int | None = None,
    ) -> go.Figure:
        """Profile of one word: its aggregated contribution to every class at once.

        The complement of :meth:`word_importance`, which fixes a class and ranks the vocabulary.
        Here the word is fixed and every class is shown, which is the view that answers "what does
        this word mean to the model" — including the multi-class case where a word pulls toward
        one class and away from another.

        Parameters
        ----------
        word : str
            The word to profile, matched the way :meth:`word_importance` keys its units.
        agg : {"mean", "sum", "mean_abs", "sum_abs"}
            How the word's occurrences collapse into one number per class. See
            :func:`~shapash.explainer.nlp_explanation.aggregate_word_contributions` for what each
            one does and does not tell you. Mean aggregations also draw the standard deviation
            across occurrences as error bars.
        lowercase : bool, optional
            Case-fold the match. ``None`` (default) defers to the model's tokenizer.
        sample_indices : list[int], optional
            Restrict the aggregation to these samples (e.g. only the misclassified ones).
        title : str, optional
            Overrides the default, which names the word and the aggregation.
        width, height : int, optional
            Figure size in pixels.

        Returns
        -------
        plotly.graph_objs.Figure

        Raises
        ------
        ValueError
            If the word does not occur in the batch (or in ``sample_indices``) — an empty chart
            here is indistinguishable from a word with zero contribution, so it says so instead.
        """
        occurrences = self._exp.word_occurrences(word, lowercase=lowercase, sample_indices=sample_indices)
        if occurrences.empty:
            scope = " in the selected samples" if sample_indices is not None else ""
            raise ValueError(f"{word!r} does not occur{scope} in this explanation.")
        stats = aggregate_word_contributions(occurrences, agg=agg)
        agg_label = WORD_AGGREGATIONS[agg][0]

        spread = None
        if WORD_AGGREGATIONS[agg][2] == "mean":
            values = occurrences["contribution"].abs() if WORD_AGGREGATIONS[agg][1] else occurrences["contribution"]
            # ddof=0: this is the spread of the occurrences actually observed, not an estimate of a
            # population — and it keeps a single-occurrence word at 0 rather than NaN.
            spread = values.groupby(occurrences["class_idx"]).std(ddof=0).sort_index()

        n_occ = len(occurrences) // max(self._exp.n_classes, 1)
        n_samples = int(occurrences["sample"].nunique())
        if title is None:
            title = f"{word!r} — {agg_label} over {n_occ} occurrence(s) in {n_samples} sample(s)"
        return plot_word_profile(
            stats,
            label_names=self._exp.label_names,
            spread=spread,
            x_title=f"{agg_label} contribution",
            title=title,
            width=width,
            height=height,
        )

    def confusion(
        self,
        normalize: str | None = None,
        title: str = "Confusion matrix",
        width: int | None = None,
        height: int | None = None,
    ) -> go.Figure:
        """Confusion matrix of predictions against ground truth.

        Parameters
        ----------
        normalize : {None, "true"}, optional
            ``"true"`` divides each row by its sum so cells show recall. ``None`` shows counts.
        title : str
            Figure title.
        width, height : int, optional
            Figure size in pixels.

        Returns
        -------
        plotly.graph_objs.Figure

        Raises
        ------
        ValueError
            If no ground truth was supplied to ``explain()``.
        """
        exp = self._exp
        if not exp.has_ground_truth:
            raise ValueError(
                "A confusion matrix needs ground truth, but this explanation has no y_true. "
                "Pass y_true to explain() to enable it."
            )
        cm = exp.confusion_matrix()
        labels = list(exp.label_to_idx)
        return plot_confusion_matrix(cm, labels, normalize=normalize, title=title, width=width, height=height)
