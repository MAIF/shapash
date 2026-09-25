"""Label-noise detection by confident learning.

Error analysis usually bottoms out in "the label is wrong". Confident learning (Northcutt, Jiang &
Chuang, *Confident Learning: Estimating Uncertainty in Dataset Labels*, JAIR 2021 — the algorithm
behind ``cleanlab``) turns a matrix of predicted probabilities plus the given labels into two
things: a **ranked list of probably-mislabelled samples**, and an **estimated noise matrix** saying
which class pairs are contaminated and by how much.

It needs nothing an explainer has not already computed: the ``(n_samples, n_classes)`` probability
matrix and the ground-truth labels. No retraining, no cross-validation, no model access — which is
also why this module is pure numpy and holds no reference to a model or a backend.

Implemented in-house rather than by depending on ``cleanlab``: the core is the ~150 lines below, and
a new dependency would have to keep its ``scikit-learn`` pin inside this project's narrow
``>=1.8,<1.9`` window for a routine the project can own outright. (Same call as the retrieval
package's, which reimplemented Captum's ``SimilarityInfluence``.)

Precondition
------------
Confident learning's guarantees assume the probabilities are **out-of-sample** for the labelled
data — produced by a model that did not train on these rows (a held-out split, or cross-validated
predictions). Applied to a model's own training data, self-confidence is inflated by memorisation
and the method systematically *under*-reports noise. Nothing here can detect that, so it is the
caller's precondition to satisfy; :func:`detect_label_issues` will happily run either way.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from shapash.compute.diagnostics.label_probe import ProbeVerdict

logger = logging.getLogger(__name__)

#: Ranking methods accepted by ``detect_label_issues(score=...)``.
SCORE_METHODS = ("self_confidence", "normalized_margin")

# Row sums are checked only loosely: a caller may legitimately pass calibrated or temperature-scaled
# scores that do not sum to exactly one.
_PROB_SUM_TOLERANCE = 1e-3

# Slack on the "clears its class threshold" test. A threshold is the *mean* of the very values it is
# compared against, so when a class's samples share a self-confidence the mean lands a few ulps off
# it and an exact ``>=`` rejects every one of them — the class then abstains wholesale and vanishes
# from the confident joint. The slack is far below any meaningful probability difference.
_THRESHOLD_EPSILON = 1e-9


@dataclass(frozen=True)
class LabelIssue:
    """One sample whose given label confident learning believes is wrong.

    Attributes
    ----------
    index : int
        Position of the sample within the analysed corpus.
    text : str
        The sample's text.
    given_label : str
        The label the dataset carries.
    suggested_label : str
        The label the probabilities point to instead.
    given_prob : float
        Predicted probability of ``given_label``.
    suggested_prob : float
        Predicted probability of ``suggested_label``.
    score : float
        Label-quality score, **lower means more likely mislabelled**. Its meaning depends on the
        ``score`` method requested (see :func:`detect_label_issues`).
    probe : ProbeVerdict or None
        A model-independent second opinion on ``given_label``, attached by a caller holding a
        :class:`~shapash.compute.diagnostics.label_probe.LabelProbe`; ``None`` when none was
        available. Reported beside ``score``, never folded into it — and it is the only field here
        that does not derive from the audited model, which is exactly what makes it able to
        distinguish a label error from a confidently-wrong model. See
        :mod:`shapash.compute.diagnostics.label_probe`.
    """

    index: int
    text: str
    given_label: str
    suggested_label: str
    given_prob: float
    suggested_prob: float
    score: float
    probe: ProbeVerdict | None = field(default=None)


@dataclass(frozen=True)
class LabelNoiseReport:
    """The outcome of a confident-learning pass over a labelled corpus.

    Attributes
    ----------
    issues : list of LabelIssue
        Flagged samples ordered worst-first, truncated to the requested ``top_n``.
    noise_matrix : numpy.ndarray
        Calibrated joint distribution of shape ``(n_classes, n_classes)`` summing to 1, where
        ``noise_matrix[i, j]`` estimates the fraction of the whole corpus that is *labelled* ``i``
        but truly ``j``. The diagonal is the correctly-labelled mass; the off-diagonal is the noise.
    thresholds : numpy.ndarray
        Per-class confidence thresholds of shape ``(n_classes,)`` — the average self-confidence of
        the samples carrying each label. ``inf`` for a class with no labelled samples.
    label_names : list of str
        Class names, ordered to match the probability columns and both matrix axes.
    n_samples : int
        Size of the analysed corpus.
    n_issues : int
        Total number of flagged samples **before** ``top_n`` truncation.
    """

    issues: list[LabelIssue]
    noise_matrix: np.ndarray
    thresholds: np.ndarray
    label_names: list[str]
    n_samples: int
    n_issues: int

    @property
    def noise_rate(self) -> float:
        """Estimated fraction of the corpus that is mislabelled (the off-diagonal mass)."""
        return float(self.noise_matrix.sum() - np.trace(self.noise_matrix))


def has_usable_probabilities(y_prob: pd.DataFrame | None) -> bool:
    """Whether a probability frame can support confident learning.

    The single predicate behind both the explainer's ``can_detect_label_noise()`` and the webapp's
    ``data:labels`` capability, so the two can never drift apart.

    Parameters
    ----------
    y_prob : pandas.DataFrame or None
        A compiled explainer's per-class probabilities.

    Returns
    -------
    bool
        ``True`` when the frame holds one column per class. ``False`` for ``None`` and for the
        single-column frame the raw-pipeline prediction path produces, which carries only the
        *winning* class's confidence — the losing classes' probabilities are exactly what confident
        learning needs, so there is nothing to work with.
    """
    if y_prob is None:
        return False
    shape = getattr(y_prob, "shape", None)
    return bool(shape is not None and len(shape) == 2 and shape[1] >= 2)


def detect_label_issues(
    probs: np.ndarray,
    labels: Sequence[str],
    texts: Sequence[str],
    label_names: list[str],
    *,
    top_n: int = 50,
    score: str = "self_confidence",
) -> LabelNoiseReport:
    """Rank probably-mislabelled samples and estimate the corpus's noise matrix.

    Runs the four steps of confident learning: per-class confidence thresholds, the confident joint,
    calibration to a joint distribution, then per-cell pruning by estimated noise rate.

    Parameters
    ----------
    probs : numpy.ndarray
        Predicted probabilities of shape ``(n_samples, n_classes)``, columns ordered like
        ``label_names``. Must be **out-of-sample** for the labelled data — see the module docstring.
    labels : sequence of str
        The given (possibly noisy) label of each sample. Every value must appear in ``label_names``.
    texts : sequence of str
        The samples' texts, aligned with ``labels``; carried onto each :class:`LabelIssue`.
    label_names : list of str
        Class names in probability-column order.
    top_n : int, optional
        Maximum number of issues to return. The full count is still reported as
        :attr:`LabelNoiseReport.n_issues`.
    score : {"self_confidence", "normalized_margin"}, optional
        How to rank the flagged samples. ``self_confidence`` is the predicted probability of the
        given label. ``normalized_margin`` subtracts the best competing class's probability, so it
        punishes a confident *alternative* rather than mere uncertainty. Lower is worse either way.

    Returns
    -------
    LabelNoiseReport

    Raises
    ------
    ValueError
        If ``probs`` is not 2-D, if there are fewer than two classes, if ``labels``/``texts``/
        ``label_names`` lengths disagree with ``probs``, if a label is absent from ``label_names``,
        or if ``score`` is not a recognised method.

    Notes
    -----
    A sample is only ever flagged toward a class the model scores *above* the given label
    (``margin > 0``). Confident learning's per-cell counts are estimates and can exceed the number
    of samples that actually favour the alternative; without this guard the tail of a cell would
    "suggest" a class the model itself ranks lower than the label already on the row.

    Examples
    --------
    >>> report = detect_label_issues(probs, y_true, texts, ["joy", "surprise"], top_n=10)
    >>> for issue in report.issues:
    ...     print(issue.score, issue.given_label, "->", issue.suggested_label, issue.text)
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D (n_samples, n_classes); got shape {probs.shape}.")
    n_samples, n_classes = probs.shape
    if n_classes < 2:
        raise ValueError(f"Label-noise detection needs at least 2 classes; got {n_classes}.")
    if len(label_names) != n_classes:
        raise ValueError(f"label_names has {len(label_names)} entries but probs has {n_classes} columns.")
    if len(labels) != n_samples:
        raise ValueError(f"labels has {len(labels)} entries but probs has {n_samples} rows.")
    if len(texts) != n_samples:
        raise ValueError(f"texts has {len(texts)} entries but probs has {n_samples} rows.")
    if score not in SCORE_METHODS:
        raise ValueError(f"Unknown score method {score!r}; expected one of {list(SCORE_METHODS)}.")

    name_to_idx = {name: i for i, name in enumerate(label_names)}
    unknown = sorted({str(v) for v in labels} - set(name_to_idx))
    if unknown:
        raise ValueError(f"Labels absent from label_names: {unknown}. Known classes: {label_names}.")
    given = np.array([name_to_idx[str(v)] for v in labels], dtype=int)

    if not np.allclose(probs.sum(axis=1), 1.0, atol=_PROB_SUM_TOLERANCE):
        logger.warning(
            "Probability rows do not sum to 1; confident learning's thresholds assume a proper "
            "distribution per sample, so results may be distorted."
        )

    thresholds = _class_thresholds(probs, given, n_classes)
    joint = _calibrate(_confident_joint(probs, given, thresholds), given, n_classes)
    scores = _quality_scores(probs, given, score)
    suggested = _prune_by_noise_rate(probs, given, joint, n_classes)

    # Worst first; the index breaks ties so the ordering is deterministic across runs.
    ranked = sorted(suggested, key=lambda i: (scores[i], i))
    issues = [
        LabelIssue(
            index=int(i),
            text=str(texts[i]),
            given_label=label_names[given[i]],
            suggested_label=label_names[suggested[i]],
            given_prob=float(probs[i, given[i]]),
            suggested_prob=float(probs[i, suggested[i]]),
            score=float(scores[i]),
        )
        for i in ranked[: max(0, top_n)]
    ]
    return LabelNoiseReport(
        issues=issues,
        noise_matrix=joint,
        thresholds=thresholds,
        label_names=list(label_names),
        n_samples=n_samples,
        n_issues=len(ranked),
    )


def _class_thresholds(probs: np.ndarray, given: np.ndarray, n_classes: int) -> np.ndarray:
    """Average self-confidence per class: ``t_j = mean{P[i, j] : given_i = j}``.

    A class carrying no labels gets ``inf``, so nothing is ever confidently assigned to it. Its
    threshold is unestimable, and substituting an arbitrary one would fabricate suggestions toward a
    class the corpus gives no evidence about.
    """
    thresholds = np.full(n_classes, np.inf)
    absent = []
    for j in range(n_classes):
        rows = given == j
        if rows.any():
            thresholds[j] = probs[rows, j].mean()
        else:
            absent.append(j)
    if absent:
        logger.warning(
            "Classes %s carry no labelled samples; their confidence thresholds cannot be estimated "
            "and they will never be suggested as a corrected label.",
            absent,
        )
    return thresholds


def _confident_joint(probs: np.ndarray, given: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Count ``(given, confidently-predicted)`` pairs — the confident joint ``C``.

    A sample votes only for classes it clears the threshold for; among those it votes for its
    argmax. A sample clearing no threshold abstains rather than being forced onto its argmax, which
    is what keeps ``C`` a *confident* count instead of a plain confusion matrix.
    """
    n_classes = thresholds.shape[0]
    above = probs >= thresholds - _THRESHOLD_EPSILON  # broadcast over rows; an inf threshold is never cleared
    has_candidate = above.any(axis=1)
    best = np.where(above, probs, -np.inf).argmax(axis=1)

    counts = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(counts, (given[has_candidate], best[has_candidate]), 1)
    return counts


def _calibrate(counts: np.ndarray, given: np.ndarray, n_classes: int) -> np.ndarray:
    """Scale the confident joint to the observed class counts, then to a joint summing to 1.

    Abstentions mean each row of ``counts`` under-counts its class. Rescaling every row to the
    number of samples actually carrying that label restores the marginal, so the result can be read
    as "fraction of the corpus labelled *i* but truly *j*".
    """
    counts = counts.astype(float)
    row_totals = counts.sum(axis=1, keepdims=True)
    class_counts = np.bincount(given, minlength=n_classes).astype(float).reshape(-1, 1)
    scaled = np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals != 0) * class_counts
    total = scaled.sum()
    return scaled / total if total > 0 else scaled


def _quality_scores(probs: np.ndarray, given: np.ndarray, method: str) -> np.ndarray:
    """Per-sample label-quality score, lower meaning more likely mislabelled."""
    rows = np.arange(probs.shape[0])
    self_confidence = probs[rows, given]
    if method == "self_confidence":
        return self_confidence
    competing = probs.copy()
    competing[rows, given] = -np.inf
    return self_confidence - competing.max(axis=1)


def _prune_by_noise_rate(probs: np.ndarray, given: np.ndarray, joint: np.ndarray, n_classes: int) -> dict[int, int]:
    """Select the flagged samples cell by cell — confident learning's ``prune_by_noise_rate``.

    Cell ``(i, j)`` of the joint estimates how *many* samples are labelled ``i`` but truly ``j``;
    this picks exactly that many, taking the class-``i`` samples with the largest
    ``P[:, j] - P[:, i]`` margin. Ranking within the cell rather than globally is what stops a
    single high-noise class pair from crowding out every other one.

    Returns
    -------
    dict[int, int]
        Sample index to suggested class index. A sample selected by several cells keeps the cell
        with the largest margin.
    """
    n_samples = probs.shape[0]
    best: dict[int, tuple[float, int]] = {}
    for i in range(n_classes):
        rows = np.flatnonzero(given == i)
        if rows.size == 0:
            continue
        for j in range(n_classes):
            if i == j:
                continue
            n_errors = int(round(joint[i, j] * n_samples))
            if n_errors <= 0:
                continue
            margins = probs[rows, j] - probs[rows, i]
            # Only samples the model actually prefers j for — see the Notes in detect_label_issues.
            candidates = np.flatnonzero(margins > 0)
            if candidates.size == 0:
                continue
            order = candidates[np.argsort(-margins[candidates])][:n_errors]
            for pos in order:
                sample, margin = int(rows[pos]), float(margins[pos])
                current = best.get(sample)
                if current is None or margin > current[0]:
                    best[sample] = (margin, j)
    return {sample: suggested for sample, (_margin, suggested) in best.items()}
