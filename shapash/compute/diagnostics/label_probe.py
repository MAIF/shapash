"""An independent second opinion on a label, from a model that is not the one under audit.

Confident learning flags a sample when the audited model confidently disagrees with its label. It
cannot say *which* of the two is wrong: a genuine label error and a confidently-wrong model produce
the same evidence. On a strong classifier the second case dominates — the emotion demo's distilbert
is 90.6% accurate against an estimated 3% label noise, so most of what it disagrees with is its own
mistake, not the corpus's.

Breaking the tie needs a signal that does not come from the audited model. This is that signal: a
cheap bag-of-words classifier (TF-IDF + multinomial logistic regression) fit on a *labelled reference
corpus*, then asked one question per flagged row — **what probability does it put on the label the
row already carries?** High means the reference corpus backs the label and the audited model is
probably just wrong; low means both models agree the label is off.

Why a separate classifier rather than nearest neighbours in the model's representation space: that
space is the one feeding the classifier head, so its neighbours restate the prediction (measured at
98.6% agreement on the emotion demo) and back the *wrong* prediction on 87% of the model's errors.
Anything derived from the audited model corroborates it by construction.

Why bag-of-words rather than sentence embeddings: measured on the emotion demo, TF-IDF + logistic
regression scores 0.658 accuracy and agrees with the audited model 66.6% of the time, against 0.636
and 64.2% for ``all-MiniLM-L6-v2`` + logistic regression — better on both axes, with no dependency
beyond the scikit-learn already in core. A weak-but-independent probe beats a strong-but-correlated
one for this job, and the probe never has to be right on its own: it only has to be wrong for
*different reasons*.

Limitations
-----------
The probe is weak in absolute terms (0.658 against the audited model's 0.906) and is **not** a
detector. Run as one it estimates a ~29% noise rate on a corpus whose real noise is a few percent.
Its output is only meaningful as a per-row second opinion on rows confident learning already flagged,
and never as a ranking, a threshold, or a noise-rate estimate.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

logger = logging.getLogger(__name__)

#: Minimum reference examples per class for the probe to be worth fitting. Below this the estimate is
#: noise, and a confident-looking second opinion drawn from three examples is worse than none.
MIN_PER_CLASS = 5


@dataclass(frozen=True)
class ProbeVerdict:
    """The independent probe's read on one row's given label.

    Attributes
    ----------
    given_prob : float
        Probability the probe assigns to the label the row already carries. **High is the
        interesting case**: the reference corpus backs the label, so the audited model's confident
        disagreement is more likely its own error than a label error.
    top_label : str
        The class the probe itself would pick.
    backs_given : bool
        Whether ``top_label`` is the given label — a threshold-free reading of "the probe sides with
        the corpus against the audited model".
    """

    given_prob: float
    top_label: str
    backs_given: bool


class LabelProbe:
    """A model-independent second opinion on given labels, fit on a labelled reference corpus.

    Parameters
    ----------
    reference_texts : sequence of str
        Texts of the reference corpus — typically the audited model's training split.
    reference_labels : sequence of str
        Labels aligned with ``reference_texts``.

    Raises
    ------
    ValueError
        If the two sequences differ in length, or the corpus holds fewer than two classes.

    Notes
    -----
    Fitting is lazy and happens once (:meth:`fit` is idempotent); on 2000 short texts it costs about
    ten seconds, which is why callers should hold one instance rather than rebuilding per query.

    The probe must not be fit on the rows it will judge — it would then be scoring its own training
    data and would back every label. Fitting on a *reference* corpus keeps its verdicts out-of-sample
    by construction, which is also why there is no "fit on the compiled batch" path: cross-validating
    a 500-row batch measured 0.460 accuracy, barely above the 0.330 majority-class rate.

    Examples
    --------
    >>> probe = LabelProbe(train_texts, train_labels)
    >>> probe.verdicts(["im feeling rather angsty and listless"], ["sadness"])[0]
    ProbeVerdict(given_prob=0.795, top_label='sadness', backs_given=True)
    """

    def __init__(self, reference_texts: Sequence[str], reference_labels: Sequence[str]) -> None:
        if len(reference_texts) != len(reference_labels):
            raise ValueError(
                f"reference_labels length ({len(reference_labels)}) must match "
                f"reference_texts length ({len(reference_texts)})."
            )
        self.reference_texts = [str(t) for t in reference_texts]
        self.reference_labels = [str(v) for v in reference_labels]
        classes, counts = np.unique(self.reference_labels, return_counts=True)
        if classes.size < 2:
            raise ValueError(f"LabelProbe needs at least 2 classes in the reference corpus; got {classes.size}.")
        thin = [str(c) for c, n in zip(classes, counts, strict=True) if n < MIN_PER_CLASS]
        if thin:
            logger.warning(
                "Reference classes %s have fewer than %d examples; the probe's verdicts for them are unreliable.",
                thin,
                MIN_PER_CLASS,
            )
        self._pipeline: Pipeline | None = None

    @property
    def classes(self) -> list[str]:
        """Classes the probe can report on — those present in the reference corpus."""
        return sorted(set(self.reference_labels))

    def fit(self) -> None:
        """Fit the TF-IDF + logistic-regression pipeline over the reference corpus. Idempotent."""
        if self._pipeline is not None:
            return
        # Word unigrams+bigrams, sublinear tf: a standard strong text baseline. Deliberately plain —
        # every knob added here is a knob that could quietly re-correlate the probe with the model.
        pipeline = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1),
            LogisticRegression(max_iter=1000, C=10.0),
        )
        pipeline.fit(self.reference_texts, self.reference_labels)
        self._pipeline = pipeline

    def verdicts(self, texts: Sequence[str], given_labels: Sequence[str]) -> list[ProbeVerdict]:
        """Return the probe's read on each ``(text, given_label)`` pair.

        Parameters
        ----------
        texts : sequence of str
            The texts to judge.
        given_labels : sequence of str
            The label each text currently carries, aligned with ``texts``. A label the reference
            corpus never shows gets ``given_prob`` 0.0 — the probe cannot speak for a class it has
            not seen, and 0.0 correctly reads as "no corpus support".

        Returns
        -------
        list[ProbeVerdict]
            One verdict per input row, in order.

        Raises
        ------
        ValueError
            If ``texts`` and ``given_labels`` differ in length.
        """
        if len(texts) != len(given_labels):
            raise ValueError(f"given_labels length ({len(given_labels)}) must match texts length ({len(texts)}).")
        if not texts:
            return []
        self.fit()
        assert self._pipeline is not None  # noqa: S101 - fit() guarantees this
        probs = self._pipeline.predict_proba(list(texts))
        classes = [str(c) for c in self._pipeline.classes_]
        position = {name: i for i, name in enumerate(classes)}
        return [
            ProbeVerdict(
                given_prob=float(row[position[label]]) if label in position else 0.0,
                top_label=classes[int(row.argmax())],
                backs_given=classes[int(row.argmax())] == label,
            )
            for row, label in ((probs[i], str(given_labels[i])) for i in range(len(texts)))
        ]
