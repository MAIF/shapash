"""Serve the NLP explainer **with the What-if Lab** (live model, cached compute).

The shape of the script follows the ``fit`` / ``explain`` split:

1. ``NlpExplainer(model, backend=...)`` — bind the model. No data, no results.
2. ``xpl.fit(reference_texts, y=reference_labels)`` — learn *reference state*: the similar-examples
   corpus and the label probe. Optional; skipped entirely with ``--n-reference 0``. This embeds the
   reference corpus (``precompute=True``), so the first run is slower here and the webapp's
   "Similar Examples" panel is instant; ``--cache-dir`` makes later runs reload instead.
3. ``explanation = xpl.explain(sentences, y=y_true, cache_dir=...)`` — the expensive step. Returns an
   immutable ``NlpExplanation``; the explainer keeps a memo, not a result.
4. Everything downstream takes that artifact as an *argument* — ``compute_projection(explanation,
   ...)``, ``can_detect_label_noise(explanation)``, ``run_app(explanation, ...)``.

The What-if Lab needs a live model in memory — editing a sample, re-predicting, and generating
counterfactuals all call the real model. That is why ``run_app`` gets both the artifact (what to
render) and ``self`` (how to compute live); serving an artifact alone makes the lab self-disable.
There is therefore no way to serve what-if without loading the model.

What *can* be avoided on every restart is recomputing the expensive stuff around the
model. This script caches both to ``--cache-dir`` (keyed by a hash of the input texts):

* **Contributions + predictions** — via ``explain(..., cache_dir=...)``.
* **The text embeddings and the 2-D projection** — via ``compute_projection(explanation, cache_dir=...)``.
  The library owns the embedding + caching (so the scatter and the similar-example neighbours are
  always in the same space); this script only supplies the reducer, PaCMAP.

The caching is automatic: each run **loads** the cache if one exists for these exact
texts, otherwise it **computes and writes** it. So the first run is slow; every run after
pays only the model load (a few seconds on CPU). Pass ``--recompute`` to force a fresh
compute (e.g. after changing the explainer). Iterating on the webapp itself? Don't
restart at all — keep this in a REPL and re-call ``run_app(explanation, ...)`` after reloading the
webapp modules; the model and the artifact both stay resident.

**The cache file is a snapshot.** ``explain``'s cache entries are ``NlpExplanation.save`` output — a
zip of plain-text ``meta.json`` plus parquet, no pickle — so any ``<hash>.xpl`` under ``--cache-dir``
can be reopened with **no model and no backend**::

    from shapash.explainer.nlp_explanation import NlpExplanation
    from shapash.webapp.nlp_app import NlpWebApp

    explanation, scatter = NlpExplanation.load("…/nlp_shap/<hash>.xpl")
    NlpWebApp(explanation, engine=None, scatter_xy=scatter).run(port=8050)

One caveat, and it is deliberate rather than an oversight: **the cached artifact carries no ground
truth.** The cache key covers ``(texts, model, backend)``, and ``y`` is not a function of that key —
it belongs to whoever called ``explain``. Serving a cache file directly therefore self-disables the
confusion matrix and the label-noise panel. For a shareable snapshot *with* labels, save the object
``explain`` returned: ``explanation.save(path, scatter_xy=projected)``.

The model runs on CPU when no GPU is present, so this serves fine on a plain box.

Model and dataset are configurable (see ``ServeConfig``): swap ``--model-name`` for any HF
sequence-classification checkpoint (its label names are read from the model's own config) and
``--dataset-name`` for any HF ``datasets`` classification dataset (point ``--text-column`` /
``--label-column`` at the right fields if they aren't ``text`` / ``label``). Cache keys cover the model
and the attribution backend as well as the texts, so switching ``--model-name`` can never reload another
model's contributions; the per-model/per-dataset subdirectories under ``--cache-dir`` below are for
human legibility (and easy pruning), not for correctness.

Dataset and model label strings aren't guaranteed to match even when their class order does (e.g.
a dataset's ``ClassLabel`` names ``"neg"``/``"pos"`` vs. a model's ``config.id2label`` values
``"NEGATIVE"``/``"POSITIVE"``) — pass ``--label-map`` to rename ground-truth labels onto the
model's spelling so predicted vs. ground-truth comparisons in the webapp line up. See "Serving a custom
dataset" below for label spaces ``--label-map`` can't fix (bucketing, dropping rows), and ``--seed`` for
how rows are sampled.

The sentence-highlight attribution method is selectable (a Captum-backed alternative to the default,
see the ``[nlp]`` extra):

* ``--attribution {shap,lig}`` — sentence-highlight method: ``shap`` (KernelSHAP, default) or ``lig``
  (Captum ``LayerIntegratedGradients``). The two are cached in **separate** subdirectories, so you can
  flip between them freely without ``--recompute``.
* ``--lig-batch-size`` — only used by ``lig``: Captum's ``internal_batch_size``, chunking each sample's
  50-step integration instead of running it through the model in one shot. Lower it (e.g. ``2`` or
  ``1``) if you hit a CUDA out-of-memory error, especially on memory-hungry architectures like DeBERTa.

``lig`` is also the method most sensitive to *truncation* actually being configured — every HF
classifier here is built through ``HFClassifierModel.from_pretrained(..., max_length="auto")``, which
resolves a safe length even when the checkpoint's tokenizer reports none.

Both counterfactual generators are offered live in the What-if Lab — ``hotflip`` (gradient-based token
substitution) and ``ablation`` (leave-one-out token removal) — and are switched from a
**method dropdown in the webapp**, so there is no CLI flag for them. They run live and are not cached.

The on-disk cache mirrors these dependencies as a hierarchy under ``--cache-dir``:
``<model>/<dataset>__<split>/`` holds the (backend-independent) embedding-store artifacts — the
vectors and the projection derived from them — and a per-backend ``<...>/nlp_shap/`` or
``<...>/nlp_captum_lig/`` subdirectory holds that method's ``<hash>.xpl`` contributions.

Serving a *private* model and dataset (both local, nothing on the hub)
----------------------------------------------------------------------
Neither ``--model-name`` nor the data has to come from the HuggingFace hub, and a local run gets the
**full** capability surface — sentence highlights (both methods), the What-if Lab, both counterfactual
generators, similar examples — exactly like a hub model, because everything is served through the same
``HFClassifierModel`` adapter:

* ``--model-name /path/to/checkpoint`` — a directory saved with ``model.save_pretrained(...)``. Weights
  and ``config.json`` are enough; the *tokenizer* need not be there. When the directory ships no
  tokenizer files, ``HFClassifierModel.from_pretrained`` falls back to the base model recorded in the
  checkpoint's own ``config.json`` (``_name_or_path``) — right for any fine-tune that kept its base
  tokenizer, which is the usual case. Pass ``--tokenizer-name`` (an id or a path) for a **customised**
  tokenizer: nothing can detect that automatically.
* ``--label-names NEGATIVE,POSITIVE`` — class names in output-column order, when the checkpoint's config
  carries none. ``trainer.model.save_pretrained(...)`` records ``id2label`` only if training set it, so a
  fine-tune often reports ``LABEL_0``/``LABEL_1``; this replaces them without needing a custom builder.
  It is applied in ``_hf_classifier_model`` — the one place every *HF classifier* is built — so it also
  overrides the hard-coded names of a ``MODEL_BUILDERS`` builder that goes through there
  (``_build_camembertv2_cls_model``). A builder assembling some other adapter directly
  (``_build_shhossain_sentiment_model``, a ``SentenceTransformerModel``) owns its label names outright
  and this flag does not reach it. Get the *order* right — verify against real labeled rows, not a
  handful (see ``--label-map`` below and the camembert cautionary tale further down).
* ``--dataset-path /path/to/dir`` — local dataframes instead of ``--dataset-name`` (the two are mutually
  exclusive; the CLI rejects both). A **directory** holds one file per split, ``<split>.parquet`` or
  ``<split>.pkl``, so ``--dataset-split``/``--reference-split`` pick files exactly as they pick hub
  splits. A **single file** works too, but only with ``--n-reference 0``: one file cannot also be the
  reference corpus, and silently retrieving "similar examples" from the served rows would be a lie.
  ``--text-column``/``--label-column`` name the columns; ``--label-map`` reconciles the dataframe's label
  spelling with the model's, as for hub datasets. (``.pkl`` is read with ``pandas.read_pickle``, which
  executes arbitrary code on load — point it only at files you produced yourself.)

Caching works the same way: a local model or dataset is keyed in the cache hierarchy by its directory
name plus a digest of its absolute path, so two checkpoints both named ``output/`` never share a cache.

Serving a *custom* model (external-head checkpoints)
----------------------------------------------------
Any standard ``AutoModelForSequenceClassification`` checkpoint loads automatically from ``--model-name``
alone — hub id or local path (``load_model`` builds an ``HFClassifierModel``; label names come from
``config.id2label``). That covers every certified encoder architecture — BERT, DistilBERT, RoBERTa,
XLM-R, DeBERTa-v1, and a MiniLM packaged as a sequence classifier.

A model whose classification **head is custom** cannot be introspected from its name — its layer
structure, weight-key convention, and input contract are non-standard (a sentence-transformer body + a
separate MLP head, a SetFit head, a hand-rolled ``nn.Module``, …). These are served through the
``MODEL_BUILDERS`` registry: a ``{--model-name: builder}`` map consulted *before* the generic path. To add
one:

1. Write a builder ``def build_x(config: ServeConfig, device: str) -> TextModel:`` that assembles the
   right adapter and returns it. Put **all** model-specific code inside it — the core ``load_model`` never
   changes. Pick the adapter by shape:

   * **sentence-transformer body + head** → :class:`~shapash.model.SentenceTransformerModel` (pass the
     ``sentence_transformers`` model + a *logits* head ``nn.Module``; drop any final ``softmax`` the head
     applies — ``predict`` softmaxes and the LIG backend needs logit space; set ``pool=`` to the body's
     pooling mode, usually ``"mean"``).
   * **raw encoder body + head** (head not fused into an ``AutoModelForSequenceClassification``) →
     :class:`~shapash.model.TorchClassifierModel` (pass ``body, head, tokenizer``; ``pool="cls"`` for a
     CLS-pooled head, ``"mean"`` otherwise).
   * **standard classifier that merely needs ``trust_remote_code``** → just call
     ``HFClassifierModel(AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=True),
     tokenizer)``.
   * **standard classifier whose config's labels can't be trusted** (missing ``id2label``, or a
     ``label2id`` that turns out backwards from the model's real behaviour) → same call as the generic
     path, but pass ``label_names=[...]`` explicitly instead of leaving it to be read from
     ``config.id2label``. Verify the ordering against a real labeled sample (dozens+, not a handful) —
     ``almanach/camembertv2-base-cls`` was first "fixed" from a 10-example hand check that (wrongly)
     looked decisive; a proper sanity check (``demo/check_camembertv2_amazon_fr.py``) later showed
     *neither* ordering is trustworthy on that checkpoint at all. A handful of examples can't tell a
     genuinely swapped label from a model that just doesn't work.

2. Register it: ``MODEL_BUILDERS["org/your-model"] = build_x``. Nothing else changes — caching, the
   What-if Lab, both attribution backends, counterfactuals, and similar-examples all work through the
   capability interface regardless of adapter.

``_build_shhossain_sentiment_model`` (registered for
``shhossain/all-MiniLM-L6-v2-sentiment-classifier``) is the worked example: a MiniLM ST body + a trained
``fc1 -> relu -> fc2 -> relu -> out`` MLP head, ``trust_remote_code`` with no HF tokenizer, so the generic
path can't touch it — but ``SentenceTransformerModel`` serves it faithfully (it reproduces the source
model's predictions exactly). ``_build_camembertv2_cls_model`` (registered for
``almanach/camembertv2-base-cls``) is the worked example of the label-names-can't-be-trusted case: an
architecturally plain ``RobertaForSequenceClassification`` whose checkpoint config omits ``id2label``.
**Its config's labels can't be trusted, but neither can the checkpoint's own predictions** — see the
builder's docstring and ``demo/check_camembertv2_amazon_fr.py`` for a sanity check showing neither label
ordering beats a trivial majority-class baseline on gold in-domain data. It stays registered only as a
worked example of this builder pattern, not as a model whose predictions should be trusted.

Serving a *custom* dataset (label logic that isn't a 1:1 rename)
------------------------------------------------------------------
Any standard HF classification dataset loads automatically from ``--dataset-name`` alone: point
``--text-column``/``--label-column`` at the right fields if they aren't ``text``/``label``, and (rows are
shuffled with ``--seed`` before ``--n``/``--n-reference`` are taken, so this is representative even when
the dataset's on-disk order groups rows by label). Ground-truth strings come from the dataset's own
``ClassLabel`` names when present, else the raw values stringified; ``--label-map`` renames them 1:1 onto
the model's own spelling when the two disagree on casing/wording despite sharing the same class order
(e.g. dataset ``"neg"`` vs. model ``"NEGATIVE"``).

A dataset whose ground truth needs **more** than a rename — bucketing several raw values into one class,
dropping rows with no mapping, or any other dataset-specific logic — is served through the
``DATASET_LOADERS`` registry: a ``{--dataset-name: loader}`` map consulted *before* the generic path
(the exact ``MODEL_BUILDERS`` pattern above, applied to data instead of the model). To add one:

1. Write a loader ``def load_x(config: ServeConfig, split: str, n: int) -> tuple[list[str], list[str]]``
   that returns ``(texts, label_strings)``. Put **all** dataset-specific code inside it, including which
   raw columns it reads — a bucketing loader's logic is tied to that dataset's exact label semantics, so
   there is no generic ``--text-column``/``--label-column`` to thread through; the core ``_load_split``
   never changes.
2. Register it: ``DATASET_LOADERS["org/your-dataset"] = load_x``. Nothing else changes — caching, the
   webapp, and the reference corpus all call ``_load_split`` regardless of which path served it.

``_load_amazon_reviews_multi_binary`` (registered for ``mteb/amazon_reviews_multi``) is the worked
example: its ``label`` is a 5-star rating minus one (0-4), not the binary sentiment space a binary
classifier like ``almanach/camembertv2-base-cls`` predicts into, so it buckets rating > 3 -> ``"positive"``,
rating < 3 -> ``"negative"``, and drops the neutral 3-star (a ground-truth convention only — see
``demo/check_camembertv2_amazon_fr.py`` for whether this specific model's predictions are trustworthy).

Usage
-----
    python demo/serve_nlp.py [--n 100] [--cache-dir demo/nlp_cache] [--port 8050]
    # Behind a reverse proxy that routes a subpath (not a whole hostname) to this process:
    python demo/serve_nlp.py --url-base-path /shapash-nlp-explainer/
    python demo/serve_nlp.py --recompute   # ignore the cache and recompute
    python demo/serve_nlp.py --attribution lig   # Captum LayerIntegratedGradients highlights
    python demo/serve_nlp.py --model-name distilbert-base-uncased-finetuned-sst-2-english \\
        --dataset-name sst2 --dataset-split validation --text-column sentence
    # A registered custom model against a registered dataset loader (bucketed 5-star -> binary
    # sentiment ground truth; --text-column/--label-column don't apply here, see DATASET_LOADERS):
    python demo/serve_nlp.py --model-name almanach/camembertv2-base-cls \\
        --dataset-name mteb/amazon_reviews_multi --dataset-config fr --dataset-split test
    python demo/serve_nlp.py --model-name lvwerra/distilbert-imdb --dataset-name stanfordnlp/imdb \\
        --label-column label --label-map '{"neg": "NEGATIVE", "pos": "POSITIVE"}'
    # A registered custom (external-head) model — served via MODEL_BUILDERS, no other flags needed:
    python demo/serve_nlp.py --model-name shhossain/all-MiniLM-L6-v2-sentiment-classifier
    # A private local checkpoint against local parquet dataframes (tokenizer resolved from the
    # checkpoint's base model; --label-map reconciles the dataframe's labels with the model's):
    python demo/serve_nlp.py --model-name demo/private_model \\
        --dataset-path data/nlp_test_parquet --dataset-split test --reference-split train \\
        --label-map '{"negative": "NEGATIVE", "positive": "POSITIVE"}'

First run needs transformers, datasets and pacmap (the ``[nlp]`` extra). A ``SentenceTransformerModel``
custom builder additionally needs the ``sentence-transformers`` package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import datasets
import pacmap
import pandas as pd
import torch
from torch import nn

from shapash.backend import NlpCaptumLigBackend
from shapash.explainer.nlp_explainer import NlpExplainer
from shapash.model import HFClassifierModel, SentenceTransformerModel, TextModel

_HERE = Path(__file__).parent

logger = logging.getLogger("serve_nlp")


@dataclass
class ServeConfig:
    """Everything needed to load data + model and serve the What-if webapp.

    A single object (rather than argparse's ``Namespace``) so the loading helpers below
    stay reusable from a REPL too — see the module docstring on iterating without restarting.
    """

    model_name: str = "bhadresh-savani/distilbert-base-uncased-emotion"
    # Tokenizer source, when it isn't the checkpoint itself: a local ``save_pretrained`` artifact often
    # ships weights + config only. ``None`` resolves it — see _resolve_tokenizer_source.
    tokenizer_name: str | None = None
    # Class names in output-column order, overriding the model config's ``id2label``. Needed for a
    # fine-tune saved without label names (its config reports "LABEL_0"/"LABEL_1").
    label_names: list[str] | None = None
    dataset_name: str = "dair-ai/emotion"
    # Local dataframes instead of the HF hub: a directory of ``<split>.parquet`` / ``<split>.pkl``, or a
    # single file (see _local_split_file). Mutually exclusive with ``dataset_name`` — the CLI rejects
    # both, and every branch between the two goes through ``is_local_dataset`` so no code path can
    # consult the unused one.
    dataset_path: Path | None = None
    dataset_config: str | None = None  # HF dataset config/subset (e.g. a language); None = dataset's default
    dataset_split: str = "test"
    text_column: str = "text"
    label_column: str = "label"
    label_map: dict[str, str] = field(default_factory=dict)
    n: int = 500
    # Rows are shuffled with this seed before taking --n / --n-reference (see _load_split) — several
    # HF dataset repos (e.g. mteb/amazon_reviews_multi) store rows grouped by label, so an unshuffled
    # "first n" slice can be entirely one class.
    seed: int = 0
    attribution: str = "shap"  # sentence-highlight method: "shap" | "lig" (Captum LayerIntegratedGradients)
    # Captum's LayerIntegratedGradients expands one sample into ``n_steps`` (50) scaled copies and runs
    # them through the model as a single batch unless told otherwise — on a memory-hungry architecture
    # (e.g. DeBERTa-v2/v3's disentangled attention) that batch of 50 can blow past a small GPU's memory.
    # ``internal_batch_size`` makes Captum chunk that batch instead; lower it further if you still OOM.
    lig_batch_size: int = 8
    # Similar-examples reference corpus: the split neighbours are retrieved from (the model's own
    # training split) and how many rows of it to bank. Set ``n_reference=0`` to disable the "Similar
    # Examples" panel.
    reference_split: str = "train"
    n_reference: int = 2000
    # Representation space the scatter projects *and* neighbours are ranked in — one setting for both,
    # so they can never disagree. ``None`` keeps whatever the model was built with ("decision").
    embedding_space: str | None = None
    cache_dir: Path = _HERE / "nlp_cache"
    port: int = 8050
    host: str = "0.0.0.0"  # noqa: S104
    # Mount point when a reverse proxy routes a subpath here rather than a whole hostname (e.g.
    # "/shapash-nlp-explainer/"). Dash rewrites its own asset and callback URLs to match, so this
    # has to equal the proxied path. None serves at the server root.
    url_base_path: str | None = None
    recompute: bool = False

    def __post_init__(self) -> None:
        """Normalise ``dataset_path`` once, so every consumer sees the same path.

        ``~`` is expanded by the *shell*, so a path only arrives here unexpanded when a caller built the
        config directly (the REPL workflow this dataclass exists for) — and a literal ``~/...`` then
        reads as a nonexistent *relative* directory, which silently defeats the ``is_file()`` tests that
        decide whether a single file can also serve as the reference corpus. Doing it here means no
        consumer has to remember to.
        """
        if self.dataset_path is not None:
            self.dataset_path = self.dataset_path.expanduser()

    @property
    def model_path(self) -> Path | None:
        """The local checkpoint directory ``model_name`` points at, or ``None`` when it is a hub id.

        The single answer to "is this model local, and where?": the cache slug, the tokenizer fallback
        and ``from_pretrained`` all ask *here* instead of each re-deriving it. They used to, and
        disagreed — two expanded ``~`` and the loader did not, so a ``~/checkpoints/...`` model resolved
        its tokenizer and its cache directory correctly and then failed to load.
        """
        path = Path(self.model_name).expanduser()
        return path if path.is_dir() else None

    @property
    def is_local_dataset(self) -> bool:
        """True when rows come from local dataframes (``--dataset-path``) rather than the HF hub.

        ``dataset_name`` and ``dataset_path`` are mutually exclusive (the CLI rejects both), so exactly
        one of them is meaningful on any given run. Every branch between the two tests *this* rather
        than comparing fields, so a stale ``dataset_name`` default cannot leak into a local run — not
        into its cache path (:func:`_dataset_slug`) and not into its loader (:func:`_load_split`).
        """
        return self.dataset_path is not None


# Short ``--attribution`` choice → the attribution backend's registered ``.name`` (used as the cache
# subdirectory *and* to pick the backend in ``build_backend``).
_ATTRIBUTION_BACKENDS = {"shap": "nlp_shap", "lig": "nlp_captum_lig"}


def _path_slug(path: Path) -> str:
    """Filesystem-safe cache tag for a local path: its name plus a digest of its absolute location.

    The name alone would collide across runs (every training run named ``output/`` or ``best_model/``
    would share one cache directory); the absolute path alone is unreadable as a directory name. The
    digest disambiguates, the name keeps ``--cache-dir`` browsable.
    """
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:8]
    return f"{resolved.name}__{digest}"


def _model_slug(config: ServeConfig) -> str:
    """Filesystem-safe tag for ``--model-name`` — an HF id or a local checkpoint directory.

    HF ids keep the historical ``org__model`` spelling, so caches built before local checkpoints were
    supported still resolve to the same directory; a local path goes through :func:`_path_slug`.
    """
    if config.model_path is not None:
        return _path_slug(config.model_path)
    return config.model_name.replace("/", "__")


def _dataset_slug(config: ServeConfig) -> str:
    """Filesystem-safe ``<dataset>__[<config>__]<split>`` tag for the cache hierarchy.

    The config segment is only included when ``--dataset-config`` is actually set, so existing cache
    directories built before that flag existed (``<dataset>__<split>``, no config with no HF dataset
    config) still resolve to the same path. A local ``--dataset-path`` is tagged by
    :func:`_path_slug` (an absolute path is neither readable nor unique as a directory name).
    """
    if config.is_local_dataset:
        return f"{_path_slug(config.dataset_path)}__{config.dataset_split}"
    parts = [config.dataset_name, *([config.dataset_config] if config.dataset_config else []), config.dataset_split]
    return "__".join(parts).replace("/", "__")


def _dataset_cache_dir(config: ServeConfig) -> Path:
    """Return ``cache_dir/<model>/<dataset>__<split>`` — the level the projection is cached at.

    The explain and projection caches both key on the model (and, for contributions, the backend), so
    these path levels are organisational rather than load-bearing: they keep a ``--cache-dir`` readable
    and prunable per model/dataset. The PaCMAP projection embeds the texts with ``model.embed`` and
    never touches the attribution backend, so every ``--attribution`` choice shares one projection cache
    at *this* level (below it, contributions split per backend).
    """
    return config.cache_dir / _model_slug(config) / _dataset_slug(config)


def _explain_cache_dir(config: ServeConfig) -> Path:
    """Return ``<dataset-dir>/<backend>`` — the level contributions are cached at.

    Contributions *do* depend on the attribution backend, so SHAP and LIG get separate subdirectories.
    Switching ``--attribution`` then reads/writes a different ``<hash>.xpl`` instead of silently
    reusing the other method's cached highlights — no ``--recompute`` needed to swap methods.
    """
    return _dataset_cache_dir(config) / _ATTRIBUTION_BACKENDS[config.attribution]


def parse_args(argv: list[str] | None = None) -> ServeConfig:
    """Parse CLI args into a ``ServeConfig``, defaulting every flag from the dataclass."""
    defaults = ServeConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default=defaults.model_name,
        help="HF model id, or a path to a local checkpoint directory saved with save_pretrained "
        "(label names are read from its config unless --label-names is given).",
    )
    parser.add_argument(
        "--tokenizer-name",
        default=defaults.tokenizer_name,
        help="HF id or path of the tokenizer, when it isn't in the checkpoint itself. Default: resolved "
        "from the checkpoint, falling back to the base model recorded in its config.",
    )
    parser.add_argument(
        "--label-names",
        default=None,
        help="Comma-separated class names in output-column order, overriding the model config's "
        "id2label — e.g. 'NEGATIVE,POSITIVE'. Needed for a fine-tune saved without label names "
        "(its config reports LABEL_0/LABEL_1).",
    )
    # One dataset source per run: a stale --dataset-name alongside --dataset-path would otherwise decide
    # the cache slug and the DATASET_LOADERS lookup for rows it did not provide.
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset-name", default=defaults.dataset_name, help="HF datasets id.")
    source.add_argument(
        "--dataset-path",
        type=Path,
        default=defaults.dataset_path,
        help="Local dataframes instead of the HF hub: a directory holding <split>.parquet or "
        "<split>.pkl (so --dataset-split / --reference-split select the file), or a single file "
        "(which then requires --n-reference 0). Reads --text-column / --label-column.",
    )
    parser.add_argument(
        "--dataset-config",
        default=defaults.dataset_config,
        help="HF dataset config/subset (e.g. a language) — needed when the dataset has more than one, "
        "such as mteb/amazon_reviews_multi (pass 'fr', 'en', ...).",
    )
    parser.add_argument("--dataset-split", default=defaults.dataset_split)
    parser.add_argument("--text-column", default=defaults.text_column, help="Column holding the raw text.")
    parser.add_argument("--label-column", default=defaults.label_column, help="Column holding the ground-truth label.")
    parser.add_argument(
        "--label-map",
        type=json.loads,
        default=defaults.label_map,
        help=(
            "JSON object renaming dataset label strings to match the model's own label names, e.g. "
            '\'{"neg": "NEGATIVE", "pos": "POSITIVE"}\'. Needed when the dataset and model disagree on '
            "label spelling/casing even though their class order matches."
        ),
    )
    parser.add_argument("--n", type=int, default=defaults.n, help="Number of samples to load.")
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults.seed,
        help="Shuffle seed applied before taking --n / --n-reference rows, so sampling is representative "
        "rather than whatever the dataset's on-disk row order puts first.",
    )
    parser.add_argument(
        "--attribution",
        choices=list(_ATTRIBUTION_BACKENDS),
        default=defaults.attribution,
        help=(
            "Sentence-highlight attribution method: 'shap' (KernelSHAP, the default) or 'lig' "
            "(Captum LayerIntegratedGradients). Cached separately, so switching needs no --recompute."
        ),
    )
    parser.add_argument(
        "--lig-batch-size",
        type=int,
        default=defaults.lig_batch_size,
        help=(
            "Captum internal_batch_size for --attribution lig: chunks each sample's n_steps=50 "
            "integration batch instead of running it through the model in one shot. Lower this "
            "(e.g. 2 or 1) if LIG hits a CUDA out-of-memory error."
        ),
    )
    parser.add_argument(
        "--reference-split",
        default=defaults.reference_split,
        help="Dataset split used as the similar-examples reference corpus (the model's training split).",
    )
    parser.add_argument(
        "--n-reference",
        type=int,
        default=defaults.n_reference,
        help="Reference rows to bank for 'Similar Examples' (0 disables the panel).",
    )
    parser.add_argument(
        "--embedding-space",
        default=defaults.embedding_space,
        help=(
            "Representation space for the 2-D scatter and similar-example retrieval: 'decision' "
            "(input to the final classification linear, class-discriminative), 'pooled' (the pooled "
            "last hidden state, semantic), or any backbone submodule name. Default: the model's own."
        ),
    )
    parser.add_argument("--cache-dir", type=Path, default=defaults.cache_dir)
    parser.add_argument("--port", type=int, default=defaults.port)
    parser.add_argument("--host", default=defaults.host)
    parser.add_argument(
        "--url-base-path",
        default=defaults.url_base_path,
        help="Serve the app under this URL prefix instead of '/', for a reverse proxy that routes a "
        "subpath to this process (e.g. '/shapash-nlp-explainer/'). Must match the proxied path; "
        "missing slashes are added.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore any cached results and recompute SHAP + projection (overwrites the cache).",
    )
    args = parser.parse_args(argv)
    # --label-names is a flat string on the CLI (one flag, no quoting rules to remember) but a list
    # everywhere else, since that is what the model adapters take.
    args.label_names = [name.strip() for name in args.label_names.split(",")] if args.label_names else None
    # Validate the *config*, not the raw namespace: ``__post_init__`` has normalised ``dataset_path`` by
    # then, so the ``is_file()`` test below sees the same path the loaders will (an unexpanded ``~/...``
    # is a nonexistent relative directory, and would slip through this check).
    config = ServeConfig(**vars(args))
    if config.is_local_dataset and config.dataset_config is not None:
        # --dataset-config only means anything to the hub loader; accepting it here would look like it
        # selected something. (It isn't in the mutually exclusive group above: that pairs the two
        # *sources*, and grouping three flags would reject the legitimate --dataset-name/--dataset-config
        # combination too.)
        parser.error(
            "--dataset-config applies to --dataset-name (a hub dataset) and has no meaning for --dataset-path."
        )
    if config.is_local_dataset and config.dataset_path.is_file() and config.n_reference > 0:
        # A single file can only serve one split; silently reusing it as the reference corpus would
        # retrieve "similar examples" from the served rows themselves. Make the caller choose. The same
        # invariant is re-checked in load_reference_corpus, which a REPL-built config does reach.
        parser.error(
            f"--dataset-path {config.dataset_path} is a single file, so there is no --reference-split "
            f"{config.reference_split!r} to load: pass --n-reference 0 to disable the Similar Examples "
            "panel, or point --dataset-path at a directory holding one file per split."
        )
    return config


def _load_hf_dataset(name: str, dataset_config: str | None, split: str) -> datasets.Dataset:
    """Load one split of a HF dataset, falling back to raw per-split JSONL for legacy-script repos.

    Some older dataset repos (e.g. ``mteb/amazon_reviews_multi``) still ship a ``<name>.py`` loading
    script, which ``datasets>=4.0`` refuses to execute (``RuntimeError: Dataset scripts are no longer
    supported``). Their raw ``<config>/<split>.jsonl`` files are usually still present in the repo, so on
    exactly that error this retries against them directly via the generic ``json`` builder, bypassing the
    broken script. Any other failure (missing dataset, bad split name, ...) is not masked.
    """
    try:
        return datasets.load_dataset(name, dataset_config, split=split)
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc) or dataset_config is None:
            raise
        logger.warning("%s ships a legacy loading script; falling back to raw %s/%s.jsonl", name, dataset_config, split)
        return datasets.load_dataset(
            "json", data_files=f"hf://datasets/{name}/{dataset_config}/{split}.jsonl", split="train"
        )


# Local dataframe formats ``--dataset-path`` recognises, in the order a ``<split>.<ext>`` file is looked
# up. ``read_pickle`` unpickles, which executes arbitrary code — only ever point ``--dataset-path`` at
# files you produced yourself (the same trust assumption as loading your own model weights).
_DATAFRAME_READERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    ".parquet": pd.read_parquet,
    ".pkl": pd.read_pickle,  # noqa: S301
    ".pickle": pd.read_pickle,  # noqa: S301
}


def _local_split_file(dataset_path: Path, split: str) -> Path:
    """Resolve ``--dataset-path`` + a split name to one dataframe file.

    A *file* is that split's data as given (the CLI only allows this when the reference corpus is
    disabled, since one file cannot also be another split). A *directory* is searched for
    ``<split>.<ext>`` over :data:`_DATAFRAME_READERS`, so ``--dataset-split``/``--reference-split``
    select files the same way they select HF splits. Where a split exists in more than one format, that
    dict's order decides (parquet first).

    ``dataset_path`` is expected already normalised — ``ServeConfig.__post_init__`` expands ``~`` so
    every caller tests the same path.
    """
    path = dataset_path
    if path.is_file():
        return path
    if not path.is_dir():
        # Distinguish "the dataset root is wrong" from "this split is missing" — the message below
        # would otherwise blame the split name for a mistyped --dataset-path.
        raise FileNotFoundError(f"--dataset-path {path} is neither a file nor a directory.")
    for suffix in _DATAFRAME_READERS:
        candidate = path / f"{split}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No '{split}' dataframe in {path}: expected one of "
        f"{', '.join(f'{split}{suffix}' for suffix in _DATAFRAME_READERS)}."
    )


def _load_local_split(config: ServeConfig, split: str, n: int) -> tuple[list[str], list[str]]:
    """Load ``n`` ``(texts, label_strings)`` rows of ``split`` from a local dataframe.

    The dataframe counterpart of the generic HF path in :func:`_load_split`, and deliberately identical
    downstream of the read: rows are shuffled with ``config.seed`` before the first ``n`` are taken,
    labels are stringified, and ``config.label_map`` renames them onto the model's spelling. Rows with a
    missing or blank text or label are dropped first — a real-world dataframe has them, an HF
    ``ClassLabel`` column cannot.
    """
    file = _local_split_file(config.dataset_path, split)
    frame = _DATAFRAME_READERS[file.suffix](file)
    for column in (config.text_column, config.label_column):
        if column not in frame.columns:
            raise KeyError(
                f"Column {column!r} not in {file} (columns: {list(frame.columns)}). "
                "Point --text-column / --label-column at the right fields."
            )
    frame = frame[[config.text_column, config.label_column]].dropna()
    # Blank-but-present is as unusable as missing, and a stray "" label would otherwise show up in the
    # webapp as a ground-truth class of its own. dropna() above only catches NaN/None.
    for column in (config.text_column, config.label_column):
        frame = frame[frame[column].astype(str).str.strip().astype(bool)]
    if frame.empty:
        raise ValueError(f"{file} has no usable rows in columns {config.text_column!r}/{config.label_column!r}.")
    frame = frame.sample(frac=1.0, random_state=config.seed).head(n)
    sentences = [str(text) for text in frame[config.text_column]]
    labels = [str(label) for label in frame[config.label_column]]
    if config.label_map:
        labels = [config.label_map.get(label, label) for label in labels]
    logger.info("Loaded %d samples from %s (split=%s)", len(sentences), file, split)
    return sentences, labels


def _load_amazon_reviews_multi_binary(config: ServeConfig, split: str, n: int) -> tuple[list[str], list[str]]:
    """``mteb/amazon_reviews_multi`` — bucket its 5-star ``label`` (0-4) onto binary sentiment.

    The raw ``label`` is a star rating minus one (0 = 1-star, ..., 4 = 5-star) — not the binary
    positive/negative space a binary sentiment model (e.g. ``almanach/camembertv2-base-cls``) predicts
    into, and not something ``--label-map`` can fix: that flag only renames labels 1:1, it can't collapse
    five classes into two or drop one. Its ``label_text`` column doesn't help either — it's just
    ``str(label)`` ("0".."4"), not a description.

    Standard star-rating bucketing: rating > 3 -> ``"positive"``, rating < 3 -> ``"negative"``; the
    3-star (rating == 3) review is neutral, has no binary ground truth, and is dropped — so this can
    return fewer than ``n`` rows. This is a *ground-truth* convention only — it says nothing about
    whether ``almanach/camembertv2-base-cls`` (or any other model served against this data) predicts
    accurately; see ``demo/check_camembertv2_amazon_fr.py`` for that question answered against a
    cleaner, in-domain gold set.

    Also fixes the "all one label" symptom on this dataset specifically: the raw file is sorted by
    ``label`` in blocks of 1000 identical values, so an unshuffled ``[:n]`` slice for any ``n`` <= 1000
    is entirely 1-star. Shuffling with ``config.seed`` before selecting fixes that regardless of ``n``.
    """
    dataset = _load_hf_dataset(config.dataset_name, config.dataset_config, split)
    dataset = dataset.shuffle(seed=config.seed).select(range(min(n, len(dataset))))
    texts, labels = [], []
    for row in dataset:
        rating = row["label"] + 1
        if rating == 3:
            continue
        texts.append(row["text"])
        labels.append("positive" if rating > 3 else "negative")
    return texts, labels


# ``--dataset-name`` -> a custom loader for a dataset whose ground-truth labels need dataset-specific
# logic (bucketing, dropping rows, ...) that the generic ``--label-column``/``--label-map`` path can't
# express. Anything not listed takes the generic path in ``_load_split``. Mirrors ``MODEL_BUILDERS``: put
# all dataset-specific code inside the loader and register it here — ``_load_split`` itself never changes.
DATASET_LOADERS: dict[str, Callable[[ServeConfig, str, int], tuple[list[str], list[str]]]] = {
    "mteb/amazon_reviews_multi": _load_amazon_reviews_multi_binary,
}


def _load_split(config: ServeConfig, split: str, n: int) -> tuple[list[str], list[str]]:
    """Load ``n`` ``(texts, label_strings)`` of ``split`` from the configured dataset.

    Three sources, checked in this order. ``--dataset-path`` (local dataframes) goes to
    :func:`_load_local_split` — first, so a leftover ``--dataset-name`` default can never route local
    rows through a hub loader. Otherwise a hub dataset registered in :data:`DATASET_LOADERS` is
    delegated to its loader entirely (see there for why ``mteb/amazon_reviews_multi`` needs one).
    Otherwise, the generic hub path applies: rows are shuffled
    with ``config.seed`` before taking the first ``n`` — so samples are representative even when the
    dataset's on-disk order groups rows by label — and ground-truth label strings come from the
    dataset's own ``ClassLabel`` feature names when present (the usual shape for HF classification
    datasets), otherwise the raw values are stringified as-is. ``config.label_map`` then renames them to
    match the model's label names when the two disagree on spelling/casing (e.g. dataset ``"neg"`` vs.
    model ``"NEGATIVE"``) despite sharing the same class order.
    """
    if config.is_local_dataset:
        return _load_local_split(config, split, n)

    loader = DATASET_LOADERS.get(config.dataset_name)
    if loader is not None:
        sentences, labels = loader(config, split, n)
        logger.info(
            "Loaded %d samples from %s (split=%s) via its registered dataset loader",
            len(sentences),
            config.dataset_name,
            split,
        )
        return sentences, labels

    dataset = _load_hf_dataset(config.dataset_name, config.dataset_config, split).shuffle(seed=config.seed)
    sentences = dataset[config.text_column][:n]
    raw_labels = dataset[config.label_column][:n]
    names = getattr(dataset.features.get(config.label_column), "names", None)
    labels = [names[i] for i in raw_labels] if names is not None else [str(v) for v in raw_labels]
    if config.label_map:
        labels = [config.label_map.get(v, v) for v in labels]
    logger.info("Loaded %d samples from %s (split=%s)", len(sentences), config.dataset_name, split)
    return sentences, labels


def load_data(config: ServeConfig) -> tuple[list[str], list[str]]:
    """Load the served ``config.n`` samples + label strings from ``config.dataset_split``."""
    return _load_split(config, config.dataset_split, config.n)


def load_reference_corpus(config: ServeConfig) -> tuple[list[str], list[str]] | None:
    """Load the similar-examples reference corpus (the model's training split).

    Returns ``(texts, labels)`` from ``config.reference_split`` for the "Similar Examples" panel,
    or ``None`` when disabled (``n_reference <= 0``). This is the pool neighbours are retrieved from,
    so it is the model's *training* set rather than the served (test) split.

    A single-file ``--dataset-path`` cannot supply it: :func:`_local_split_file` returns that one file
    for *any* split name, so the bank would be built from the served rows themselves and every "similar
    example" would be a lookalike of the corpus it came from. The CLI rejects that combination up front
    with a friendlier message, but the invariant is enforced here as well — a ``ServeConfig`` built
    directly (the REPL workflow this module is designed for) never goes through ``parse_args``.
    """
    if config.n_reference <= 0:
        return None
    if config.is_local_dataset and config.dataset_path.is_file():
        raise ValueError(
            f"dataset_path {config.dataset_path} is a single file, so it cannot also provide the "
            f"reference_split {config.reference_split!r}. Set n_reference=0, or point dataset_path at a "
            "directory holding one file per split."
        )
    return _load_split(config, config.reference_split, config.n_reference)


def _hf_classifier_model(config: ServeConfig, device: str, **kwargs) -> HFClassifierModel:
    """Adapt this demo's ``ServeConfig`` onto :meth:`HFClassifierModel.from_pretrained`.

    A thin translation layer: the library classmethod owns everything that used to live here — resolving
    a tokenizer (including the base-model fallback for a ``save_pretrained`` directory that shipped none),
    defeating the ``model_max_length`` no-op trap (``max_length="auto"``), and the ``label_names`` arity
    check. All this demo adds is its own conventions: ``--model-name`` may be a local path or a hub id,
    ``--tokenizer-name`` maps to the ``tokenizer=`` override, and ``--label-names`` beats a builder's own
    hard-coded ``label_names`` (passing the flag is a deliberate request to override).

    Every path that produces an ``HFClassifierModel`` — the generic loader *and* every custom builder in
    :data:`MODEL_BUILDERS` — goes through here, so no builder can bypass ``max_length`` resolution by
    constructing the adapter directly. Other ``kwargs`` are forwarded untouched.
    """
    label_names = config.label_names or kwargs.pop("label_names", None)
    source = str(config.model_path) if config.model_path is not None else config.model_name
    return HFClassifierModel.from_pretrained(
        source,
        tokenizer=config.tokenizer_name,
        label_names=label_names,
        device=device,
        **kwargs,
    )


def _load_hf_classifier(config: ServeConfig, device: str) -> HFClassifierModel:
    """The generic default: any standard ``AutoModelForSequenceClassification`` checkpoint.

    Covers every certified encoder architecture (BERT/DistilBERT/RoBERTa/XLM-R/DeBERTa-v1, and MiniLM
    packaged as a sequence classifier), whether ``--model-name`` is a hub id or a local checkpoint
    directory. Label names come from the model's own ``config.id2label``, so a new ``--model-name``
    brings its own classes with no extra flags — unless ``--label-names`` overrides them (applied in
    :func:`_hf_classifier_model`, so builders honour it too), which a fine-tune saved without label
    names needs (its config reports ``LABEL_0``/``LABEL_1``).
    """
    return _hf_classifier_model(config, device)


# Reconcile the shhossain checkpoint's own class label spelling with the dair-ai/emotion dataset it is
# demoed against ("sad" -> "sadness"); every other class name already matches.
_SHHOSSAIN_LABEL_ALIASES = {"sad": "sadness"}


# ── Custom-model builders ──────────────────────────────────────────────────────────────────────────
# Standard checkpoints load generically above. A model whose classification *head* is custom (its layer
# structure / weight keys / IO contract are not a standard, so they can't be introspected from a name
# alone) needs a per-model builder. Register one function per such checkpoint below; the generic path is
# never touched. Each builder takes (config, device) and returns any TextModel adapter.


def _build_shhossain_sentiment_model(config: ServeConfig, device: str) -> SentenceTransformerModel:
    """Example builder — ``shhossain/all-MiniLM-L6-v2-sentiment-classifier`` (ST body + MLP head).

    This checkpoint is a custom ``trust_remote_code`` model: a standard ``all-MiniLM-L6-v2`` body plus a
    trained ``fc1 -> relu -> fc2 -> relu -> out -> softmax`` MLP head, and it ships no HF tokenizer — so the
    generic HF path cannot load it. Everything model-specific (the head's weight-key convention, dropping
    its final softmax to recover logits, mean pooling) is contained *here*, not in the core loader.
    """
    huggingface_hub = __import__("huggingface_hub")
    sentence_transformers = __import__("sentence_transformers")
    safetensors_torch = __import__("safetensors.torch", fromlist=["load_file"])

    with open(huggingface_hub.hf_hub_download(config.model_name, "config.json")) as fh:
        cfg = json.load(fh)
    class_map = cfg["class_map"]
    labels = [class_map[str(i)] for i in range(len(class_map))]
    # This checkpoint's class_map spells one emotion as "sad" while its natural demo dataset
    # (dair-ai/emotion) names the same class "sadness"; align them so predicted-vs-ground-truth
    # comparisons in the webapp line up. (Same intent as the --label-map flag, but done here since
    # the mismatch is intrinsic to this model's own labels, not a dataset choice.)
    labels = [_SHHOSSAIN_LABEL_ALIASES.get(label, label) for label in labels]

    st_model = sentence_transformers.SentenceTransformer(cfg["embedding_model"], device=device)
    weights = safetensors_torch.load_file(huggingface_hub.hf_hub_download(config.model_name, "model.safetensors"))
    # Rebuild the trained MLP as a *logits* head (sizes from weight shapes; the checkpoint's final softmax
    # is dropped — SentenceTransformerModel.predict softmaxes and the LIG backend needs logit space).
    head = nn.Sequential(
        nn.Linear(weights["fc1.weight"].shape[1], weights["fc1.weight"].shape[0]),
        nn.ReLU(),
        nn.Linear(weights["fc2.weight"].shape[1], weights["fc2.weight"].shape[0]),
        nn.ReLU(),
        nn.Linear(weights["out.weight"].shape[1], weights["out.weight"].shape[0]),
    )
    for idx, name in ((0, "fc1"), (2, "fc2"), (4, "out")):
        head[idx].load_state_dict({"weight": weights[f"{name}.weight"], "bias": weights[f"{name}.bias"]})
    return SentenceTransformerModel(st_model, head.to(device).eval(), label_names=labels, pool="mean")


def _build_camembertv2_cls_model(config: ServeConfig, device: str) -> HFClassifierModel:
    """``almanach/camembertv2-base-cls`` — standard checkpoint, but don't trust its predictions either.

    Architecturally this is a plain ``RobertaForSequenceClassification`` (CamemBERT v2) that would load
    fine through the generic path — the config problem is only ``label_names``. Its ``config.json`` ships
    ``label2id`` but no ``id2label``; when only one of the pair is present, transformers regenerates the
    missing side (and *overwrites* the other) via a ``num_labels`` setter, so ``config.id2label`` resolves
    to meaningless ``{0: "LABEL_0", 1: "LABEL_1"}`` at load time — the generic ``_load_hf_classifier``
    path (which reads ``label_names`` from exactly that) would show those in the webapp.

    Passing ``label_names`` explicitly only fixes the *display* problem, not the model. A first attempt
    at picking the right order used a 10-example hand-written probe (documented order 2/10, flipped
    8/10) and shipped the flip — that was wrong to trust: ``demo/check_camembertv2_amazon_fr.py`` scores
    both orderings against 388 gold-labeled reviews from this checkpoint's own FLUE-CLS fine-tuning
    domain and finds **neither beats the trivial majority-class baseline**, and the model is confidently
    wrong about as often as confidently right (mean top-class probability ~0.79 regardless of
    correctness). The 10-example probe was noise, not signal — a small hand check can't tell a genuinely
    swapped label from a model that just doesn't work. ``label_names`` below keeps the checkpoint's own
    documented convention (no evidence favours the flip over it); treat every prediction from this model
    as illustrative, not accurate — this builder exists as a worked example of the
    label-names-can't-be-trusted registration pattern, not as a working sentiment classifier.

    Only ``label_names`` is overridden — everything else, truncation included, comes from
    :func:`_hf_classifier_model`. This checkpoint's tokenizer reports no ``model_max_length`` and its
    ``max_position_embeddings`` is 1025, so building the model directly (as this once did) left
    truncation disabled and let any review over 1024 tokens abort the CUDA context.
    """
    logger.warning(
        "%s: this checkpoint's predictions are near chance-level and confidently wrong on gold in-domain "
        "data (neither label ordering beats the majority-class baseline) — see "
        "demo/check_camembertv2_amazon_fr.py. Treat its output in this demo as illustrative only.",
        config.model_name,
    )
    return _hf_classifier_model(config, device, label_names=["negative", "positive"])


# ``--model-name`` -> a custom builder for a checkpoint whose head can't be introspected from its name.
# Anything not listed takes the generic ``_load_hf_classifier`` path. To serve a new custom model, add a
# ``build(config, device) -> TextModel`` function above and one entry here — the core loader is untouched.
# See the module docstring's "Serving a custom model" section for the per-adapter recipe.
MODEL_BUILDERS: dict[str, Callable[[ServeConfig, str], TextModel]] = {
    "shhossain/all-MiniLM-L6-v2-sentiment-classifier": _build_shhossain_sentiment_model,
    "almanach/camembertv2-base-cls": _build_camembertv2_cls_model,
}


def load_model(config: ServeConfig) -> TextModel:
    """Load the model on the best available device: a registered custom builder, else the generic HF path.

    Standard ``AutoModelForSequenceClassification`` checkpoints of any certified architecture load with no
    configuration; a checkpoint whose head is custom (so it can't be introspected from its name) is built
    by its entry in :data:`MODEL_BUILDERS`.
    """
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA available — using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        logger.info("No CUDA device found — using CPU")

    builder = MODEL_BUILDERS.get(config.model_name)
    return builder(config, device) if builder is not None else _load_hf_classifier(config, device)


def build_backend(config: ServeConfig, model: TextModel) -> NlpCaptumLigBackend | None:
    """Return the attribution backend selected by ``--attribution``.

    Returns ``None`` for ``"shap"`` so ``NlpExplainer`` builds its default ``NlpShapBackend`` (which
    needs the model's ``shap_callable`` wiring); ``"lig"`` returns an explicit ``NlpCaptumLigBackend``.
    """
    if config.attribution == "lig":
        # LIG runs one integration per class per sample — show a progress bar over the batch.
        # internal_batch_size chunks each sample's n_steps=50 scaled-copies batch so it doesn't OOM
        # memory-hungry architectures (e.g. DeBERTa) on a small GPU — see ServeConfig.lig_batch_size.
        return NlpCaptumLigBackend(
            model,
            label_names=model.label_names,
            explainer_compute_args={"internal_batch_size": config.lig_batch_size},
            show_progress=True,
        )
    return None


def build_reducer() -> pacmap.PaCMAP:
    """The dimensionality reducer for the scatter — a *demo* choice, not a library one.

    ``NlpExplainer.compute_projection`` owns everything that must stay consistent (which space the
    texts are embedded in, and the caching of both the vectors and the coordinates) and takes the
    reducer as an argument, because which manifold method suits your data is a modelling decision.
    PaCMAP is a good default for text clusters; swap in UMAP, t-SNE, or drop the argument entirely for
    the built-in PCA.
    """
    return pacmap.PaCMAP(n_components=2, n_neighbors=5, MN_ratio=0.5, FP_ratio=2.0)


def main() -> None:
    """Load config + data + model, fit, explain (reusing the cache), and serve the artifact."""
    config = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    # Quiet noisy third-party INFO logs (HF Hub freshness checks, download chatter) so the
    # device/cache messages below stand out.
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    sentences, y_true = load_data(config)
    dataset_dir = _dataset_cache_dir(config)  # projection lives here (shared across attribution backends)
    explain_dir = _explain_cache_dir(config)  # contributions live here (one subdir per attribution backend)

    model = load_model(config)
    if config.embedding_space is not None:
        # One setting drives both the scatter projection and similar-example retrieval. Assigning it
        # here (rather than threading it through every MODEL_BUILDERS entry) is safe because the
        # setter validates the name against the backbone — a typo raises now, not mid-forward-pass.
        model.embedding_space = config.embedding_space
    logger.info("Embedding space: %s (scatter projection + similar examples)", model.resolve_space())

    # Similar-examples reference corpus (the training split) + a cache dir for its embedding bank,
    # keyed by corpus/space/model so it is embedded once and reloaded on later runs.
    reference_corpus = load_reference_corpus(config)
    similar_cache_dir = dataset_dir / "similar"

    # Step 1 — bind the model. No data and no results: the constructor takes what is true for every
    # batch. No cf_generator is passed, so NlpExplainer auto-discovers every built-in compatible with
    # the model (HotFlip + AblationFlip on an HF classifier) and offers them from the What-if Lab's
    # method dropdown.
    xpl = NlpExplainer(
        model,
        label_names=model.label_names,
        backend=build_backend(config, model),
    )

    # Step 2 — fit() learns reference state, which for text is the corpus. Two panels come from it,
    # with different requirements: "Similar Examples" needs an embedding-capable model, while the
    # label-noise probe needs only the *labels* and so works even behind a prediction-only pipeline.
    # Passing y is what enables the second. Both are skipped when --n-reference 0 leaves this None.
    # This is where the reference corpus gets embedded (one forward pass per example, cached under
    # similar_cache_dir) — deliberately here rather than inside the first callback that opens the
    # panel. Pass precompute=False to trade a fast start for a slow first click.
    reference_texts, reference_labels = reference_corpus if reference_corpus is not None else (None, None)
    xpl.fit(reference_texts, y=reference_labels, cache_dir=similar_cache_dir)

    if config.recompute:
        # Drop this text set's cached contributions so the steps below recompute + overwrite. The
        # explainer owns the key (it covers the model and backend), so it also owns the deletion —
        # only *this* model+backend's entry goes, other backends' caches are left intact. The
        # embeddings + projection are dropped by compute_projection(recompute=True) below.
        logger.info("--recompute: dropping cached %s contributions + projection for these texts", config.attribution)
        xpl.clear_cache(sentences, explain_dir)

    # Both steps load from cache if one exists for these exact texts, otherwise they compute and write
    # it. So the first run is slow; later runs only pay the model load.
    explain_cache = xpl.cache_path(sentences, explain_dir)
    if explain_cache.exists():
        logger.info("Contributions cache hit (%s) — loading %s", config.attribution, explain_cache)
    else:
        logger.info(
            "Contributions cache miss (%s) — computing for %d texts (this is the slow part)",
            config.attribution,
            len(sentences),
        )
    # Step 3 — the expensive call, and the only one that produces results. The returned artifact is
    # immutable and holds no model, no backend and no explainer handle; ground truth rides on it
    # (never on xpl), which is why a cache hit still gets *this* run's y_true attached.
    dataset_label = str(config.dataset_path) if config.is_local_dataset else config.dataset_name
    explanation = xpl.explain(sentences, y=y_true, cache_dir=explain_dir)

    # Step 4 — everything downstream reads the artifact as an argument.
    projected = xpl.compute_projection(
        explanation, reducer=build_reducer(), cache_dir=dataset_dir, recompute=config.recompute
    )

    # (No bank warm-up here: fit(precompute=True) above already embedded the reference corpus, under
    # the same condition this used to test — can_find_similar() is "a retriever was built", which is
    # exactly what fit's precompute branch keys on.)

    logger.info(
        "attribution=%s | counterfactual=%s | can_edit=%s | can_counterfactual=%s | can_find_similar=%s"
        " | can_detect_label_noise=%s | can_probe_labels=%s",
        config.attribution,
        ",".join(name for name, _ in xpl.available_cf_generators()) or "none",
        xpl.can_edit(),
        xpl.can_counterfactual(),
        xpl.can_find_similar(),
        # Takes the artifact: label-noise detection needs ground truth + per-class probabilities, and
        # both live on the explanation rather than on the explainer.
        xpl.can_detect_label_noise(explanation),
        xpl.can_probe_labels(),
    )
    # Log the path the app actually mounts on, normalised the way NlpWebApp will normalise it, so a
    # subpath run shows the URL to open rather than a root that would 404.
    base_path = f"/{config.url_base_path.strip().strip('/')}/" if config.url_base_path else "/"
    logger.info("Serving on http://%s:%d%s (Ctrl+C to stop)", config.host, config.port, base_path)
    xpl.run_app(
        explanation,
        port=config.port,
        debug=False,
        host=config.host,
        scatter_xy=projected,
        url_base_pathname=config.url_base_path,
        info={"Dataset": dataset_label},
    )


if __name__ == "__main__":
    main()
