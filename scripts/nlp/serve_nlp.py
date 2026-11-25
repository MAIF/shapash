"""Serve the NLP explainer webapp, What-if Lab included, on a live model with cached compute.

Loads a HF classifier and dataset, runs ``fit`` / ``explain`` / ``compute_embeddings``, caches all
of it under ``--cache-dir``, and serves the result. Usage and examples: ``scripts/nlp/README.md``.
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

from shapash.backend import NlpBackend, NlpCaptumLigBackend, NlpShapBackend
from shapash.compute.embeddings import Embedding
from shapash.explainer.nlp_explainer import NlpExplainer
from shapash.model import HFClassifierModel, SentenceTransformerModel, TextModel

_HERE = Path(__file__).parent

logger = logging.getLogger("serve_nlp")


@dataclass
class ServeConfig:
    """Everything needed to load data + model and serve the What-if webapp.

    A dataclass rather than argparse's ``Namespace`` so the loading helpers stay usable from a REPL.
    """

    model_name: str = "bhadresh-savani/distilbert-base-uncased-emotion"
    # Tokenizer source when the checkpoint ships none; ``None`` lets ``from_pretrained`` resolve it.
    tokenizer_name: str | None = None
    # Class names in output-column order, overriding the model config's ``id2label``.
    label_names: list[str] | None = None
    dataset_name: str = "dair-ai/emotion"
    # Local dataframes instead of the hub; mutually exclusive with ``dataset_name``.
    dataset_path: Path | None = None
    dataset_config: str | None = None  # HF dataset config/subset (e.g. a language)
    dataset_split: str = "test"
    text_column: str = "text"
    label_column: str = "label"
    label_map: dict[str, str] = field(default_factory=dict)
    n: int = 500
    # Rows are shuffled before taking --n / --n-reference: some datasets store rows grouped by label.
    seed: int = 0
    attribution: str = "shap"  # "shap" | "lig" (Captum LayerIntegratedGradients)
    # Captum ``internal_batch_size``: chunks LIG's 100-step batch so memory-hungry models don't OOM.
    lig_batch_size: int = 8
    # "probability" | "logit"; ``None`` keeps the backend's own (probability for SHAP, logit for LIG).
    output_space: str | None = None
    # Similar-examples corpus: the split neighbours are retrieved from. ``n_reference=0`` disables it.
    reference_split: str = "train"
    n_reference: int = 2000
    # Space for both the scatter and neighbour ranking; ``None`` keeps the model's own ("decision").
    embedding_space: str | None = None
    cache_dir: Path = _HERE / "nlp_cache"
    port: int = 8050
    host: str = "0.0.0.0"  # noqa: S104
    # Mount point behind a reverse proxy routing a subpath (e.g. "/shapash-nlp-explainer/").
    url_base_path: str | None = None
    recompute: bool = False

    def __post_init__(self) -> None:
        """Expand ``~`` in ``dataset_path``: a REPL-built config bypasses the shell's expansion."""
        if self.dataset_path is not None:
            self.dataset_path = self.dataset_path.expanduser()

    @property
    def model_path(self) -> Path | None:
        """The local checkpoint directory ``model_name`` points at, or ``None`` for a hub id."""
        path = Path(self.model_name).expanduser()
        return path if path.is_dir() else None

    @property
    def is_local_dataset(self) -> bool:
        """True when rows come from local dataframes (``--dataset-path``) rather than the HF hub."""
        return self.dataset_path is not None


# ``--attribution`` choice -> the backend's registered ``.name`` (also its cache subdirectory).
_ATTRIBUTION_BACKENDS = {"shap": "nlp_shap", "lig": "nlp_captum_lig"}


def _path_slug(path: Path) -> str:
    """Cache tag for a local path: its name (readable) plus a digest of its absolute path (unique)."""
    resolved = path.expanduser().resolve()
    digest = hashlib.sha256(str(resolved).encode()).hexdigest()[:8]
    return f"{resolved.name}__{digest}"


def _model_slug(config: ServeConfig) -> str:
    """Cache tag for ``--model-name``: ``org__model`` for a hub id, :func:`_path_slug` for a path."""
    if config.model_path is not None:
        return _path_slug(config.model_path)
    return config.model_name.replace("/", "__")


def _dataset_slug(config: ServeConfig) -> str:
    """Cache tag ``<dataset>__[<config>__]<split>``; the config segment only when it is set."""
    if config.is_local_dataset:
        return f"{_path_slug(config.dataset_path)}__{config.dataset_split}"
    parts = [config.dataset_name, *([config.dataset_config] if config.dataset_config else []), config.dataset_split]
    return "__".join(parts).replace("/", "__")


def _dataset_cache_dir(config: ServeConfig) -> Path:
    """Return ``cache_dir/<model>/<dataset>``: embeddings and projection, shared by every backend.

    The library cache keys already cover model, backend and texts; these levels only keep the cache
    browsable and prunable.
    """
    return config.cache_dir / _model_slug(config) / _dataset_slug(config)


def _explain_cache_dir(config: ServeConfig) -> Path:
    """Return ``<dataset-dir>/<backend>``: contributions, one subdirectory per attribution backend."""
    return _dataset_cache_dir(config) / _ATTRIBUTION_BACKENDS[config.attribution]


def parse_args(argv: list[str] | None = None) -> ServeConfig:
    """Parse CLI args into a ``ServeConfig``, defaulting every flag from the dataclass."""
    defaults = ServeConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        default=defaults.model_name,
        help="HF model id, or a local checkpoint directory saved with save_pretrained.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default=defaults.tokenizer_name,
        help="HF id or path of the tokenizer, when it isn't in the checkpoint. Default: the checkpoint, "
        "falling back to the base model recorded in its config.",
    )
    parser.add_argument(
        "--label-names",
        default=None,
        help="Comma-separated class names in output-column order, overriding the config's id2label "
        "(e.g. 'NEGATIVE,POSITIVE' for a fine-tune whose config reports LABEL_0/LABEL_1).",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset-name", default=defaults.dataset_name, help="HF datasets id.")
    source.add_argument(
        "--dataset-path",
        type=Path,
        default=defaults.dataset_path,
        help="Local dataframes instead of the hub: a directory of <split>.parquet / <split>.pkl, or a "
        "single file (requires --n-reference 0).",
    )
    parser.add_argument(
        "--dataset-config",
        default=defaults.dataset_config,
        help="HF dataset config/subset (e.g. a language), when the dataset has several.",
    )
    parser.add_argument("--dataset-split", default=defaults.dataset_split)
    parser.add_argument("--text-column", default=defaults.text_column, help="Column holding the raw text.")
    parser.add_argument("--label-column", default=defaults.label_column, help="Column holding the ground-truth label.")
    parser.add_argument(
        "--label-map",
        type=json.loads,
        default=defaults.label_map,
        help='JSON object renaming dataset labels to the model\'s, e.g. \'{"neg": "NEGATIVE", "pos": "POSITIVE"}\'.',
    )
    parser.add_argument("--n", type=int, default=defaults.n, help="Number of samples to serve.")
    parser.add_argument(
        "--seed", type=int, default=defaults.seed, help="Shuffle seed applied before taking --n / --n-reference rows."
    )
    parser.add_argument(
        "--attribution",
        choices=list(_ATTRIBUTION_BACKENDS),
        default=defaults.attribution,
        help="Sentence-highlight method: 'shap' (Partition SHAP) or 'lig' (Captum LayerIntegratedGradients).",
    )
    parser.add_argument(
        "--lig-batch-size",
        type=int,
        default=defaults.lig_batch_size,
        help="Captum internal_batch_size for --attribution lig. Lower it (2 or 1) on CUDA out-of-memory.",
    )
    parser.add_argument(
        "--output-space",
        choices=["probability", "logit"],
        default=defaults.output_space,
        help="Space the contributions are explained in. Default: the backend's own (lig: logit only).",
    )
    parser.add_argument(
        "--reference-split",
        default=defaults.reference_split,
        help="Split used as the similar-examples corpus (the model's training split).",
    )
    parser.add_argument(
        "--n-reference",
        type=int,
        default=defaults.n_reference,
        help="Reference rows for 'Similar Examples' (0 disables the panel).",
    )
    parser.add_argument(
        "--embedding-space",
        default=defaults.embedding_space,
        help="Space for the scatter and similar examples: 'decision', 'pooled', or a backbone submodule "
        "name. Default: the model's own.",
    )
    parser.add_argument("--cache-dir", type=Path, default=defaults.cache_dir)
    parser.add_argument("--port", type=int, default=defaults.port)
    parser.add_argument("--host", default=defaults.host)
    parser.add_argument(
        "--url-base-path",
        default=defaults.url_base_path,
        help="URL prefix to serve under, matching a reverse proxy's subpath (e.g. '/shapash-nlp-explainer/').",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore the cache and recompute contributions + projection (overwrites the cache).",
    )
    args = parser.parse_args(argv)
    args.label_names = [name.strip() for name in args.label_names.split(",")] if args.label_names else None
    # Validate the config rather than the namespace, so the checks see the ``~``-expanded path.
    config = ServeConfig(**vars(args))
    if config.attribution == "lig" and config.output_space == "probability":
        parser.error("--attribution lig attributes raw logits only; drop --output-space or pass 'logit'.")
    if config.is_local_dataset and config.dataset_config is not None:
        parser.error("--dataset-config applies to --dataset-name only, not --dataset-path.")
    if config.is_local_dataset and config.dataset_path.is_file() and config.n_reference > 0:
        parser.error(
            f"--dataset-path {config.dataset_path} is a single file, so there is no --reference-split "
            f"{config.reference_split!r} to load: pass --n-reference 0, or point --dataset-path at a "
            "directory holding one file per split."
        )
    return config


# Formats ``--dataset-path`` recognises, in lookup order. ``read_pickle`` executes arbitrary code:
# point ``--dataset-path`` only at files you produced yourself.
_DATAFRAME_READERS: dict[str, Callable[[Path], pd.DataFrame]] = {
    ".parquet": pd.read_parquet,
    ".pkl": pd.read_pickle,  # noqa: S301
    ".pickle": pd.read_pickle,  # noqa: S301
}


def _local_split_file(dataset_path: Path, split: str) -> Path:
    """Resolve ``--dataset-path`` + a split name to one dataframe file.

    A file is returned as is; a directory is searched for ``<split>.<ext>`` in
    :data:`_DATAFRAME_READERS` order.
    """
    path = dataset_path
    if path.is_file():
        return path
    if not path.is_dir():
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

    Rows with a missing or blank text or label are dropped, then sampled like the hub path.
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
    # A blank label would otherwise show up as a ground-truth class of its own.
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


def _load_split(config: ServeConfig, split: str, n: int) -> tuple[list[str], list[str]]:
    """Load ``n`` ``(texts, label_strings)`` of ``split`` from the configured dataset.

    Hub labels come from the ``ClassLabel`` names when present, else the raw values stringified;
    ``config.label_map`` then renames them onto the model's spelling.
    """
    if config.is_local_dataset:
        return _load_local_split(config, split, n)

    dataset = datasets.load_dataset(config.dataset_name, config.dataset_config, split=split).shuffle(seed=config.seed)
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
    """Load the similar-examples corpus from ``config.reference_split``, or ``None`` when disabled.

    A single-file ``dataset_path`` is refused: it would return the served rows themselves. ``parse_args``
    checks this too, but a REPL-built config never goes through it.
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


def _load_hf_classifier(config: ServeConfig, device: str) -> HFClassifierModel:
    """Load any standard ``AutoModelForSequenceClassification`` checkpoint, hub id or local path.

    Label names come from the config's ``id2label`` unless ``--label-names`` overrides them.
    """
    source = str(config.model_path) if config.model_path is not None else config.model_name
    return HFClassifierModel.from_pretrained(
        source,
        tokenizer=config.tokenizer_name,
        label_names=config.label_names,
        device=device,
    )


# The shhossain checkpoint spells one dair-ai/emotion class differently.
_SHHOSSAIN_LABEL_ALIASES = {"sad": "sadness"}


def _build_shhossain_sentiment_model(config: ServeConfig, device: str) -> SentenceTransformerModel:
    """``shhossain/all-MiniLM-L6-v2-sentiment-classifier``: a MiniLM ST body + a trained MLP head.

    A ``trust_remote_code`` model with no HF tokenizer, so the generic path cannot load it. The head
    is rebuilt without its final softmax: ``predict`` softmaxes and LIG needs logits.
    """
    huggingface_hub = __import__("huggingface_hub")
    sentence_transformers = __import__("sentence_transformers")
    safetensors_torch = __import__("safetensors.torch", fromlist=["load_file"])

    with open(huggingface_hub.hf_hub_download(config.model_name, "config.json")) as fh:
        cfg = json.load(fh)
    class_map = cfg["class_map"]
    labels = [class_map[str(i)] for i in range(len(class_map))]
    labels = [_SHHOSSAIN_LABEL_ALIASES.get(label, label) for label in labels]

    st_model = sentence_transformers.SentenceTransformer(cfg["embedding_model"], device=device)
    weights = safetensors_torch.load_file(huggingface_hub.hf_hub_download(config.model_name, "model.safetensors"))
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


# ``--model-name`` -> builder for a checkpoint the generic HF path cannot load (custom head).
MODEL_BUILDERS: dict[str, Callable[[ServeConfig, str], TextModel]] = {
    "shhossain/all-MiniLM-L6-v2-sentiment-classifier": _build_shhossain_sentiment_model,
}


def load_model(config: ServeConfig) -> TextModel:
    """Load the model on the best available device: a registered builder, else the generic HF path."""
    if torch.cuda.is_available():
        device = "cuda"
        logger.info("CUDA available — using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
        logger.info("No CUDA device found — using CPU")

    builder = MODEL_BUILDERS.get(config.model_name)
    return builder(config, device) if builder is not None else _load_hf_classifier(config, device)


def build_backend(config: ServeConfig) -> NlpBackend | None:
    """Return the backend for ``--attribution`` / ``--output-space``; ``None`` keeps the SHAP default."""
    if config.attribution == "lig":
        return NlpCaptumLigBackend(
            explainer_compute_args={"internal_batch_size": config.lig_batch_size},
            show_progress=True,
        )
    if config.output_space == "logit":
        return NlpShapBackend(output_space="logit")
    return None


def build_reducer() -> pacmap.PaCMAP:
    """The scatter's reducer: a choice of this script, not of the library (default is PCA)."""
    return pacmap.PaCMAP(n_components=2, n_neighbors=5, MN_ratio=0.5, FP_ratio=2.0)


def main() -> None:
    """Load config + data + model, fit, explain (reusing the cache), and serve the artifact."""
    config = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    sentences, y_true = load_data(config)
    dataset_dir = _dataset_cache_dir(config)
    explain_dir = _explain_cache_dir(config)

    model = load_model(config)
    if config.embedding_space is not None:
        # The setter validates the name against the backbone, so a typo fails here.
        model.embedding_space = config.embedding_space
    logger.info("Embedding space: %s (scatter projection + similar examples)", model.resolve_space())

    # No cf_generator: every built-in generator compatible with the model is offered in the webapp.
    xpl = NlpExplainer(model, label_names=model.label_names, backend=build_backend(config))

    # Embeds the reference corpus now (cached), rather than on the first "Similar Examples" click.
    reference_corpus = load_reference_corpus(config)
    reference_texts, reference_labels = reference_corpus if reference_corpus is not None else (None, None)
    xpl.fit(reference_texts, y=reference_labels, cache_dir=dataset_dir / "similar")

    explain_cache = xpl.cache_path(sentences, explain_dir)
    if config.recompute:
        logger.info("--recompute: recomputing %s contributions + projection for these texts", config.attribution)
    elif explain_cache.exists():
        logger.info("Contributions cache hit (%s) — loading %s", config.attribution, explain_cache)
    else:
        logger.info(
            "Contributions cache miss (%s) — computing for %d texts (this is the slow part)",
            config.attribution,
            len(sentences),
        )
    explanation = xpl.explain(sentences, y=y_true, cache_dir=explain_dir, recompute=config.recompute)

    # One projection file per space; refitted when the texts changed (``corpus_id`` differs).
    projection_file = dataset_dir / f"projection.{model.resolve_space()}.npz"
    projection = Embedding.load(projection_file) if projection_file.exists() and not config.recompute else None
    if projection is not None and projection.corpus_id == explanation.corpus_id:
        logger.info("Projection loaded from %s", projection_file)
    else:
        embeddings = xpl.compute_embeddings(explanation, cache_dir=dataset_dir, recompute=config.recompute)
        projection = embeddings.project(build_reducer())
        projection.save(projection_file)

    logger.info(
        "attribution=%s (%s, masking=%s) | counterfactual=%s | can_edit=%s | can_counterfactual=%s"
        " | can_find_similar=%s | can_detect_label_noise=%s | can_probe_labels=%s",
        config.attribution,
        xpl.backend.output_space,
        getattr(xpl.backend, "masking", "-"),
        ",".join(name for name, _ in xpl.available_cf_generators()) or "none",
        xpl.can_edit(),
        xpl.can_counterfactual(),
        xpl.can_find_similar(),
        xpl.can_detect_label_noise(explanation),
        xpl.can_probe_labels(),
    )
    # Normalised the way NlpWebApp does, so a subpath run logs the URL to open.
    base_path = f"/{config.url_base_path.strip().strip('/')}/" if config.url_base_path else "/"
    logger.info("Serving on http://%s:%d%s (Ctrl+C to stop)", config.host, config.port, base_path)
    dataset_label = str(config.dataset_path) if config.is_local_dataset else config.dataset_name
    xpl.run_app(
        explanation,
        port=config.port,
        debug=False,
        host=config.host,
        projection=projection,
        url_base_pathname=config.url_base_path,
        info={"Dataset": dataset_label},
    )


if __name__ == "__main__":
    main()
