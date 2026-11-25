# Serving the NLP explainer

`serve_nlp.py` explains a text classifier and serves the NLP webapp. The What-if Lab (live edits,
counterfactuals) needs the model loaded, so the script always loads it. The expensive compute is
cached instead. Library concepts (`fit` / `explain`, `.xpl` snapshots, backends) are covered in the
[NLP Explainability wiki page](../../docs/wiki/NLP-Explainability.md).

Requires the `[nlp]` extra:

```bash
pip install "shapash[nlp]"
python scripts/nlp/serve_nlp.py          # emotion model + dair-ai/emotion, http://127.0.0.1:8050
python scripts/nlp/serve_nlp.py --help   # every flag
```

It uses the GPU if there is one, otherwise the CPU.

## Caching

The first run computes contributions, embeddings and the 2-D projection, and writes them under
`--cache-dir` (default `scripts/nlp/nlp_cache/`). Later runs with the same texts only pay the model
load. `--recompute` forces a fresh compute. The layout is:

```
<cache-dir>/<model>/<dataset>__<split>/
├── projection.<space>.npz      # 2-D scatter, shared by all attribution methods
├── similar/                    # similar-examples embedding bank
├── nlp_shap/<hash>.xpl         # contributions, one directory per method
└── nlp_captum_lig/<hash>.xpl
```

Switching `--attribution`, `--output-space` or `--model-name` never reuses another configuration's
results, so no `--recompute` is needed. A cached `.xpl` can be served without the model
(`NlpExplanation.load`), but it carries no ground truth and the What-if Lab is disabled.

## Examples

```bash
# Captum LayerIntegratedGradients instead of SHAP; SHAP on raw logits
python scripts/nlp/serve_nlp.py --attribution lig
python scripts/nlp/serve_nlp.py --output-space logit

# Another hub model and dataset
python scripts/nlp/serve_nlp.py --model-name distilbert-base-uncased-finetuned-sst-2-english \
    --dataset-name sst2 --dataset-split validation --text-column sentence

# Dataset and model spell labels differently: rename the ground truth
python scripts/nlp/serve_nlp.py --model-name lvwerra/distilbert-imdb --dataset-name stanfordnlp/imdb \
    --label-map '{"neg": "NEGATIVE", "pos": "POSITIVE"}'

# Behind a reverse proxy routing a subpath to this process
python scripts/nlp/serve_nlp.py --url-base-path /shapash-nlp-explainer/
```

Rows are shuffled with `--seed` before `--n` / `--n-reference` are taken, since some datasets store
rows grouped by label.

## Private model and data

Nothing has to come from the HuggingFace hub, and a local run gets the full webapp:

```bash
python scripts/nlp/serve_nlp.py --model-name path/to/checkpoint \
    --dataset-path path/to/data --dataset-split test --reference-split train \
    --label-map '{"negative": "NEGATIVE", "positive": "POSITIVE"}'
```

- `--model-name` is a `save_pretrained` directory. When it ships no tokenizer, the base model
  recorded in its `config.json` is used; pass `--tokenizer-name` for a customised one.
- `--label-names NEGATIVE,POSITIVE` sets the class names (in output order) when the config only
  reports `LABEL_0`/`LABEL_1`. Check the order against a good number of labelled rows, not a handful.
- `--dataset-path` is a directory with one `<split>.parquet` or `<split>.pkl` per split, or a single
  file with `--n-reference 0`. `.pkl` files are unpickled, so only use files you produced yourself.

## Custom models

Any standard `AutoModelForSequenceClassification` checkpoint loads from `--model-name` alone. A
model with a custom head goes through the `MODEL_BUILDERS` registry in `serve_nlp.py`: write a
`build(config, device) -> TextModel` function and register it under the model name. Pick the
adapter by shape:

- sentence-transformer body + head: `SentenceTransformerModel`, with a head returning logits (drop
  any final softmax). `_build_shhossain_sentiment_model` is a worked example.
- raw encoder body + head: `TorchClassifierModel(body, head, tokenizer)`.
- standard classifier needing `trust_remote_code`: `HFClassifierModel.from_pretrained(name,
  trust_remote_code=True)`.

Caching, the What-if Lab, both attribution methods and similar examples then work unchanged.

## Iterating on the webapp

Run `main()`'s body in a REPL once, then call `xpl.run_app(explanation, ...)` again after reloading
the changed webapp modules. The model and the explanation stay in memory.
