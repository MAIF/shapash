# Shapash NLP Explainer — Demo

Interactive webapp that explains a DistilBERT emotion classifier at the token level using SHAP
values, with a live **What-if Lab**: edit a sample's text for a live prediction + token-contribution
highlight, and get auto-generated minimal token flips (counterfactuals) that change the prediction.

**Webapp layout**, three panels:
- Left: control panel with **Dataset** (Dash AG Grid to explore the dataset and click on texts),
  **Embeddings** (scatter plot, used to lasso/box select, highlight word importance), and **Data Editor**
- Upper Right: **Word Importance** (global explainability) tab and **Counterfactuals** tab.
- Lower Right: **Local contributions** divided as sentence highlights tab and waterfall plot tab.

**`demo/serve_nlp.py` is the serving script** — everything below runs it with its defaults (the
emotion model/dataset). `Dockerfile`/`docker-compose.yml` are just a convenience wrapper around it:
they never run SHAP themselves, they only copy in a cache built by running the script once (Step 1
below). Model, dataset, split, columns, sample count, and a ground-truth label-renaming map are all
configurable via `ServeConfig` in that file — see its module docstring for
`--model-name`/`--dataset-name`/`--dataset-split`/`--text-column`/`--label-column`/`--label-map`/`--n`
and worked examples (e.g. swapping in an IMDB sentiment model/dataset). **Nothing has to come from the
HuggingFace hub**: `--model-name /path/to/checkpoint` serves a local `save_pretrained` directory (its
tokenizer is resolved from the checkpoint, falling back to the base model in its `config.json`; pass
`--tokenizer-name` for a customised one, and `--label-names` when the config has no class names), and
`--dataset-path` serves local `<split>.parquet` / `<split>.pkl` dataframes instead of `--dataset-name`.
A private model + private data gets the full webapp — highlights, What-if Lab, counterfactuals,
similar examples — like any hub model. `Dockerfile` itself stays
hardcoded to the emotion example on purpose (it's meant to run out of the box for anyone cloning the
repo) — its build-time warm-up (`RUN python -c "..."`) downloads exactly that model/dataset/split, so
it always matches `ServeConfig`'s defaults. Serving a different model/dataset combo in Docker means
editing that warm-up line to match, then redoing Step 1 below with the same flags before rebuilding.

---

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running (for the Docker path), or the
  `[nlp]` extra installed locally (`pip install "shapash[nlp]"`)

## Step 1 — Pre-compute the SHAP cache (one-time, GPU optional)

Needs the `[nlp]` extra (torch, transformers, datasets, pacmap). Run the script once — it computes SHAP
+ the projection (a few minutes on CPU, faster on GPU), writes the cache, then starts serving. Once it
prints the local URL, the cache is on disk — press `Ctrl+C`:

```bash
pip install "shapash[nlp]"
python demo/serve_nlp.py   # Ctrl+C once it starts serving
```

This writes `demo/nlp_cache/` — SHAP contributions, predictions, and the 2-D projection, keyed by a
hash of the input texts. The Docker build copies it in, so this expensive step never runs in Docker.

> **Already have `demo/nlp_cache/`?** Skip to Step 2.

## Step 2 — Build and run

Run from the **repo root**:

```bash
docker compose -f demo/docker-compose.yml up --build shapash-nlp
```

The build downloads the model + dataset into the image and copies in the SHAP cache; at **run** time it
serves fully offline — no downloads, no SHAP, no GPU. (The build fails fast if `demo/nlp_cache/` is
missing — do Step 1 first.)

Then open **http://localhost:8050** in your browser. Press `Ctrl+C` to stop.

## Run without Docker

Just run the same script — it serves on http://127.0.0.1:8050 and reuses the `demo/nlp_cache/`
from Step 1 (recomputing only if the input texts change, or if you pass `--recompute`):

```bash
python demo/serve_nlp.py             # ServeConfig defaults (500 samples, dair-ai/emotion test split)
python demo/serve_nlp.py --n 1000    # more samples (recomputes + re-caches)
```

It uses the GPU if present, otherwise CPU.

> **Iterating on the webapp code?** Don't restart the process — run `serve_nlp.py`'s body in a REPL
> once, then re-call `xpl.run_app(...)` after reloading the changed webapp modules. The model stays
> resident, so you never pay the reload.
