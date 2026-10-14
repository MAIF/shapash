# NLP Explainer image (CPU-only, live model — the What-if Lab needs it in memory).
#
# This is a convenience wrapper around `demo/serve_nlp.py`, the actual serving script —
# run that once to generate `demo/nlp_cache/` before building. The expensive part — SHAP +
# the projection — is computed ONCE outside Docker and copied in as a cache (see demo/README.md);
# the build never runs SHAP. The model + dataset are downloaded in a build layer (fast, ~270 MB,
# cached like any pip layer), so runtime is fully offline: no downloads, no SHAP, no GPU at run time.
#
# Build from the repo root (fails fast if nlp_cache is missing — do Step 1 first):
#   docker build -f demo/Dockerfile -t shapash-nlp .
#   docker run --rm -p 8050:8050 shapash-nlp
# Then open http://localhost:8050

FROM python:3.11-slim

WORKDIR /app

# CPU-only torch first (from PyTorch's CPU index — smaller, no CUDA), so the [nlp] extra
# sees torch already satisfied and doesn't pull the CUDA build from PyPI.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.12.1"

COPY pyproject.toml README.md MANIFEST.in ./
COPY shapash/ ./shapash/
# setuptools-scm can't find .git in the build context; any placeholder works since the
# running app doesn't use the installed package version at runtime.
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SHAPASH=0.0.0.dev
RUN pip install --no-cache-dir ".[nlp]" "pyarrow>=17.0.0"

# Download the model + dataset into the image (a plain download, not SHAP compute).
ENV HF_HOME=/app/hf_cache
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
import datasets; \
m = 'bhadresh-savani/distilbert-base-uncased-emotion'; \
AutoTokenizer.from_pretrained(m, use_fast=True); \
AutoModelForSequenceClassification.from_pretrained(m); \
datasets.load_dataset('dair-ai/emotion', split='test')"

COPY demo/serve_nlp.py ./serve_nlp.py
# Pre-computed SHAP + projection cache (built by running serve_nlp.py once — see README).
COPY demo/nlp_cache/ ./nlp_cache/

# Serve fully offline: model from the baked HF cache, SHAP from nlp_cache, no network.
# PYTHONUNBUFFERED streams logs; TOKENIZERS_PARALLELISM avoids a fork deadlock.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1

EXPOSE 8050

CMD ["python", "serve_nlp.py", "--n", "1000", "--n-reference", "5000"]
