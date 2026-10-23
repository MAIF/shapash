"""Generate the files the Sphinx documentation needs before it is built.

Sphinx only reads sources under ``docs/``, so this script:

- copies the tutorial notebooks from ``tutorial/`` into ``docs/tutorials/``, with an ``index.rst`` per folder;
- generates the demo report published at ``report.html`` (linked from the README) into ``docs/_generated/``.

Both output folders are gitignored. Run it before ``sphinx-build``:

    python scripts/prepare_docs.py
    sphinx-build docs docs/_build/html
"""

import argparse
import shutil
from pathlib import Path

from generate_report import generate_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = PROJECT_ROOT / "tutorial"
DOCS_DIR = PROJECT_ROOT / "docs"
TUTORIALS_DOC_DIR = DOCS_DIR / "tutorials"
REPORT_PATH = DOCS_DIR / "_generated" / "report.html"

# Section title of each tutorial folder; other folders get their name, capitalized
TITLES = {
    "overview": "First steps",
    "common": "Feature groups and colors",
    "debug_and_what_if": "Debugging and what-if analysis",
    "domain_examples": "Domain examples",
    "explainability_quality": "Explainability quality",
    "explainer_and_backend": "Backends",
    "generate_report": "Shapash Report",
    "generate_webapp": "Web app",
    "nlp": "Text data (NLP)",
    "plots_and_charts": "Plots and charts",
    "postprocess": "Postprocessing",
    "predictor_to_production": "SmartPredictor",
    "production_and_ops": "Production and operations",
    "use_encoders": "Encoders",
}


def _copy_notebooks(src_dir: Path, dst_dir: Path, name: str) -> None:
    """Copy the notebooks of ``src_dir`` into ``dst_dir`` and write an ``index.rst`` listing them."""
    title = TITLES.get(name, name.replace("_", " ").capitalize())
    dst_dir.mkdir(parents=True, exist_ok=True)
    for notebook in sorted(src_dir.glob("*.ipynb")):
        shutil.copyfile(notebook, dst_dir / notebook.name)
    index = "\n".join([title, "=" * len(title), "", ".. toctree::", "    :maxdepth: 1", "    :glob:", "", "    *", ""])
    (dst_dir / "index.rst").write_text(index, encoding="utf-8")


def copy_tutorials() -> None:
    """Mirror ``tutorial/`` into ``docs/tutorials/``: one folder per tutorial folder, plus the overview notebooks."""
    shutil.rmtree(TUTORIALS_DOC_DIR, ignore_errors=True)
    for src_dir in sorted(TUTORIAL_DIR.iterdir()):
        if src_dir.is_dir() and not src_dir.name.startswith("."):
            _copy_notebooks(src_dir, TUTORIALS_DOC_DIR / src_dir.name, src_dir.name)
    _copy_notebooks(TUTORIAL_DIR, TUTORIALS_DOC_DIR, "overview")


def main() -> None:
    """Parse CLI options and prepare the documentation sources."""
    parser = argparse.ArgumentParser(description="Prepare the generated sources of the Sphinx documentation.")
    parser.add_argument(
        "--skip-report",
        action="store_true",
        help="do not regenerate the demo report (faster; keeps a previously generated one)",
    )
    args = parser.parse_args()

    copy_tutorials()
    print(f"Tutorials copied to {TUTORIALS_DOC_DIR}")
    if not args.skip_report:
        generate_report("titanic", "classification", output_file=REPORT_PATH)
        print(f"Report generated: {REPORT_PATH}")


if __name__ == "__main__":
    main()
