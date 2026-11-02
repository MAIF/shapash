"""Generate the files the Sphinx documentation needs before it is built.

Sphinx only reads sources under ``docs/``, so this script:

- copies the tutorial notebooks from ``tutorial/`` into ``docs/tutorials/``, with an ``index.rst`` per folder;
- generates the demo report published at ``report.html`` (linked from the README) into ``docs/_generated/``;
- generates ``switcher.json``, the list of versions of the version switcher, from the git tags into ``docs/_generated/``.

Both output folders are gitignored. Run it before ``sphinx-build``:

    python scripts/prepare_docs.py
    sphinx-build docs docs/_build/html
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from generate_report import generate_report
from packaging.version import InvalidVersion, Version

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DIR = PROJECT_ROOT / "tutorial"
DOCS_DIR = PROJECT_ROOT / "docs"
TUTORIALS_DOC_DIR = DOCS_DIR / "tutorials"
REPORT_PATH = DOCS_DIR / "_generated" / "report.html"
SWITCHER_PATH = DOCS_DIR / "_generated" / "switcher.json"
DOCS_URL = "https://shapash.readthedocs.io/en"
# Older releases are not built on Read the Docs
OLDEST_LISTED_RELEASE = (2, 4)

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


def _release_tags() -> list[Version]:
    """Return the final release versions tagged in git, newest first."""
    try:
        tags = subprocess.run(
            ["git", "tag", "--list", "v*"],  # noqa: S607 (git from PATH, constant arguments)
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        print("Could not list the git tags: the version switcher lists the development version only")
        return []
    versions = []
    for tag in tags:
        try:
            version = Version(tag)
        except InvalidVersion:
            continue
        if not version.is_prerelease and version.release[:2] >= OLDEST_LISTED_RELEASE:
            versions.append(version)
    return sorted(versions, reverse=True)


def write_version_switcher() -> None:
    """Write the version switcher entries: the development version, then the latest patch of each minor release.

    The newest release is marked as preferred and links to ``stable``. An entry's ``version`` is the Read the
    Docs version slug, which ``docs/conf.py`` matches against the version being built.
    """
    entries = [{"name": "dev", "version": "latest", "url": f"{DOCS_URL}/latest/"}]
    seen_minors = set()
    for version in _release_tags():
        minor = version.release[:2]
        if minor in seen_minors:
            continue
        seen_minors.add(minor)
        slug = f"v{version}"
        if len(entries) == 1:  # newest release
            entries.append(
                {"name": f"{slug} (stable)", "version": slug, "url": f"{DOCS_URL}/stable/", "preferred": True}
            )
        else:
            entries.append({"name": slug, "version": slug, "url": f"{DOCS_URL}/{slug}/"})
    SWITCHER_PATH.parent.mkdir(parents=True, exist_ok=True)
    SWITCHER_PATH.write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


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
    write_version_switcher()
    print(f"Version switcher written: {SWITCHER_PATH}")
    if not args.skip_report:
        generate_report("titanic", "classification", output_file=REPORT_PATH)
        print(f"Report generated: {REPORT_PATH}")


if __name__ == "__main__":
    main()
