# How to contribute to Shapash

Thank you for your interest in Shapash! Bug reports, documentation fixes, tutorials and new features
are all welcome. This guide walks you through the whole process, from opening an issue to getting your
pull request merged.

By contributing, you agree that your contributions are licensed under the project's
[Apache 2.0 license](LICENSE).

- [Open an issue first](#open-an-issue-first)
- [Set up your environment](#set-up-your-environment)
    - [Get the code](#get-the-code)
    - [Install the dependencies](#install-the-dependencies)
    - [Set up pre-commit hooks](#set-up-pre-commit-hooks)
- [Make your changes](#make-your-changes)
    - [Create a branch](#create-a-branch)
    - [Follow the code conventions](#follow-the-code-conventions)
    - [Run code quality checks](#run-code-quality-checks)
    - [Run the tests](#run-the-tests)
    - [Update the documentation and tutorials](#update-the-documentation-and-tutorials)
    - [Change dependencies](#change-dependencies)
- [Submit your changes](#submit-your-changes)
    - [Commit and push](#commit-and-push)
    - [Open a pull request](#open-a-pull-request)
    - [Review and merge](#review-and-merge)
- [Contributing using AI](#contributing-using-ai)

# Open an issue first

Every pull request must be linked to an issue. The issue is where we agree on whether and how a change
should be made, so you don't spend time on a pull request that cannot be merged.

Before opening an issue:

- Check the Project tab to see whether the item already exists in the roadmap.
- Search the existing issues to avoid duplicates.

Then open an issue with the template that fits: **Bug report**, **Feature request** or
**Documentation**. For a bug, include a minimal example that reproduces it, and your Shapash and
Python versions.

The Shapash team or community members will reply and help decide whether a pull request is needed. For
anything beyond a small fix, wait for that feedback before you start coding.

# Set up your environment

## Get the code

Fork the [MAIF/shapash](https://github.com/MAIF/shapash) repository to your GitHub account with the
**Fork** button, then clone your fork and register the official repository as `upstream`:

```
git clone https://github.com/<your-github-username>/shapash.git
cd shapash
git remote add upstream https://github.com/MAIF/shapash.git
```

## Install the dependencies

Python 3.11 or newer is required (CI tests 3.11 to 3.14).

We recommend [uv](https://docs.astral.sh/uv/getting-started/installation/). This command creates a
`.venv` virtual environment with shapash in editable mode and the `dev` and `test` dependency groups:

```
uv sync
```

It installs the versions pinned in `uv.lock`, the same ones CI tests against.

Alternatively, with pip (>= 25.1), create and activate a virtual environment, then run:

```
pip install -e . --group dev --group test
```

pip does not read `uv.lock`: it installs the newest versions `pyproject.toml` allows, which may differ
from CI.

The commands in the rest of this guide run inside the virtual environment. Either activate it
(`source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows), or prefix each command with
`uv run` (for example `uv run pytest`).

Common tasks have `make` targets, the same ones CI runs; `make help` lists them. Without `make` (on
Windows, for example), run the command the target wraps, as written in the `Makefile`.

This covers classical (tabular) shapash. The NLP modules and their tests need the heavier `nlp` extra
(torch, transformers, ...); without it, those tests are skipped. To work on NLP:

```
uv sync --extra nlp            # or: pip install -e ".[nlp]" --group dev --group test
```

A later plain `uv sync` removes the NLP packages again (it syncs exactly), while `uv run` and
`uv sync --inexact` leave them in place.

## Set up pre-commit hooks

Pre-commit hooks check your changes automatically before each commit (formatting, linting, type
checking, ...). Enable them once:

```
uv run pre-commit install      # or, with pip in an activated venv: pre-commit install
```

The ruff and mypy hooks use the tools installed in your development environment (`.venv`, or the
activated virtual environment), so they run the same versions as CI.

# Make your changes

## Create a branch

Pull requests target the `develop` branch, so start your branch from the latest `develop` of the
official repository:

```
git fetch upstream
git switch -c feature/my-contribution upstream/develop
```

Name your branch **feature/...** for a new feature or **fix/...** for a bug fix.

## Follow the code conventions

- Code must run on Python 3.11, the oldest supported version.
- Public functions and classes have [numpy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).
- Every bug fix or new feature comes with tests: a bug fix with a test that fails without it. At least
  90% of the lines you add or change must be covered (see [Run the tests](#run-the-tests)).
- A new third-party dependency that only some features need goes into an extra in
  `[project.optional-dependencies]` (like `lime` or `report`), not into the core dependencies. Import
  it with `import_optional_module` from `shapash/_optional.py`, which tells users which extra to
  install when it is missing.

## Run code quality checks

The pre-commit hooks run these checks on the files you commit. To run them on the whole codebase, as
CI does:

```
make lint                      # ruff check (`ruff check --fix` fixes what it can)
make format                    # ruff format --check (run `ruff format` to reformat the files)
make typecheck                 # mypy shapash
```

The ruff and mypy settings are in `pyproject.toml`. CI type-checks on Python 3.11, so code that only
works on newer Python versions fails there.

CI also audits the GitHub workflows with zizmor. If you change files in `.github/`, run
`make workflows` before pushing.

## Run the tests

While you work, run the tests related to your change:

```
pytest tests/unit_tests/path/to/test_file.py     # one file
pytest tests/unit_tests -k "name_of_a_test"      # tests matching a name
```

Before pushing, run the whole suite with coverage:

```
make coverage
```

This runs the unit tests and the slower integration tests in `tests/integration_tests/`, and reports
line and branch coverage with the lines each file leaves uncovered. With the `nlp` extra installed,
the NLP tests download their models from the Hugging Face Hub on their first run.

CI runs every test, NLP included, and checks coverage twice:

- The whole test run fails if total coverage drops below 75%.
- On the pull request, [Codecov](https://about.codecov.io/) checks that at least 90% of the lines you
  add or change are covered by tests, and comments with the uncovered ones.

CI runs the suite on every supported Python version. To do the same locally, use
[tox](https://tox.wiki) with the `tox-uv` plugin:

```
uv tool install tox --with tox-uv
make test-all
```

tox installs the newest dependency versions rather than the ones in `uv.lock`, and runs without the
NLP tests.

Finally, check that the package builds:

```
uv build                       # or: python -m build (needs the `build` package)
```

## Update the documentation and tutorials

If your change affects how users work with Shapash, update the documentation along with the code:

- Docstrings feed the API reference, so keep them in line with the code.
- The documentation pages are in `docs/` and are built with Sphinx.
- The tutorial notebooks are in `tutorial/`. If you change a feature a tutorial demonstrates, update
  and re-run that notebook. With uv, add the Jupyter kernel with `--group notebook` (for example
  `uv sync --group notebook`); a plain `uv sync` removes it.

To build the documentation locally:

```
uv sync --group doc
make docs
```

This builds the HTML pages in `docs/_build/html/` and opens them in your browser. Without `make`
(on Windows, for example), run the two steps yourself:

```
python scripts/prepare_docs.py
sphinx-build -W --keep-going docs docs/_build/html
```

`scripts/prepare_docs.py` copies the tutorial notebooks into `docs/tutorials/` and generates the demo
report; add `--skip-report` to reuse the report from a previous run.

The build downloads the Python, numpy, pandas, scikit-learn and plotly inventories to link their types.
Without network access, for example behind a proxy, set the `SHAPASH_DOCS_OFFLINE` environment variable to `1`
(`SHAPASH_DOCS_OFFLINE=1 make docs`, or `set SHAPASH_DOCS_OFFLINE=1` first on Windows) to skip them.

Like CI, which builds the documentation on every pull request, these commands fail on any Sphinx
warning (`--keep-going` still lists them all).

To check the external links, for example before a release, run `make -C docs linkcheck` after a
documentation build.

## Change dependencies

Dependencies are declared in `pyproject.toml`.

- With uv, after any change there, run `uv lock` and commit the updated `uv.lock` with it: CI fails
  when the lock file is out of date.
- CI also tests every dependency at the lowest version `pyproject.toml` allows. If that job fails
  because your change needs a newer release of a dependency, raise its lower bound.

# Submit your changes

## Commit and push

Use clear commit messages and group related changes into logical commits. Once the code quality
checks, the tests and the build pass, push your branch to your fork:

```
git add <file>
git commit -m "Fix ..."
git push -u origin feature/my-contribution
```

## Open a pull request

A pull request asks the Shapash team to review your changes and merge them into the `develop` branch
of the official repository.

At the top of your fork's page on GitHub, click **Compare & pull request**.

<img width="912" height="59" alt="shapash-compare-pr" src="https://github.com/user-attachments/assets/c7d1649d-64f0-4669-9243-627c2285b7e9" />

Set the pull request branches as follows:

- Base repository: MAIF/shapash
- Base branch: develop
- Head repository: your-github-username/shapash
- Head branch: your contribution branch

<img width="1237" height="181" alt="shapash-pr-branch" src="https://github.com/user-attachments/assets/5cee69b7-cec4-4c24-bbe6-3deb41bab925" />

The pull request description is pre-filled with a template. Complete it, link the issue it addresses
(for example `Fixes #123`), then click the green **Create pull request** button.

<img width="1242" height="722" alt="shapash-pr-description" src="https://github.com/user-attachments/assets/8977e2b9-e3dd-4c1c-b904-eb6817bdfcb0" />

## Review and merge

CI runs on your pull request: code quality checks, the tests on every supported Python version, the
tests with the lowest supported dependency versions, and the Codecov coverage check. If a job fails, open it from the
**Checks** tab of the pull request to see why, then push a fix to the same branch: the pull request
updates automatically.

A member of the Shapash team will review your pull request and may ask for changes; push them to the
same branch too. If `develop` moves on in the meantime and your branch needs updating, bring it up to
date with:

```
git fetch upstream
git merge upstream/develop
git push origin feature/my-contribution
```

Once the pull request is approved and CI passes, a maintainer merges it.

# Contributing using AI

AI tools are welcome, as an assistant, not a replacement for your own work. We adopt the rules of the
[unitaryHACK guide to ethical AI contributions](https://unitaryhack.dev/ai-guide/), summed up here:

- **You own what you submit.** Only submit code you understand and could explain to a maintainer in
  a live discussion.
- **Verify everything.** Run the tests locally and check that they pass and cover the edge cases. AI
  output that was never run, functions or libraries that don't exist, and "fixes" that delete or
  weaken a failing test lead to the pull request being closed.
- **Clean up.** Remove superfluous AI-generated comments and verbose explanations.
- **Disclose it.** Say in the pull request how you used AI, for example: "I used [tool] to draft
  [part], then reviewed and tested it myself." Presenting AI-generated work as entirely your own is
  not acceptable.
- **No mass submissions.** Don't use automation to open AI-generated issues or pull requests in bulk.

Good uses include drafting docstrings or test scaffolding that you then check, brainstorming the cause
of an error, and suggesting refactorings that keep the behaviour unchanged. Unverified AI output shifts
the debugging onto maintainers: be a contributor, not a burden.

Thank you for contributing to Shapash!
