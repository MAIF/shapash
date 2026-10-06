# Repository Agent Instructions

## Collaboration
- The user usually expects implementation when requesting a code change. Start from the named file or nearest owning code path, inspect relevant tests, state a concise local hypothesis, then make a focused change.
- Before editing, provide a short update describing the investigation and intended change. After editing, run the narrowest relevant validation before widening the scope.
- Prefer concise communication in French, matching the user's language. Keep technical deliverables, issue text, and PR text in the language the user requests.

## Git and Scope
- Inspect the current branch and worktree before committing, switching branches, or pushing.
- Preserve existing staged and unstaged changes. Do not include unrelated files in a commit, and do not discard user or formatter changes without checking their origin.
- Do not change pre-commit configuration unless explicitly requested. If a hook fails, distinguish a code failure from a missing local tool and report it clearly.
- Do not commit, push, or create GitHub issues/PRs unless explicitly asked. Before creating an issue or PR, check for duplicates and confirm the intended repository and base branch.

## Issue and PR Writing
- Always write GitHub issue titles and descriptions, and pull request titles and descriptions, in English.
- Keep issue and PR text concise and grounded in the current code behavior.
- Separate existing problems from expected behavior or proposed changes; do not present implementation details as if they were the problem.
- When the user asks for only problems, describe only current symptoms and impact. Prefer present tense and avoid describing completed work in the past tense.

## Smart Report Context
- Report runtimes should depend on the core `Explainer`, not the `SmartExplainer` wrapper, when the underlying explainer is sufficient.
- Pass the report palette as `colors_dict`; do not substitute the plotter's derived `_style_dict` for it.
- For report changes, inspect the built-in YAML templates, custom-block tutorial, navigation behavior, and report-generation tests when relevant.

## Repository and Code Style
- Shapash is a Python library for model explainability. Core package code lives in `shapash/`; unit tests live in `tests/unit_tests/`; tutorials and report examples live in `tutorial/` and `scripts/`.
- Supported Python versions are `>=3.11,<3.15`. Follow the project's existing snake_case naming, type annotations, and NumPy-style docstring conventions.
- Write all source-code comments and docstrings in English. Keep comments concise and limited to context that is not clear from the code.
- Ruff is the linter and formatter, configured for a 120-character line length. Use `ruff check` and `ruff format --check` for validation.
- Mypy checks the `shapash` package; use `mypy shapash` after changes to typed package code.
- Tests use pytest, with both pytest-style tests and `unittest.TestCase` suites. Prefer focused commands such as `pytest tests/unit_tests/report/test_report_generation.py -k <case>` before broader runs.
- Common project checks are available through `make test`, `make lint`, `make format`, and `make typecheck`; pre-commit runs the repository's configured hooks.
- Report rendering uses optional Panel dependencies (`shapash[report]`). Preserve the optional-dependency behavior when changing report imports or runtime setup.
