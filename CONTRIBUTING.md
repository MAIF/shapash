# How to contribute to Shapash

This guide explains how to contribute to Shapash. If you found a bug, identified an improvement, or want to propose a new feature, follow the steps below.

- [How to open an issue](#how-to-open-an-issue)
- [Create your contribution and submit a pull request](#create-your-contribution-and-submit-a-pull-request)
    - [Fork the repository](#fork-the-repository)
    - [Clone your fork](#clone-your-fork)
    - [Keep your fork up to date](#keep-your-fork-up-to-date)
    - [Install dependencies and set up your environment](#install-dependencies-and-set-up-your-environment)
    - [Set up pre-commit hooks](#set-up-pre-commit-hooks)
    - [Start coding on a dedicated branch](#start-coding-on-a-dedicated-branch)
    - [Run tests and build checks](#run-tests-and-build-checks)
    - [Run code quality checks](#run-code-quality-checks)
    - [Commit and push your changes](#commit-and-push-your-changes)
    - [Create a pull request](#create-a-pull-request)
    - [Submit your pull request](#submit-your-pull-request)

# How to open an issue

Opening an issue starts a discussion to evaluate whether your bug report or feature request should be implemented in Shapash.

Before opening an issue:

- Check the Project tab to see whether the item already exists in the roadmap.
- Search existing open issues to avoid duplicates.
- Specify whether your request is a feature or a bug fix.

Every pull request must be linked to an issue.

After you open an issue, the Shapash team or community members will provide feedback and help decide whether a pull request is needed.

# Setup your environment

## Get the code
- Fork the MAIF/shapash repository to your personal GitHub account using the **Fork** button.
- Clone your fork into your workspace.
- Synchronnize your fork with the upstream repository before creating a new branch.

```
git remote add upstream https://github.com/MAIF/shapash.git
git pull upstream develop
```

## Install dependencies and set up your environment

Python `>=3.11, <3.15` is required.

After creating a virtual environment, run:
```
pip install -e ".[dev,test]"
```

Alternatively, we recommend using `uv`:
```
uv sync --extra dev --extra test
```

See `pyproject.toml` for other dependency groups.

## Set up pre-commit hooks

We use pre-commit hooks to catch issues automatically before each commit. After installing development dependencies, enable the hooks:

```
pre-commit install
```

# Create your contribution and submit a pull request

## Start coding on a dedicated branch

Create a personal branch for your contribution:

```
git checkout -b feature/my-contribution-branch
```

Recommended branch naming convention:

- **feature/your_branch_name** for a new feature
- **bugfix/your_branch_name** for a bug fix

## Run tests and build checks

Run pytest to ensure tests pass and to collect coverage:

```
pytest --cov=shapash
```

To test against multiple Python versions, use `tox`:

```
pipx install tox
pipx inject tox tox-uv
tox
```

`tox-uv` allows tox to use `uv` for interpreter and environment management. Missing Python versions are skipped automatically.

Make sure Shapash builds correctly:

```
python -m build

or

uv build
```

## Run code quality checks

Check linting and formatting with ruff:

```
ruff check
ruff format
```

Check type annotations with mypy:

```
mypy shapash
```

## Commit and push your changes

Use clear commit messages and group related changes logically.

Once all checks pass (code quality, tests, and build), push your changes to your fork:

```
git add <file>
git commit -m "Fix bug in ..."
git push origin feature/my-contribution-branch
```

Your branch is now available on your remote fork.

Next, create a pull request so the Shapash team can review and merge your changes.

## Create a pull request

A pull request asks the Shapash team to review your changes and merge them into the `develop` branch of the official repository.

At the top of your forked repository page, click **Compare & pull request**.

<img src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/assets/images/contributing/shapash-compare-pr.png" alt="pull request" />

Configure the pull request branches as follows:

- Base repository: MAIF/shapash
- Base branch: develop
- Head repository: your-github-username/shapash
- Head branch: your-contribution-branch

<img src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/assets/images/contributing/shapash-pr-branch.png" alt="select pull request branches" />

After selecting the correct branches, click the green **Create pull request** button.

<img src="https://raw.githubusercontent.com/MAIF/shapash/master/docs/assets/images/contributing/shapash-pr-description.png" alt="pull request description" />

The pull request description is pre-filled with a template. Please complete it with all relevant details about your contribution.

## Submit your pull request

Your pull request is now ready to be submitted. A member of the Shapash team will review it and contact you if changes are needed.

Thank you for contributing to Shapash.
