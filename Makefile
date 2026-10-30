.PHONY: help clean clean-build clean-pyc clean-test lint format typecheck workflows test test-all coverage coverage-report docs dist
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -not -path './.venv/*' -exec rm -fr {} +
	find . -name '*.egg' -not -path './.venv/*' -exec rm -f {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -not -path './.venv/*' -exec rm -f {} +
	find . -name '*.pyo' -not -path './.venv/*' -exec rm -f {} +
	find . -name '*~' -not -path './.venv/*' -exec rm -f {} +
	find . -name '__pycache__' -not -path './.venv/*' -exec rm -fr {} +

clean-test: ## remove test, coverage and linter cache artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -f coverage.xml
	rm -fr htmlcov/
	rm -fr .pytest_cache
	rm -fr .mypy_cache
	rm -fr .ruff_cache

lint: ## check style with ruff
	ruff check

format: ## check formatting with ruff
	ruff format --check

typecheck: ## check type annotations with mypy
	mypy shapash

workflows: ## audit the GitHub Actions workflows with zizmor
	zizmor .github

test: ## run tests quickly with the default Python
	pytest

test-all: ## run tests on every Python version with tox
	tox

coverage: ## run tests with coverage
	pytest --cov=shapash --cov-branch

coverage-report: ## run tests with coverage and generate xml report
	pytest --cov=shapash --cov-branch --cov-report=xml

docs: ## generate Sphinx HTML documentation, including API docs
	python scripts/prepare_docs.py
	$(MAKE) -C docs clean
	$(MAKE) -C docs html SPHINXOPTS="-W --keep-going"
	python -m webbrowser "file://$(CURDIR)/docs/_build/html/index.html"

dist: clean ## builds source and wheel package
	uv build
	ls -l dist
