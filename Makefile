SHELL := /bin/bash
PYTHON_BIN ?= python3.11
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
PRE_COMMIT := $(VENV)/bin/pre-commit
STREAMLIT := $(VENV)/bin/streamlit

.PHONY: help setup install-deps dev lint format typecheck test test-unit test-integration run download-models verify-deps cleanup-cache sync-cast smoke clean pre-commit hooks backup-voice-bank

help:
	@grep -E '^[a-zA-Z_-]+:.*?#' Makefile | awk 'BEGIN {FS = ":.*?#"} {printf "%-22s %s\n", $$1, $$2}' | sort

$(VENV)/bin/activate: pyproject.toml
	$(PYTHON_BIN) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e .[dev]
	$(PRE_COMMIT) install
	@touch $(VENV)/bin/activate

setup: $(VENV)/bin/activate ## Create virtual environment and install dependencies

install-deps: $(VENV)/bin/activate ## Alias for setup

dev: $(VENV)/bin/activate ## Run the Streamlit dev server
	$(STREAMLIT) run ui/streamlit_app/app.py

lint: $(VENV)/bin/activate ## Run Ruff and Black
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/black --check src tests

format: $(VENV)/bin/activate ## Auto-format the codebase
	$(VENV)/bin/ruff check --fix src tests
	$(VENV)/bin/black src tests

pre-commit: $(VENV)/bin/activate ## Run all pre-commit hooks across the repo
	$(PRE_COMMIT) run --all-files

hooks: $(VENV)/bin/activate ## Install pre-commit hooks
	$(PRE_COMMIT) install

typecheck: $(VENV)/bin/activate ## Run mypy
	$(VENV)/bin/mypy src

pytest-base := $(VENV)/bin/pytest -ra --strict-markers

test: $(VENV)/bin/activate ## Run entire test suite
	$(pytest-base)

test-unit: $(VENV)/bin/activate ## Run unit tests only
	$(pytest-base) -m "not integration"

test-integration: $(VENV)/bin/activate ## Run integration tests
	$(pytest-base) -m integration

run: $(VENV)/bin/activate ## Execute the processing CLI (usage: make run INPUT=path/to/file)
	@[ -n "$(INPUT)" ] || (echo "INPUT parameter required" >&2 && exit 1)
	$(PYTHON) -m show_scribe.cli process "$(INPUT)"

download-models: $(VENV)/bin/activate ## Download Whisper, Pyannote, and Resemblyzer models
	$(PYTHON) scripts/setup/download_models.py

verify-deps: $(VENV)/bin/activate ## Check required binaries and Python packages
	$(PYTHON) scripts/setup/verify_dependencies.py

cleanup-cache: $(VENV)/bin/activate ## Remove cache and temp directories
	$(PYTHON) scripts/maintenance/cleanup_cache.py

sync-cast: $(VENV)/bin/activate ## Regenerate cast_list.json files from show_config.json
	$(PYTHON) scripts/maintenance/sync_cast_configs.py $(SHOW)

smoke: $(VENV)/bin/activate ## Quick smoke test suite
	$(pytest-base) -m "not slow"

clean: ## Remove caches and build artifacts
	rm -rf $(VENV) \
		.pytest_cache \
		.mypy_cache \
		.ruff_cache \
		build \
		dist \
		*.egg-info

backup-voice-bank: $(VENV)/bin/activate ## Backup the voice bank database
	$(PYTHON) scripts/rebuild_voice_bank.py --backup
