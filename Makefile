.PHONY: install test lint demo demo-dry demo-gif clean

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# kuzu==0.11.3 (the `kuzu`/`all` extra) has no cp314 wheels and its sdist
# doesn't build, so a bare `python3` that happens to resolve to 3.14 (as
# Homebrew's does once it's the latest) makes `make install` fail. Pick
# the first 3.13/3.12/3.11 interpreter found on PATH instead; override
# with `make install PYTHON=python3.12` to force a specific one.
PYTHON ?= $(shell for p in python3.13 python3.12 python3.11; do command -v $$p >/dev/null 2>&1 && echo $$p && break; done)

install:
	@if [ -z "$(PYTHON)" ]; then \
		echo "error: no python3.11/3.12/3.13 found on PATH (kuzu==0.11.3 has no cp314 wheels, so 3.14 won't work)."; \
		echo "       install one (e.g. 'brew install python@3.12') or pass it explicitly: make install PYTHON=python3.12"; \
		exit 1; \
	fi
	@echo "using $(PYTHON) ($(shell $(PYTHON) --version 2>&1))"
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e '.[dev,all]'

test:
	$(PY) -m pytest -q

lint:
	$(VENV)/bin/ruff check .

# Real demo: regenerates docs/assets/demo.cast + demo.svg using the real
# fastembed embedder. First run downloads the ~70MB embedding model
# (BAAI/bge-small-en-v1.5) — requires network the first time only.
demo:
	$(PY) scripts/make_demo.py

# Fast, offline layout dry-run: deterministic fake embedder, watermarked
# SVG, writes outside docs/assets. Use this to iterate on timing/layout
# without the model download.
demo-dry:
	$(PY) scripts/make_demo.py --fake-embedder

# Alternative GIF path via VHS (https://github.com/charmbracelet/vhs).
# Requires the `vhs` CLI and memstrand installed in your shell.
demo-gif:
	vhs demo.tape

clean:
	rm -rf $(VENV) dist build *.egg-info
	find . -name '__pycache__' -not -path './.venv/*' -exec rm -rf {} +
