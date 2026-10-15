.PHONY: install test lint demo demo-dry demo-gif clean

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

install:
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e '.[dev,all]'

test:
	$(PY) -m pytest -q

lint:
	$(VENV)/bin/ruff check src tests

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
