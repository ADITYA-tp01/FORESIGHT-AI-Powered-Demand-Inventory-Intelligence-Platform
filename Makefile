.PHONY: install pipeline test lint app api env

PYTHON ?= python
PIP ?= $(PYTHON) -m pip

install:
	$(PIP) install -e ".[dev]"

env:
	$(PYTHON) -m foresight.pipeline --phase 0

pipeline:
	$(PYTHON) -m foresight.pipeline

test:
	$(PYTHON) -m pytest tests -q

lint:
	$(PYTHON) -m ruff check src tests api app

app:
	$(PYTHON) -m streamlit run app/streamlit_app.py

api:
	$(PYTHON) -m uvicorn api.main:app --reload --port 8000
