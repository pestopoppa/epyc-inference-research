.PHONY: help setup lint test health docs docs-check

UV ?= uv
PYTHON_SMOKE := scripts/research/xmas_winner_table.py \
	scripts/research/xmas_function_axis_sweep.py \
	scripts/docs/generate_docs_index.py
PYTEST_SMOKE := scripts/research/test_xmas_winner_table.py
PYTEST_SMOKE += scripts/docs/test_generate_docs_index.py

help:
	@printf '%s\n' 'Targets: setup lint test health docs docs-check'

setup:
	scripts/setup.sh

lint:
	$(UV) run --with ruff ruff check $(PYTHON_SMOKE) $(PYTEST_SMOKE)

test:
	$(UV) run --with pytest --with pyyaml pytest -q $(PYTEST_SMOKE)

health:
	scripts/session/health_check.sh

docs:
	$(UV) run python scripts/docs/generate_docs_index.py

docs-check:
	$(UV) run python scripts/docs/generate_docs_index.py --check
