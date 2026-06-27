.PHONY: help setup lint test

UV ?= uv
PYTHON_SMOKE := scripts/research/xmas_winner_table.py \
	scripts/research/xmas_function_axis_sweep.py
PYTEST_SMOKE := scripts/research/test_xmas_winner_table.py

help:
	@printf '%s\n' 'Targets: setup lint test'

setup:
	scripts/setup.sh

lint:
	$(UV) run --with ruff ruff check $(PYTHON_SMOKE) $(PYTEST_SMOKE)

test:
	$(UV) run --with pytest --with pyyaml pytest -q $(PYTEST_SMOKE)
