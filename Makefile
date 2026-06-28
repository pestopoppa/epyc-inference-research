.PHONY: help setup lint test health docs docs-check analysis analysis-check security-check autopilot-gate

UV ?= uv
PYTHON_SMOKE := scripts/research/xmas_winner_table.py \
	scripts/research/xmas_function_axis_sweep.py \
	scripts/docs/generate_docs_index.py \
	scripts/analysis/generate_analysis_reports_index.py \
	scripts/security/audit_repository.py \
	scripts/autopilot/candidate_eval_gate.py
PYTEST_SMOKE := scripts/research/test_xmas_winner_table.py
PYTEST_SMOKE += scripts/docs/test_generate_docs_index.py
PYTEST_SMOKE += scripts/analysis/test_generate_analysis_reports_index.py
PYTEST_SMOKE += scripts/security/test_audit_repository.py
PYTEST_SMOKE += scripts/autopilot/test_candidate_eval_gate.py

help:
	@printf '%s\n' 'Targets: setup lint test health docs docs-check analysis analysis-check security-check autopilot-gate'

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

analysis:
	$(UV) run python scripts/analysis/generate_analysis_reports_index.py

analysis-check:
	$(UV) run python scripts/analysis/generate_analysis_reports_index.py --check

security-check:
	$(UV) run python scripts/security/audit_repository.py

autopilot-gate:
	$(UV) run python scripts/autopilot/candidate_eval_gate.py --execute
