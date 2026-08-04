.PHONY: help setup lint test health docs docs-check analysis analysis-check security-check autopilot-gate

UV ?= uv
PYTHON_SMOKE := scripts/research/xmas_winner_table.py \
	scripts/research/xmas_function_axis_sweep.py \
	scripts/docs/generate_docs_index.py \
	scripts/analysis/generate_analysis_reports_index.py \
	scripts/security/audit_repository.py \
	scripts/autopilot/candidate_eval_gate.py \
	scripts/halo/closed_loop_observation_surface.py \
	scripts/halo/convert_tap_to_otel.py
CAPTURE_CONTRACT_SOURCES := scripts/benchmark/v7_quality_gate_runner.py \
	scripts/benchmark/capture_integrity_watchdog.py \
	scripts/benchmark/score_with_claude.py \
	scripts/benchmark/agentic_swe_harness.py \
	artifacts/architect-code-eval-20260724/convert_sr_to_patch.py
PYTHON_SMOKE += $(CAPTURE_CONTRACT_SOURCES)
PYTEST_SMOKE := scripts/research/test_xmas_winner_table.py
PYTEST_SMOKE += scripts/docs/test_generate_docs_index.py
PYTEST_SMOKE += scripts/analysis/test_generate_analysis_reports_index.py
PYTEST_SMOKE += scripts/security/test_audit_repository.py
PYTEST_SMOKE += scripts/autopilot/test_candidate_eval_gate.py
PYTEST_SMOKE += scripts/halo/test_closed_loop_observation_surface.py
PYTEST_SMOKE += scripts/benchmark/test_capture_contract_guard.py
PYTEST_SMOKE += scripts/benchmark/test_capture_integrity_watchdog.py
PYTEST_SMOKE += scripts/benchmark/test_agentic_swe_harness.py
PYTEST_SMOKE += scripts/kernel_rnd/test_kernel_store.py
PYTEST_SMOKE += scripts/kernel_rnd/test_c6_reward_integrity.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/test_schemas.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/test_journal.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/test_storage.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/test_integration.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/resource/test_device_claim.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/resource/test_preflight.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/resource/test_claim_witness.py
# AK3 — the trusted tiered evaluator (P-AK-SEARCH-1). The two cross-module
# suites come first: they are the ones that fail when two modules disagree.
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/evaluator/test_conformance.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/evaluator/test_integration.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/evaluator/test_api.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/evaluator/test_controls.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/evaluator/test_controls_redteam.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/evaluator/test_correctness.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/evaluator/test_integrity.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/evaluator/test_recipes.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/evaluator/test_statistics.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/evaluator/test_surface.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/evaluator/test_devices.py
# AK4 — the planner/critic/controller plane. Same ordering rule as AK3: the two
# cross-module suites come first, because they are the ones that fail when two
# modules that are each green disagree at the seam between them.
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/controller/test_ak4_conformance.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/controller/test_loop_integration.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/controller/test_state_machine.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/controller/test_guards.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/controller/test_context.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/controller/test_hypotheses.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/controller/test_planner_critic.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/controller/test_selection.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/controller/test_composition.py
# AK5/AK6 — the release plane (plan, readiness, T3 gate, packager) and AK8/AK9 —
# the backend adapters. Same ordering rule again, and it earns itself here more
# than anywhere else: `test_release_integration.py` is the only suite that can see
# a SEAM, and every defect it caught passed both of the modules' own suites first.
# It also carries the cardinal-rule audit — no module in either plane can write a
# production branch, move a stable kernel symlink, write an era-registry row or
# apply an AutoPilot baseline — so it is the one that must not be skipped.
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/release/test_release_integration.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/release/test_plan.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/release/test_readiness.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/release/test_t3.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/release/test_t3_protocol_binding_redteam.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/release/test_t3_waiver_authority_redteam.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/release/test_packager.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/adapters/test_serving_runtime.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/adapters/test_whisper_stt.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/adapters/test_qwentts_tts.py
# AK6 — the /kernel operator surface. It belongs in the gate for the same reason
# the surface exists: the guarantees it holds (a dead loop cannot read as fresh, a
# blocking panel cannot read clear while the sections beside it read blocked, the
# one writer cannot reach a checkout or a production tree) are only guarantees
# while something runs them.
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/surface/test_dashboard_contract.py
# The SEAM: the only suite where the real producer writes a file the real hub
# reads. Both halves were green while disagreeing about a field the producer
# owns, so this one runs first among the surface suites in spirit and must never
# be dropped — a seam nobody checks is two modules drifting in private.
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/surface/test_surface_seam.py
# AK3 execution layer — the modules that actually build a candidate and run a
# measurement. `test_execution_chain.py` comes FIRST for the reason every other
# seam suite above it does: it is the only suite that composes all five executors
# with the evaluator that reads them, and the two BuildProvenance classes it
# reconciles were each green in their own module while naming different records.
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/execution/test_execution_chain.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/execution/test_cpu_region_claim.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/execution/test_cpu_region_claim_redteam.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/execution/test_worktree.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/execution/test_t0_provider.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/execution/test_microbench.py
PYTEST_SMOKE += scripts/kernel_rnd/autokernel/execution/test_control_runner.py

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
