"""Original serial selection/accounting; child hardware observations are fixtures."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from . import cpu_screen, scheduling, serial_roster, serial_run as sr, serial_scheduling as ss
from .test_serial_scheduling import manifest
from .test_serial_roster import _inputs, _build
from .test_serial_run import CHILD
from .test_shared_history import _record


@pytest.mark.parametrize("cpu_list,reason", [(None, "inherits owned affinity"),
                                            ("0-47,96-143", "SMT sibling geometry")])
def test_unsupported_reduced_geometry_retains_original_full_target(tmp_path, cpu_list, reason):
    _resolved, _owners, argv = _inputs(tmp_path, backends=("cpu",))
    targets, _skipped, _cpus = _build(argv)
    target = targets[0]
    from .resolved_recipe import CanonicalResolvedRecipe, resolve_canonical_launch
    path = Path(sr.option(target, "--cpu-serving-launch"))
    original = CanonicalResolvedRecipe.from_dict(json.loads(path.read_text()))
    template = replace(original.template, cpu_list=cpu_list)
    full = resolve_canonical_launch(template, build_dir=original.build_dir,
        command_argv=original.command_argv,
        topology_prefix=() if cpu_list is None else ("taskset", "-c", cpu_list),
        launch_environment=dict(original.launch_env),
        artifact_identities={"model": original.model.to_dict(), "drafter": None,
            "executable": original.executable.to_dict(), "dsos": [x.to_dict() for x in original.dsos]},
        backend="cpu", environment_policy=original.environment_policy, port=original.port,
        runtime_binary_dir=original.runtime_binary_dir, runtime_ld_paths=original.runtime_ld_paths,
        provenance=dict(original.provenance))
    path.write_text(json.dumps(full.to_dict()))
    _record(Path(sr.option(target, "--store")), statement="remove redundant arithmetic",
            research_scope={"model": {"path": full.model.path}, "backend": "cpu"})
    preview = cpu_screen.preview_batch(target, None)
    assert preview["scope"] == "full" and reason in preview["reason"]
    assert CanonicalResolvedRecipe.from_dict(json.loads(path.read_text())) == full


def _quarter():
    return {"scope": "quarter", "candidate": None, "cpu_list": "0-23", "region_fraction": .25,
            "mechanism_family": "local_work", "basis": "fixture original mechanism"}


def test_original_scheduler_uses_actual_scope_cost_without_assuming_shorter_duration():
    source = manifest()
    before = source.to_dict()
    state = scheduling.initial_state(source.config, source.scheduler_id)
    next_state, selection, index = ss.select_target(source, state, ("cpu",), now=1,
        stage_number=0, scope_previews={"cpu": _quarter()})
    assert index == 0 and selection.proposal.estimated_claims.physical_region_fraction == .25
    assert selection.proposal.estimated_duration_seconds == source.proposals["cpu"].estimated_duration_seconds
    assert not selection.proposal.full_region
    assert next_state.issued_selection_digests == (selection.digest,)
    assert source.to_dict() == before
    _, full, _ = ss.select_target(source, state, ("cpu",), now=1, stage_number=0,
                                 scope_previews={"cpu": {"scope": "full", "candidate": None}})
    assert selection.proposal.proposal_id != full.proposal.proposal_id
    with pytest.raises(cpu_screen.ScreenRefused, match="GPU"):
        cpu_screen.scoped_proposal(source.proposals["gpu"], _quarter())


def test_unaffordable_full_confirmation_remains_pending_but_other_target_can_select():
    source = manifest()
    state = scheduling.initial_state(source.config, source.scheduler_id)
    pending = {"scope": "full_confirmation", "candidate": {"path": "/original", "sha256": "a" * 64}}
    original = source.proposals["cpu"]
    state = replace(state, campaign_charged_seconds=source.config.campaign_charged_seconds_cap - 0.01)
    assert "remaining campaign" in cpu_screen.confirmation_debt(source.config, state, original, pending)
    assert cpu_screen.confirmation_debt(source.config, state, original, {"scope": "full"}) is None
    next_state, selection, _ = ss.select_target(source, state, ("gpu",), now=1, stage_number=0,
        scope_previews={"gpu": {"scope": "full", "candidate": None}})
    assert selection.proposal.backend == "gpu"
    assert next_state.campaign_charged_seconds == state.campaign_charged_seconds
    capacity = replace(state.capacity, physical_region_fraction=.01)
    tiny_capacity = replace(state, capacity=capacity, capacity_digest=capacity.digest)
    assert "capacity" in cpu_screen.confirmation_debt(source.config, tiny_capacity, original, pending)


def test_actual_owned_roster_child_scope_matches_original_selection_and_accounting(tmp_path, monkeypatch):
    _resolved, _owners, argv = _inputs(tmp_path, backends=("cpu", "gpu"))
    targets, _skipped, _cpus = _build(argv)
    cpu = next(row for row in targets if sr.option(row, "--cpu-serving-launch"))
    from .resolved_recipe import CanonicalResolvedRecipe
    full = CanonicalResolvedRecipe.from_dict(json.loads(Path(sr.option(cpu, "--cpu-serving-launch")).read_text()))
    _record(Path(sr.option(cpu, "--store")), statement="remove redundant arithmetic in vec_dot",
            research_scope={"model": {"path": full.model.path}, "backend": "cpu"})
    # Existing actual tiny-child harness produces explicit synthetic held evidence.
    # It does not claim a model, build, source improvement or placement observation.
    child_source = CHILD.replace('selected = legacy_targets',
        'gpu = sr.option(argv, "--gpu-serving-launch") is not None\n'
        'experimental = cpu or (gpu and sr.option(argv, "--experimental-branch") is not None)\n'
        'selected = legacy_targets')
    child_source = child_source.replace('if cpu else "legacy_gpu_screen"',
        'if cpu else "gpu_serving_selected_workload" if gpu else "legacy_gpu_screen"')
    child_source = child_source.replace('if cpu else sr.champion.CANONICAL_BRANCH',
        'if experimental else sr.champion.CANONICAL_BRANCH')
    child_source = child_source.replace('None if cpu else anchor', 'None if experimental else anchor')
    child_source = child_source.replace('None if cpu else "a" * 40', 'None if experimental else "a" * 40')
    child_source = child_source.replace('row = sr.continuation(', '''scope = sr.option(argv, "--cpu-screen-scope")
screen = None
if scope:
    from scripts.kernel_rnd.autokernel.loop import cpu_screen, resolved_recipe
    full = resolved_recipe.CanonicalResolvedRecipe.from_dict(json.loads(
        Path(sr.option(argv, "--cpu-serving-launch")).read_text()))
    measured = cpu_screen.prepare_launch(full, scope, resolved.resources.cpu_logical)["launch"]
    screen = {"scope": scope, "candidate": None, "full_execution_digest": full.execution_digest,
              "measured_execution_digest": measured.execution_digest}
row = sr.continuation(cpu_screen=screen,''')
    child = tmp_path / "scope_child.py"
    child.write_text(child_source)
    monkeypatch.setattr(sr, "_child_command", lambda args: [sr.sys.executable, str(child), *args])
    monkeypatch.setenv("PYTHONPATH", str(Path(sr.__file__).resolve().parents[4]))
    assert sr.main(argv) == 0
    router = tmp_path / "router"
    state = json.loads((router / "serial-state.json").read_text())
    assert state["next_batch"] == 4 and state["active"] is None
    accounting = scheduling.SchedulerState.from_dict(state["scheduler_state"])
    assert accounting.campaign_attempts == 4
    assert not accounting.issued_selection_digests and not accounting.successor_fences
    receipts, covered_gpu = [], []
    for directory in sorted((router / "batches").iterdir()):
        row, _sha = sr.load_completed(directory / "loop-continuation.json")
        selected = scheduling.Selection.from_dict(json.loads((directory / "scheduler-selection.json").read_text()))
        assert row["held_claim_evidence"]["selection_digest"] == selected.digest
        if sr.option(row["input_argv"], "--cpu-serving-launch"):
            assert row["cpu_screen"]["scope"] == "quarter"
            assert selected.proposal.estimated_claims.physical_region_fraction == .25
            original = json.loads((directory / "cpu-screen-selection.json").read_text())
            assert original["selected"]["mechanism_family"] == "local_work"
            assert original["source"]["binding"] == row["binding"]
            receipts.append(row)
        else:
            assert "cpu_screen" not in row and selected.proposal.backend == "gpu"
            assert selected.slot_kind == "coverage" and selected.proposal.production_frontier
            covered_gpu.append(selected.proposal.frontier_id)
    # Original resource-time/seed policy need not give equal ITERATION counts to
    # a quarter CPU claim and a GPU claim in this tiny four-attempt budget.
    assert receipts
    assert covered_gpu and set(covered_gpu) <= set(accounting.frozen_frontier)
    # Restart does not replay any completed child or reread its changed hint.
    before = (router / "seen.jsonl").read_bytes()
    assert sr.main(argv) == 0
    assert (router / "seen.jsonl").read_bytes() == before
