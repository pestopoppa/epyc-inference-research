"""Declared runtime arms, runtime-surface epoch input and adoption receipt (fakes only).

No server, model or hardware: canonical launches are fixture identities, the
runtime-selection state is written the way `RuntimeAdmission._checkpoint`
writes it, and strict admission itself is exercised by test_runtime_admission.
"""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from . import accumulate, epoch_aliases, measurement_capture as mc, resolved_recipe as rr
from . import run, runtime_arms as ra, serial_run, serving
from .test_resolved_recipe import _artifacts, _policy

KEYS = ("GGML_IQK", "GGML_IQK_Q8_0", "GGML_REPACK_THREADS", "OMP_PLACES")


def _launch(build: Path, env=None, *, port=18641):
    recipe = serving.Recipe(name="ds41-fixture", model="/models/ds41-fixture.gguf", device="none",
                            ngl=0, threads=48, cpu_list="0-95", np=1, ctx=8192, n_predict=256,
                            temperature=0.0, top_k=1)
    command = recipe.server_argv(build, port)[3:] + ["--no-mmap", "--no-webui"]
    prefix = ["taskset", "-c", "0-95", "numactl", "--interleave=all"]
    template = rr.canonical_recipe_projection(name="ds41-fixture", command_argv=command,
        topology_prefix=prefix, n_predict=256, temperature=0.0, top_k=1)
    launch_env = {"LD_LIBRARY_PATH": str(build / "bin"), "GGML_IQK": "1", "OMP_PLACES": "cores",
                  **(env or {})}
    return rr.resolve_canonical_launch(template, build_dir=build, command_argv=command,
        topology_prefix=prefix, launch_environment=launch_env,
        artifact_identities=_artifacts(template, build=build), backend="cpu",
        environment_policy=_policy(*KEYS), port=port,
        runtime_binary_dir=str(build / "bin"), runtime_ld_paths=(str(build / "bin"),),
        provenance={"export_sha256": "a" * 64, "instance_mode": "full",
                    "source:fixture": "b" * 64})


def _declaration(*arms, policy="bit_exact_only", campaign="ak-ds41"):
    return ra.RuntimeArmDeclaration.from_dict({
        "schema": ra.SCHEMA, "campaign_id": campaign, "numerics_policy": policy,
        "arms": [dict(arm) for arm in arms]})


A2 = {"arm_id": "omp-places-48-sib", "kind": "env",
      "candidate": {"key": "OMP_PLACES", "value": "{2}:47:2,{1}"}, "numerics": "bit_exact",
      "rationale": "48 places for 48 threads: libgomp spins at barriers"}
A1 = {"arm_id": "omp-places-48-even", "kind": "env",
      "candidate": {"key": "OMP_PLACES", "value": "{0}:48:2"}, "numerics": "bit_exact",
      "rationale": "48 even places"}
A3 = {"arm_id": "repack-threads-48", "kind": "env",
      "candidate": {"key": "GGML_REPACK_THREADS", "value": "48"}, "numerics": "bit_exact",
      "rationale": "cap the load-time repack team at the serving thread count"}
IQK = {"arm_id": "iqk-q8-0", "kind": "env",
       "candidate": {"key": "GGML_IQK_Q8_0", "value": "1"}, "numerics": "not_bit_exact",
       "rationale": "iqk dense Q8_0 route; activations quantize to Q8_2_X4"}


# ------------------------------------------------------------------ declaration

def test_declaration_roundtrip_and_refusals():
    declaration = _declaration(A2, A1, A3)
    assert ra.RuntimeArmDeclaration.from_dict(declaration.to_dict()) == declaration
    assert declaration.env_keys() == {"OMP_PLACES", "GGML_REPACK_THREADS"}
    assert declaration.digest() == ra.RuntimeArmDeclaration.from_dict(
        json.loads(json.dumps(declaration.to_dict()))).digest()
    with pytest.raises(ra.ArmDeclarationRefused, match="bit_exact_only refuses"):
        _declaration(A2, IQK)
    assert _declaration(A2, IQK, policy="evaluator_coherence_gate").arms[1].numerics == \
        "not_bit_exact"
    with pytest.raises(ra.ArmDeclarationRefused, match="repeat"):
        _declaration(A2, dict(A1, arm_id=A2["arm_id"]))
    with pytest.raises(ra.ArmDeclarationRefused, match="same treatment"):
        _declaration(A2, dict(A2, arm_id="copy"))
    with pytest.raises(ra.ArmDeclarationRefused, match="installed runtime field"):
        _declaration(dict(A2, kind="batch"))
    with pytest.raises(ra.ArmDeclarationRefused, match="env candidate"):
        _declaration(dict(A2, candidate={"key": "OMP_PLACES", "value": 48}))
    with pytest.raises(ra.ArmDeclarationRefused, match="threads must be"):
        _declaration(dict(A2, kind="threads", candidate=0))
    with pytest.raises(ra.ArmDeclarationRefused, match="exactly"):
        ra.RuntimeArmDeclaration.from_dict({**declaration.to_dict(), "extra": 1})


def test_preflight_refuses_wrong_campaign_uninstalled_key_and_budget():
    declaration = _declaration(A2, A3)
    declaration.preflight(campaign_id="ak-ds41",
                          runtime_env_keys={"OMP_PLACES", "GGML_REPACK_THREADS"}, max_candidates=10)
    with pytest.raises(ra.ArmDeclarationRefused, match="belong to"):
        declaration.preflight(campaign_id="other", runtime_env_keys={"OMP_PLACES",
                              "GGML_REPACK_THREADS"}, max_candidates=10)
    with pytest.raises(ra.ArmDeclarationRefused, match="GGML_REPACK_THREADS"):
        declaration.preflight(campaign_id="ak-ds41", runtime_env_keys={"OMP_PLACES"},
                              max_candidates=10)
    with pytest.raises(ra.ArmDeclarationRefused, match="max_candidates"):
        declaration.preflight(campaign_id="ak-ds41", runtime_env_keys={"OMP_PLACES",
                              "GGML_REPACK_THREADS"}, max_candidates=1)


def test_backend_allowlist_admits_load_thread_cap_only_through_policy(tmp_path):
    launch = _launch(tmp_path / "b")
    keys = run._runtime_env_keys(launch, launch)
    assert keys == {"GGML_IQK", "GGML_IQK_Q8_0", "GGML_REPACK_THREADS", "OMP_PLACES"}
    # A policy that does not list the key leaves it inert.
    narrow = SimpleNamespace(environment_policy=SimpleNamespace(measurement_keys=("OMP_PLACES",)))
    assert run._runtime_env_keys(narrow, narrow) == {"OMP_PLACES"}
    assert run._runtime_env_keys(narrow, None) == {"OMP_PLACES"}
    assert run._runtime_env_keys(None, None) == set()
    assert "GGML_REPACK_THREADS" not in run.GPU_RUNTIME_ENV_KEYS


# ------------------------------------------------------------------ surface

def test_runtime_surface_ignores_build_identity_but_not_runtime_fields(tmp_path):
    first, rebuilt = _launch(tmp_path / "gen-1"), _launch(tmp_path / "gen-2")
    assert first.to_dict()["build_dir"] != rebuilt.to_dict()["build_dir"]
    assert ra.surface_digest(first) == ra.surface_digest(rebuilt)
    assert ra.surface_digest(first) == ra.surface_digest(first.to_dict())
    placed = _launch(tmp_path / "gen-1", {"OMP_PLACES": "{2}:47:2,{1}"})
    assert ra.surface_digest(placed) != ra.surface_digest(first)
    assert placed.execution_digest != first.execution_digest


def test_epoch_input_moves_only_with_a_carried_selection(tmp_path):
    base = dict(cpu_execution_digest="e" * 64, frozen_prompt_digest="f" * 64)
    historical = epoch_aliases.launch_epoch_inputs(**base)
    assert "runtime_recipe_surface_digest" not in historical
    assert epoch_aliases.launch_epoch_inputs(**base, runtime_recipe_surface_digest=None) == historical
    selected = epoch_aliases.launch_epoch_inputs(**base, runtime_recipe_surface_digest="1" * 64)
    other = epoch_aliases.launch_epoch_inputs(**base, runtime_recipe_surface_digest="2" * 64)
    epoch = lambda inputs: run.archive.epoch_for(anchor_commit="c" * 40,
        build_recipe=run.build_recipe.NATIVE_CPU_RECIPE.to_dict(), host_state=inputs)
    assert len({epoch(historical), epoch(selected), epoch(other)}) == 3
    # The measurement epoch (OP-60) carries it too: it is a measured input.
    assert run.measurement_epoch_inputs(selected)["runtime_recipe_surface_digest"] == "1" * 64


def test_selection_current_recipe_reads_retained_selection_before_claim(tmp_path):
    adopted = _launch(tmp_path / "gen-1", {"OMP_PLACES": "{2}:47:2,{1}"})
    (tmp_path / "store").mkdir()
    store = mc.ArtifactStore(tmp_path / "store" / "runtime-preparation")
    try:
        reference = store.write("direct-runtime-selection", {
            "schema": "epyc.autokernel.direct_runtime_selection.v1",
            "owner": {"fixture": True}, "admission": {"fixture": True},
            "current_recipe": adopted.to_dict(), "current_source_commit": "c" * 40}).to_dict()
        wrong = store.write("direct-runtime-default", {"schema": "other"}).to_dict()
    finally:
        store.close()
    recipe = ra.selection_current_recipe(tmp_path / "store", reference)
    assert ra.surface_digest(recipe) == ra.surface_digest(adopted)
    with pytest.raises(ra.ArmDeclarationRefused, match="not a runtime selection"):
        ra.selection_current_recipe(tmp_path / "store", wrong)
    with pytest.raises(ra.ArmDeclarationRefused, match="no runtime store"):
        ra.selection_current_recipe(tmp_path / "elsewhere", reference)
    with pytest.raises(ra.ArmDeclarationRefused, match="shape"):
        ra.selection_current_recipe(tmp_path / "store", {"locator": "x"})


# ------------------------------------------------------------------ planner

class _Ordinary:
    def __init__(self):
        self.calls = 0

    def propose(self, context):
        self.calls += 1
        return "planner-hypothesis"

    def author(self, hypothesis, context):
        return ("authored",)


def _context(anchor, *, observation_only=False):
    return {"runtime_anchor": anchor.to_dict(), "runtime_observation_only": observation_only,
            "runtime_env_keys": sorted(KEYS)}


def _write_state(root: Path, name: str, attempts):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"runtime-selection-{name}.json").write_text(json.dumps(
        {"schema": "epyc.autokernel.direct_runtime_admission.v1", "scope": name,
         "attempts": attempts, "selected": None}))


def test_declared_arms_are_served_first_then_settled_then_planner(tmp_path):
    anchor = _launch(tmp_path / "gen-1")
    root = tmp_path / "runtime-preparation"
    events = []
    ordinary = _Ordinary()
    planner = ra.DeclaredArmPlanner(ordinary, _declaration(A2, A1), store_root=lambda: root,
                                    on_event=events.append)
    first = planner.propose(_context(anchor))
    assert first.mechanism_id == "runtime-arm-omp-places-48-sib"
    assert first.runtime_pair.dimension.candidate == {"key": "OMP_PLACES", "value": "{2}:47:2,{1}"}
    assert first.runtime_pair.dimension.anchor == {"key": "OMP_PLACES", "value": "cores"}
    assert first.runtime_pair.anchor.to_dict() == anchor.to_dict()
    assert ordinary.calls == 0 and events[-1]["status"] == "served"
    # compare() appends its attempt before any launch: a pending attempt is a
    # continuation (budget-interrupted calibration), served again, not counted.
    _write_state(root, "s1", [{"pair": first.runtime_pair.to_dict(), "candidate_id": "akc-1",
                               "result": None}])
    again = planner.propose(_context(anchor))
    assert again.mechanism_id == first.mechanism_id and events[-1]["continuation"] is True
    assert ra.Ledger(root).count(ra.surface_digest(anchor), A2["arm_id"]) == 1
    # A finished attempt settles the arm for this surface; the next arm follows.
    _write_state(root, "s1", [{"pair": first.runtime_pair.to_dict(), "candidate_id": "akc-1",
                               "result": {"locator": "x.json", "sha256": "0" * 64,
                                          "verified": True}}])
    second = planner.propose(_context(anchor))
    assert second.mechanism_id == "runtime-arm-omp-places-48-even"
    _write_state(root, "s2", [{"pair": second.runtime_pair.to_dict(), "candidate_id": "akc-2",
                               "result": {"locator": "y.json", "sha256": "1" * 64,
                                          "verified": True}}])
    assert planner.propose(_context(anchor)) == "planner-hypothesis"
    assert ordinary.calls == 1
    assert ra.arm_state(root, planner.declaration, anchor) == {
        A2["arm_id"]: "settled", A1["arm_id"]: "settled"}
    # A source keep rebuilds the anchor: same runtime surface, arms stay settled.
    rebuilt = _launch(tmp_path / "gen-2")
    assert ra.arm_state(root, planner.declaration, rebuilt) == {
        A2["arm_id"]: "settled", A1["arm_id"]: "settled"}
    assert planner.author("h", {}) == ("authored",)


def test_after_adoption_remaining_arms_reopen_against_the_champion_recipe(tmp_path):
    anchor = _launch(tmp_path / "gen-1")
    root = tmp_path / "runtime-preparation"
    planner = ra.DeclaredArmPlanner(_Ordinary(), _declaration(A2, A1), store_root=lambda: root)
    first = planner.propose(_context(anchor))
    second_old = ra.DeclaredArmPlanner(_Ordinary(), _declaration(A1), store_root=lambda: root
                                       ).propose(_context(anchor))
    done = {"locator": "x.json", "sha256": "0" * 64, "verified": True}
    _write_state(root, "s1", [{"pair": first.runtime_pair.to_dict(), "result": done},
                              {"pair": second_old.runtime_pair.to_dict(), "result": done}])
    # A2 was adopted: the current recipe now carries it.
    adopted = first.runtime_pair.candidate
    state = ra.arm_state(root, planner.declaration, adopted)
    assert state == {A2["arm_id"]: "open", A1["arm_id"]: "open"}
    challenger = planner.propose(_context(adopted))
    # A2 IS the current value (exact no-op) and is skipped; A1 challenges the champion.
    assert challenger.mechanism_id == "runtime-arm-omp-places-48-even"
    assert challenger.runtime_pair.dimension.anchor == {"key": "OMP_PLACES",
                                                        "value": "{2}:47:2,{1}"}


def test_arm_refused_before_any_attempt_is_bounded_durably(tmp_path):
    anchor = _launch(tmp_path / "gen-1")
    root = tmp_path / "runtime-preparation"
    served = []
    for _batch in range(ra.MAX_UNATTEMPTED_SERVES + 2):
        # A new process per batch (one-iteration batches): only the ledger persists.
        planner = ra.DeclaredArmPlanner(_Ordinary(), _declaration(A3), store_root=lambda: root)
        served.append(planner.propose(_context(anchor)))
    assert [getattr(row, "mechanism_id", row) for row in served] == (
        ["runtime-arm-repack-threads-48"] * ra.MAX_UNATTEMPTED_SERVES
        + ["planner-hypothesis"] * 2)


def test_planner_defers_when_runtime_is_not_strict_or_arm_uninstalled(tmp_path):
    anchor = _launch(tmp_path / "gen-1")
    root = tmp_path / "runtime-preparation"
    ordinary = _Ordinary()
    planner = ra.DeclaredArmPlanner(ordinary, _declaration(A2), store_root=lambda: root)
    assert planner.propose(_context(anchor, observation_only=True)) == "planner-hypothesis"
    assert planner.propose({"runtime_observation_only": False}) == "planner-hypothesis"
    closed = ra.DeclaredArmPlanner(ordinary, _declaration(A2), store_root=lambda: None)
    assert closed.propose(_context(anchor)) == "planner-hypothesis"
    events = []
    uninstalled = ra.DeclaredArmPlanner(ordinary, _declaration(A2), store_root=lambda: root,
                                        on_event=events.append)
    context = {**_context(anchor), "runtime_env_keys": ["GGML_IQK"]}
    assert uninstalled.propose(context) == "planner-hypothesis"
    assert events[-1]["status"] == "skipped" and "installed runtime keys" in events[-1]["reason"]
    assert not (root / ra.LEDGER_NAME).exists()


# ------------------------------------------------------------------ adoption

def test_adoption_receipt_names_both_recipes_epoch_transition_and_rebase(tmp_path):
    anchor = _launch(tmp_path / "gen-1")
    root = tmp_path / "runtime-preparation"
    declaration = _declaration(A2)
    pair = ra.DeclaredArmPlanner(_Ordinary(), declaration, store_root=lambda: root
                                 ).propose(_context(anchor)).runtime_pair
    body = ra.adoption_receipt(
        campaign_id="ak-ds41", previous=pair.anchor, adopted=pair.candidate,
        admission={"locator": "adm.json", "sha256": "a" * 64, "verified": True},
        selection_reference={"locator": "sel.json", "sha256": "b" * 64, "verified": True},
        runtime_pair=pair.to_dict(), comparison={"effect": 0.095, "pairs": 9, "decisive": True,
                                                 "noise_floor_pct": 2.1, "extra": "dropped"},
        epoch="e" * 64, measurement_epoch="m" * 64, anchor_commit="c" * 40,
        statistics_sha256="s" * 64, declaration=declaration, invalidated_floor="/floor.json",
        accumulator={"champion_of_record": "a", "tip": "b"}, adopted_at="2026-09-27T00:00:00Z")
    assert body["schema"] == ra.ADOPTION_SCHEMA
    assert body["surface_change"] == {"launch_env": {"OMP_PLACES": {
        "before": "cores", "after": "{2}:47:2,{1}"}}}
    assert body["declared_arm"]["arm_id"] == A2["arm_id"]
    assert body["epoch_transition"]["next_epoch_input"] == {
        "runtime_recipe_surface_digest": ra.surface_digest(pair.candidate)}
    assert body["measured_under"]["measurement_epoch"] == "m" * 64
    assert "extra" not in body["comparison"] and body["comparison"]["effect"] == 0.095
    assert "never a production" in body["authority"]
    path = ra.write_adoption_receipt(tmp_path / "store", body)
    assert json.loads(path.read_text()) == body
    assert path.parent.name == ra.ADOPTION_DIR


def test_stale_runtime_recipe_bundle_never_fires_threshold_and_roundtrips(tmp_path):
    bundle = accumulate.Bundle(champion_of_record="a" * 40, tip="b" * 40, keeps=["k1"],
                               compounded_bench_pct=50.0, keeps_since_serving_gate=1)
    policy = accumulate.AccumulatorPolicy(every_keeps=4)
    assert accumulate.gate_trigger(bundle, 2.0, policy) == "threshold"
    bundle.measurement_validity = accumulate.MEASUREMENT_STALE_RUNTIME_RECIPE
    assert accumulate.gate_trigger(bundle, 2.0, policy) is None
    path = bundle.save(tmp_path)
    reloaded = accumulate.Bundle.from_dict(json.loads(Path(path).read_text()))
    assert reloaded.measurement_validity == accumulate.MEASUREMENT_STALE_RUNTIME_RECIPE
    bundle.add_keep("k2", "c" * 40, 3.0)
    assert bundle.measurement_validity == accumulate.MEASUREMENT_CURRENT


# ------------------------------------------------------------------ serial binding

def test_enabling_the_runtime_protocol_keeps_the_continuation_binding(tmp_path):
    launch = tmp_path / "launch.json"
    launch.write_text("{}")
    base = ["--worktree", "/w", "--cpu-serving-launch", str(launch), "--workers", "1"]
    enabled = base + ["--runtime-statistics", "/s.json", "--runtime-calibration-max-launches",
                      "640", "--runtime-arms", "/a.json", "--calibrate-runtime"]
    assert serial_run.resume_binding(enabled) == serial_run.resume_binding(base)
    # Execution documents stay bound: a changed launch is a different continuation.
    before = serial_run.resume_binding(enabled)
    launch.write_text('{"changed": true}')
    assert serial_run.resume_binding(enabled) != before
    assert serial_run.resume_binding(enabled + ["--workers", "2"]) != \
        serial_run.resume_binding(base)


# ------------------------------------------------------------------ through run.main

def _fixture_admission(keeps):
    class FixtureAdmission:
        """Synthetic admission boundary; strict admission is test_runtime_admission's."""

        def __init__(self, **kwargs):
            self.original = kwargs["original"]
            self.state = {"selected": None}
            self.default = kwargs["store"].write("fixture-default", self.original.to_dict())

        def selected(self):
            return self.original

        def compare(self, pair):
            self.pair = pair
            return {"recipe": pair.anchor.template.name,
                    "recipe_hash": pair.anchor.template.recipe_hash, "pairs": 2, "effect": .1,
                    "decisive": True, "noise_floor_pct": 1, "runtime_pair": pair.to_dict(),
                    "runtime_admission": {"fixture": True}}

        def retain(self, row, current):
            keeps.append(self.pair)
            return self.pair.candidate

        def selection_reference(self, *args, **kwargs):
            return {"locator": "fixture-runtime-selection.json", "sha256": "b" * 64,
                    "verified": True}

        def retained_build(self, reference):
            return Path(keeps[0].anchor.build_dir) if keeps else None

        def pending_pair(self):
            return None

    return FixtureAdmission


def test_runtime_keep_writes_an_auditable_adoption_receipt(monkeypatch, tmp_path):
    from . import claim, pool, runtime_admission
    # The fixture's 0-95 CPU list includes q3: never wait on the host's real MI210 lock.
    monkeypatch.setattr(claim, "DEVICE_LOCK", tmp_path / "gpu_device.fixture.lock")
    from .test_existing_cpu_run import \
        test_existing_main_cpu_five_iterations_preserves_canonical_champion as run_fixture
    keeps = []

    def checked(result, measured, builds):
        assert len(keeps) == 1 and result["iterations"][0]["status"] == "kept"
        adoptions = result["runtime_adoptions"]
        assert len(adoptions) == 1
        path = Path(adoptions[0]["receipt"])
        body = json.loads(path.read_text())
        assert body["schema"] == ra.ADOPTION_SCHEMA and body["campaign_id"] == "ak-loop"
        assert body["adopted_recipe"]["runtime_surface_digest"] == \
            ra.surface_digest(keeps[0].candidate) == adoptions[0]["adopted_surface_digest"]
        assert body["previous_recipe"]["runtime_surface_digest"] == \
            ra.surface_digest(keeps[0].anchor)
        assert body["dimension"] == keeps[0].dimension.to_dict()
        assert body["measured_under"]["epoch"] == result["epoch"]
        assert body["measured_under"]["measurement_epoch"] == result["measurement_epoch"]
        assert body["epoch_transition"]["next_epoch_input"][
            "runtime_recipe_surface_digest"] == ra.surface_digest(keeps[0].candidate)
        assert body["selection_reference"]["locator"] == "fixture-runtime-selection.json"
        assert body["statistics_sha256"] and len(body["statistics_sha256"]) == 64
        # This batch STARTED without a carried selection: its epoch has no runtime key.
        assert "runtime_recipe_surface_digest" not in result
        assert result["runtime_preparation"]["adoptions"] == adoptions

    monkeypatch.setattr(runtime_admission, "RuntimeAdmission", _fixture_admission(keeps))
    monkeypatch.setattr(pool, "prune_anchor_generations",
                        lambda *a, **k: pool.PruneReport("complete"))
    run_fixture(False, runtime_transition=checked)


def _with_arms(monkeypatch, declaration_body, *, expect_exit=None):
    from .test_existing_cpu_run import \
        test_existing_main_cpu_five_iterations_preserves_canonical_champion as run_fixture
    original_main = run.main
    seen = {}

    def with_arms(argv):
        path = Path(argv[argv.index("--store") + 1]).parent / "runtime-arms.json"
        path.write_text(json.dumps(declaration_body))
        seen["path"] = path
        return original_main([*argv, "--runtime-arms", str(path), "--runtime-arm-evidence", "strict"])

    monkeypatch.setattr(run, "main", with_arms)
    if expect_exit is not None:
        with pytest.raises(SystemExit, match=expect_exit):
            run_fixture(True)
    else:
        run_fixture(True)
    return seen


def test_main_preflights_declared_arms_before_any_claim(monkeypatch):
    threads = {"schema": ra.SCHEMA, "campaign_id": "ak-loop", "numerics_policy": "bit_exact_only",
               "arms": [{"arm_id": "threads-plus-one", "kind": "threads", "candidate": 49,
                         "numerics": "bit_exact", "rationale": "fixture thread arm"}]}
    checked, original = [], ra.RuntimeArmDeclaration.preflight

    def preflight(self, **kwargs):
        checked.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(ra.RuntimeArmDeclaration, "preflight", preflight)
    _with_arms(monkeypatch, threads)     # the dry run: wiring proven, nothing claimed
    assert len(checked) == 1 and checked[0]["campaign_id"] == "ak-loop"
    assert checked[0]["max_candidates"] == 10


def test_main_refuses_declared_env_arm_outside_the_environment_policy(monkeypatch):
    uninstalled = {"schema": ra.SCHEMA, "campaign_id": "ak-loop",
                   "numerics_policy": "bit_exact_only", "arms": [A3]}
    _with_arms(monkeypatch, uninstalled, expect_exit="2")


def test_main_refuses_declared_arms_for_another_campaign(monkeypatch):
    other = {"schema": ra.SCHEMA, "campaign_id": "ak-ds41", "numerics_policy": "bit_exact_only",
             "arms": [dict(A2, kind="threads", candidate=49)]}
    _with_arms(monkeypatch, other, expect_exit="2")


def test_main_refuses_an_unreadable_carried_selection_before_any_claim(monkeypatch, tmp_path):
    from .test_existing_cpu_run import \
        test_existing_main_cpu_five_iterations_preserves_canonical_champion as run_fixture
    original_main = run.main
    reference = tmp_path / "runtime-recipe-reference.json"
    reference.write_text(json.dumps({"locator": "absent.json", "sha256": "0" * 64,
                                     "verified": True}))

    def carried(argv):
        return original_main([*argv, "--runtime-recipe-reference", str(reference)])

    monkeypatch.setattr(run, "main", carried)
    # No runtime store holds that selection: the epoch cannot be derived, so the
    # launch refuses before claiming rather than measuring under a guessed epoch.
    with pytest.raises(SystemExit, match="2"):
        run_fixture(True)


def test_env_arm_on_a_key_the_template_declares_is_enumerable(tmp_path):
    """DS41's canonical recipe DECLARES its OMP stack in the template env. An env arm on such
    a key used to fail the canonical consistency check (template env != frozen launch)."""
    from dataclasses import replace
    from . import actors
    build = tmp_path / "gen-1"
    base = _launch(build)
    template = replace(base.template, env={"GGML_IQK": "1", "OMP_PLACES": "cores"})
    declared = rr.resolve_canonical_launch(
        template, build_dir=build, command_argv=base.command_argv,
        topology_prefix=base.topology_prefix, launch_environment=dict(base.launch_env),
        artifact_identities=_artifacts(template, build=build), backend="cpu",
        environment_policy=_policy("GGML_IQK", "OMP_PLACES", witness="process_environ"),
        port=base.port, runtime_binary_dir=base.runtime_binary_dir,
        runtime_ld_paths=base.runtime_ld_paths, provenance=dict(base.provenance))
    assert declared.capability.supported
    context = {"runtime_anchor": declared.to_dict(), "runtime_env_keys": ["OMP_PLACES"]}
    pair = actors._runtime_pair(ra.RuntimeArm.from_dict(A2).treatment(), context, "arm")
    assert pair.candidate.template.env["OMP_PLACES"] == "{2}:47:2,{1}"
    assert pair.candidate.template.name == declared.template.name
    assert pair.candidate.template.recipe_hash != declared.template.recipe_hash
    assert pair.anchor.to_dict() == declared.to_dict()
    assert pair.candidate.capability.supported


# ------------------------------------------------------------------ keep-grade evidence

def _fake_measure(calls, *, candidate_bonus=0.0):
    import time

    def measure(recipe, build, port, *, evidence, **kwargs):
        started = time.time()
        calls.append((recipe.threads, dict(recipe.env or {})))
        evidence.append({"schema": serving.RESIDENCY_SCHEMA, "backend": "cpu",
                         "status": "not_applicable", "window_start": started,
                         "window_end": time.time(), "samples": 0})
        base = 10 + (len(calls) % 7) / 100
        return base * (1 + candidate_bonus) if recipe.threads == 49 else base
    return measure


def _keep_grade_inputs(tmp_path, monkeypatch, *, bonus):
    from .unified_planner import RuntimeDimension, enumerate_runtime_dimensions
    anchor = _launch(tmp_path / "gen-1")
    requests = (("p", json.dumps({"prompt": [1], "n_predict": 256, "temperature": 0.0,
                                  "top_k": 1}).encode()),)
    calls = []
    monkeypatch.setattr(serving, "_measure_once", _fake_measure(calls, candidate_bonus=bonus))
    floor = serving.calibrate_floor(anchor.template, Path(anchor.build_dir), samples=24,
        port=anchor.port, resolved_recipe=anchor, frozen_requests=requests,
        instrument=serving.MATCHED_INSTRUMENT, pairs=5)
    pair = enumerate_runtime_dimensions(anchor, (RuntimeDimension(
        "runtime-arm-threads-49", "threads", 48, 49, "declared"),))[0]
    return anchor, requests, floor, pair, calls


@pytest.mark.parametrize("bonus", [0.10, 0.0])
def test_keep_grade_runtime_ab_uses_the_matched_floor_of_the_current_recipe(
        tmp_path, monkeypatch, bonus):
    anchor, requests, floor, pair, calls = _keep_grade_inputs(tmp_path, monkeypatch, bonus=bonus)
    assert len(calls) == 48
    out = serving.compare(pair.anchor.template, Path(anchor.build_dir), Path(anchor.build_dir),
        pairs=5, port=anchor.port, floor_pct=floor["floor_pct"], floor_unit=serving.COMPARE_EFFECT_UNIT,
        floor_record=floor, instrument=serving.MATCHED_INSTRUMENT,
        anchor_resolved_recipe=pair.anchor, candidate_resolved_recipe=pair.candidate,
        frozen_requests=requests,
        floor_request_digest=serving.request_digest(anchor.template, requests),
        runtime_pair=pair, runtime_evidence="keep_grade")
    assert len(calls) == 58 and out["admission"] == "keep_grade_matched_serving_floor"
    assert out["measurement_plan"]["instrument"] == serving.MATCHED_INSTRUMENT
    assert out["floor_sha256"] == floor["content_sha256"]
    assert sorted(threads for threads, _ in calls[48:]) == [48] * 5 + [49] * 5
    assert out["decisive"] == (abs(out["effect"]) * 100 >= floor["floor_pct"])
    assert out["decisive"] is (bonus > 0)


def test_keep_grade_refusals_keep_every_other_runtime_path_closed(tmp_path, monkeypatch):
    anchor, requests, floor, pair, calls = _keep_grade_inputs(tmp_path, monkeypatch, bonus=0.1)
    common = dict(pairs=5, port=anchor.port, anchor_resolved_recipe=pair.anchor,
                  candidate_resolved_recipe=pair.candidate, frozen_requests=requests,
                  runtime_pair=pair)
    digest = serving.request_digest(anchor.template, requests)
    build = Path(anchor.build_dir)
    anchor = SimpleNamespace(template=pair.anchor.template)
    # Without keep-grade evidence, a runtime pair still may not borrow the floor.
    with pytest.raises(serving.ServingFloorMismatch, match="cannot qualify a runtime"):
        serving.compare(anchor.template, build, build, floor_pct=floor["floor_pct"],
                        floor_unit=serving.COMPARE_EFFECT_UNIT, floor_record=floor,
                        instrument=serving.MATCHED_INSTRUMENT, floor_request_digest=digest,
                        **common)
    # Keep-grade needs the matched instrument and a floor record.
    with pytest.raises(serving.ServingFloorMismatch, match="keep-grade runtime evidence needs"):
        serving.compare(anchor.template, build, build, floor_pct=None,
                        instrument=serving.MATCHED_INSTRUMENT, runtime_evidence="keep_grade",
                        **common)
    with pytest.raises(serving.RecipeError, match="unknown runtime evidence"):
        serving.compare(anchor.template, build, build, floor_pct=None,
                        runtime_evidence="loose", **common)
    # A floor calibrated under ANOTHER recipe (the candidate's) is refused.
    other = dict(floor, recipe_hash=pair.candidate.template.recipe_hash)
    with pytest.raises(serving.ServingFloorMismatch):
        serving.compare(anchor.template, build, build, floor_pct=floor["floor_pct"],
                        floor_unit=serving.COMPARE_EFFECT_UNIT, floor_record=other,
                        instrument=serving.MATCHED_INSTRUMENT, floor_request_digest=digest,
                        runtime_evidence="keep_grade", **common)
    assert len(calls) == 48     # nothing launched by any refusal


def test_is_declared_matches_mechanism_and_exact_treatment(tmp_path):
    anchor = _launch(tmp_path / "gen-1")
    declaration = _declaration(A2)
    context = _context(anchor)
    from . import actors
    declared = actors._runtime_pair(ra.RuntimeArm.from_dict(A2).treatment(), context,
                                    "runtime-arm-" + A2["arm_id"])
    assert ra.is_declared(declared, declaration)
    renamed = actors._runtime_pair(ra.RuntimeArm.from_dict(A2).treatment(), context, "planner-id")
    assert not ra.is_declared(renamed, declaration)
    other = actors._runtime_pair(ra.RuntimeArm.from_dict(A1).treatment(), context,
                                 "runtime-arm-" + A2["arm_id"])
    assert not ra.is_declared(other, declaration)
    assert not ra.is_declared(declared, None)


def test_keep_grade_attempt_ledger_settles_the_arm(tmp_path):
    anchor = _launch(tmp_path / "gen-1")
    root = tmp_path / "runtime-preparation"
    declaration = _declaration(A2)
    planner = ra.DeclaredArmPlanner(_Ordinary(), declaration, store_root=lambda: root,
                                    evidence="keep_grade")
    # keep-grade: planner treatments are observation-only, declared arms are not.
    served = planner.propose(_context(anchor, observation_only=True))
    assert served.mechanism_id == "runtime-arm-" + A2["arm_id"]
    ra.record_attempt(root, pair=served.runtime_pair.to_dict(),
                      comparison={"effect": 0.01, "decisive": False, "extra": 1},
                      declaration=declaration)
    assert ra.arm_state(root, declaration, anchor) == {A2["arm_id"]: "settled"}
    assert planner.propose(_context(anchor, observation_only=True)) == "planner-hypothesis"
    body = json.loads((root / ra.ATTEMPTS_NAME).read_text())
    assert "extra" not in body["attempts"][0]["comparison"]
    with pytest.raises(ra.ArmDeclarationRefused, match="evidence"):
        ra.DeclaredArmPlanner(_Ordinary(), declaration, store_root=lambda: root, evidence="x")


def test_keep_grade_selection_restores_and_rebinds_to_a_new_build(tmp_path):
    from dataclasses import replace
    anchor = _launch(tmp_path / "gen-1")
    context = _context(anchor)
    from . import actors
    pair = actors._runtime_pair(ra.RuntimeArm.from_dict(A2).treatment(), context,
                                "runtime-arm-" + A2["arm_id"])
    (tmp_path / "store").mkdir()
    store = mc.ArtifactStore(tmp_path / "store" / "runtime-preparation")
    try:
        body = ra.keep_grade_selection(adopted=pair.candidate, previous=pair.anchor,
            runtime_pair=pair.to_dict(), comparison={"effect": 0.09, "decisive": True},
            declaration=_declaration(A2), current_source_commit="c" * 40)
        reference = store.write(ra.KEEP_GRADE_SELECTION_NAMESPACE, body).to_dict()
        same = ra.restore_keep_grade_selection(store, reference, build=Path(anchor.build_dir),
                                               rebind=lambda *a: pytest.fail("no rebind"))
        assert same.to_dict() == pair.candidate.to_dict()
        rebuilt = _launch(tmp_path / "gen-2", {"OMP_PLACES": "{2}:47:2,{1}"})
        moved = ra.restore_keep_grade_selection(store, reference, build=tmp_path / "gen-2",
                                                rebind=lambda recipe, build: rebuilt)
        assert moved is rebuilt and ra.surface_digest(moved) == body["adopted_surface_digest"]
        with pytest.raises(ra.ArmDeclarationRefused, match="rebind"):
            ra.restore_keep_grade_selection(store, reference, build=tmp_path / "gen-2",
                                            rebind=lambda recipe, build: _launch(build))
        strict = store.write("direct-runtime-selection", {
            "schema": ra.STRICT_SELECTION_SCHEMA, "current_recipe": pair.candidate.to_dict()}
        ).to_dict()
        with pytest.raises(ra.ArmDeclarationRefused):
            ra.restore_keep_grade_selection(store, strict, build=tmp_path / "gen-1",
                                            rebind=lambda *a: None)
    finally:
        store.close()
    # The pre-claim epoch read is evidence-scoped.
    assert ra.surface_digest(ra.selection_current_recipe(
        tmp_path / "store", reference, evidence="keep_grade")) == body["adopted_surface_digest"]
    with pytest.raises(ra.ArmDeclarationRefused, match="strict evidence"):
        ra.selection_current_recipe(tmp_path / "store", reference, evidence="strict")


def test_main_keep_grade_arm_is_adopted_then_restored_into_a_new_epoch(monkeypatch, tmp_path):
    """run.main end to end with pool.drive replaced: the declared arm is served by the
    wrapped planner, measured at keep-grade against the matched floor, adopted through the
    real keep closure, and the next launch restores it under a new epoch."""
    from contextlib import contextmanager
    from . import claim, cpu_profile, pool, test_promotion_targets as fixtures
    from .test_cpu_screen import _inputs as cpu_inputs
    monkeypatch.setattr(claim, "DEVICE_LOCK", tmp_path / "mi210.lock")
    fixture = fixtures.TheKeepBuildsAProductionCompleteAnchor()
    fixture.setUp()
    try:
        launch, prompts, options, _ = cpu_inputs(fixture)
        arms_path = fixture.root / "runtime-arms.json"
        arms_path.write_text(json.dumps({
            "schema": ra.SCHEMA, "campaign_id": "unified-ak-test",
            "numerics_policy": "bit_exact_only",
            "arms": [{"arm_id": "threads-plus-one", "kind": "threads",
                      "candidate": launch.template.threads + 1, "numerics": "bit_exact",
                      "rationale": "fixture thread arm"}]}))
        calls, seen = [], {}
        reference_file = fixture.root / "runtime-recipe-reference.json"

        def measure_once(recipe, build, port, *, evidence, **kwargs):
            import time
            calls.append(recipe.threads)
            evidence.append({"schema": serving.RESIDENCY_SCHEMA, "backend": "cpu",
                             "status": "not_applicable", "window_start": time.time(),
                             "window_end": time.time(), "samples": 0})
            base = 10 + (len(calls) % 7) / 100
            return base * 1.2 if recipe.threads == launch.template.threads + 1 else base

        @contextmanager
        def hold(*_args):
            yield {"device_id": "synthetic-cpu-claim"}

        def drive(**kwargs):
            context = kwargs["build_context"]()
            seen.setdefault("anchors", []).append(context["runtime_anchor"])
            seen.setdefault("preparation", []).append(dict(context["runtime_preparation"]))
            worker = SimpleNamespace(name="lane0", worktree=fixture.root / "lane0",
                                     build_dir=fixture.startup_anchor)
            planner = kwargs["make_planner"](worker)
            if reference_file.exists():
                # Launch 2: the adopted value IS the current recipe, so the arm is an exact
                # no-op and the ordinary planner is consulted; a source comparison then
                # recalibrates the matched floor under the adopted recipe first.
                with pytest.raises(AssertionError, match="ordinary planner consulted"):
                    planner.propose(context)
                seen["source_row"] = kwargs["make_measure"](worker)(
                    SimpleNamespace(runtime_pair=None), ()).row
                return pool.PoolResult()
            hypothesis = planner.propose(context)
            seen.setdefault("served", []).append(getattr(hypothesis, "mechanism_id", None))
            if not isinstance(hypothesis, run.loop.Hypothesis) or hypothesis.runtime_pair is None:
                return pool.PoolResult()
            comparison = kwargs["make_measure"](worker)(hypothesis, ())
            seen.setdefault("rows", []).append(comparison.row)
            if comparison.decisive and comparison.effect > 0:
                kwargs["commit"](worker, hypothesis, (), comparison)
            return pool.PoolResult()

        class _NoActor:
            def __init__(self, *a, **k):
                pass

            def propose(self, context):
                raise AssertionError("ordinary planner consulted while a declared arm is open")

        def invoke(argv):
            argv += options + ["--serving-pairs", "5", "--serving-instrument",
                               serving.MATCHED_INSTRUMENT, "--runtime-arms", str(arms_path)]
            if reference_file.exists():
                argv += ["--runtime-recipe-reference", str(reference_file)]
            with mock.patch.object(run.claim, "hold_cpu", hold), \
                    mock.patch.object(run.os, "sched_getaffinity", return_value={0, 1}), \
                    mock.patch.object(run.os, "sched_setaffinity"), \
                    mock.patch.object(run.workload_contract, "read_census", return_value=SimpleNamespace(
                        n_embd=1536, dominant_quant="Q5_0")), \
                    mock.patch.object(cpu_profile, "profile_loop",
                                      side_effect=cpu_profile.CpuProfileRefused("fixture")), \
                    mock.patch.object(serving, "_measure_once", measure_once), \
                    mock.patch.object(run.actors, "AgentPlanner", _NoActor), \
                    mock.patch.object(run.gates, "op_correctness",
                                      return_value=run.gates.Verdict("correctness", True, "fixture")), \
                    mock.patch.object(pool, "provision", return_value=[]), \
                    mock.patch.object(pool, "drive", drive):
                return real_main(argv)

        real_main = run.main
        with mock.patch.object(run, "main", invoke):
            assert fixture._run_one_keep()[0] == 0
        # Launch 1: 48-launch floor on the original recipe, then ONE 10-launch keep-grade A/B.
        assert len(calls) == 58 and seen["served"] == ["runtime-arm-threads-plus-one"]
        row = seen["rows"][0]
        assert row["admission"] == "keep_grade_matched_serving_floor" and row["decisive"]
        assert seen["preparation"][0]["status"] == "keep_grade_declared_arms"
        receipts = sorted((fixture.store / ra.ADOPTION_DIR).glob("*.json"))
        assert len(receipts) == 1
        receipt = json.loads(receipts[0].read_text())
        assert receipt["evidence"] == "keep_grade"
        assert receipt["comparison"]["floor_sha256"] == row["floor_sha256"]
        reference_file.write_text(json.dumps(receipt["selection_reference"]))
        attempts = json.loads((fixture.store / "runtime-preparation" / ra.ATTEMPTS_NAME).read_text())
        assert len(attempts["attempts"]) == 1
        # Launch 2 carries the selection: the adopted recipe is the anchor, its own floor is
        # calibrated (48 launches), and the arm is an exact no-op now, so the planner runs.
        with mock.patch.object(run, "main", invoke):
            assert fixture._run_one_keep()[0] == 0
        assert seen["anchors"][1]["template"]["threads"] == launch.template.threads + 1
        # Floor recompute: the adopted recipe got its own 24-pair matched floor (48 launches),
        # every one of them under the adopted thread count.
        assert len(calls) == 58 + 48 + 10
        assert set(calls[58:]) == {launch.template.threads + 1}
        assert seen["source_row"]["recipe_hash"] != row["recipe_hash"]
        assert ra.surface_digest(seen["anchors"][1]) == \
            receipt["epoch_transition"]["next_epoch_input"]["runtime_recipe_surface_digest"]
    finally:
        fixture.doCleanups()


def test_serial_preview_keeps_batches_full_while_declared_arms_are_unsettled(tmp_path):
    from . import cpu_screen
    anchor = _launch(tmp_path / "gen-1")
    launch_file, arms_file = tmp_path / "launch.json", tmp_path / "arms.json"
    launch_file.write_text(json.dumps(anchor.to_dict()))
    arms_file.write_text(json.dumps(_declaration(A2).to_dict()))
    (tmp_path / "store").mkdir()
    argv = ["--cpu-serving-launch", str(launch_file), "--resolved-campaign", str(tmp_path / "c.json"),
            "--store", str(tmp_path / "store"), "--runtime-arms", str(arms_file)]
    selected = cpu_screen.preview_batch(argv, None)
    assert selected["scope"] == "full" and "unsettled" in selected["reason"]
    arms_file.write_text("{not json")
    selected = cpu_screen.preview_batch(argv, None)
    assert selected["scope"] == "full" and "unreadable" in selected["reason"]


# ------------------------------------------------------------------ load_threads (argv arm)

LOAD = {"arm_id": "load-threads-48", "kind": "load_threads", "candidate": 48,
        "numerics": "bit_exact",
        "rationale": "model-loader reader team = -t, so no smaller OpenMP team precedes compute"}


def test_load_threads_argv_arm_is_a_single_field_launch_delta(tmp_path):
    from . import actors
    anchor = _launch(tmp_path / "gen-1")
    assert "--load-threads" not in anchor.command_argv
    context = _context(anchor)
    pair = actors._runtime_pair(ra.RuntimeArm.from_dict(LOAD).treatment(), context,
                                "runtime-arm-load-threads-48")
    assert pair.dimension.anchor is None and pair.dimension.candidate == 48
    assert pair.candidate.command_argv[-2:] == ("--load-threads", "48")
    assert pair.candidate.template.extra_flags[-2:] == ("--load-threads", "48")
    assert pair.candidate.launch_env == pair.anchor.launch_env
    assert pair.candidate.template.threads == pair.anchor.template.threads
    assert pair.candidate.capability.supported
    assert ra.surface_digest(pair.candidate) != ra.surface_digest(pair.anchor)
    assert ra.is_declared(pair, _declaration(LOAD))
    # After adoption the arm is an exact no-op on the adopted recipe (champion/challenger).
    adopted = {**context, "runtime_anchor": pair.candidate.to_dict()}
    with pytest.raises(actors.ProviderTransient):
        actors._runtime_pair(ra.RuntimeArm.from_dict(LOAD).treatment(), adopted, "again")
    # ... and the OMP_PLACES fallback then challenges ON TOP of --load-threads 48.
    fallback = actors._runtime_pair(ra.RuntimeArm.from_dict(A2).treatment(), adopted,
                                    "runtime-arm-" + A2["arm_id"])
    assert fallback.candidate.command_argv[-2:] == ("--load-threads", "48")
    assert dict(fallback.candidate.launch_env)["OMP_PLACES"] == "{2}:47:2,{1}"
    with pytest.raises(ra.ArmDeclarationRefused, match="positive int"):
        _declaration(dict(LOAD, candidate=0))


def test_canonical_grammar_accepts_load_threads_and_refuses_negative(tmp_path):
    anchor = _launch(tmp_path / "gen-1")
    command = tuple(anchor.command_argv) + ("--load-threads", "32")
    projected = rr.canonical_recipe_projection(name="x", command_argv=command,
        topology_prefix=anchor.topology_prefix, n_predict=256, temperature=0.0, top_k=1)
    assert projected.extra_flags[-2:] == ("--load-threads", "32")
    with pytest.raises(rr.ResolutionError, match="non-negative"):
        rr.canonical_recipe_projection(name="x", command_argv=tuple(anchor.command_argv)
            + ("--load-threads", "-1"), topology_prefix=anchor.topology_prefix,
            n_predict=256, temperature=0.0, top_k=1)


DS41_LAUNCH = Path("/mnt/raid0/llm/autokernel/campaigns/ak-ds41-cpu-decode-20260923/inputs/"
                   "ds41-cpu-t48-dspark.launch.json")


@pytest.mark.skipif(not DS41_LAUNCH.exists(), reason="DS41 campaign inputs not on this host")
def test_declared_ds41_arms_enumerate_against_the_live_launch():
    """Read-only: the prepared DS41 arms are launch-valid single-field deltas."""
    from . import actors
    launch = rr.CanonicalResolvedRecipe.from_dict(json.loads(DS41_LAUNCH.read_text()))
    context = {"runtime_anchor": launch.to_dict(), "runtime_observation_only": True,
               "runtime_env_keys": sorted(run._runtime_env_keys(launch, launch))}
    load = actors._runtime_pair(ra.RuntimeArm.from_dict(LOAD).treatment(), context,
                                "runtime-arm-load-threads-48")
    places = actors._runtime_pair(ra.RuntimeArm.from_dict(A2).treatment(), context,
                                  "runtime-arm-" + A2["arm_id"])
    assert load.candidate.capability.supported and places.candidate.capability.supported
    assert dict(load.candidate.launch_env) == dict(launch.launch_env)
    assert dict(places.candidate.launch_env)["OMP_PLACES"] == "{2}:47:2,{1}"


def test_serial_common_args_admit_the_runtime_protocol_options(tmp_path):
    from . import serial_roster, serial_run as sr
    from .test_serial_roster import _inputs
    _resolved, _owners, argv = _inputs(tmp_path, backends=("cpu",))
    common = tmp_path / "common.json"
    common.write_text(json.dumps(["--workers", "1", "--runtime-arms", str(tmp_path / "arms.json"),
                                  "--runtime-arm-evidence", "keep_grade"]))
    targets, _skipped, _cpus = serial_roster.build_targets(
        Path(sr.option(argv, "--resolved-campaign")), Path(sr.option(argv, "--owned-targets")),
        target_root=Path(sr.option(argv, "--state-dir")) / "targets", common_path=common)
    assert sr.option(targets[0], "--runtime-arm-evidence") == "keep_grade"
    assert sr.option(targets[0], "--runtime-arms") == str(tmp_path / "arms.json")
    # Still refused: identities (the launch) may never ride in common args.
    common.write_text(json.dumps(["--cpu-serving-launch", "/x.json"]))
    with pytest.raises(sr.SerialRefused, match="not a shared"):
        serial_roster.build_targets(
            Path(sr.option(argv, "--resolved-campaign")), Path(sr.option(argv, "--owned-targets")),
            target_root=Path(sr.option(argv, "--state-dir")) / "targets", common_path=common)
