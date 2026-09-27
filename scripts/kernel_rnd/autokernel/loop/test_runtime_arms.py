"""Declared runtime arms, runtime-surface epoch input and adoption receipt (fakes only).

No server, model or hardware: canonical launches are fixture identities, the
runtime-selection state is written the way `RuntimeAdmission._checkpoint`
writes it, and strict admission itself is exercised by test_runtime_admission.
"""
import json
from pathlib import Path
from types import SimpleNamespace

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
        return original_main([*argv, "--runtime-arms", str(path)])

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
