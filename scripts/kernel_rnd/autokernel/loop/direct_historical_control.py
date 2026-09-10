"""Original Qwen CPU IQK control, held by the existing direct-loop owner.

This is a control of the evaluator, not evidence about a serving hypothesis.
Its pp512 tiny-graph calibration and 0.03--0.60 reference band never move to
GLM. Reopening derives the observation from original records without issuing
work, reattesting an old window or manufacturing a RegionClaimReceipt.
"""
from contextlib import nullcontext
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from .. import campaign, schemas, storage
from ..evaluator import api, controls, correctness, recipes, statistics as st
from ..execution import control_runner as cr, inference_window, live_controls as lc, microbench, sandbox
from ..execution import t0_provider as tp
from . import observation_binding as ob, runtime_window as rw, serial_run
from .claim import HeldCpuClaim
from .measurement_capture import ArtifactStore, StoredArtifact

SCHEMA = "epyc.autokernel.direct_historical_control.v1"
NAMESPACE = "direct-historical-control"
LOCATOR = ("data/kernel-v8-candidate/cpu-prefill-regression/"
    "run-20260725T155655Z-v4-waive-q8-kfd-procrace-swapoff/summary.json")


class HistoricalControlRefused(ValueError):
    pass


def _digest(value):
    return schemas.content_hash(value)


def _ref(value):
    return f"{value.locator}#sha256={value.sha256}"


def _read(store, namespace, reference):
    item = StoredArtifact(**reference)
    body = ob._plain(store.read(item.locator, item.sha256))
    if store.verify(namespace, body) != item:
        raise HistoricalControlRefused("original historical artifact namespace differs")
    return body


def _check(ok, reason, *, unavailable=False):
    return schemas.Check(schemas.PASS if ok else schemas.COULD_NOT_CHECK
                         if unavailable else schemas.FAIL, (reason,))


def _deadline(value):
    if value is not None:
        if type(value) not in (int, float) or not math.isfinite(value):
            raise HistoricalControlRefused("original invocation deadline is not finite")
        if time.monotonic() >= value:
            raise HistoricalControlRefused("original invocation budget exhausted before historical control launch")


def _frame():
    if lc.RECIPE_ID != lc.PREFILL_RECIPE_ID or lc.PROMPT_TOKENS != 512:
        raise HistoricalControlRefused("historical pp512 frame cannot inherit another live-control recipe")
    return {"recipe_id": lc.PREFILL_RECIPE_ID, "model": str(lc.MODEL),
        "cpu_list": lc.CPU_LIST, "calibration": lc._calibration_frame(),
        "calibration_blocks": lc.CALIBRATION_BLOCKS, "neutral_blocks": lc.NEUTRAL_BLOCKS,
        "control_arms": list(lc.CONTROL_ARM_IQK["historical_win_replay"]),
        "production_commit": lc.PRODUCTION_COMMIT, "instrument_commit": lc.INSTRUMENT_COMMIT,
        "instrument_root": str(lc.INSTRUMENT_ROOT), "instrument_binary": str(lc.INSTRUMENT_BINARY),
        "reference_band": {"low": .03, "high": .60}, "phase": "prefill",
        "cell_class": recipes.CELL_CLASS_TINY_GRAPH}


def source_identity():
    from . import lifecycle_observation as lo
    modules = (lc, microbench, tp, cr, api, controls, correctness, st, rw, inference_window)
    paths = (Path(__file__), *(Path(module.__file__) for module in modules))
    return {"files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "loaded": [lo.callable_identity(fn) for fn in
                   (run_or_reopen, _collect, _compose, _t0_material, _window, _frame)]}


def _resolve(store):
    root = Path(__file__).resolve().parents[4]
    declaration = controls.HistoricalWinReplayDeclaration(
        win_id="iqk-prefill-port", backend="llama_cpu", phase="prefill",
        reference_direction="higher_better", reference_band=controls.ReferenceBand(.03, .60),
        evidence_locator=str(root / LOCATOR), durability_class="carried_in_git")
    resolution = controls.resolve_historical_win_replay(declarations=(declaration,),
        backend="llama_cpu", tracked_index=storage.GitTrackedIndex(root))
    # The 29 MiB original source is already carried in Git. Retain its exact
    # byte identity, not another multi-megabyte copy for every control window.
    # Replay verifies those original bytes, never a newly healthier Git index.
    raw = serial_run._read(root / LOCATOR, limit=32 * 1024 * 1024) if resolution.available else None
    return store.write("direct-historical-resolution", {
        "resolution": resolution.to_dict(), "repo_relative_locator": LOCATOR,
        "original_size_bytes": None if raw is None else len(raw),
        "original_sha256": None if raw is None else hashlib.sha256(raw).hexdigest()})


def _resolution(store, reference):
    body = _read(store, "direct-historical-resolution", reference)
    row = body["resolution"]
    declaration = None
    if row["declaration"] is not None:
        declaration, issues = controls.HistoricalWinReplayDeclaration.parse(row["declaration"])
        if issues or declaration is None:
            raise HistoricalControlRefused("original historical declaration is malformed")
    if body["repo_relative_locator"] != LOCATOR:
        raise HistoricalControlRefused("original historical evidence locator differs")
    if row["available"]:
        raw = serial_run._read(Path(__file__).resolve().parents[4] / LOCATOR, limit=32 * 1024 * 1024)
        if len(raw) != body["original_size_bytes"] or hashlib.sha256(raw).hexdigest() != body["original_sha256"]:
            raise HistoricalControlRefused("retained historical evidence bytes changed")
    return controls.HistoricalWinResolution(row["backend"], row["available"], declaration,
        schemas.Check(row["outcome"], tuple(row["reasons"])), row["marker"], row["durability_outcome"])


class _T0Claim:
    """Adapt the original observation, not a second grant or legacy receipt."""
    def __init__(self, adapter, deadline):
        self.adapter, self.deadline, self.claim_id = adapter, deadline, adapter.claim_id

    def verify_held(self):
        _deadline(self.deadline)
        return self.adapter.attest().check

    def covers(self, cpu_list):
        from ..execution.cpu_region_claim import parse_cpu_list
        return parse_cpu_list(cpu_list) <= parse_cpu_list(self.adapter.cpu_list)


class _Sink:
    def __init__(self, store):
        self.owner, self.references = store, []

    def store(self, capture):
        reference = self.owner.write("direct-historical-t0-process", capture.to_dict())
        self.references.append(reference.to_dict())
        return _ref(reference)


class _BoundedRunner:
    def __init__(self, root, deadline, store):
        self.deadline, self.store = deadline, store
        self.runner = tp.SubprocessRunner(sandbox_policy=sandbox.SandboxPolicy(writable_root=str(root)))

    def run(self, argv, *, env, cwd, timeout_s):
        _deadline(self.deadline)
        is_generation = str(lc.INSTRUMENT_BINARY.parent / "llama-completion") in argv and "-p" in argv
        wait_bound = 600.0 if self.deadline is None else max(0., self.deadline - time.monotonic())
        window = (inference_window.InferenceCallWindow(timeout_s=wait_bound).acquire()
                  if is_generation else nullcontext())
        with window:
            _deadline(self.deadline)
            if self.deadline is not None:
                timeout_s = min(timeout_s, self.deadline - time.monotonic())
            result = self.runner.run(argv, env=env, cwd=cwd, timeout_s=timeout_s)
        if result.orphans:
            self.store.write("direct-historical-t0-process", result.to_dict())
            raise tp.ProcessEscaped("historical T0 cleanup left an owned process alive")
        return result


def _t0_plan(source_sha, *, arm):
    binary = lc.INSTRUMENT_BINARY.parent / "llama-completion"
    candidate = tp.CandidateBuild(str(lc.INSTRUMENT_ROOT), str(binary.parent.parent),
        lc.INSTRUMENT_COMMIT, source_sha, str(binary), str(binary.parent),
        str(binary.parent / "test-backend-ops"))
    return tp.T0ExecutionPlan(candidate=candidate, tools=campaign.HostOps._t0_tools(),
        op_suite=tp.OpSuitePlan(backend_filter="CPU", ops=correctness.MANDATORY_BACKEND_OPS,
            suite_id="test-backend-ops/v1", suite_source_sha256=source_sha, suite_seed=2026081101),
        dispatch=tp.DispatchTracePlan(derived_surface=correctness.MANDATORY_BACKEND_OPS),
        anchor=tp.AnchorBuild(str(lc.INSTRUMENT_ROOT), lc.INSTRUMENT_COMMIT, str(binary), str(binary.parent)),
        generation=campaign.HostOps._t0_generation_plan(type("HistoricalModel", (), {"model": str(lc.MODEL)})()),
        determinism_runs=2, backend="llama_cpu",
        parameter_env=(("GGML_IQK", "0" if arm == "anchor" else "1"),))


def _collect_t0(store, adapter, root, source_sha, deadline):
    sink = _Sink(store)
    root.mkdir()
    runner = _BoundedRunner(root, deadline, store)
    held = _T0Claim(adapter, deadline)
    anchor = tp.capture_anchor(plan=_t0_plan(source_sha, arm="anchor"), runner=runner,
        claim=held, sink=sink, generation_seeds=(42, 42))
    anchor_references = list(sink.references)
    sink.references.clear()
    provider = tp.ExecutedT0EvidenceProvider(plan=_t0_plan(source_sha, arm="candidate"),
        runner=runner, claim=held, sink=sink, anchor_capture=anchor)
    collected = tp._Collected()
    provider.collect_op_suite(collected)
    provider.collect_coherence(collected)
    provider.collect_determinism(collected)
    return store.write("direct-historical-t0", {"source_sha256": source_sha,
        "anchor": asdict(anchor), "anchor_captures": anchor_references,
        "candidate_captures": sink.references,
        "claim_observations": list(adapter.observations)})


def _t0_material(store, reference):
    """Existing parsers replay original process output; no process or file probe."""
    body = _read(store, "direct-historical-t0", reference)
    anchor_row = dict(body["anchor"])
    for key in ("resolved_libraries", "output_digests", "output_lengths", "oracle_ids", "capture_refs", "notes"):
        anchor_row[key] = tuple(tuple(row) if isinstance(row, list) else row for row in anchor_row[key])
    anchor = tp.AnchorCapture(**anchor_row)
    captures, anchor_captures = [], []
    for ref in (*body["anchor_captures"], *body["candidate_captures"]):
        row = _read(store, "direct-historical-t0-process", ref)
        if row["orphans"]:
            raise HistoricalControlRefused("original historical T0 process did not drain")
    for ref in body["anchor_captures"]:
        row = _read(store, "direct-historical-t0-process", ref)
        row.update(argv=tuple(row["argv"]), env=tuple(map(tuple, row["env"])), orphans=tuple(row["orphans"]))
        anchor_captures.append(tp.CompletedProcess(**row))
    plan = _t0_plan(body["source_sha256"], arm="anchor")
    # The anchor's successful repeats, not a stored 'stable' assertion, own
    # its output identity. The linkage capture is separate and remains raw.
    invocation = tp.build_generation_invocation(binary=plan.anchor.binary,
        library_path=plan.anchor.library_path, plan=plan.generation, base_env=plan.base_env,
        seed=42, parameter_env=plan.parameter_env)
    generations = [row for row in anchor_captures if row.argv == invocation.argv]
    if len(generations) != 2 or any(tuple(sorted(row.env)) != tuple(sorted(invocation.env_dict().items()))
            or row.cwd != plan.candidate.worktree for row in generations):
        raise HistoricalControlRefused("original anchor generation command/environment differs")
    digests, lengths = [], []
    for row in generations:
        if tp.ExecutedT0EvidenceProvider._generation_defect(row) is None:
            digests.append(tp.sha256_text(row.stdout))
            lengths.append(len(row.stdout))
    stability = ("not_measured" if len(digests) < 2 else "bitwise_stable"
                 if len(set(digests)) == 1 else "bitwise_unstable")
    if (tuple(digests), tuple(lengths), stability) != (
            anchor.output_digests, anchor.output_lengths, anchor.determinism_class):
        raise HistoricalControlRefused("original anchor generation output does not rederive")
    for ref in body["candidate_captures"]:
        row = _read(store, "direct-historical-t0-process", ref)
        row.update(argv=tuple(row["argv"]), env=tuple(map(tuple, row["env"])), orphans=tuple(row["orphans"]))
        captures.append(tp.CompletedProcess(**row))

    class OriginalRunner:
        def __init__(self):
            self.index = 0

        def run(self, argv, *, env, cwd, timeout_s):
            if self.index >= len(captures):
                raise HistoricalControlRefused("original historical T0 capture is missing")
            captured = captures[self.index]
            self.index += 1
            if (tuple(argv) != captured.argv or tuple(sorted(env.items())) != tuple(sorted(captured.env))
                    or cwd != captured.cwd):
                raise HistoricalControlRefused("original historical T0 command/environment differs")
            return captured

    class RecordedWindow:
        def __init__(self):
            originals = [_read(store, "direct-held-observation", ref)
                         for ref in body["claim_observations"]]
            if not originals or any(row["cpu_list"] != lc.CPU_LIST for row in originals):
                raise HistoricalControlRefused("original T0 held-footprint observations are absent")
            self.claim_id = originals[0]["claim_id"]
            self.check = _check(all(row["claim_id"] == self.claim_id
                and row["observation"]["status"] == "held" for row in originals),
                "original historical T0 held observations")

        def verify_held(self):
            # This parser is never installed in a live runner. Original physical
            # ownership is independently rederived by _window from its records.
            return self.check

    runner = OriginalRunner()
    provider = tp.ExecutedT0EvidenceProvider(plan=_t0_plan(body["source_sha256"], arm="candidate"),
        runner=runner, claim=RecordedWindow(), sink=tp.MemoryCaptureSink(), anchor_capture=anchor)
    collected = tp._Collected()
    ops = provider.collect_op_suite(collected)
    coherence = provider.collect_coherence(collected)
    determinism = provider.collect_determinism(collected)
    if runner.index != len(captures):
        raise HistoricalControlRefused("unconsumed original historical T0 captures")
    return body, anchor, ops, coherence, determinism, provider._change_surface()


def _t0_gates(store, reference, declaration):
    body, anchor, ops, coherence, determinism, surface = _t0_material(store, reference)
    request = api.EvaluationRequest(
        event_id="ake-historical-t0", campaign_id=declaration["control_campaign_id"],
        candidate_id="akc-historical-iqk", tier="T0", backend="llama_cpu", phase="prefill",
        cell_class=recipes.CELL_CLASS_TINY_GRAPH, protocol_id=api.PROTOCOL_VERSIONED_ID,
        artifact=api.ArtifactIdentity(body["source_sha256"], anchor.binary_sha256, anchor.linkage_sha256),
        anchor=anchor.identity(), evaluator=api.EvaluatorIdentity("P-AK-SEARCH-1/v1",
            _digest(declaration["source"]), _ref(StoredArtifact(**reference))),
        scope_denominator=api.ScopeDenominator("full", (), (), 96),
        scope_manifest_sha256=_digest({"cpu_list": lc.CPU_LIST}), co_residency="single",
        determinism=api.DeterminismReport(determinism.measured_class(), determinism.runs),
        metric=recipes.get_recipe(lc.PREFILL_RECIPE_ID).metric, metric_direction="higher_better",
        reps=lc.CALIBRATION_REPS, change_class="parameter", anchor_tier="T0", transfer_ratio_to=(),
        created_at=declaration["issued_at"], campaign_controls=None, calibration=None)
    policy = campaign.HostOps._t0_evaluator_policy(request)
    gates = (correctness.check_backend_op_units(request, ops, surface, policy),
        correctness.check_determinism_class(request, determinism, policy)[0],
        correctness.check_output_coherence(request, coherence, policy, determinism)[0])
    return body, gates


def _stopping_commitment(store, declaration, rule, *, reference=None, measured_at=None):
    if reference is None:
        commitment = st.StoppingRuleCommitment.commit(rule,
            campaign_id=declaration["control_campaign_id"],
            committed_at=datetime.now(timezone.utc).isoformat())
        return store.write("direct-historical-stopping-rule", {
            "declaration_sha256": _digest(declaration), "rule": rule.to_dict(),
            "commitment": commitment.to_dict()})
    original = _read(store, "direct-historical-stopping-rule", reference)
    commitment = st.StoppingRuleCommitment(**original["commitment"])
    if (original["declaration_sha256"] != _digest(declaration)
            or original["rule"] != rule.to_dict()
            or commitment.campaign_id != declaration["control_campaign_id"]
            or commitment.verify(rule).outcome != schemas.PASS):
        raise HistoricalControlRefused("original historical stopping commitment differs")
    if measured_at is not None and datetime.fromisoformat(commitment.committed_at) > datetime.fromisoformat(measured_at):
        raise HistoricalControlRefused("historical stopping rule was not committed before measurement")
    return commitment


def _collect(store, held, declaration, root, deadline):
    """Use existing tools only; never build, copy an instrument or reacquire."""
    _deadline(deadline)
    try:
        adapter = rw.DirectHeldClaimAdapter(held, cpu_list=lc.CPU_LIST, store=store)
    except ValueError as exc:
        # This constructor only checks the original declared footprint; it has
        # not observed or launched anything. Insufficient coverage is not an
        # unavailable historical backend and never licenses a smaller replay.
        raise HistoricalControlRefused(f"original historical CPU footprint {lc.CPU_LIST} is not covered: {exc}") from exc
    for name in lc.INSTRUMENT_BUILD_TARGETS:
        if not (lc.INSTRUMENT_BINARY.parent / name).is_file():
            raise HistoricalControlRefused(f"original historical instrument is missing {name}; no implicit build")
    identity = lc.LiveCampaignIdentity(declaration["control_campaign_id"], str(root))
    instrument_sha = lc._sha256_file(lc.INSTRUMENT_BINARY)
    linkage_sha, linkage_text = lc._linkage(lc.INSTRUMENT_BINARY, lc.INSTRUMENT_BINARY.parent)
    lc._write_preflight(root, instrument_sha=instrument_sha, copy_sha=instrument_sha,
                        host_state=microbench.read_host_state)
    source = subprocess.run(["git", "-C", str(lc.INSTRUMENT_ROOT), "ls-tree", "-r", "-z", lc.INSTRUMENT_COMMIT],
        capture_output=True, check=True, timeout=30).stdout
    if not source or len(source) > 16 * 1024 * 1024:
        raise HistoricalControlRefused("original instrument source manifest is unavailable")
    source_sha = hashlib.sha256(source).hexdigest()
    config = rw.configuration(store, held, storage_floor_bytes_free=200 * 1024 ** 3)
    opened = store.write("direct-historical-boundary", rw.boundary(config, held, marker="open"))
    t0 = _collect_t0(store, adapter, root / "t0", source_sha, deadline)
    _body, gates = _t0_gates(store, t0.to_dict(), declaration)
    original_gates = store.write("direct-historical-t0-gates", {
        "t0": t0.to_dict(), "gates": [row.to_dict() for row in gates]})
    if any(row.check.outcome != schemas.PASS for row in gates):
        raise HistoricalControlRefused(f"original historical T0 failed before performance collection: {_ref(original_gates)}")
    binding = recipes.ToolBinding(str(lc.INSTRUMENT_BINARY), str(lc.INSTRUMENT_ROOT), str(lc.INSTRUMENT_BINARY.parent))
    anchor = api.AnchorIdentity(lc.INSTRUMENT_COMMIT, instrument_sha, linkage_sha, tool="llama-bench")
    originals, materials = {}, {}

    def host_state(**kwargs):
        _deadline(deadline)
        return microbench.read_host_state(**kwargs)

    for label, count in (("aa_calibration", lc.CALIBRATION_BLOCKS), ("neutral_calibration", lc.NEUTRAL_BLOCKS)):
        _deadline(deadline)
        materials[label] = lc._measure(label=label, blocks=count, claim=held, held_claim=adapter,
            candidate_binding=binding, anchor_binding=binding, anchor=anchor,
            candidate_iqk="0", anchor_iqk="0", output_root=root, host_state=host_state, identity=identity)
        raw = json.loads((root / "raw" / (label + ".json")).read_text())
        originals[label] = store.write("direct-historical-microbench", raw).to_dict()
    _declared, rule, _construction, _split, solve = lc._campaign_inputs(
        materials["aa_calibration"], materials["neutral_calibration"], identity)
    solve.require_accepted()
    commitment = _stopping_commitment(store, declaration, rule)
    count = rule.max_total_blocks(solve.outputs.b_min_blocks)
    label = "historical_win_replay"
    _deadline(deadline)
    lc._measure(label=label, blocks=count, claim=held, held_claim=adapter,
        candidate_binding=binding, anchor_binding=binding, anchor=anchor,
        candidate_iqk="1", anchor_iqk="0", output_root=root, host_state=host_state, identity=identity)
    originals[label] = store.write("direct-historical-microbench", json.loads(
        (root / "raw" / (label + ".json")).read_text())).to_dict()
    closed = store.write("direct-historical-boundary", rw.boundary(config, held, marker="close"))
    (root / "close").mkdir()
    lc._write_preflight(root / "close", instrument_sha=instrument_sha,
        copy_sha=lc._sha256_file(lc.INSTRUMENT_BINARY), host_state=microbench.read_host_state)
    return {"identity": asdict(identity), "source_sha256": source_sha,
        "source_tree_hex": source.hex(), "instrument_sha256": instrument_sha,
        "linkage_sha256": linkage_sha, "linkage_text": linkage_text,
        "anchor": anchor.to_dict(), "t0": t0.to_dict(), "raw": originals,
        "stopping_commitment": commitment.to_dict(),
        "boundaries": {"open": opened.to_dict(), "close": closed.to_dict()},
        "claim_footprint": {key: held[key] for key in ("device_id", "cpu_list", "regions")},
        "claim_observations": adapter.observations,
        "preflights": [store.write("direct-historical-preflight", json.loads(path.read_text())).to_dict()
                       for path in (root / "preflight.json", root / "close" / "preflight.json")]}


def _window(store, material, *, anchor, solve, rule, commitment, raw_ref):
    """Reduce original owner observations; no live attestation during replay."""
    from ..resource import preflight
    from . import search_window as sw
    boundaries = [_read(store, "direct-historical-boundary", material["boundaries"][name])
                  for name in ("open", "close")]
    claims = [row["claim"] for row in boundaries]
    first = claims[0]
    same = all(row["owner_pid"] == first["owner_pid"] and
        [(item["path"], item["device"], item["inode"]) for item in row["locks"]] ==
        [(item["path"], item["device"], item["inode"]) for item in first["locks"]]
        for row in claims)
    preflights = []
    for index, boundary in enumerate(boundaries):
        expected = "open" if index == 0 else "close"
        if boundary["snapshot"]["marker"] != expected:
            raise HistoricalControlRefused("original historical boundary marker differs")
        result = sw.captured_preflight(boundary["snapshot"], self_pid=first["owner_pid"],
            scope=preflight.PreflightScope(label="original direct CPU serving window",
                cpu_regions=frozenset(material["claim_footprint"]["regions"]), protocol_id="P-AK-SEARCH-1"))
        if result.to_dict() != boundary["preflight"]:
            raise HistoricalControlRefused("original historical preflight does not rederive")
        preflights.append(result.as_check())
    originals = [_read(store, "direct-historical-microbench", ref) for ref in material["raw"].values()]
    # The native runner owns which host observations act. Idle frequency is
    # legitimately deferred; its during-load check must pass before complete.
    # Reapplying worst-of to diagnostic checks would create a new host gate.
    host = _check(all(row["complete"] is True and not row["refusals"]
        for original in originals for row in (original, *original["blocks"])),
        "all original native legs completed their owning host/claim checks")
    identity_checks = []
    for ref in material["preflights"]:
        original = _read(store, "direct-historical-preflight", ref)
        identity_checks.extend(schemas.Check(item["outcome"], tuple(item["reasons"]))
                               for item in original["checks"].values())
    source_ok = _check(hashlib.sha256(bytes.fromhex(material["source_tree_hex"])).hexdigest()
        == material["source_sha256"], "original measured instrument source tree")
    identity_checks.append(source_ok)
    identity = schemas.Check.worst_of(identity_checks)
    historical = _read(store, "direct-historical-microbench", material["raw"]["historical_win_replay"])
    blocks = [lc._paired_block_from_raw(row["paired_block"]) for row in historical["blocks"]]
    control_bootstrap = schemas.Check(schemas.PASS,
        ("internal evaluator-control bootstrap, never a candidate control panel",))
    panel = api.ControlPanel(*(control_bootstrap for _ in range(5)))
    current_anchors = tuple(st.median(block.anchor_samples) for block in blocks)

    def storage_check(boundary):
        value = boundary["snapshot"]["storage"]
        return _check(bool(value) and value["free_bytes"] >= 200 * 1024 ** 3,
                      "original historical storage floor", unavailable=not value)

    return api.WindowAttestations(resource_claim_receipt=raw_ref,
        resource_claim_open=_check(claims[0]["status"] == "held", "original claim open"),
        resource_claim_close=_check(claims[1]["status"] == "held", "original claim close"),
        resource_claim_same_holder=_check(same and bool(first["locks"]), "original same holder and physical locks"),
        no_concurrent_inference=schemas.Check.worst_of(preflights), preflight_attestation_ref=raw_ref,
        host_receipt=raw_ref, host_health=host,
        anchor_at_open=anchor if identity.outcome == schemas.PASS else None,
        anchor_at_close=anchor if identity.outcome == schemas.PASS else None,
        anchor_gate=st.anchor_gate_check(current_anchors, band=solve.outputs, b_min=solve.outputs.b_min_blocks),
        evaluator_bundle=source_ok, runtime_source_label=identity,
        recipe=api.RecipeReceipt(historical["candidate_receipt"]["constructor_id"],
            historical["candidate_receipt"]["constructor_sha256"],
            historical["candidate_receipt"]["argv_sha256"]),
        storage_open=storage_check(boundaries[0]), storage_close=storage_check(boundaries[1]),
        strata=_check(all(block.stratum == api.STRATUM_SELECTION for block in blocks), "original selection stratum"),
        stopping_rule_id=rule.rule_id,
        rule_immutability=commitment.verify(rule),
        order_randomized=schemas.Check(historical["order_control"]["outcome"], tuple(historical["order_control"]["reasons"])),
        order_seed=material["identity"]["campaign_id"],
        aa_cadence=control_bootstrap, controls=panel,
        calibration=_check(solve.accepted, "original Qwen A/A and neutral calibration solve"),
        control_definitions_immutable=controls.verify_control_definitions(), raw_evidence_ref=raw_ref)


def _compose(store, declaration, material, raw_ref):
    identity = lc.LiveCampaignIdentity(**material["identity"])
    root = Path(identity.evidence_ref)
    if identity.campaign_id != declaration["control_campaign_id"] or root != Path(declaration["root"]):
        raise HistoricalControlRefused("original historical campaign/output identity differs")
    materials = {}
    def same_instrument(raw):
        if raw.get("anchor_identity") != material["anchor"]:
            raise HistoricalControlRefused("original historical raw anchor differs from its measured instrument")
        for arm in ("candidate_receipt", "anchor_receipt"):
            receipt = raw[arm]
            expected = {"binary_path": declaration["frame"]["instrument_binary"],
                "binary_sha256": material["instrument_sha256"],
                "source_root": declaration["frame"]["instrument_root"],
                "library_path": str(Path(declaration["frame"]["instrument_binary"]).parent)}
            if any(receipt.get(key) != value for key, value in expected.items()):
                raise HistoricalControlRefused("original historical raw arm differs from its measured instrument")
    for label, count in (("aa_calibration", lc.CALIBRATION_BLOCKS), ("neutral_calibration", lc.NEUTRAL_BLOCKS)):
        loaded, raw = lc._load_recorded_material(root, identity=identity, label=label,
            expected_blocks=count, prompt=512, candidate_iqk="0", anchor_iqk="0")
        if raw != _read(store, "direct-historical-microbench", material["raw"][label]):
            raise HistoricalControlRefused("original historical raw vector differs from its retained artifact")
        same_instrument(raw)
        materials[label] = loaded
    declared, rule, construction, split, solve = lc._campaign_inputs(
        materials["aa_calibration"], materials["neutral_calibration"], identity)
    solve.require_accepted()
    historical, raw = lc._load_recorded_material(root, identity=identity, label="historical_win_replay",
        expected_blocks=rule.max_total_blocks(solve.outputs.b_min_blocks), prompt=512,
        candidate_iqk="1", anchor_iqk="0")
    if raw != _read(store, "direct-historical-microbench", material["raw"]["historical_win_replay"]):
        raise HistoricalControlRefused("original historical control raw vector changed")
    same_instrument(raw)
    commitment = _stopping_commitment(store, declaration, rule,
        reference=material["stopping_commitment"], measured_at=raw["started_at"])
    statistics = st.CampaignStatistics(campaign_id=identity.campaign_id, campaign_seed=identity.campaign_seed,
        effect_scale=st.EFFECT_SCALE_RELATIVE, hypothesis=st.HYPOTHESIS_IMPROVEMENT, margin=0.0,
        stopping_rule=rule, stopping_rule_commitment=commitment,
        split_rule=split, construction=construction, calibration=solve.outputs,
        aa_effect_pool=solve.aa_effect_pool, anchor_calibration_values=solve.anchor_calibration_values)
    anchor_row = dict(material["anchor"])
    anchor_row["measurement_event_ids"] = tuple(anchor_row["measurement_event_ids"])
    anchor = api.AnchorIdentity(**anchor_row)
    fixture = lc._fixture(controls.CONTROL_HISTORICAL_WIN_REPLAY, historical.blocks, tier="T2",
        source_sha=material["source_sha256"], binary_sha=material["instrument_sha256"],
        linkage_sha=material["linkage_sha256"], measured_at=raw["ended_at"])
    fixtures = cr.resolve_fixture_set(fixtures=(fixture,),
        pinned_digest=_digest(cr._fixture_payload((fixture,))), source_label=raw_ref)
    binding = cr.CampaignBinding(identity.campaign_id, "llama_cpu", "prefill", recipes.CELL_CLASS_TINY_GRAPH,
        api.PROTOCOL_VERSIONED_ID, api.EvaluatorIdentity("P-AK-SEARCH-1/v1", _digest(declaration["source"]), raw_ref),
        api.ScopeDenominator("full", (), (), 96), _digest({"cpu_list": lc.CPU_LIST}), "single",
        recipes.get_recipe(lc.PREFILL_RECIPE_ID).metric, "higher_better", lc.CALIBRATION_REPS,
        "parameter", anchor, declared, solve.outputs)
    t0_body, original_gates = _t0_gates(store, material["t0"], declaration)
    if t0_body["source_sha256"] != material["source_sha256"]:
        raise HistoricalControlRefused("original T0 and measured control source differ")
    class Gates:
        def run_gates(self, request):
            return original_gates

    stages = []
    dispatcher = api.TierDispatcher(gate_runners={tier: Gates() for tier in ("T0", "T1", "T2")})
    window = _window(store, material, anchor=anchor, solve=solve, rule=rule,
                     commitment=commitment, raw_ref=raw_ref)
    reducer = st.PairedBlockReducer(statistics)

    class Pipeline:
        def evaluate(self, submission):
            for tier in ("T0", "T1", "T2"):
                request = replace(submission.request, tier=tier, anchor_tier=tier)
                effect = None if tier == "T0" else reducer.reduce_blocks(request, submission.blocks)
                outcome = dispatcher.dispatch(request, submission.window, effect=effect)
                stages.append(outcome.durable_payload)
                if outcome.verdict.status != api.STATUS_PASS:
                    return outcome
            return outcome

    runner = cr.ExecutedControlRunner(pipeline=Pipeline(), fixtures=fixtures, binding=binding,
                                      campaign_statistics=statistics)
    definition = next(row for row in controls.CONTROL_DEFINITIONS
                      if row.control_id == controls.CONTROL_HISTORICAL_WIN_REPLAY)
    runner.open_window(window_id=identity.window_id, window=window)
    try:
        observation = runner.run_control(definition, controls.ControlRunContext(identity.campaign_id,
            "llama_cpu", "prefill", recipes.CELL_CLASS_TINY_GRAPH, identity.window_id, "T2",
            identity.campaign_seed, anchor, _resolution(store, declaration["resolution"]).declaration))
    finally:
        runner.close_window()
    return observation, stages


def run_or_reopen(*, store, held_claim, campaign_id, window_index, reference=None,
                  deadline_monotonic_s=None):
    """Installed runtime consumer entrypoint; replay never executes work."""
    if type(store) is not ArtifactStore or type(held_claim) is not HeldCpuClaim:
        raise TypeError("historical control requires the original store and acquired CPU context")
    if type(campaign_id) is not str or not campaign_id.startswith("ak-"):
        raise HistoricalControlRefused("historical control requires the original campaign identity")
    if type(window_index) is not int or window_index < 0:
        raise HistoricalControlRefused("historical control window index must be nonnegative")
    key = _digest({"campaign_id": campaign_id, "window_index": window_index})
    if reference is not None:
        result = _read(store, NAMESPACE, reference)
        declaration = _read(store, "direct-historical-declaration", result["declaration"])
        if (result["schema"] != SCHEMA or declaration["parent_campaign_id"] != campaign_id
                or declaration["window_index"] != window_index):
            raise HistoricalControlRefused("historical receipt belongs to another original campaign/window")
        if declaration["frame"] != _frame() or declaration["source"] != source_identity():
            raise HistoricalControlRefused("original historical source/frame differs; no receipt relabeling")
        resolution = _resolution(store, declaration["resolution"])
        if result["error"] is not None:
            observation = controls.ControlObservation(controls.CONTROL_HISTORICAL_WIN_REPLAY,
                False, could_not_run_reason=result["error"], evidence_ref=_ref(StoredArtifact(**result["declaration"])))
            if observation.to_dict() != result["observation"]:
                raise HistoricalControlRefused("original historical refusal differs")
            return resolution, observation, StoredArtifact(**reference)
        original = _read(store, "direct-historical-material", result["material"])
        observation, stages = _compose(store, declaration, original, _ref(StoredArtifact(**result["material"])))
        if observation.to_dict() != result["observation"] or stages != result["tier_evaluations"]:
            raise HistoricalControlRefused("original historical observation does not rederive")
        return resolution, observation, StoredArtifact(**reference)
    root = store.root / ("historical-control-" + key)
    declaration = {"schema": SCHEMA, "parent_campaign_id": campaign_id, "window_index": window_index,
        "control_campaign_id": "ak-historical-" + key[:32], "root": str(root),
        "frame": _frame(), "source": source_identity(), "resolution": _resolve(store).to_dict(),
        "deadline_monotonic_s": deadline_monotonic_s,
        "issued_at": datetime.now(timezone.utc).isoformat()}
    if root.exists():
        raise HistoricalControlRefused("historical control already issued; reopen its original reference, never rerun")
    declared = store.write("direct-historical-declaration", declaration)
    root.mkdir(exist_ok=False)
    resolution = _resolution(store, declaration["resolution"])
    material_ref, stages, error = None, [], None
    try:
        if not resolution.available:
            raise HistoricalControlRefused(resolution.reason())
        material = _collect(store, held_claim, declaration, root, deadline_monotonic_s)
        material_ref = store.write("direct-historical-material", material)
        observation, stages = _compose(store, declaration, material, _ref(material_ref))
    except tp.ProcessEscaped:
        raise  # Possibly live owned work cannot be converted into a retryable control result.
    except (HistoricalControlRefused, OSError, st.CalibrationFailed,
            tp.InstrumentCapabilityError, tp.AnchorCaptureIncomplete) as exc:
        error = f"{type(exc).__name__}: {exc}"
        observation = controls.ControlObservation(controls.CONTROL_HISTORICAL_WIN_REPLAY,
            False, could_not_run_reason=error, evidence_ref=_ref(declared))
    result = store.write(NAMESPACE, {"schema": SCHEMA, "declaration": declared.to_dict(),
        "material": None if material_ref is None else material_ref.to_dict(), "error": error,
        "observation": observation.to_dict(), "tier_evaluations": stages})
    return resolution, observation, result
