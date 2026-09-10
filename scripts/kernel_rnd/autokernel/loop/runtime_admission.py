"""Existing-loop runtime-only selection, using the original evaluator policy.

This owns no resource grants, source promotion, builder, or alternative grader.
The loop supplies its actual held context and enrolled request/build. Every
admission is rederived from the original calibration, HTTP and control records.
"""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import statistics
import time
from types import SimpleNamespace

from .. import campaign, schemas
from ..evaluator import api, controls, correctness, statistics as st
from ..execution.cpu_region_claim import parse_cpu_list
from ..resource import preflight
from . import gates, observation_binding as ob, runtime_calibration as rc
from . import runtime_window as rw, search_window as sw, serial_run, status
from .measurement_capture import StoredArtifact
from .resolved_recipe import CanonicalResolvedRecipe
from .serving_preparation import PreparationArmPair
from .unified_planner import RuntimeArmPair

SCHEMA = "epyc.autokernel.direct_runtime_admission.v1"


def _ref(value):
    return f"{value.locator}#sha256={value.sha256}"


def _check(condition, reason, *, unavailable=False):
    return schemas.Check(schemas.PASS if condition else
        schemas.COULD_NOT_CHECK if unavailable else schemas.FAIL, (reason,))


def source_identity():
    # Whole-file pins include supporting callables, not just their callers' names.
    from . import native_server_response, resolved_recipe, serving, serving_beliefs
    from .run import _cpu_arm, _rebind_build_dso
    modules = (api, controls, correctness, st, rc, rw, gates,
               native_server_response, resolved_recipe, serving, serving_beliefs)
    paths = (Path(__file__), *(Path(module.__file__) for module in modules))
    from . import lifecycle_observation as lo
    return {"files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "loaded": [lo.callable_identity(function) for function in (
            _check, _outputs, _window_checks, source_snapshot, DirectGates.run_gates,
            RuntimeAdmission.compare, RuntimeAdmission._compare_attempt, RuntimeAdmission._observation,
            RuntimeAdmission.calibration, RuntimeAdmission._evaluate_pair,
            RuntimeAdmission._controls, RuntimeAdmission.reopen_admission,
            RuntimeAdmission.retain, RuntimeAdmission.selected, RuntimeAdmission.selection_reference,
            RuntimeAdmission.retained_build, restore_selection, _cpu_arm, _rebind_build_dso)]}


def source_snapshot(worktree, commit, store):
    """SHA256 of the actual Git tree manifest, not SHA256 of a commit label."""
    result = subprocess.run(["git", "-C", str(worktree), "ls-tree", "-r", "-z", commit],
                            capture_output=True, check=True, timeout=30)
    if not result.stdout or len(result.stdout) > 16 * 1024 * 1024:
        raise rc.RuntimeCalibrationRefused("original source tree manifest unavailable or exceeds 16 MiB")
    return store.write("direct-source-snapshot", {"commit": commit,
        "tree_manifest_hex": result.stdout.hex(),
        "tree_manifest_sha256": hashlib.sha256(result.stdout).hexdigest()})


def _anchor(recipe, commit, measurements=()):
    return api.AnchorIdentity(commit, recipe.executable.sha256,
        rc._digest([row.to_dict() for row in recipe.dsos]),
        measurement_event_ids=measurements, tool="llama-server")


def _read(store, namespace, reference):
    artifact = StoredArtifact(**reference)
    body = ob._plain(store.read(artifact.locator, artifact.sha256))
    if store.verify(namespace, body) != artifact:
        raise rc.RuntimeCalibrationRefused("original " + namespace + " namespace differs")
    return body


def _outputs(rows, arm):
    """Original per-prompt repeated token/content vectors; timing is not output."""
    values = {}
    for kind, _index, actual_arm, _body, raw in rows:
        if kind != "pair" or actual_arm != arm:
            continue
        for row in raw:
            if row["raw"]["phase"] != "measurement":
                continue
            response = row["response"]
            tokens, content = response.get("tokens"), response.get("content")
            if type(tokens) is list and tokens and all(type(token) is int for token in tokens):
                payload = {"tokens": tokens, "content": content}
                length = len(tokens)
            elif type(content) is str and content:
                payload, length = {"content": content}, len(content.encode())
            else:
                payload, length = None, 0
            values.setdefault(row["raw"]["prompt_id"], []).append({
                "digest": None if payload is None else rc._digest(payload), "length": length,
                "tokens": tokens, "request": ob._plain(row["request"]),
                "generated": response.get("timings", {}).get("predicted_n"),
                "reference": ob._plain(row["artifact"])})
    return values


class DirectGates:
    """Applicable original runtime gates; no synthetic source-build T0 report."""
    def __init__(self, *, rows, operations, raw_ref, original_prompts):
        self.rows, self.operations = rows, operations
        self.raw_ref, self.original_prompts = raw_ref, original_prompts

    def run_gates(self, request):
        policy = campaign.HostOps._t0_evaluator_policy(SimpleNamespace(backend=request.backend))
        result = []
        for row in self.operations:
            value = row["verdict"]
            result.append(api.GateResult("direct_" + row["arm"] + "_" + row["op"],
                api.GATE_CORRECTNESS, _check(value["passed"], value["detail"],
                    unavailable=value["gate"] == "oracle_unavailable"), evidence_ref=self.raw_ref))
        anchor, candidate = _outputs(self.rows, "anchor"), _outputs(self.rows, "candidate")
        for prompt in self.original_prompts.prompts:
            a, c = anchor.get(prompt.prompt_id, ()), candidate.get(prompt.prompt_id, ())
            original = json.loads(prompt.body)
            work = bool(a and c) and all(row["request"] == original
                and row["generated"] == original["n_predict"] for row in (*a, *c))
            result.append(api.GateResult("direct_work_" + prompt.prompt_id, api.GATE_CORRECTNESS,
                _check(work, "original request/cache/seed/work count must be unchanged"),
                requires_anchor=True, evidence_ref=self.raw_ref))
            aa = tuple(row["digest"] for row in a if row["digest"] is not None)
            cc = tuple(row["digest"] for row in c if row["digest"] is not None)
            stable = "bitwise_stable" if len(aa) >= 2 and len(set(aa)) == 1 else "bitwise_unstable"
            evidence = None
            seed = original.get("seed")
            if type(seed) is int and seed >= 0 and len(aa) == len(a) and len(cc) == len(c) and min(len(a), len(c)) >= 2:
                evidence = correctness.DeterminismEvidence(seed, len(cc), cc, aa, stable,
                    request.anchor.source_commit, request.anchor.binary_sha256,
                    request.anchor.linkage_sha256, False, None, self.raw_ref, "evaluator")
            prompt_request = request if evidence is None else replace(request, determinism=
                api.DeterminismReport(evidence.measured_class(), evidence.runs))
            gate, _properties = correctness.check_determinism_class(prompt_request, evidence, policy)
            result.append(replace(gate, gate_id=gate.gate_id + ":" + prompt.prompt_id))
            # Check every original output, not only the first pair. Tolerance is
            # available only through the owning reducer and actual token vectors.
            for index, row in enumerate(c):
                baseline = a[index] if index < len(a) else None
                coherence = None
                if baseline is not None:
                    tokens_a, tokens_c = baseline["tokens"], row["tokens"]
                    agreement, divergence = None, None
                    if isinstance(tokens_a, list) and tokens_a and isinstance(tokens_c, list):
                        agreement = sum(x == y for x, y in zip(tokens_a, tokens_c)) / max(len(tokens_a), len(tokens_c))
                        divergence = next((n for n, (x, y) in enumerate(zip(tokens_a, tokens_c)) if x != y), None)
                    coherence = correctness.CoherenceEvidence(row["digest"], row["length"],
                        baseline["digest"], baseline["length"], "original frozen sampler",
                        original.get("temperature") == 0 or original.get("top_k") == 1,
                        seed, original["n_predict"], agreement, divergence, stable,
                        request.anchor.source_commit, request.anchor.binary_sha256,
                        request.anchor.linkage_sha256, prompt.prompt_id, self.raw_ref, "evaluator")
                gate, _verdict = correctness.check_output_coherence(request, coherence, policy, evidence)
                result.append(replace(gate, gate_id=f"{gate.gate_id}:{prompt.prompt_id}:{index}"))
            if not c:
                gate = correctness.check_output_coherence(request, None, policy)[0]
                result.append(replace(gate, gate_id=gate.gate_id + ":" + prompt.prompt_id + ":missing"))
        return tuple(result)


def _window_checks(window, rows, reduction, panel, anchor, raw_ref):
    opened = _read(window.store, "direct-window-boundary", window.boundaries["open"])
    closed = _read(window.store, "direct-window-boundary", window.boundaries["close"])
    for name, boundary in (("open", opened), ("close", closed)):
        if boundary["window"] != window.identity or boundary["snapshot"]["marker"] != name:
            raise rc.RuntimeCalibrationRefused("original window boundary identity differs")
    claims = [opened["claim"], closed["claim"]]
    same = claims[0]["owner_pid"] == claims[1]["owner_pid"] and [
        (r["path"], r["device"], r["inode"]) for r in claims[0]["locks"]] == [
        (r["path"], r["device"], r["inode"]) for r in claims[1]["locks"]]
    physical, health = [], []
    for _kind, _index, arm, body, raw in rows:
        facts = body["during_work"]
        health.append(rw.health(facts))
        measurements = [r for r in raw if r["raw"]["phase"] == "measurement"]
        start = min(r["raw"]["started_monotonic_s"] for r in measurements)
        end = max(r["raw"]["ended_monotonic_s"] for r in measurements)
        samples = [r["snapshot"] for r in facts["samples"] if r["phase_contained"]
            and start <= r["snapshot"]["started"] <= r["snapshot"]["ended"] <= end]
        gap = window.window_config.limits.max_gap_s
        coverage = len(samples) >= 2 and samples[0]["started"] - start <= gap and end - samples[-1]["ended"] <= gap
        coverage = coverage and all(b["started"] - a["ended"] <= gap for a, b in zip(samples, samples[1:]))
        health.append(_check(coverage, "original during-request frequency coverage", unavailable=True))
        recipe = getattr(window.pair, arm)
        maps = []
        for sample in samples:
            loaded = sample.get("loaded_maps", {})
            target = loaded.get("target", {})
            text = sw._text(loaded.get("file"))
            stable_pid = (target.get("pid") == body["observations"][0]["process_pid"]
                and target.get("start_ticks") == loaded.get("before_start_ticks") == loaded.get("after_start_ticks"))
            maps.append(stable_pid and text is not None and all(item.path in text for item in recipe.dsos))
        physical.append(_check(len(maps) >= 2 and all(maps), "original live PID and exact declared DSO maps", unavailable=True))
    actual_artifacts = all(row["stable"] and row["error"] is None
        and row["sha256"] == row["expected"]["sha256"] for row in (*opened["artifacts"], *closed["artifacts"]))
    identities = _check(actual_artifacts and bool(opened["artifacts"]), "original executable/DSO identities at open and close")
    physical.append(identities)
    preflights = []
    for boundary in (opened, closed):
        original = sw.captured_preflight(boundary["snapshot"], self_pid=boundary["claim"]["owner_pid"],
            scope=preflight.PreflightScope(label="original direct CPU serving window",
                cpu_regions=frozenset(window.held_claim["regions"]), protocol_id="P-AK-SEARCH-1"))
        if original.to_dict() != boundary["preflight"]:
            raise rc.RuntimeCalibrationRefused("original preflight does not rederive")
        preflights.append(original.as_check())
    def storage(boundary):
        value = boundary["snapshot"]["storage"]
        return _check(bool(value) and value["free_bytes"] >= window.statistical.controls.storage_floor_bytes_free,
                      "original free storage meets declared floor", unavailable=not value)
    anchors = tuple(body["value"] for kind, _index, _arm, body, _raw in rows if kind == "anchor_gate")
    source_ok = _check(window.frame["calculation_source"] == rc.calculation_source()
        and window.frame["host_window_source"] == rw.source_identity(), "original loaded calculation/window source")
    return api.WindowAttestations(resource_claim_receipt=raw_ref,
        resource_claim_open=_check(claims[0]["status"] == "held", "original claim observed at open"),
        resource_claim_close=_check(claims[1]["status"] == "held", "original claim observed at close"),
        resource_claim_same_holder=_check(same, "same actual holder and physical locks"),
        no_concurrent_inference=schemas.Check.worst_of(preflights), preflight_attestation_ref=raw_ref,
        host_receipt=raw_ref, host_health=schemas.Check.worst_of(health),
        anchor_at_open=anchor if identities.outcome == schemas.PASS else None,
        anchor_at_close=anchor if identities.outcome == schemas.PASS else None,
        anchor_gate=st.anchor_gate_check(anchors, band=window.outputs, b_min=window.outputs.b_min_blocks),
        evaluator_bundle=source_ok, runtime_source_label=schemas.Check.worst_of([source_ok, *physical]),
        recipe=api.RecipeReceipt("direct-canonical-runtime/v1", rc._digest(window.pair.to_dict()),
            rc._digest([list(window.pair.anchor.command_argv), list(window.pair.candidate.command_argv)])),
        storage_open=storage(opened), storage_close=storage(closed),
        stopping_rule_id=window.statistical.stopping_rule.rule_id,
        order_seed=window.statistical.campaign_seed, raw_evidence_ref=raw_ref,
        **reduction.window_checks, **panel)


class RuntimeAdmission:
    def __init__(self, *, store, held_claim, campaign_id, epoch, original, prompts,
                 statistical, host_state, worktree, source_commit, escalation=None,
                 deadline_monotonic_s=None):
        self.store, self.held_claim = store, held_claim
        self.deadline_monotonic_s = deadline_monotonic_s
        if type(campaign_id) is not str or not campaign_id.startswith("ak-"):
            raise rc.RuntimeCalibrationRefused("prospective runtime campaign_id must use owning 'ak-' grammar; do not rename old records")
        self.campaign_id, self.epoch, self.prompts = campaign_id, epoch, prompts
        self.statistical, self.host_state = statistical, host_state
        self.worktree, self.source_commit, self.escalation = worktree, source_commit, escalation
        self.original = original
        self.owner_inputs = {"campaign_id": campaign_id, "epoch": epoch,
            "original": original.to_dict(), "prompts": prompts.to_dict(),
            "statistical": statistical.to_dict(), "host_state": host_state,
            "worktree": str(worktree), "source_commit": source_commit}
        self.scope = rc._digest({"campaign": campaign_id, "epoch": epoch,
            "recipe": original.to_dict(), "prompts": prompts.to_dict(), "statistics": statistical.to_dict()})
        self.state_name = f"runtime-selection-{self.scope}.json"
        path = store.root / self.state_name
        if path.exists():
            self.state, _sha = serial_run._json(path, limit=128 * 1024)
            if set(self.state) != {"schema", "scope", "attempts", "selected"} or self.state["schema"] != SCHEMA or self.state["scope"] != self.scope:
                raise rc.RuntimeCalibrationRefused("runtime selection belongs to another original frame")
        else:
            self.state = {"schema": SCHEMA, "scope": self.scope, "attempts": [], "selected": None}
            self._checkpoint()
        if len(self.state["attempts"]) > statistical.controls.max_candidates:
            raise rc.RuntimeCalibrationRefused("runtime candidate allocation exceeds original budget")
        self._frames = {}
        self._windows = {}
        self._historical = {}
        self._issued = {}
        self.default = store.write("direct-runtime-default", {
            "schema": "epyc.autokernel.direct_runtime_default.v1", "scope": self.scope,
            "campaign_id": campaign_id, "epoch": epoch, "recipe": original.to_dict(),
            "prompt_manifest_digest": prompts.digest, "statistics": statistical.to_dict(),
            "selection": "original_default", "qualified": False})

    def _checkpoint(self):
        status.write_json(self.store.root, self.state_name, self.state, prefix=".runtime-selection-")

    def calibration(self, anchor, *, reopen_only=False):
        if anchor.snapshot_digest not in self._frames:
            self._frames[anchor.snapshot_digest] = rc.DirectCalibration(store=self.store,
                held_claim=self.held_claim, campaign_id=self.campaign_id, epoch=self.epoch,
                anchor=anchor, neutral=rc.neutral_material(store=self.store, anchor=anchor),
                prompts=self.prompts, statistical=self.statistical, host_state=self.host_state,
                deadline_monotonic_s=self.deadline_monotonic_s)
        frame = self._frames[anchor.snapshot_digest]
        if reopen_only and frame.solution is None:
            raise rc.RuntimeCalibrationRefused("retained original calibration is incomplete; no replay launch")
        reference = frame.collect()
        frame.reopen(reference).require_accepted()
        health = frame.validity()
        if health.outcome != schemas.PASS:
            raise rc.RuntimeCalibrationRefused("original calibration host validity is " + health.outcome
                + ": " + "; ".join(health.reasons))
        return frame, reference

    def _evaluate_pair(self, pair, candidate_id, *, panel, stratum="selection", reopening=None):
        frame, reference = self.calibration(pair.anchor, reopen_only=reopening is not None)
        key = (rc._digest(pair.to_dict()), candidate_id, stratum)
        if reopening is None:
            if key not in self._windows:
                if self.deadline_monotonic_s is not None and time.monotonic() >= self.deadline_monotonic_s:
                    raise rc.RuntimeLaunchBudgetExhausted("original invocation launch deadline reached; completed setup retained")
                window = rc.DirectPairWindow(calibration=frame, calibration_reference=reference,
                                             pair=pair, candidate_id=candidate_id, stratum=stratum)
                if window._reopened:
                    raise rc.RuntimeCalibrationRefused("original pair window already exists; no replay launch")
                operations = [{"arm": arm, "op": op, "verdict": asdict(gates.op_correctness(
                    Path(getattr(pair, arm).build_dir), op=op, backend="CPU", resolved_recipe=getattr(pair, arm)))}
                    for arm in ("anchor", "candidate") for op in correctness.MANDATORY_BACKEND_OPS]
                source = source_snapshot(self.worktree, self.source_commit, self.store)
                self._windows[key] = window, operations, source
            window, operations, source = self._windows[key]
        else:
            window = rc.DirectPairWindow(calibration=frame, calibration_reference=reference,
                                         pair=pair, candidate_id=candidate_id, stratum=stratum)
            operations, source = reopening["operations"], StoredArtifact(**reopening["source_snapshot"])
        sequence = st.SequentialEvaluation(rule=self.statistical.stopping_rule,
            commitment=self.statistical.commitment, construction=window.statistics.construction,
            b_min=window.outputs.b_min_blocks, threshold=window.outputs.threshold_for(stratum),
            hypothesis=self.statistical.hypothesis, margin=self.statistical.margin,
            metric_direction="higher_better", effect_scale=self.statistical.effect_scale,
            order_schedule=window.schedule)
        count = window.outputs.b_min_blocks
        while True:
            anchors, blocks, rows = window.reopen(count) if reopening else window.collect_to(count)
            for block in blocks[len(sequence.blocks):]:
                sequence.next_block_request()
                sequence.submit_block(block)
            if sequence.terminal:
                break
            count += self.statistical.stopping_rule.extension.blocks_per_round
        if reopening is None:
            window.finish()
        source_body = _read(self.store, "direct-source-snapshot", source.to_dict())
        if source_body["commit"] != self.source_commit:
            raise rc.RuntimeCalibrationRefused("original source snapshot names another commit")
        anchor = _anchor(pair.anchor, self.source_commit, tuple(
            _ref(StoredArtifact(**row["artifact"])) for row in window.launches
            if row["membership"][2] == "anchor"))
        raw = {"schema": SCHEMA, "window": window.declaration.to_dict(), "count": count,
            "operations": operations, "source_snapshot": source.to_dict(),
            "boundaries": window.boundaries, "launches": window.launches,
            "source": source_identity(), "stop": sequence.decide().to_dict()}
        original = self.store.verify("direct-runtime-window-result", raw) if reopening else self.store.write("direct-runtime-window-result", raw)
        if reopening is not None and raw != reopening:
            raise rc.RuntimeCalibrationRefused("original runtime evidence/source/stopping rule differs")
        outputs = _outputs(rows, "candidate")
        complete = bool(outputs) and all(row["digest"] is not None for values in outputs.values() for row in values)
        count_runs = min((len(values) for values in outputs.values()), default=0) if complete else 0
        vectors = tuple(rc._digest([outputs[name][i]["digest"] for name in sorted(outputs)]) for i in range(count_runs))
        # One repeat is the whole original request vector, not individual slots.
        determinism = api.DeterminismReport("not_measured" if count_runs < 2 else
            "bitwise_stable" if len(set(vectors)) == 1 else "bitwise_unstable", count_runs)
        request = api.EvaluationRequest("ake-" + original.sha256, self.campaign_id, candidate_id,
            "T1", "llama_cpu", "decode", "serving", api.PROTOCOL_VERSIONED_ID,
            api.ArtifactIdentity(source_body["tree_manifest_sha256"], pair.candidate.executable.sha256,
                rc._digest([row.to_dict() for row in pair.candidate.dsos])), anchor,
            api.EvaluatorIdentity("direct-runtime/v1", rc._digest(source_identity()), _ref(original)),
            api.ScopeDenominator("partial", (), ("cpu",), len(parse_cpu_list(
                pair.anchor.template.cpu_list or self.held_claim["cpu_list"]))),
            rc._digest(window.frame), "single", determinism,
            "aggregate_tok_s", "higher_better", 1, "parameter", "T1", (),
            rows[-1][3]["ended_at"], self.statistical.controls, window.outputs)
        reduction = st.PairedBlockReducer(window.statistics).reduce(request, blocks, raw_samples_ref=_ref(original))
        attestations = _window_checks(window, rows, reduction, panel, anchor, _ref(original))
        runner = DirectGates(rows=rows, operations=operations, raw_ref=_ref(original), original_prompts=self.prompts)
        outcome = api.TierDispatcher(gate_runners={"T1": runner}).dispatch(request, attestations, effect=reduction.estimate)
        # Preserve the emitted per-case vector, not just the summary verdict.
        event = {"original_window": original.to_dict(), "event": outcome.event,
                 "evaluation": outcome.durable_payload}
        if reopening is None:
            self.store.write("direct-runtime-evaluation", event)
        else:
            self.store.verify("direct-runtime-evaluation", event)
        return original, outcome, reduction

    def _controls(self, anchor, index, *, reopening=None):
        owner = self
        try:
            from . import direct_historical_control
        except ImportError as exc:
            raise rc.RuntimeCalibrationRefused("original historical control runner is not installed") from exc
        frame, reference = self.calibration(anchor, reopen_only=reopening is not None)
        solved = frame.reopen(reference)
        declared = self.statistical.commitment.committed_at
        completed = (reopening if reopening is not None else
                     self.state["attempts"][index].setdefault("controls", {}))
        historical_reference = completed.get(controls.CONTROL_HISTORICAL_WIN_REPLAY)
        if reopening is not None or index not in self._historical:
            historical_result = direct_historical_control.run_or_reopen(
                store=self.store, held_claim=self.held_claim, campaign_id=self.campaign_id,
                window_index=index, reference=historical_reference,
                deadline_monotonic_s=self.deadline_monotonic_s)
            if reopening is None:
                self._historical[index] = historical_result
        else:
            historical_result = self._historical[index]
        historical, historical_observation, historical_ref = historical_result
        if reopening is None:
            completed[controls.CONTROL_HISTORICAL_WIN_REPLAY] = historical_ref.to_dict()
            self._checkpoint()
        bundle = controls.resolve_control_bundle(pinned_definitions_digest=controls.CONTROL_DEFINITIONS_DIGEST,
            aa_cadence=controls.AACadence(1, 3600, declared),
            seed_rotation=controls.SeedRotationSchedule(1, declared),
            historical_win_replays=() if historical.declaration is None else (historical.declaration,),
            source_label="direct original runtime controls")
        originals = {controls.CONTROL_HISTORICAL_WIN_REPLAY: historical_ref.to_dict()}
        # Cycle-breaking carrier only, as in execution.live_controls. Never put
        # this panel in a result, selected recipe, or candidate evaluation.
        provisional = schemas.Check(schemas.PASS, ("internal control-evaluation bootstrap only",))
        bootstrap = {"controls": api.ControlPanel(provisional, provisional, provisional, provisional, provisional),
            "aa_cadence": provisional, "control_definitions_immutable": bundle.reverify(
                pinned_definitions_digest=bundle.definitions_digest)}
        class Runner:
            def run_control(self, definition, context):
                if definition.control_id == controls.CONTROL_HISTORICAL_WIN_REPLAY:
                    return historical_observation
                pairs = {controls.CONTROL_POSITIVE: lambda: rc.positive_control_pair(anchor),
                    controls.CONTROL_NEUTRAL: lambda: frame.pairs["neutral"],
                    controls.CONTROL_AA: lambda: PreparationArmPair("aa", anchor, anchor, None),
                    controls.CONTROL_DEGRADED_NEGATIVE: lambda: rc.degraded_work_control(anchor, owner.prompts)}
                pair = pairs[definition.control_id]()
                retained = completed.get(definition.control_id)
                original = None if retained is None else _read(owner.store, "direct-runtime-window-result", retained)
                ref, outcome, reduction = owner._evaluate_pair(pair, "akc-control-" + context.seed,
                    panel=bootstrap, reopening=original)
                originals[definition.control_id] = ref.to_dict()
                if reopening is None:
                    completed[definition.control_id] = ref.to_dict()
                    owner._checkpoint()
                return controls.ControlObservation(definition.control_id, True, outcome.verdict,
                    abs_effects=tuple(abs(value) for value in reduction.block_effects), evidence_ref=_ref(ref))
        harness = controls.ControlHarness(bundle=bundle, runner=Runner())
        context = controls.ControlContext(self.campaign_id, "llama_cpu", "decode", "serving",
            self.scope + ":" + str(index), historical, controls.neutral_dispersion_check(solved), solved.outputs)
        observations = harness.run_all(run_context=controls.ControlRunContext(self.campaign_id,
            "llama_cpu", "decode", "serving", context.window_id, "T1", "derived by harness",
            _anchor(anchor, self.source_commit), historical.declaration), historical=historical,
            campaign_seed=self.statistical.campaign_seed, windows_completed=index)
        result = harness.evaluate(observations=observations, context=context,
            aa_cadence=_check(any(row.control_id == controls.CONTROL_AA and row.ran for row in observations),
                             "original A/A executed at this window boundary"),
            escalation=self.escalation, pinned_definitions_digest=bundle.definitions_digest,
            pinned_campaign_digest=bundle.campaign_digest)
        return result, originals

    def compare(self, pair):
        pair = RuntimeArmPair.from_dict(pair.to_dict())
        pending = next((i for i, row in enumerate(self.state["attempts"])
                        if row["result"] is None and row["pair"] == pair.to_dict()), None)
        if pending is None:
            if len(self.state["attempts"]) >= self.statistical.controls.max_candidates:
                raise rc.RuntimeCalibrationRefused("original runtime candidate budget exhausted")
            index = len(self.state["attempts"])
            candidate_id = "akc-runtime-" + rc._digest({"scope": self.scope, "index": index, "pair": pair.to_dict()})
            self.state["attempts"].append({"pair": pair.to_dict(), "candidate_id": candidate_id, "result": None})
            self._checkpoint()  # Exact original intent before any launch.
        else:
            index = pending
            candidate_id = self.state["attempts"][index]["candidate_id"]
        def continuation(*, retried=False):
            from .loop import MeasurementInvalid
            try:
                return self._compare_attempt(pair, index, candidate_id)
            except MeasurementInvalid as exc:
                if not retried:
                    # Existing iterate owns whether this one same-tail reschedule
                    # receives its next iteration budget; no hidden local retry.
                    exc.reschedule = lambda: continuation(retried=True)
                raise
        return continuation()

    def _compare_attempt(self, pair, index, candidate_id):
        panel, control_refs = self._controls(pair.anchor, index)
        if panel.panel is None:
            # Missing historical disposition is visible. Candidate observation
            # remains useful through serving.compare; no bootstrap escapes here.
            raise rc.RuntimeCalibrationRefused(panel.blocked_reason or "original measured control panel unavailable")
        attempt = self.state["attempts"][index]
        original_selection = (None if attempt.get("selection") is None else
            _read(self.store, "direct-runtime-window-result", attempt["selection"]))
        selected, outcome, reduction = self._evaluate_pair(pair, candidate_id,
            panel=controls.window_control_attestations(panel), reopening=original_selection)
        attempt["selection"] = selected.to_dict()
        self._checkpoint()
        admitted = (outcome.event is not None and not outcome.event_violations
            and not panel.halts_campaign and panel.may_rank and outcome.verdict.speed_rank_admissible
            and outcome.verdict.effect_resolution == api.EFFECT_IMPROVEMENT)
        confirmation = None
        if admitted:
            prior_confirmations = sum(bool(row.get("confirmation")) for row in self.state["attempts"])
            if (not attempt.get("confirmation")
                    and prior_confirmations >= self.statistical.controls.confirmation_admission_count):
                admitted = False
            else:
                self.state["attempts"][index]["confirmation"] = True
                self._checkpoint()
                confirmation, confirmed, _reduction = self._evaluate_pair(pair, candidate_id,
                    panel=controls.window_control_attestations(panel), stratum="confirmation",
                    reopening=None if attempt.get("confirmation_result") is None else
                        _read(self.store, "direct-runtime-window-result", attempt["confirmation_result"]))
                attempt["confirmation_result"] = confirmation.to_dict()
                self._checkpoint()
                admitted = (confirmed.event is not None and not confirmed.event_violations
                    and confirmed.verdict.speed_rank_admissible and confirmed.verdict.effect_resolution == api.EFFECT_IMPROVEMENT)
        body = {"schema": SCHEMA, "scope": self.scope, "index": index, "pair": pair.to_dict(),
            "source_commit": self.source_commit, "prompt_manifest_digest": self.prompts.digest,
            "controls": control_refs, "panel": panel.to_dict(), "selection": selected.to_dict(),
            "confirmation": None if confirmation is None else confirmation.to_dict(),
            "admitted": admitted, "source": source_identity()}
        result = self.store.write("direct-runtime-admission", body)
        self.state["attempts"][index]["result"] = result.to_dict()
        self._checkpoint()
        self._issued[result.sha256] = pair
        effect = reduction.estimate
        row = {"recipe": pair.anchor.template.name, "recipe_hash": pair.anchor.template.recipe_hash,
            "pairs": len(reduction.blocks), "effect": 0.0 if effect is None else effect.value,
            "decisive": True if admitted else None, "noise_floor_pct": reduction.noise_floor * 100,
            "runtime_pair": pair.to_dict(), "runtime_admission": result.to_dict(),
            "runtime_status": "admitted" if admitted else "observed_not_admitted",
            "control_panel": panel.to_dict(), "evaluation": outcome.durable_payload,
            "evaluation_event": outcome.event,
            "epoch": self.epoch, "qualified": admitted}
        try:
            row["belief_export_receipt"] = str(self._observation(pair, selected, result))
        except Exception as exc:
            row["belief_capture_error"] = f"{type(exc).__name__}: {exc}"[:256]
        return row

    def _observation(self, pair, selected, admission):
        """Project the SAME original launches through the existing unqualified source."""
        from . import serving, serving_beliefs
        material = _read(self.store, "direct-runtime-window-result", selected.to_dict())
        launches = [_read(self.store, "direct-calibration-launch", item["artifact"])
                    for item in material["launches"] if item["membership"][0] == "pair"]
        samples, windows = {}, {}
        for arm in ("anchor", "candidate"):
            rows = [row for row in launches if row["membership"][2] == arm]
            samples[arm] = [row["value"] for row in rows]
            windows[arm] = [row["observations"][0]["residency"] for row in rows]
        a, c = statistics.median(samples["anchor"]), statistics.median(samples["candidate"])
        requests = self.prompts.requests(tuple(p.prompt_id for p in self.prompts.prompts), pair.anchor.template)
        native = {"schema": "epyc.autokernel.serving_runtime_ab.v1",
            "recipe": pair.anchor.template.name, "recipe_hash": pair.anchor.template.recipe_hash,
            "candidate_recipe_hash": pair.candidate.template.recipe_hash,
            "runtime_pair": pair.to_dict(), "metric": "aggregate_tok_s", "np": pair.anchor.template.np,
            "pairs": len(samples["anchor"]), "anchor_tok_s": a, "candidate_tok_s": c,
            "effect": c / a - 1, "effect_pct": (c / a - 1) * 100,
            "anchor_samples": samples["anchor"], "candidate_samples": samples["candidate"],
            "anchor_residency": windows["anchor"], "candidate_residency": windows["candidate"],
            "decisive": None, "noise_floor_pct": None, "qualified": False,
            "request_digest": serving.request_digest(pair.anchor.template, requests),
            "floor_request_digest": None,
            # This source does not carry strict admission; the separate original
            # receipt is linked for attribution, never projected as belief grade.
            "admission": "observation_only_original_strict_evidence_unavailable",
            "separate_runtime_admission": admission.to_dict()}
        inputs = serving_beliefs.prepare(pair.anchor.template, anchor=pair.anchor,
            candidate=pair.candidate, anchor_build=pair.anchor.build_dir,
            candidate_build=pair.candidate.build_dir, frozen_requests=requests,
            pairs=native["pairs"], candidate_recipe=pair.candidate.template, runtime_pair=pair)
        native["belief_capture"] = serving_beliefs.finish(native, inputs)
        return serving_beliefs.export(self.store.root.parent,
            {"comparison": native, "mechanism_id": pair.dimension.dimension_id},
            campaign_id=self.campaign_id, epoch=self.epoch, recorded_at=rc._now())

    def reopen_admission(self, reference):
        body = _read(self.store, "direct-runtime-admission", reference)
        if (body["scope"] != self.scope or body["source"] != source_identity()
                or body["source_commit"] != self.source_commit or body["prompt_manifest_digest"] != self.prompts.digest):
            raise rc.RuntimeCalibrationRefused("retained recipe evidence/source/original frame differs")
        index = body["index"]
        if type(index) is not int or not 0 <= index < len(self.state["attempts"]):
            raise rc.RuntimeCalibrationRefused("retained admission has no original allocated attempt")
        attempt = self.state["attempts"][index]
        if (attempt["pair"] != body["pair"] or attempt["result"] != reference
                or bool(attempt.get("confirmation")) != (body["confirmation"] is not None)):
            raise rc.RuntimeCalibrationRefused("retained admission differs from its original allocated attempt")
        pair = RuntimeArmPair.from_dict(body["pair"])
        panel, refs = self._controls(pair.anchor, body["index"], reopening=body["controls"])
        if refs != body["controls"] or panel.to_dict() != body["panel"] or panel.panel is None:
            raise rc.RuntimeCalibrationRefused("original measured controls do not rederive")
        admitted = not panel.halts_campaign and panel.may_rank
        for stratum, key in (("selection", "selection"), ("confirmation", "confirmation")):
            if body[key] is None:
                admitted = False
                continue
            original = _read(self.store, "direct-runtime-window-result", body[key])
            ref, outcome, _reduction = self._evaluate_pair(pair,
                self.state["attempts"][body["index"]]["candidate_id"],
                panel=controls.window_control_attestations(panel), stratum=stratum, reopening=original)
            admitted = (admitted and ref.to_dict() == body[key] and outcome.event is not None
                and not outcome.event_violations and outcome.verdict.speed_rank_admissible
                and outcome.verdict.effect_resolution == api.EFFECT_IMPROVEMENT)
        if admitted != body["admitted"]:
            raise rc.RuntimeCalibrationRefused("retained admission does not rederive from original measurements")
        return pair, admitted

    def retain(self, comparison, current):
        reference = comparison["runtime_admission"]
        pair, admitted = self.reopen_admission(reference)
        if not admitted or pair.anchor.to_dict() != current.to_dict():
            raise rc.RuntimeCalibrationRefused("recipe keep lacks original admission or names another current anchor")
        self.state["selected"] = reference
        self._checkpoint()
        return pair.candidate

    def selected(self):
        if self.state["selected"] is None:
            return self.original
        pair, admitted = self.reopen_admission(self.state["selected"])
        if not admitted:
            raise rc.RuntimeCalibrationRefused("selected recipe was not admitted")
        return CanonicalResolvedRecipe.from_dict(pair.candidate.to_dict())

    def retained_build(self, reference):
        """Protect only the selected admission's original measured build."""
        if reference is None:
            return None
        retained = _read(self.store, "direct-runtime-selection", reference)
        original = CanonicalResolvedRecipe.from_dict(retained["owner"]["original"])
        return Path(original.build_dir)

    def selection_reference(self, current, *, current_source_commit, origin=None):
        """Bounded operational continuation, never an admission for a new build."""
        if self.state["selected"] is not None:
            _pair, admitted = self.reopen_admission(self.state["selected"])
            if not admitted:
                raise rc.RuntimeCalibrationRefused("operational selection lacks original admission")
            original = {"owner": self.owner_inputs, "admission": self.state["selected"]}
        elif origin is not None:
            retained = _read(self.store, "direct-runtime-selection", origin)
            original = {key: retained[key] for key in ("owner", "admission")}
        else:
            return None
        return self.store.write("direct-runtime-selection", {
            "schema": "epyc.autokernel.direct_runtime_selection.v1", **original,
            "current_recipe": current.to_dict(), "current_source_commit": current_source_commit}).to_dict()


def restore_selection(*, store, held_claim, reference, worktree, source_commit, build, prompts,
                      source_anchor=None):
    """Reopen the original admission, then verify its exact current-build rebind."""
    from .planned_serving import FrozenPromptManifest
    from .serving_preparation import ServingStatisticsDeclaration
    from .run import _cpu_arm
    body = _read(store, "direct-runtime-selection", reference)
    if (set(body) != {"schema", "owner", "admission", "current_recipe", "current_source_commit"}
            or body["schema"] != "epyc.autokernel.direct_runtime_selection.v1"
            or body["owner"]["worktree"] != str(worktree)
            or body["owner"]["prompts"] != prompts.to_dict()):
        raise rc.RuntimeCalibrationRefused("runtime continuation source/request ownership differs")
    # run.main supplies this only from its digest-reopened, exact-anchor-verified
    # source_resumed record. It names an operational build, NOT a new admission.
    if body["current_source_commit"] != source_commit and (
            not isinstance(source_anchor, dict)
            or source_anchor.get("commit") != source_commit
            or Path(source_anchor.get("path", "")) != Path(build)):
        raise rc.RuntimeCalibrationRefused("runtime continuation source/request ownership differs")
    kwargs = dict(body["owner"])
    kwargs["original"] = CanonicalResolvedRecipe.from_dict(kwargs["original"])
    kwargs["prompts"] = FrozenPromptManifest.from_dict(kwargs["prompts"])
    kwargs["statistical"] = ServingStatisticsDeclaration.from_dict(kwargs["statistical"])
    owner = RuntimeAdmission(store=store, held_claim=held_claim, **kwargs)
    pair, admitted = owner.reopen_admission(body["admission"])
    if not admitted:
        raise rc.RuntimeCalibrationRefused("runtime continuation original admission is absent")
    current = CanonicalResolvedRecipe.from_dict(body["current_recipe"])
    # A same-build continuation preserves the original snapshot exactly. Rebinding
    # it again would mint another experimental_parent_snapshot for no build change.
    rebound = (pair.candidate if Path(current.build_dir) == Path(pair.candidate.build_dir)
               else _cpu_arm(pair.candidate, Path(current.build_dir)))
    # Source keeps can form several operational rebinds. Their snapshot ancestry
    # changes, but every normalized executable/runtime/workload field must still
    # equal the original admitted treatment rebound to that exact saved build.
    if (current.execution_digest != rebound.execution_digest
            or current.template != rebound.template):
        raise rc.RuntimeCalibrationRefused("runtime continuation differs from original current build rebind")
    if body["current_source_commit"] == source_commit:
        if Path(current.build_dir) != Path(build):
            raise rc.RuntimeCalibrationRefused("runtime continuation current build differs")
        return current
    return _cpu_arm(current, Path(build))
