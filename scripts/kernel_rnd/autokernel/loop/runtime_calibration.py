"""Direct serving anchor/instrument material for the existing calibration solver.

No threshold is estimated here and a numerical solve alone is not admission. The
working loop's acquired claim and exact HTTP producer supply the raw observations;
there is no synthetic native plan, fence, grant, or replacement controller.
"""
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

from .. import schemas
from ..evaluator import api, controls, statistics as st
from . import claim, measurement_capture as mc, native_server_response as response
from . import observation_binding as ob, planned_serving as ps, serial_run, serving, status
from . import lifecycle_observation as lo, runtime_window
from .serving_preparation import PreparationArmPair, ServingStatisticsDeclaration
from .resolved_recipe import resolve_canonical_launch

SCHEMA = "epyc.autokernel.direct_runtime_calibration.v1"
MAX_BLOCKS = 1024
MAX_NEUTRAL_EXECUTABLE_BYTES = 64 * 1024 * 1024


def _digest(value):
    return hashlib.sha256(json.dumps(ob._plain(value), sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


class RuntimeCalibrationRefused(RuntimeError):
    pass


class RuntimeLaunchBudgetExhausted(RuntimeCalibrationRefused):
    """No new launch after the original invocation bound; completed rows remain."""


def declare_statistics(*, store, campaign_id, epoch, supplied=None):
    """Freeze campaign inputs once, before any direct candidate is measured.

    These configurable input defaults match the existing campaign preset, not
    its measured outputs. All phi/B_min/error thresholds are freshly solved from
    this frame's own A/A and neutral material by the unchanged evaluator.
    """
    if any(type(value) is not str or not value.strip() for value in (campaign_id, epoch)):
        raise RuntimeCalibrationRefused("original campaign and epoch are required")
    identity = _digest({"campaign_id": campaign_id, "epoch": epoch})
    name = f"direct-statistics-{identity}.json"
    path = store.root / name
    if path.exists():
        original, _sha = serial_run._json(path, limit=32_768)
        result = ServingStatisticsDeclaration.from_dict(original)
        if result.commitment.campaign_id != campaign_id:
            raise RuntimeCalibrationRefused("original statistical campaign differs")
        if supplied is not None and result.to_dict() != supplied.to_dict():
            raise RuntimeCalibrationRefused("statistical inputs changed after declaration")
        return result
    if supplied is None:
        declared = api.CampaignControls(200, 0.03, 10, 2, 20, 200 * 1024 ** 3)
        rule = st.StoppingRule(
            rule_id="ak-stop-direct-serving/v1", final_table="t1_paired_block_table",
            decisions=(("evidence_threshold_crossed", "compose_into_champion_lineage"),
                       ("extension_exhausted", "abandon"),
                       ("block_ceiling_reached", "abandon")),
            extension=st.BoundedExtension(max_rounds=1, blocks_per_round=5),
            max_blocks_per_candidate=declared.max_blocks_per_candidate)
        rep_floor = st.reps_floor_for_relative_effect(declared.contribution_floor)
        result = ServingStatisticsDeclaration(
            identity, declared, rule, st.StoppingRuleCommitment.commit(
                rule, campaign_id=campaign_id, committed_at=_now()),
            st.StratumSplitRule("ak-split-direct-serving/v1", identity, 0.3,
                                st.RotationSchedule("ak-rotation-direct-serving/v1", 4)),
            "sign_martingale_predictable_lambda/v1", st.EFFECT_SCALE_RELATIVE,
            st.HYPOTHESIS_IMPROVEMENT, 0.0,
            st.OwningProtocolRepRule("P-AK-SEARCH-1", st.REP_RULE_FLOOR,
                                    rep_floor.blocks, rep_floor.citation))
    else:
        result = ServingStatisticsDeclaration.from_dict(supplied.to_dict())
        if result.commitment.campaign_id != campaign_id:
            raise RuntimeCalibrationRefused("supplied statistical campaign differs")
    status.write_json(store.root, name, result.to_dict(), prefix=".direct-statistics-")
    return result


def neutral_material(*, store, anchor):
    """An actual byte-identical executable alias; original model and DSOs stay put.

    This does not build, rewrite RUNPATH, hash a model, or claim control-2 PASS.
    The independent neutral launches are still required by the owning solve.
    """
    anchor.validate_launch(anchor.template, anchor.build_dir, anchor.port)
    source = Path(anchor.executable.path)
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_NEUTRAL_EXECUTABLE_BYTES:
            raise RuntimeCalibrationRefused("neutral executable exceeds finite copy capacity")
        chunks, remaining = [], before.st_size
        while remaining:
            block = os.read(descriptor, min(1 << 20, remaining))
            if not block:
                raise RuntimeCalibrationRefused("original executable changed during neutral copy")
            chunks.append(block)
            remaining -= len(block)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (any(getattr(before, key) != getattr(after, key) for key in keys) or os.read(descriptor, 1)
                or hashlib.sha256(payload).hexdigest() != anchor.executable.sha256):
            raise RuntimeCalibrationRefused("original executable differs from enrolled identity")
    finally:
        os.close(descriptor)
    root = store.root / "direct-neutral" / anchor.snapshot_digest
    binary = root / "bin" / "llama-server"
    binary.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(binary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o500)
    except FileExistsError:
        descriptor = os.open(binary, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            identity = os.fstat(descriptor)
            if (not stat.S_ISREG(identity.st_mode) or identity.st_size != len(payload)
                    or os.read(descriptor, len(payload) + 1) != payload):
                raise RuntimeCalibrationRefused("retained neutral executable differs")
        finally:
            os.close(descriptor)
    else:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    original = store.write("direct-neutral-material", {
        "schema": SCHEMA, "anchor_snapshot_digest": anchor.snapshot_digest,
        "original_executable": anchor.executable.to_dict(),
        "copied_executable": str(binary), "sha256": anchor.executable.sha256})
    artifacts = {"model": anchor.model.to_dict(),
                 "drafter": None if anchor.drafter is None else anchor.drafter.to_dict(),
                 "executable": {**anchor.executable.to_dict(), "path": str(binary)},
                 "dsos": [item.to_dict() for item in anchor.dsos]}
    neutral = resolve_canonical_launch(anchor.template, build_dir=root,
        command_argv=(str(binary), *anchor.command_argv[1:]), topology_prefix=anchor.topology_prefix,
        launch_environment=dict(anchor.launch_env), artifact_identities=artifacts,
        backend=anchor.backend, environment_policy=anchor.environment_policy,
        port=anchor.port, runtime_binary_dir=str(binary.parent),
        runtime_ld_paths=anchor.runtime_ld_paths,
        provenance={**dict(anchor.provenance), "neutral_material": original.sha256})
    return PreparationArmPair("neutral", anchor, neutral,
                              f"{original.locator}#sha256={original.sha256}")


def positive_control_pair(original):
    """Parameterize the existing IQK0->1 control, not the experiment's anchor.

    Both control arms retain the original request/model/instrument/topology.
    Their two explicit recipe identities are separate from the candidate
    experiment, whose own anchor remains unchanged even when IQK is already on.
    This declares material only; real correctness, measurement and the unchanged
    positive-control predicate must still establish its expected direction.
    """
    from .unified_planner import RuntimeDimension, enumerate_runtime_dimensions
    if original.backend != "cpu" or "GGML_IQK" not in original.environment_policy.measurement_keys:
        raise RuntimeCalibrationRefused("IQK positive-control environment is not in the original CPU contract")
    definition = next(item for item in controls.CONTROL_DEFINITIONS
                      if item.control_id == controls.CONTROL_POSITIVE)
    dimension = RuntimeDimension(definition.fixture_id, "env",
        {"key": "GGML_IQK", "value": "0"}, {"key": "GGML_IQK", "value": "1"},
        "execution/live_controls.py:CONTROL_ARM_IQK:positive; " + controls.CONTROL_DEFINITIONS_DIGEST)
    return enumerate_runtime_dimensions(original, (dimension,))[0]


@dataclass(frozen=True)
class _DegradedWorkControl:
    """The declared reduced-work negative, never a runtime optimization arm."""
    anchor: object
    candidate: object
    original_prompts: ps.FrozenPromptManifest
    candidate_prompts: ps.FrozenPromptManifest

    def to_dict(self):
        return {"control_id": controls.CONTROL_DEGRADED_NEGATIVE,
            "definitions_digest": controls.CONTROL_DEFINITIONS_DIGEST,
            "anchor": self.anchor.to_dict(), "candidate": self.candidate.to_dict(),
            "original_prompts": self.original_prompts.to_dict(),
            "candidate_prompts": self.candidate_prompts.to_dict()}


def degraded_work_control(anchor, prompts):
    """Actually request half the tokens, retaining the unchanged expected workload.

    This is the existing reduced-work control mechanism. Its wrong request and
    resulting HTTP bytes are captured as wrong work, not relabelled as original
    full-work samples. The direct gate compares these against original_prompts.
    """
    if anchor.template.n_predict <= 1:
        raise RuntimeCalibrationRefused("original workload cannot supply a reduced-work negative")
    reduced = anchor.template.n_predict // 2
    template = replace(anchor.template, n_predict=reduced)
    candidate = resolve_canonical_launch(template, build_dir=anchor.build_dir,
        command_argv=anchor.command_argv, topology_prefix=anchor.topology_prefix,
        launch_environment=dict(anchor.launch_env),
        artifact_identities={"model": anchor.model.to_dict(),
            "drafter": None if anchor.drafter is None else anchor.drafter.to_dict(),
            "executable": anchor.executable.to_dict(), "dsos": [row.to_dict() for row in anchor.dsos]},
        backend=anchor.backend, environment_policy=anchor.environment_policy, port=anchor.port,
        runtime_binary_dir=anchor.runtime_binary_dir, runtime_ld_paths=anchor.runtime_ld_paths,
        provenance=dict(anchor.provenance))
    changed = []
    for prompt in prompts.prompts:
        row = prompt.to_dict()
        request = json.loads(prompt.body)
        request["n_predict"] = reduced
        if prompts.schema == ps.PROMPT_SCHEMA_V2:
            row["request"] = request
        else:
            row["n_predict"] = reduced
        row["request_digest"] = hashlib.sha256(ps._canonical(request)).hexdigest()
        changed.append(row)
    body = {"schema": prompts.schema, "version": prompts.version, "prompts": changed}
    altered = ps.FrozenPromptManifest.from_dict({**body, "digest": _digest(body)})
    return _DegradedWorkControl(anchor, candidate, prompts, altered)


def _solve(frame, statistical, blocks, samples_ref):
    """Unchanged owning policy; the candidate never supplies or raises phi."""
    return controls.run_calibration_block(st.CalibrationInputs(
        backend=frame["backend"], phase=frame["phase"], cell_class=frame["cell_class"],
        campaign_seed=statistical.campaign_seed, controls=statistical.controls,
        stopping_rule=statistical.stopping_rule,
        construction=st.select_construction(statistical.construction_id),
        effect_scale=statistical.effect_scale, metric_direction="higher_better",
        hypothesis=statistical.hypothesis, margin=statistical.margin,
        aa_blocks=tuple(blocks["aa"]), neutral_blocks=tuple(blocks["neutral"]),
        anchor_calibration_values=tuple(block.anchor_samples[0] for block in blocks["aa"]),
        samples_ref=samples_ref, owning_rep_rule=statistical.owning_rep_rule))


def calculation_source():
    from . import runtime_recovery
    return ob._freeze({"callables": [lo.callable_identity(function) for function in (
        _solve, controls.run_calibration_block, st.solve_calibration,
        DirectCalibration._launch, DirectCalibration._reopen_launch,
        DirectCalibration._material, DirectCalibration.reopen, DirectCalibration.validity,
        DirectPairWindow.__init__, DirectPairWindow.collect_to, DirectPairWindow.reopen,
        DirectPairWindow._context, DirectPairWindow._checkpoint, DirectPairWindow.finish,
        DirectPairWindow._prompts_for,
        positive_control_pair, degraded_work_control)],
        "statistics_module": st.STATISTICS_MODULE_ID,
        "recovery": runtime_recovery.source_identity()})


class DirectCalibration:
    """One original anchor frame, reusable by candidate recipes only after admission.

    The original statistical choices are explicit existing declarations. Collection
    is caller-held and serial. A partially returned object retains every completed
    launch; invoking collect again on it continues those original members, not a new
    draw. Durable reopening below rederives material and solve without relaunching.
    """

    def __init__(self, *, store, held_claim, campaign_id, epoch, anchor, neutral,
                 prompts, statistical, host_state, deadline_monotonic_s=None, on_progress=None):
        if type(store) is not mc.ArtifactStore or type(held_claim) is not claim.HeldCpuClaim:
            raise RuntimeCalibrationRefused("direct calibration requires original store and acquired CPU context")
        if type(statistical) is not ServingStatisticsDeclaration:
            raise RuntimeCalibrationRefused("original statistical declaration is required")
        statistical = ServingStatisticsDeclaration.from_dict(statistical.to_dict())
        if statistical.commitment.campaign_id != campaign_id:
            raise RuntimeCalibrationRefused("statistical declaration belongs to a different campaign")
        n = statistical.controls.calibration_block_count
        if not 1 <= n <= MAX_BLOCKS:
            raise RuntimeCalibrationRefused("calibration block count exceeds direct collection capacity")
        if type(prompts) is not ps.FrozenPromptManifest:
            raise RuntimeCalibrationRefused("original prompt manifest is required")
        prompts = ps.FrozenPromptManifest.from_dict(prompts.to_dict())
        aa = PreparationArmPair("aa", anchor, anchor, None)
        if type(neutral) is not PreparationArmPair or neutral.kind != "neutral":
            raise RuntimeCalibrationRefused("explicit byte-identical neutral material is required")
        neutral = PreparationArmPair.from_dict(neutral.to_dict())
        if neutral.anchor.to_dict() != anchor.to_dict() or anchor.backend != "cpu":
            raise RuntimeCalibrationRefused("neutral and direct CPU anchor identities differ")
        if anchor.template.metric != "aggregate_tok_s":
            raise RuntimeCalibrationRefused("direct calibration does not own this metric")
        requests = prompts.requests(tuple(row.prompt_id for row in prompts.prompts), anchor.template)
        if len(requests) != anchor.template.np:
            raise RuntimeCalibrationRefused("original requests differ from serving slot count")
        if any(type(value) is not str or not value.strip() for value in (campaign_id, epoch)):
            raise RuntimeCalibrationRefused("original campaign and epoch are required")
        if not host_state or not isinstance(host_state, dict):
            raise RuntimeCalibrationRefused("original host-state frame is required")
        self.store, self.held_claim = store, held_claim
        self.deadline_monotonic_s = deadline_monotonic_s
        self._on_progress = on_progress
        self.window_config = runtime_window.configuration(store, held_claim,
            storage_floor_bytes_free=statistical.controls.storage_floor_bytes_free,
            nominal_khz=host_state.get("nominal_khz", runtime_window.DEFAULT_NOMINAL_KHZ))
        self.statistical, self.prompts, self.requests = statistical, prompts, requests
        self.pairs = {"aa": aa, "neutral": neutral}
        self.frame = ob._freeze({"campaign_id": campaign_id, "epoch": epoch,
            "backend": "llama_cpu", "phase": "decode", "cell_class": "serving",
            "metric": anchor.template.metric, "metric_direction": "higher_better",
            "anchor": anchor.to_dict(), "prompt_manifest": prompts.to_dict(),
            "statistical": statistical.to_dict(), "host_state": host_state,
            # A frame belongs to the resource geometry, not to a process that
            # might later reopen its completed records. Per-launch original
            # holder identities remain in the immutable launch evidence.
            "claim_footprint": {key: held_claim[key] for key in ("device_id", "cpu_list", "regions")},
            "neutral": neutral.to_dict(),
            "host_window_configuration": self.window_config.to_dict(),
            "host_window_source": ob._plain(runtime_window.source_identity()),
            "calculation_source": ob._plain(calculation_source()),
            "response_source": ob._plain(response.source_identity())})
        self.identity = _digest(self.frame)
        self.declaration = store.write("direct-calibration-declaration", {
            "schema": SCHEMA, "frame": ob._plain(self.frame)})
        self.launches = []
        self.failures = []
        self.pending = None
        self.solution = None
        self.state_name = f"direct-calibration-{self.identity}.json"
        path = store.root / self.state_name
        if path.exists():
            original, _sha = serial_run._json(path, limit=16_384 + 4096 * n)
            if (set(original) != {"schema", "frame_digest", "declaration", "launches", "failures", "pending", "solution"}
                    or original["schema"] != SCHEMA or original["frame_digest"] != self.identity
                    or original["declaration"] != self.declaration.to_dict()
                    or type(original["launches"]) is not list or len(original["launches"]) > 4 * n):
                raise RuntimeCalibrationRefused("original calibration checkpoint differs")
            self.launches = original["launches"]
            self.failures = original["failures"]
            self.pending = original["pending"]
            self.solution = original["solution"]
            if self.pending is not None:
                from .loop import RunAborted
                raise RunAborted("original calibration launch cleanup unresolved; no fallback measurement")
            # Reopen every completed prefix member before considering any new
            # launch. A moved or invalid original is never replaced silently.
            self._material(self.launches, complete=False)
        else:
            self._checkpoint()

    def _checkpoint(self):
        status.write_json(self.store.root, self.state_name, {
            "schema": SCHEMA, "frame_digest": self.identity,
            "declaration": self.declaration.to_dict(), "launches": self.launches,
            "failures": self.failures,
            "pending": self.pending, "solution": self.solution}, prefix=".direct-calibration-")
        self._progress()

    def _progress(self):
        # Diagnostics only: no new artifact reads or authority, and callback
        # failure cannot change a durable checkpoint or an execution exception.
        try:
            if self._on_progress is not None:
                pair_window = isinstance(self, DirectPairWindow)
                limit = (self.outputs.b_min_blocks + 2 * self.maximum if pair_window
                         else 4 * self.statistical.controls.calibration_block_count)
                self._on_progress({
                    "observed_at": _now(), "phase": "pair_window" if pair_window else "calibration",
                    "frame_digest": self.identity, "completed_launches": len(self.launches),
                    "launch_limit": limit, "limit_is_upper_bound": pair_window,
                    "failed_launches": len(self.failures),
                    "pending": list(self.pending["membership"]) if self.pending else None})
        except Exception:
            pass

    def collect(self):
        self._sources()
        if self.pending is not None:
            from .loop import RunAborted
            raise RunAborted("original calibration launch cleanup unresolved; no fallback measurement")
        if self.solution is not None:
            reference = mc.StoredArtifact(**self.solution)
            self.reopen(reference)
            self._progress()
            return reference
        n = self.statistical.controls.calibration_block_count
        for kind, pair in self.pairs.items():
            schedule = st.OrderSchedule.derive(campaign_seed=self.statistical.campaign_seed,
                candidate_id=f"{self.identity}:{kind}", base_blocks=n)
            for index in range(n):
                arms = ("anchor", "candidate") if schedule.order_for(index) == st.ORDER_ANCHOR_FIRST \
                    else ("candidate", "anchor")
                for arm in arms:
                    membership = (kind, index, arm)
                    if any(tuple(row["membership"]) == membership for row in self.launches):
                        continue
                    recipe = getattr(pair, arm)
                    context = {"campaign_id": self.frame["campaign_id"], "epoch": self.frame["epoch"],
                        "comparison_id": f"{self.identity}:{kind}", "arm": arm, "launch_index": index}
                    self._launch(membership, recipe, context)
        material, blocks = self._material(self.launches)
        raw = self.store.write("direct-calibration-material", material)
        solve = _solve(self.frame, self.statistical, blocks, f"{raw.locator}#sha256={raw.sha256}")
        body = {"schema": SCHEMA, "frame_digest": self.identity, "material": raw.to_dict(),
                "numeric_solve": solve.to_dict(), "qualified": False,
                "remaining_admission": ["original_control_panel", "complete_original_t0", "original_host_window"]}
        result = self.store.write("direct-calibration-solve", body)
        self.solution = result.to_dict()
        self._checkpoint()
        return result

    def _launch(self, membership, recipe, context, *, prompts=None):
        """One original server launch, also used by direct candidate/control windows."""
        if self.deadline_monotonic_s is not None and time.monotonic() >= self.deadline_monotonic_s:
            raise RuntimeLaunchBudgetExhausted("original invocation launch deadline reached; completed prefix retained")
        prompts = self.prompts if prompts is None else prompts
        requests = prompts.requests(tuple(row.prompt_id for row in prompts.prompts), recipe.template)
        capture = response.ServerResponseCapture.for_direct(store=self.store,
            context=context, recipe=recipe, prompts=prompts)
        opened = self.held_claim.observe()
        if opened["status"] != "held":
            raise RuntimeCalibrationRefused("original CPU claim is not held before launch")
        self.pending = {"membership": list(membership), "context": context, "claim_open": opened}
        self._checkpoint()
        observations = []
        observer = runtime_window.DuringWork(self.window_config, recipe)
        failure, value, operational_error = None, None, None
        try:
            value = serving._measure_once(recipe.template, Path(recipe.build_dir), recipe.port,
                resolved_recipe=recipe, frozen_requests=requests,
                observation=observations, response_capture=capture, observation_session=observer)
            from .residency import cpu_lifecycle_invalidity
            contradictions = cpu_lifecycle_invalidity(observer.body()["cpu_lifecycle"], recipe.template.cpu_list)
            if contradictions:
                from .loop import MeasurementInvalid
                raise MeasurementInvalid("direct CPU measurement has observed placement contradictions", {
                    "failed_conditions": contradictions, "context": context,
                    "recipe": recipe.to_dict(), "observations": observations})
            rows = response.reopen_direct_unit(observations[0]["server_responses"], store=self.store,
                context=context, recipe=recipe, prompts=prompts,
                expected_pid=observations[0]["process_pid"])
            health = runtime_window.launch_health(observer.body(), rows,
                max_gap_s=self.window_config.limits.max_gap_s)
            if health.outcome != schemas.PASS:
                from .loop import MeasurementInvalid
                raise MeasurementInvalid("direct CPU launch host validity is " + health.outcome, {
                    "health": {"outcome": health.outcome, "reasons": list(health.reasons)},
                    "context": context, "recipe": recipe.to_dict(), "observations": observations})
        except BaseException as exc:
            operational_error = exc
            failure = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                closed = self.held_claim.observe()
                body = {"schema": SCHEMA, "declaration": self.declaration.to_dict(),
                    "membership": list(membership), "context": context,
                    "recipe": recipe.to_dict(), "value": value, "observations": observations,
                    "claim_open": opened, "claim_close": closed,
                    "during_work": observer.body(),
                    "ended_at": _now(), "failure": failure}
                original = self.store.write("direct-calibration-launch", body)
                if operational_error is None:
                    self.launches.append({"membership": list(membership), "artifact": original.to_dict()})
                    self.pending = None
                else:
                    from .loop import MeasurementInvalid
                    terminal = isinstance(operational_error, MeasurementInvalid) and observer.shutdown_resolved
                    self.failures.append({"membership": list(membership),
                        "artifact": original.to_dict(), "terminal_invalid": terminal})
                    if terminal:
                        self.pending = None
                    operational_error.add_note("original failed launch retained: " + original.locator)
                self._checkpoint()
            except Exception as retention_error:
                if operational_error is None:
                    raise
                operational_error.add_note(
                    f"original calibration retention also failed: {retention_error}")

    def _material(self, launches, *, complete=True):
        n = self.statistical.controls.calibration_block_count
        if len(launches) > 4 * n or complete and len(launches) != 4 * n:
            raise RuntimeCalibrationRefused("original calibration membership is incomplete")
        blocks, originals = {"aa": [], "neutral": []}, []
        position = 0
        for kind, pair in self.pairs.items():
            schedule = st.OrderSchedule.derive(campaign_seed=self.statistical.campaign_seed,
                candidate_id=f"{self.identity}:{kind}", base_blocks=n)
            for index in range(n):
                arms = ("anchor", "candidate") if schedule.order_for(index) == st.ORDER_ANCHOR_FIRST \
                    else ("candidate", "anchor")
                values, dates = {}, []
                for arm in arms:
                    if position == len(launches) and not complete:
                        return None, blocks
                    ref = launches[position]
                    position += 1
                    recipe = getattr(pair, arm)
                    context = {"campaign_id": self.frame["campaign_id"], "epoch": self.frame["epoch"],
                        "comparison_id": f"{self.identity}:{kind}", "arm": arm, "launch_index": index}
                    body, _rows = self._reopen_launch(ref, [kind, index, arm], recipe, context)
                    values[arm] = body["value"]
                    dates.append(body["ended_at"])
                    originals.append(ref)
                unit = f"{self.identity}:{kind}:{index}"
                blocks[kind].append(st.PairedBlock(index, unit,
                    self.statistical.split_rule.assign(unit), schedule.order_for(index),
                    (values["anchor"],), (values["candidate"],), measured_at=max(dates)))
        return {"schema": SCHEMA, "frame": ob._plain(self.frame), "launches": originals,
                "blocks": {kind: [asdict(block) for block in rows] for kind, rows in blocks.items()}}, blocks

    def _reopen_launch(self, ref, membership, recipe, context, *, prompts=None):
        prompts = self.prompts if prompts is None else prompts
        artifact = ref["artifact"]
        body = ob._plain(self.store.read(artifact["locator"], artifact["sha256"]))
        if self.store.verify("direct-calibration-launch", body).to_dict() != artifact:
            raise RuntimeCalibrationRefused("original calibration launch namespace differs")
        if (body["membership"] != membership or ref["membership"] != membership
                or body["context"] != context or body["recipe"] != recipe.to_dict()
                or body["declaration"] != self.declaration.to_dict() or body["failure"] is not None
                or len(body["observations"]) != 1):
            raise RuntimeCalibrationRefused("original calibration launch identity or completion differs")
        opened, closed = body["claim_open"], body["claim_close"]
        if (opened["status"] != "held" or closed["status"] != "held"
                or opened["owner_pid"] != closed["owner_pid"]
                or [(r["path"], r["device"], r["inode"]) for r in opened["locks"]] != [
                    (r["path"], r["device"], r["inode"]) for r in closed["locks"]]):
            raise RuntimeCalibrationRefused("original claim continuity was not observed")
        observation = body["observations"][0]
        rows = response.reopen_direct_unit(observation["server_responses"], store=self.store,
            context=context, recipe=recipe, prompts=prompts,
            expected_pid=observation["process_pid"])
        receipt = observation["server_responses"]
        if not (opened["ended_monotonic_s"] <= receipt["request_started_monotonic_s"]
                <= receipt["retention_ended_monotonic_s"] <= closed["started_monotonic_s"]):
            raise RuntimeCalibrationRefused("original held observations do not enclose HTTP capture")
        rates = []
        for row in rows:
            if row["raw"]["phase"] != "measurement":
                continue
            output = row["response"]
            if row["raw"]["error"] is not None or output is None or output.get("stop") is not True:
                raise RuntimeCalibrationRefused("original HTTP response is incomplete")
            timings = output.get("timings", {})
            rate = timings.get("predicted_per_second")
            if (type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0
                    or timings.get("predicted_n") != row["request"]["n_predict"]):
                raise RuntimeCalibrationRefused("original HTTP rate or generated count differs")
            rates.append(rate)
        if sum(rates) != body["value"]:
            raise RuntimeCalibrationRefused("calibration value differs from original HTTP rates")
        from .residency import cpu_lifecycle_invalidity
        if cpu_lifecycle_invalidity(body["during_work"]["cpu_lifecycle"], recipe.template.cpu_list):
            raise RuntimeCalibrationRefused("original launch has observed CPU placement contradictions")
        return body, rows

    def validity(self):
        """During-request host validity, separate from the numerical solve."""
        checks = []
        for ref in self.launches:
            kind, index, arm = ref["membership"]
            recipe = getattr(self.pairs[kind], arm)
            context = {"campaign_id": self.frame["campaign_id"], "epoch": self.frame["epoch"],
                "comparison_id": f"{self.identity}:{kind}", "arm": arm, "launch_index": index}
            body, rows = self._reopen_launch(ref, [kind, index, arm], recipe, context)
            facts = body["during_work"]
            checks.append(runtime_window.launch_health(facts, rows,
                max_gap_s=self.window_config.limits.max_gap_s))
        if not checks:
            checks.append(schemas.Check(schemas.COULD_NOT_CHECK, ("original calibration host evidence absent",)))
        return schemas.Check.worst_of(checks)

    def reopen(self, reference):
        self._sources()
        body = ob._plain(self.store.read(reference.locator, reference.sha256))
        if self.store.verify("direct-calibration-solve", body).to_dict() != reference.to_dict():
            raise RuntimeCalibrationRefused("retained solve namespace differs")
        material_ref = body["material"]
        material = ob._plain(self.store.read(material_ref["locator"], material_ref["sha256"]))
        if self.store.verify("direct-calibration-material", material).to_dict() != material_ref:
            raise RuntimeCalibrationRefused("retained calibration material namespace differs")
        reconstructed, blocks = self._material(material["launches"])
        if _digest(material) != _digest(reconstructed) or body["frame_digest"] != self.identity:
            raise RuntimeCalibrationRefused("retained calibration material/frame differs")
        solve = _solve(self.frame, self.statistical, blocks,
                       f"{material_ref['locator']}#sha256={material_ref['sha256']}")
        if body["numeric_solve"] != solve.to_dict() or body["qualified"] is not False:
            raise RuntimeCalibrationRefused("retained calibration solve/admission differs")
        return solve

    def _sources(self):
        if (self.frame["calculation_source"] != calculation_source()
                or self.frame["response_source"] != response.source_identity()
                or self.frame["host_window_source"] != runtime_window.source_identity()):
            raise RuntimeCalibrationRefused("original calibration producer/calculation source moved")


class DirectPairWindow:
    """The same original launch path for a candidate or a declared control.

    The first B_min launches are the window's anchor check, not candidate
    samples. Subsequent launches follow the original randomized paired order.
    This object collects facts; neither its existence nor its checkpoint admits
    a candidate. Interrupted windows cannot regain launch authority on restart.
    """

    _launch = DirectCalibration._launch
    _reopen_launch = DirectCalibration._reopen_launch
    _sources = DirectCalibration._sources
    _progress = DirectCalibration._progress

    def __init__(self, *, calibration, calibration_reference, pair, candidate_id,
                 stratum="selection", replacement=None):
        from .unified_planner import RuntimeArmPair
        from . import runtime_recovery
        if type(calibration) is not DirectCalibration:
            raise RuntimeCalibrationRefused("original direct calibration owner is required")
        if type(pair) is RuntimeArmPair:
            pair = RuntimeArmPair.from_dict(pair.to_dict())
        elif type(pair) is PreparationArmPair:
            pair = PreparationArmPair.from_dict(pair.to_dict())
        elif type(pair) is _DegradedWorkControl:
            original = degraded_work_control(pair.anchor, calibration.prompts)
            if original.to_dict() != pair.to_dict():
                raise RuntimeCalibrationRefused("degraded control differs from the declared reduced-work fixture")
            pair = original
        else:
            raise RuntimeCalibrationRefused("original candidate/control arm pair is required")
        if pair.anchor.to_dict() != ob._plain(calibration.frame["anchor"]):
            raise RuntimeCalibrationRefused("window anchor differs from its original calibration")
        if type(candidate_id) is not str or not candidate_id.startswith("akc-"):
            raise RuntimeCalibrationRefused("original evaluator candidate identity is required")
        if stratum not in api.STRATA:
            raise RuntimeCalibrationRefused("original selection or confirmation stratum is required")
        solve = calibration.reopen(calibration_reference)
        outputs = solve.require_accepted()
        self.store, self.held_claim = calibration.store, calibration.held_claim
        self.deadline_monotonic_s = calibration.deadline_monotonic_s
        self._on_progress = calibration._on_progress
        self.window_config = calibration.window_config
        self.prompts, self.requests = calibration.prompts, calibration.requests
        self.statistical = calibration.statistical
        self.pair, self.candidate_id, self.stratum = pair, candidate_id, stratum
        self.calibration, self.calibration_reference = calibration, calibration_reference
        self.outputs = outputs
        self.maximum = self.statistical.stopping_rule.max_total_blocks(outputs.b_min_blocks)
        if not 1 <= self.maximum <= MAX_BLOCKS:
            raise RuntimeCalibrationRefused("direct pair window exceeds finite collection capacity")
        self.statistics = st.CampaignStatistics(
            campaign_id=calibration.frame["campaign_id"],
            campaign_seed=self.statistical.campaign_seed,
            effect_scale=self.statistical.effect_scale, hypothesis=self.statistical.hypothesis,
            margin=self.statistical.margin, stopping_rule=self.statistical.stopping_rule,
            stopping_rule_commitment=self.statistical.commitment,
            split_rule=self.statistical.split_rule,
            construction=st.select_construction(self.statistical.construction_id),
            calibration=outputs, aa_effect_pool=solve.aa_effect_pool,
            anchor_calibration_values=solve.anchor_calibration_values,
            owning_rep_rule=self.statistical.owning_rep_rule)
        self.schedule = self.statistics.order_schedule(candidate_id)
        # Choose the material partition before launch, using the owning split.
        # The request itself is unchanged; these IDs identify independent blocks.
        units = []
        for index in range(1024 * self.maximum):
            unit = f"{candidate_id}:{stratum}:{index}"
            if self.statistical.split_rule.assign(unit) == stratum:
                units.append(unit)
                if len(units) == self.maximum:
                    break
        if len(units) != self.maximum:
            raise RuntimeCalibrationRefused("bounded original stratum allocation is incomplete")
        self.frame = ob._freeze({"campaign_id": calibration.frame["campaign_id"],
            "epoch": calibration.frame["epoch"], "calibration": calibration_reference.to_dict(),
            "pair": pair.to_dict(), "candidate_id": candidate_id, "stratum": stratum,
            "units": units, "prompt_manifest": self.prompts.to_dict(),
            "host_window_source": ob._plain(runtime_window.source_identity()),
            "calculation_source": ob._plain(calculation_source()),
            "response_source": ob._plain(response.source_identity()),
            **({"replacement": replacement} if replacement is not None else {})})
        self.identity = _digest(self.frame)
        self.state_name = f"direct-pair-window-{self.identity}.json"
        self.launches, self.failures, self.pending, self.solution = [], [], None, None
        self.boundaries = {"open": None, "close": None}
        path = self.store.root / self.state_name
        self._reopened = path.exists()
        if self._reopened:
            original, _sha = serial_run._json(path, limit=16_384 + 4096 * (3 * self.maximum))
            if (set(original) != {"schema", "frame_digest", "declaration", "launches", "failures", "pending", "solution", "boundaries"}
                    or original["schema"] != SCHEMA or original["frame_digest"] != self.identity
                    or type(original["launches"]) is not list
                    or len(original["launches"]) > outputs.b_min_blocks + 2 * self.maximum):
                raise RuntimeCalibrationRefused("original pair-window checkpoint differs")
            self.declaration = mc.StoredArtifact(**original["declaration"])
            declared = ob._plain(self.store.read(self.declaration.locator, self.declaration.sha256))
            if declared.get("schema") != SCHEMA or declared.get("frame") != ob._plain(self.frame):
                raise RuntimeCalibrationRefused("original pair-window declaration differs")
            self.launches, self.pending, self.solution = (
                original["launches"], original["pending"], original["solution"])
            self.boundaries = original["boundaries"]
            self.failures = original["failures"]
            if self.pending is not None or any(row.get("terminal_invalid") is not True for row in self.failures):
                from .loop import RunAborted
                raise RunAborted("original pair launch cleanup unresolved; no fallback measurement")
        else:
            if replacement is not None and replacement.get("holder") != runtime_recovery.holder_identity(self.held_claim):
                raise RuntimeCalibrationRefused("replacement was selected for another original holder")
            self.declaration = self.store.write("direct-pair-window", {
                "schema": SCHEMA, "frame": ob._plain(self.frame),
                "holder": runtime_recovery.holder_identity(self.held_claim)})
            self._checkpoint()

    def _context(self, kind, index, arm):
        return {"campaign_id": self.frame["campaign_id"], "epoch": self.frame["epoch"],
            "comparison_id": f"{self.identity}:{kind}", "arm": arm, "launch_index": index}

    def _checkpoint(self):
        status.write_json(self.store.root, self.state_name, {
            "schema": SCHEMA, "frame_digest": self.identity,
            "declaration": self.declaration.to_dict(), "launches": self.launches,
            "failures": self.failures,
            "pending": self.pending, "solution": self.solution,
            "boundaries": self.boundaries}, prefix=".direct-pair-window-")
        self._progress()

    def finish(self):
        if self.boundaries["close"] is None:
            if self._reopened:
                raise RuntimeCalibrationRefused("original pair window has no original close receipt")
            self.boundaries["close"] = self.store.write("direct-window-boundary",
                {"window": self.identity, **runtime_window.boundary(
                    self.window_config, self.held_claim, marker="close"),
                 "artifacts": runtime_window.artifacts(self.pair)}).to_dict()
            self._checkpoint()

    def collect_to(self, count):
        self._sources()
        if (type(count) is not int or not self.outputs.b_min_blocks <= count <= self.maximum):
            raise RuntimeCalibrationRefused("pair count exceeds original calibrated stopping window")
        expected = self.outputs.b_min_blocks + 2 * count
        if len(self.launches) >= expected and self.pending is None:
            return self.reopen(count)
        if self._reopened or self.pending is not None:
            raise RuntimeCalibrationRefused("unfinished original pair window cannot be replay-launched")
        if self.boundaries["close"] is not None:
            raise RuntimeCalibrationRefused("closed original pair window cannot launch a successor")
        if self.boundaries["open"] is None:
            self.boundaries["open"] = self.store.write("direct-window-boundary",
                {"window": self.identity, **runtime_window.boundary(
                    self.window_config, self.held_claim, marker="open"),
                 "artifacts": runtime_window.artifacts(self.pair)}).to_dict()
            self._checkpoint()
        for index in range(self.outputs.b_min_blocks):
            if index >= len(self.launches):
                self._launch(("anchor_gate", index, "anchor"), self.pair.anchor,
                             self._context("anchor_gate", index, "anchor"))
        anchors, _blocks, _rows = self.reopen(0)
        checked = st.anchor_gate_check(anchors, band=self.outputs, b_min=self.outputs.b_min_blocks)
        if checked.outcome != "PASS":
            raise RuntimeCalibrationRefused("original anchor window is invalid: " + "; ".join(checked.reasons))
        for index in range(count):
            arms = ("anchor", "candidate") if self.schedule.order_for(index) == st.ORDER_ANCHOR_FIRST \
                else ("candidate", "anchor")
            for offset, arm in enumerate(arms):
                position = self.outputs.b_min_blocks + 2 * index + offset
                if position >= len(self.launches):
                    self._launch(("pair", index, arm), getattr(self.pair, arm),
                                 self._context("pair", index, arm),
                                 prompts=self._prompts_for(arm))
        return self.reopen(count)

    def _prompts_for(self, arm):
        return self.pair.candidate_prompts if arm == "candidate" \
            and type(self.pair) is _DegradedWorkControl else self.prompts

    def reopen(self, count):
        self._sources()
        if type(count) is not int or not 0 <= count <= self.maximum:
            raise RuntimeCalibrationRefused("original pair-window count differs")
        if self.pending is not None or len(self.launches) < self.outputs.b_min_blocks + 2 * count:
            raise RuntimeCalibrationRefused("original pair-window membership is incomplete")
        anchors, blocks, rows = [], [], []
        for index in range(self.outputs.b_min_blocks):
            body, raw = self._reopen_launch(self.launches[index], ["anchor_gate", index, "anchor"],
                self.pair.anchor, self._context("anchor_gate", index, "anchor"))
            anchors.append(body["value"])
            rows.append(("anchor_gate", index, "anchor", body, raw))
        for index in range(count):
            arms = ("anchor", "candidate") if self.schedule.order_for(index) == st.ORDER_ANCHOR_FIRST \
                else ("candidate", "anchor")
            values, dates = {}, []
            for offset, arm in enumerate(arms):
                position = self.outputs.b_min_blocks + 2 * index + offset
                body, raw = self._reopen_launch(self.launches[position], ["pair", index, arm],
                    getattr(self.pair, arm), self._context("pair", index, arm),
                    prompts=self._prompts_for(arm))
                values[arm], dates = body["value"], [*dates, body["ended_at"]]
                rows.append(("pair", index, arm, body, raw))
            extended = index >= self.outputs.b_min_blocks
            extension = ((index - self.outputs.b_min_blocks)
                         // self.statistical.stopping_rule.extension.blocks_per_round + 1) if extended else None
            blocks.append(st.PairedBlock(index, self.frame["units"][index], self.stratum,
                self.schedule.order_for(index), (values["anchor"],), (values["candidate"],),
                st.SEGMENT_EXTENSION if extended else st.SEGMENT_BASE, extension, max(dates)))
        return tuple(anchors), tuple(blocks), tuple(rows)
