"""Prospective replay of the retained, single-mechanism GPU x4-load control.

The reference tolerance is declared engineering input, not confidence or a
candidate threshold. Original Qwen/bench controls never become evidence about
another model's runtime treatment. The current direct owner supplies both held
components; no extra claim, production build, or retrospective PASS is issued.
"""
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace

from .. import schemas, storage
from ..evaluator import api, controls, devices, statistics as st
from ..execution import control_runner as cr, microbench, t0_provider as tp
from ..evaluator import correctness
from .. import campaign
from . import bench, observation_binding as ob, runtime_calibration as rc
from . import runtime_window as rw, serial_run
from .claim import DEVICE_ID, HeldCpuClaim
from .measurement_capture import StoredArtifact

LOCATOR = "data/autokernel-gpu-controls/x4-load-fe881-bff30.json"
FIXTURE_SHA256 = "a09969e60e7e563ffd06f37c7bca6436ba6c5b7b81b43a62bcdf3145168d504a"
NAMESPACE = "direct-gpu-control"
SCHEMA = "epyc.autokernel.direct_gpu_control.v1"
BELIEF_SCHEMA = "epyc.autokernel.direct_gpu_control_beliefs.v1"
PRODUCER_ID = "autokernel.loop.direct_gpu_control/v1"
MAX_RAW_BYTES = 16 * 1024 * 1024


class GpuControlRefused(rc.RuntimeCalibrationRefused):
    pass


def _ref(reference):
    return f"{reference.locator}#sha256={reference.sha256}"


def _read(store, namespace, reference):
    reference = StoredArtifact(**reference)
    body = ob._plain(store.read(reference.locator, reference.sha256))
    if store.verify(namespace, body) != reference:
        raise GpuControlRefused("original GPU control artifact namespace differs")
    return body


def _check(ok, reason, *, unavailable=False):
    return schemas.Check(schemas.PASS if ok else schemas.COULD_NOT_CHECK
                         if unavailable else schemas.FAIL, (reason,))


def _deadline(value):
    if value is not None and time.monotonic() >= value:
        raise rc.RuntimeLaunchBudgetExhausted("original invocation budget exhausted before GPU control launch")


def fixture():
    root = Path(__file__).resolve().parents[4]
    raw = serial_run._read(root / LOCATOR, limit=MAX_RAW_BYTES)
    if hashlib.sha256(raw).hexdigest() != FIXTURE_SHA256:
        raise GpuControlRefused("prospective GPU fixture declaration changed")
    body = json.loads(raw)
    original_bytes = body["original_payload"].encode()
    if hashlib.sha256(original_bytes).hexdigest() != body["original_payload_sha256"]:
        raise GpuControlRefused("original GPU historical payload digest differs")
    original = json.loads(original_bytes)["comparison"]
    a, c = original["anchor_samples"], original["candidate_samples"]
    if len(a) != len(c) or len(a) != 20:
        raise GpuControlRefused("original GPU historical pair count differs")
    effects = [right / left - 1 for left, right in zip(a, c)]
    if (body["approved_reference_band"] != {"low": min(effects), "high": max(effects)}
            or body["original_paired_effect_median"] != statistics.median(effects)
            or body["original_ratio_of_medians"] != original["effect"]):
        raise GpuControlRefused("approved engineering tolerance differs from original paired range")
    return body


def resolution():
    body = fixture()
    root = Path(__file__).resolve().parents[4]
    declaration = controls.HistoricalWinReplayDeclaration(
        win_id=body["win_id"], backend=body["backend"], phase=body["phase"],
        reference_direction=body["reference_direction"],
        reference_band=controls.ReferenceBand(**body["approved_reference_band"]),
        evidence_locator=str(root / LOCATOR), durability_class="carried_in_git",
        evidence_sha256=FIXTURE_SHA256,
        evidence_provenance="Original 20 paired effects; fixed engineering replay tolerance, not a confidence interval")
    return controls.resolve_historical_win_replay(declarations=(declaration,),
        backend="llama_gpu", tracked_index=storage.GitTrackedIndex(root))


class OriginalClaims:
    """Structural T0/measurement protocol over the two actually acquired owners."""
    def __init__(self, cpu, gpu, *, store, cpu_list):
        if (type(gpu) is not HeldCpuClaim or gpu["device_id"] != DEVICE_ID
                or type(cpu) is not HeldCpuClaim or cpu._domain != gpu._domain):
            raise GpuControlRefused("GPU control requires both original same-owner component claims")
        try:
            self.cpu = rw.DirectHeldClaimAdapter(cpu, cpu_list=cpu_list, store=store)
        except (TypeError, ValueError) as exc:
            raise GpuControlRefused(str(exc)) from exc
        self.gpu, self.store, self.cpu_list = gpu, store, cpu_list
        self.claim_id = self.cpu.claim_id + ":gpu:" + gpu._context_id
        self.observations = []

    def covers(self, cpu_list):
        return rw.parse_cpu_list(cpu_list) <= rw.parse_cpu_list(self.cpu_list)

    def observe(self):
        if len(self.observations) >= 16384:
            raise GpuControlRefused("original GPU control observation capacity exhausted")
        body = {"cpu": self.cpu.owner.observe(), "gpu": self.gpu.observe(),
            "cpu_holder": {"context_id": self.cpu.owner._context_id, "domain": self.cpu.owner._domain},
            "gpu_holder": {"context_id": self.gpu._context_id, "domain": self.gpu._domain},
            "cpu_list": self.cpu_list}
        reference = self.store.write("direct-gpu-held-observation", body)
        self.observations.append(reference.to_dict())
        return body, reference

    def verify_held(self):
        body, reference = self.observe()
        return _check(all(body[key]["status"] == "held" for key in ("cpu", "gpu")),
                      "original component observations " + _ref(reference))


def require_available(*, store, held_claim, gpu_claim):
    declared = resolution()
    if not declared.available:
        raise GpuControlRefused("GPU historical fixture is not yet durably installed: " + declared.reason())
    body = fixture()
    frame = body["frame"]
    original = OriginalClaims(held_claim, gpu_claim, store=store, cpu_list=frame["cpu_list"])
    if original.verify_held().outcome != schemas.PASS:
        raise GpuControlRefused("GPU control requires both currently held original components")
    if bench.CPU_LIST != frame["cpu_list"]:
        raise GpuControlRefused("original GPU bench constructor footprint changed")
    for arm in ("anchor", "candidate"):
        build = Path(frame[arm + "_build"])
        row, _sha = serial_run._json(build / "provenance.json", limit=32768)
        if row["champion_commit"] != frame[arm + "_commit"]:
            raise GpuControlRefused("retained GPU instrument source identity differs")
        for name in ("llama-bench", "test-backend-ops", "llama-cli"):
            if not (build / "bin" / name).is_file():
                raise GpuControlRefused("retained GPU instrument missing " + name + "; no implicit build")
    return declared, original


class NativeCapture:
    """Capture only; the original bench owns argv, subprocess and scalar output."""
    def __init__(self, store, claims, config, *, membership, deadline):
        self.store, self.claims, self.config = store, claims, config
        self.membership, self.deadline = membership, deadline
        self.reference = None
        self.raw = None

    @contextmanager
    def invocation(self, argv, env, attempt):
        _deadline(self.deadline)
        if attempt:
            # A strict control retains a failed launch and returns to its owner;
            # it never hides a process retry inside a successful native scalar.
            raise GpuControlRefused("original GPU control launch failed; explicit owner reschedule required")
        opened, claim_reference = self.claims.observe()
        if any(opened[key]["status"] != "held" for key in ("cpu", "gpu")):
            raise GpuControlRefused("original GPU control component claim was lost before launch")
        actual = SimpleNamespace(template=SimpleNamespace(cpu_list=self.claims.cpu_list))
        during = rw.GpuDuringWork(self.config, actual)
        self.raw = {"membership": self.membership, "argv": list(argv), "env": dict(env),
            "attempt": attempt, "claim_open": claim_reference.to_dict(), "error": None,
            "started_at": datetime.now(timezone.utc).isoformat()}
        try:
            during.start()
            during.phase("measurement")
            self.raw["started_monotonic_s"] = time.monotonic()
            yield
        except BaseException as exc:
            self.raw["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.raw["ended_monotonic_s"] = time.monotonic()
            during.phase("measurement_end")
            during.finish()
            self.raw["during_work"] = during.body()
            self.raw["shutdown_resolved"] = during.shutdown_resolved
            _closed, reference = self.claims.observe()
            self.raw["claim_close"] = reference.to_dict()
            self.raw["ended_at"] = datetime.now(timezone.utc).isoformat()
            self.reference = self.store.write("direct-gpu-bench-launch", self.raw)

    def completed(self, done, residency):
        if len(done.stdout.encode()) + len(done.stderr.encode()) > MAX_RAW_BYTES:
            raise GpuControlRefused("original GPU control raw output exceeds finite capacity")
        self.raw.update(returncode=done.returncode, stdout=done.stdout,
                        stderr=done.stderr, residency=residency)


def source_identity():
    from . import lifecycle_observation as lo
    return {"fixture_sha256": FIXTURE_SHA256,
        "files": {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                  for module in (bench, microbench, tp, controls, st, rw)},
        "self": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "loaded": [lo.callable_identity(value) for value in (
            fixture, resolution, require_available, OriginalClaims.observe,
            OriginalClaims.verify_held, NativeCapture.invocation.__wrapped__, NativeCapture.completed,
            BenchWindow.launch, BenchWindow.pairs, _native, _collect, _compose, _identities,
            _t0_gates, _collect_t0, _t0_material, _window, _belief_receipt, run_or_reopen)]}


def _native(store, reference, *, frame, arm, membership, config):
    """Reopen the original JSON vector and process-mean quantity, not a new estimator."""
    row = _read(store, "direct-gpu-bench-launch", reference)
    binary = str(Path(frame[arm + "_build"]) / "bin/llama-bench")
    expected = [*frame["topology"], binary, "-m", frame["model"], *frame["explicit_flags"]]
    if row["argv"] != expected or row["membership"] != membership:
        raise GpuControlRefused("original GPU process recipe or membership differs")
    if (row["error"] is not None or row.get("returncode") != 0 or not row["shutdown_resolved"]
            or row.get("residency", {}).get("resident") is not True):
        raise GpuControlRefused("original GPU control process did not complete and drain")
    originals = [_read(store, "direct-gpu-held-observation", row[key])
                 for key in ("claim_open", "claim_close")]
    if any(not rw.same_claim(originals[0][key], originals[1][key]) for key in ("cpu", "gpu")):
        raise GpuControlRefused("original GPU control component continuity failed")
    parsed = microbench.parse_llama_bench_json(row["stdout"])
    if len(parsed) != 1:
        raise GpuControlRefused("original GPU control is one tg128 point, not a sweep")
    measured = parsed[0]
    if (measured.n_prompt != frame["pp"] or measured.n_gen != frame["tg"]
            or measured.n_gpu_layers != 99 or not measured.flash_attn
            or len(measured.samples_ts) != frame["reps"]
            or len(measured.samples_ns) != frame["reps"]
            or not frame[arm + "_commit"].startswith(measured.build_commit)
            or not measured.build_commit
            or Path(measured.model_filename).name != Path(frame["model"]).name):
        raise GpuControlRefused("original GPU control raw workload/build differs")
    # The retained historical fixture's observation unit was one native process
    # mean. Preserve it; the native repetition vector remains alongside it and
    # is never substituted with median(repetitions) from another instrument.
    result = measured.avg_ts
    if not (result > 0 and abs(result - statistics.mean(measured.samples_ts)) < 1e-5):
        raise GpuControlRefused("native printed process mean does not match its six-decimal repetition vector")
    responses = [{"raw": {"phase": "measurement", "started_monotonic_s": row["started_monotonic_s"],
        "ended_monotonic_s": row["ended_monotonic_s"]}}]
    health = rw.launch_health(row["during_work"], responses, max_gap_s=config.limits.max_gap_s)
    duration = devices.GFX90A_RANKED_DURATION_ADMISSION.check(measured.samples_ns, device_id="ROCm0")
    if health.outcome != schemas.PASS or duration.outcome != schemas.PASS:
        raise GpuControlRefused("original GPU control window unusable: " + "; ".join((*health.reasons, *duration.reasons)))
    return result, row, rw.gpu_device_state(row["during_work"], responses)


class BenchWindow:
    """Original per-launch checkpoint owner for one frozen control frame."""
    def __init__(self, *, store, claims, declaration, statistical, root, deadline):
        from . import status
        self.store, self.claims = store, claims
        self.declaration, self.statistical, self.root = declaration, statistical, root
        self.deadline, self._write = deadline, status.write_json
        self.frame = declaration["fixture"]["frame"]
        self.config = rw.configuration(store, claims.cpu.owner,
            storage_floor_bytes_free=statistical.controls.storage_floor_bytes_free)
        self.path = root / "window.json"
        if self.path.exists():
            self.state, _sha = serial_run._json(self.path, limit=4 * 1024 * 1024)
            if self.state["declaration"] != schemas.content_hash(declaration):
                raise GpuControlRefused("GPU control checkpoint declaration differs")
            if self.state["pending"] is not None:
                from .loop import RunAborted
                raise RunAborted("GPU control has an unresolved original launch; reconcile child and claims before restart")
        else:
            self.state = {"declaration": schemas.content_hash(declaration), "pending": None,
                          "launches": [], "invalid": []}
            self._checkpoint()

    def _checkpoint(self):
        self._write(self.root, "window.json", self.state, prefix=".gpu-control-")

    def launch(self, membership, arm):
        existing = next((row for row in self.state["launches"] if row["membership"] == membership), None)
        if existing is not None:
            return _native(self.store, existing["reference"], frame=self.frame,
                arm=arm, membership=membership, config=self.config)[0]
        _deadline(self.deadline)
        self.state["pending"] = {"membership": membership, "arm": arm}
        self._checkpoint()
        capture = NativeCapture(self.store, self.claims, self.config,
            membership=membership, deadline=self.deadline)
        try:
            bench.run_once(Path(self.frame[arm + "_build"]) / "bin/llama-bench",
                Path(self.frame["model"]), pp=self.frame["pp"], tg=self.frame["tg"],
                reps=self.frame["reps"], capture=capture)
            value, _body, _device = _native(self.store, capture.reference.to_dict(),
                frame=self.frame, arm=arm, membership=membership, config=self.config)
        except BaseException as exc:
            if capture.reference is not None and capture.raw["shutdown_resolved"]:
                self.state["invalid"].append({"membership": membership, "arm": arm,
                                             "reference": capture.reference.to_dict()})
                self.state["pending"] = None
                self._checkpoint()
                if isinstance(exc, bench.BenchFailed):
                    raise GpuControlRefused(f"original benchmark failed after teardown: {exc}") from exc
            raise
        self.state["launches"].append({"membership": membership, "arm": arm,
                                      "reference": capture.reference.to_dict()})
        self.state["pending"] = None
        self._checkpoint()
        return value

    def pairs(self, label, count, candidate_arm, *, schedule, stratum=api.STRATUM_SELECTION):
        units = []
        for attempt in range(1024 * count):
            unit = f"{label}:{attempt}"
            if self.statistical.split_rule.assign(unit) == stratum:
                units.append(unit)
                if len(units) == count:
                    break
        if len(units) != count:
            raise GpuControlRefused("bounded original control stratum allocation incomplete")
        blocks = []
        for index in range(count):
            order = schedule.order_for(index)
            values = {}
            for key in (("anchor", "candidate") if order == "AB" else ("candidate", "anchor")):
                actual = "anchor" if key == "anchor" else candidate_arm
                values[key] = self.launch([label, index, key], actual)
            completed = [_read(self.store, "direct-gpu-bench-launch", row["reference"])
                for row in self.state["launches"] if row["membership"][:2] == [label, index]]
            blocks.append(st.PairedBlock(index, units[index], stratum, order,
                (values["anchor"],), (values["candidate"],),
                measured_at=max(row["ended_at"] for row in completed)))
        return tuple(blocks)


def _t0_plan(frame, identities, arm):
    row = identities[arm]
    build = Path(frame[arm + "_build"]) / "bin"
    anchor = Path(frame["anchor_build"]) / "bin"
    return tp.T0ExecutionPlan(
        candidate=tp.CandidateBuild(frame["source_root"], str(build.parent),
            frame[arm + "_commit"], row["source_sha256"], str(build / "llama-cli"),
            str(build), str(build / "test-backend-ops")),
        tools=campaign.HostOps._t0_tools(),
        op_suite=tp.OpSuitePlan(backend_filter="ROCm0", ops=correctness.MANDATORY_BACKEND_OPS,
            suite_id="test-backend-ops/v1", suite_source_sha256=row["source_sha256"], suite_seed=2026091001),
        dispatch=tp.DispatchTracePlan(derived_surface=correctness.MANDATORY_BACKEND_OPS),
        anchor=tp.AnchorBuild(frame["source_root"], frame["anchor_commit"],
            str(anchor / "llama-cli"), str(anchor)),
        generation=tp.GenerationPlan(prompt="The capital of France is", prompt_ref="ak-prompt-001",
            n_predict=32, seed=42, threads=8, extra_argv=("-m", frame["model"], "-ngl", "99")),
        determinism_runs=2, backend="llama_gpu")


class T0Runner:
    """Original trusted retained tools under this owner's existing affinity/claims."""
    def __init__(self, claims, deadline):
        self.claims, self.deadline = claims, deadline
        self.runner = tp.SubprocessRunner()

    def run(self, argv, *, env, cwd, timeout_s):
        _deadline(self.deadline)
        if self.claims.verify_held().outcome != schemas.PASS:
            raise GpuControlRefused("original GPU control claims lost before T0")
        prior = os.sched_getaffinity(0)
        try:
            os.sched_setaffinity(0, rw.parse_cpu_list(self.claims.cpu_list))
            result = self.runner.run(argv, env=env, cwd=cwd, timeout_s=timeout_s)
        finally:
            os.sched_setaffinity(0, prior)
        if result.orphans:
            raise tp.ProcessEscaped("original GPU control T0 left an owned process alive")
        return result


class T0Sink:
    def __init__(self, store):
        self.owner, self.references = store, []

    def store(self, capture):
        reference = self.owner.write("direct-gpu-t0-process", capture.to_dict())
        self.references.append(reference.to_dict())
        return _ref(reference)


def _collect_t0(store, claims, frame, identities, deadline):
    sink, runner = T0Sink(store), T0Runner(claims, deadline)
    anchor = tp.capture_anchor(plan=_t0_plan(frame, identities, "anchor"), runner=runner,
        claim=claims, sink=sink, generation_seeds=(42, 42))
    anchor_refs = list(sink.references)
    sink.references.clear()
    provider = tp.ExecutedT0EvidenceProvider(plan=_t0_plan(frame, identities, "candidate"),
        runner=runner, claim=claims, sink=sink, anchor_capture=anchor)
    collected = tp._Collected()
    provider.collect_op_suite(collected)
    provider.collect_coherence(collected)
    provider.collect_determinism(collected)
    return store.write("direct-gpu-t0", {"anchor": asdict(anchor), "anchor_captures": anchor_refs,
        "candidate_captures": sink.references, "claim_observations": list(claims.observations)})


def _t0_material(store, reference, declaration, identities):
    body = _read(store, "direct-gpu-t0", reference)
    frame = declaration["fixture"]["frame"]
    anchor_row = dict(body["anchor"])
    for key in ("resolved_libraries", "output_digests", "output_lengths", "oracle_ids", "capture_refs", "notes"):
        anchor_row[key] = tuple(tuple(row) if isinstance(row, list) else row for row in anchor_row[key])
    anchor = tp.AnchorCapture(**anchor_row)
    expected_cli = str((Path(frame["anchor_build"]) / "bin/llama-cli").resolve())
    if (anchor.source_commit != frame["anchor_commit"] or anchor.binary_sha256 != next(
            row["sha256"] for row in identities["anchor"]["files"] if row["path"] == expected_cli)):
        raise GpuControlRefused("original T0 anchor differs from retained GPU instrument")
    originals = [_read(store, "direct-gpu-held-observation", ref) for ref in body["claim_observations"]]
    if not originals or any(row["cpu_list"] != frame["cpu_list"] or
            any(row[key]["status"] != "held" for key in ("cpu", "gpu")) for row in originals):
        raise GpuControlRefused("original GPU T0 component observations absent or lost")

    def captures(key):
        result = []
        for ref in body[key]:
            row = _read(store, "direct-gpu-t0-process", ref)
            if row["orphans"]:
                raise GpuControlRefused("original GPU T0 process did not drain")
            row.update(argv=tuple(row["argv"]), env=tuple(map(tuple, row["env"])), orphans=tuple(row["orphans"]))
            result.append(tp.CompletedProcess(**row))
        return result

    plan = _t0_plan(frame, identities, "anchor")
    invocation = tp.build_generation_invocation(binary=plan.anchor.binary,
        library_path=plan.anchor.library_path, plan=plan.generation, base_env=plan.base_env,
        seed=42, cpu_prefix=False)
    generations = [row for row in captures("anchor_captures") if row.argv == invocation.argv]
    if len(generations) != 2 or any(tuple(sorted(row.env)) != tuple(sorted(invocation.env_dict().items()))
            or row.cwd != plan.candidate.worktree for row in generations):
        raise GpuControlRefused("original GPU anchor generation recipe differs")
    digests, lengths = [], []
    for row in generations:
        if tp.ExecutedT0EvidenceProvider._generation_defect(row) is None:
            digests.append(tp.sha256_text(row.stdout))
            lengths.append(len(row.stdout))
    stability = ("not_measured" if len(digests) < 2 else "bitwise_stable"
                 if len(set(digests)) == 1 else "bitwise_unstable")
    if (tuple(digests), tuple(lengths), stability) != (
            anchor.output_digests, anchor.output_lengths, anchor.determinism_class):
        raise GpuControlRefused("original GPU anchor output identity does not rederive")
    rows = captures("candidate_captures")

    class Reader:
        index = 0

        def run(self, argv, *, env, cwd, timeout_s):
            if self.index >= len(rows):
                raise GpuControlRefused("original GPU T0 capture missing")
            row = rows[self.index]
            self.index += 1
            if tuple(argv) != row.argv or tuple(sorted(env.items())) != tuple(sorted(row.env)) or cwd != row.cwd:
                raise GpuControlRefused("original GPU T0 command/environment differs")
            return row

    class RecordedClaims:
        def verify_held(self):
            return _check(bool(originals) and all(row[key]["status"] == "held"
                for row in originals for key in ("cpu", "gpu")),
                "retained original CPU/GPU observations")

    reader = Reader()
    provider = tp.ExecutedT0EvidenceProvider(plan=_t0_plan(frame, identities, "candidate"),
        runner=reader, claim=RecordedClaims(), sink=tp.MemoryCaptureSink(), anchor_capture=anchor)
    collected = tp._Collected()
    ops = provider.collect_op_suite(collected)
    coherence = provider.collect_coherence(collected)
    determinism = provider.collect_determinism(collected)
    if reader.index != len(rows):
        raise GpuControlRefused("unconsumed original GPU T0 captures")
    return anchor, ops, coherence, determinism, provider._change_surface()


def _identities(store, frame):
    """Small retained instruments only; never reread the model payload."""
    result = {}
    for arm in ("anchor", "candidate"):
        build = Path(frame[arm + "_build"]) / "bin"
        files = sorted({path.resolve(strict=True) for path in build.iterdir()
            if path.name in {"llama-bench", "llama-cli", "test-backend-ops"}
            or path.name.startswith(("libggml", "libllama"))})
        if len(files) > 32:
            raise GpuControlRefused("original GPU instrument file census exceeds capacity")
        artifacts = []
        for path in files:
            raw = serial_run._read(path, limit=512 * 1024 * 1024)
            artifacts.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        source = subprocess.run(["git", "-C", frame["source_root"], "ls-tree", "-r", "-z",
            frame[arm + "_commit"]], capture_output=True, check=True, timeout=30).stdout
        if not source or len(source) > MAX_RAW_BYTES:
            raise GpuControlRefused("original GPU source manifest missing or exceeds capacity")
        source_ref = store.write("direct-gpu-source", {"commit": frame[arm + "_commit"],
            "tree_manifest_hex": source.hex()})
        binary = next(row for row in artifacts if row["path"] == str((build / "llama-bench").resolve()))
        libraries = [row for row in artifacts if Path(row["path"]).name.startswith("lib")]
        if not libraries or not any("libggml-hip" in row["path"] for row in libraries):
            raise GpuControlRefused("original GPU loader closure lacks its HIP provider")
        result[arm] = {"source_sha256": hashlib.sha256(source).hexdigest(),
            "source": source_ref.to_dict(), "binary_sha256": binary["sha256"],
            "linkage_sha256": schemas.content_hash(libraries), "files": artifacts}
    return result


def _t0_gates(store, reference, declaration, identities):
    anchor, ops, coherence, determinism, surface = _t0_material(store, reference, declaration, identities)
    request = api.EvaluationRequest(event_id="ake-gpu-control-t0",
        campaign_id=declaration["control_campaign_id"], candidate_id="akc-gpu-x4-load", tier="T0",
        backend="llama_gpu", phase="decode", cell_class="tiny_graph",
        protocol_id=api.PROTOCOL_VERSIONED_ID,
        artifact=api.ArtifactIdentity(**{key: identities["candidate"][key]
            for key in ("source_sha256", "binary_sha256", "linkage_sha256")}),
        anchor=anchor.identity(), evaluator=api.EvaluatorIdentity("P-AK-SEARCH-1/v1",
            schemas.content_hash(declaration["source"]), _ref(StoredArtifact(**reference))),
        scope_denominator=api.ScopeDenominator("full", (), (), 8),
        scope_manifest_sha256=schemas.content_hash(declaration["fixture"]["frame"]), co_residency="single",
        determinism=api.DeterminismReport(determinism.measured_class(), determinism.runs),
        metric="decode_tokens_per_s", metric_direction="higher_better", reps=9,
        change_class="layout", anchor_tier="T0", transfer_ratio_to=(),
        created_at=declaration["issued_at"], campaign_controls=None, calibration=None)
    policy = campaign.HostOps._t0_evaluator_policy(request)
    return (correctness.check_backend_op_units(request, ops, surface, policy),
        correctness.check_determinism_class(request, determinism, policy)[0],
        correctness.check_output_coherence(request, coherence, policy, determinism)[0])


def _statistical(declaration):
    from .serving_preparation import ServingStatisticsDeclaration
    return ServingStatisticsDeclaration.from_dict(declaration["statistical"])


def _statistics(declaration, solve):
    statistical = _statistical(declaration)
    return st.CampaignStatistics(campaign_id=declaration["control_campaign_id"],
        campaign_seed=statistical.campaign_seed, effect_scale=statistical.effect_scale,
        hypothesis=statistical.hypothesis, margin=statistical.margin,
        stopping_rule=statistical.stopping_rule, stopping_rule_commitment=statistical.commitment,
        split_rule=statistical.split_rule, construction=st.select_construction(statistical.construction_id),
        calibration=solve.outputs, aa_effect_pool=solve.aa_effect_pool,
        anchor_calibration_values=solve.anchor_calibration_values)


def _blocks(rows):
    return tuple(st.PairedBlock(**{**row, "anchor_samples": tuple(row["anchor_samples"]),
        "candidate_samples": tuple(row["candidate_samples"])}) for row in rows)


def _solve_material(declaration, material, reference):
    calibration = {key: _blocks(material["pairs"][key]) for key in ("aa", "neutral")}
    return rc._solve({"backend": "llama_gpu", "phase": "decode", "cell_class": "tiny_graph"},
        _statistical(declaration), calibration, reference)


def _collect(store, claims, declaration, root, deadline):
    """Checkpoint original launches; known failed arms are not valid membership."""
    statistical = _statistical(declaration)
    window = BenchWindow(store=store, claims=claims, declaration=declaration,
        statistical=statistical, root=root, deadline=deadline)
    state = window.state
    if "open" not in state:
        state["open"] = store.write("direct-gpu-boundary", rw.boundary(window.config,
            claims.cpu.owner, marker="open", gpu_claim=claims.gpu)).to_dict()
        state["identities"] = _identities(store, window.frame)
        window._checkpoint()
    if "t0" not in state:
        state["t0"] = _collect_t0(store, claims, window.frame, state["identities"], deadline).to_dict()
        window._checkpoint()
    gates = _t0_gates(store, state["t0"], declaration, state["identities"])
    if any(row.check.outcome != schemas.PASS for row in gates):
        raise GpuControlRefused("original GPU control T0 failed: " + "; ".join(
            reason for row in gates for reason in row.check.reasons))
    pairs = {}
    count = statistical.controls.calibration_block_count
    if not 1 <= count <= rc.MAX_BLOCKS:
        raise GpuControlRefused("GPU control calibration exceeds original collection capacity")
    # As in the original live-control constructor, both neutral arms are the
    # exact unchanged anchor instrument; separate fresh launches, not A/A reuse.
    for label in ("aa", "neutral"):
        schedule = st.OrderSchedule.derive(campaign_seed=statistical.campaign_seed,
            candidate_id="akc-gpu-control-" + label, base_blocks=count)
        pairs[label] = window.pairs(label, count, "anchor", schedule=schedule)
    material = {"pairs": {key: [ob._plain(asdict(row)) for row in value] for key, value in pairs.items()}}
    calibration = store.write("direct-gpu-calibration-material", material)
    solve = _solve_material(declaration, material, _ref(calibration))
    solve.require_accepted()
    statistics = _statistics(declaration, solve)
    count = statistical.stopping_rule.max_total_blocks(solve.outputs.b_min_blocks)
    for control_id in (controls.CONTROL_POSITIVE, controls.CONTROL_HISTORICAL_WIN_REPLAY):
        candidate_id = "akc-control-" + control_id.replace("_", "-")
        pairs[control_id] = window.pairs(control_id, count, "candidate",
            schedule=statistics.order_schedule(candidate_id))
    closed = store.write("direct-gpu-boundary", rw.boundary(window.config, claims.cpu.owner,
        marker="close", gpu_claim=claims.gpu))
    final_identities = _identities(store, window.frame)
    if final_identities != state["identities"]:
        raise GpuControlRefused("original GPU source/binary/loader identity changed during control window")
    return {"pairs": {key: [ob._plain(asdict(row)) for row in value] for key, value in pairs.items()},
        "calibration": calibration.to_dict(), "identities": state["identities"], "t0": state["t0"],
        "boundaries": {"open": state["open"], "close": closed.to_dict()},
        "final_identities": final_identities, "launches": state["launches"], "invalid": state["invalid"],
        "config": window.config.to_dict()}


def _window(store, declaration, material, blocks, statistics, control_id, raw_ref):
    from ..resource import preflight
    from . import search_window as sw
    boundaries = [_read(store, "direct-gpu-boundary", material["boundaries"][key]) for key in ("open", "close")]
    same = all(rw.same_claim(boundaries[0][key], boundaries[1][key]) for key in ("claim", "gpu_claim"))
    preflights, health = [], []
    for index, boundary in enumerate(boundaries):
        if boundary["snapshot"]["marker"] != ("open", "close")[index]:
            raise GpuControlRefused("original GPU window marker differs")
        original = sw.captured_preflight(boundary["snapshot"], self_pid=boundary["claim"]["owner_pid"],
            scope=preflight.PreflightScope(label="original direct CPU serving window",
                cpu_regions=frozenset(declaration["cpu_footprint"]["regions"]), protocol_id="P-AK-SEARCH-1"))
        if original.to_dict() != boundary["preflight"]:
            raise GpuControlRefused("original GPU preflight does not rederive")
        preflights.append(original.as_check())
    frame = declaration["fixture"]["frame"]
    config = SimpleNamespace(limits=SimpleNamespace(max_gap_s=material["config"]["limits"]["max_gap_s"]))
    for entry in material["launches"]:
        _value, raw, _device = _native(store, entry["reference"], frame=frame, arm=entry["arm"],
            membership=entry["membership"], config=config)
        for key in ("claim_open", "claim_close"):
            original = _read(store, "direct-gpu-held-observation", raw[key])
            same = same and all(rw.same_claim(boundaries[0][boundary_key], original[component])
                for boundary_key, component in (("claim", "cpu"), ("gpu_claim", "gpu")))
        responses = [{"raw": {"phase": "measurement", "started_monotonic_s": raw["started_monotonic_s"],
            "ended_monotonic_s": raw["ended_monotonic_s"]}}]
        health.append(rw.launch_health(raw["during_work"], responses, max_gap_s=config.limits.max_gap_s))
    identities = material["identities"]
    identity_ok = _check(identities == material["final_identities"] and bool(identities),
                         "original source/binary/loader closures at open and close")
    source_ok = _check(declaration["source"] == source_identity(), "original loaded GPU control supplier")
    anchor = api.AnchorIdentity(frame["anchor_commit"], identities["anchor"]["binary_sha256"],
        identities["anchor"]["linkage_sha256"], tool="llama-bench")
    def free(boundary):
        row = boundary["snapshot"]["storage"]
        return _check(bool(row) and row["free_bytes"] >= _statistical(declaration).controls.storage_floor_bytes_free,
            "original declared free-storage floor", unavailable=not row)
    bootstrap = schemas.Check(schemas.PASS, ("internal evaluator-control bootstrap only; never candidate evidence",))
    return api.WindowAttestations(resource_claim_receipt=raw_ref,
        resource_claim_open=_check(all(boundaries[0][key]["status"] == "held" for key in ("claim", "gpu_claim")), "original component open"),
        resource_claim_close=_check(all(boundaries[1][key]["status"] == "held" for key in ("claim", "gpu_claim")), "original component close"),
        resource_claim_same_holder=_check(same, "original same held components across every control launch"),
        no_concurrent_inference=schemas.Check.worst_of(preflights), preflight_attestation_ref=raw_ref,
        host_receipt=raw_ref, host_health=schemas.Check.worst_of(health),
        anchor_at_open=anchor if identity_ok.outcome == schemas.PASS else None,
        anchor_at_close=anchor if identity_ok.outcome == schemas.PASS else None,
        anchor_gate=st.anchor_gate_check(tuple(row.anchor_samples[0] for row in blocks),
            band=statistics.calibration, b_min=statistics.b_min),
        evaluator_bundle=source_ok, runtime_source_label=schemas.Check.worst_of((source_ok, identity_ok)),
        recipe=api.RecipeReceipt("autokernel.loop.bench.run_once", declaration["source"]["files"][bench.__name__],
            schemas.content_hash(frame)), storage_open=free(boundaries[0]), storage_close=free(boundaries[1]),
        strata=_check(all(row.stratum == api.STRATUM_SELECTION for row in blocks), "original control selection stratum"),
        stopping_rule_id=statistics.stopping_rule.rule_id,
        rule_immutability=statistics.stopping_rule_commitment.verify(statistics.stopping_rule),
        order_randomized=statistics.order_schedule(
            "akc-control-" + control_id.replace("_", "-")).check_observed(blocks),
        order_seed=statistics.campaign_seed, aa_cadence=bootstrap,
        controls=api.ControlPanel(*(bootstrap for _ in range(5))),
        calibration=_check(statistics.calibration.accepted, "original measured calibration solve"),
        control_definitions_immutable=controls.verify_control_definitions(), raw_evidence_ref=raw_ref)


def _compose(store, declaration, material, raw_ref):
    """Original reducers mint each verdict; no supplied promoted flag."""
    calibration = _read(store, "direct-gpu-calibration-material", material["calibration"])
    if calibration["pairs"] != {key: material["pairs"][key] for key in ("aa", "neutral")}:
        raise GpuControlRefused("original GPU calibration vectors differ")
    solve = _solve_material(declaration, calibration, _ref(StoredArtifact(**material["calibration"])))
    solve.require_accepted()
    statistics = _statistics(declaration, solve)
    frame = declaration["fixture"]["frame"]
    config = SimpleNamespace(limits=SimpleNamespace(max_gap_s=material["config"]["limits"]["max_gap_s"]))
    seen = set()
    device_rows = []
    for entry in material["launches"]:
        membership = tuple(entry["membership"])
        if membership in seen:
            raise GpuControlRefused("original GPU launch membership duplicated")
        seen.add(membership)
        value, raw, _device = _native(store, entry["reference"], frame=frame, arm=entry["arm"],
            membership=entry["membership"], config=config)
        label, index, arm = membership
        expected_arm = "candidate" if label in (controls.CONTROL_POSITIVE,
            controls.CONTROL_HISTORICAL_WIN_REPLAY) and arm == "candidate" else "anchor"
        if entry["arm"] != expected_arm or material["pairs"][label][index][arm + "_samples"] != [value]:
            raise GpuControlRefused("original GPU paired value differs from its native process")
        device_rows.append((label, index, arm, {"during_work": raw["during_work"]}, [{"raw": {
            "phase": "measurement", "started_monotonic_s": raw["started_monotonic_s"],
            "ended_monotonic_s": raw["ended_monotonic_s"]}}]))
    expected = {(label, index, arm) for label, rows in material["pairs"].items()
        for index in range(len(rows)) for arm in ("anchor", "candidate")}
    if seen != expected:
        raise GpuControlRefused("original GPU paired table lacks exact native launch membership")
    from .runtime_admission import _device_state
    device = _device_state(device_rows, raw_ref)
    gates = _t0_gates(store, material["t0"], declaration, material["identities"])
    frame, identities = declaration["fixture"]["frame"], material["identities"]
    for arm in ("anchor", "candidate"):
        source = _read(store, "direct-gpu-source", identities[arm]["source"])
        if source["commit"] != frame[arm + "_commit"] or hashlib.sha256(
                bytes.fromhex(source["tree_manifest_hex"])).hexdigest() != identities[arm]["source_sha256"]:
            raise GpuControlRefused("original GPU source manifest does not rederive")
    anchor = api.AnchorIdentity(frame["anchor_commit"], identities["anchor"]["binary_sha256"],
        identities["anchor"]["linkage_sha256"], tool="llama-bench")
    class Gates:
        def run_gates(self, request):
            return gates
    dispatcher = api.TierDispatcher(gate_runners={tier: Gates() for tier in ("T0", "T1", "T2")})
    reducer = st.PairedBlockReducer(statistics)
    observations, evaluations = {}, {}
    for control_id, final_tier in ((controls.CONTROL_POSITIVE, "T1"),
                                  (controls.CONTROL_HISTORICAL_WIN_REPLAY, "T2")):
        blocks = _blocks(material["pairs"][control_id])
        definition = next(row for row in controls.CONTROL_DEFINITIONS if row.control_id == control_id)
        fixture = cr.ControlFixture(definition.fixture_id, control_id, final_tier,
            "akc-control-" + control_id.replace("_", "-"),
            api.ArtifactIdentity(**{key: identities["candidate"][key]
                for key in ("source_sha256", "binary_sha256", "linkage_sha256")}),
            api.DeterminismReport("not_measured", 0), declaration["issued_at"],
            max(row.measured_at for row in blocks), api.STRATUM_SELECTION,
            tuple(row.anchor_samples for row in blocks), tuple(row.candidate_samples for row in blocks))
        fixtures = cr.resolve_fixture_set(fixtures=(fixture,),
            pinned_digest=schemas.content_hash(cr._fixture_payload((fixture,))), source_label=raw_ref)
        binding = cr.CampaignBinding(declaration["control_campaign_id"], "llama_gpu", "decode", "tiny_graph",
            api.PROTOCOL_VERSIONED_ID, api.EvaluatorIdentity("P-AK-SEARCH-1/v1",
                schemas.content_hash(declaration["source"]), raw_ref),
            api.ScopeDenominator("full", (), (), 8), schemas.content_hash(frame), "single",
            "decode_tokens_per_s", "higher_better", frame["reps"], "layout", anchor,
            _statistical(declaration).controls, solve.outputs)
        stages = []
        class Pipeline:
            def evaluate(self, submission):
                # Static fixture transport must preserve the observed vectors,
                # schedule and original stopping commitment. It cannot turn a
                # different launch order or later declaration into this run.
                if [row.order for row in submission.blocks] != [row.order for row in blocks]:
                    raise GpuControlRefused("original GPU control launch order differs from evaluator schedule")
                if datetime.fromisoformat(statistics.stopping_rule_commitment.committed_at) > min(
                        datetime.fromisoformat(row.measured_at) for row in blocks):
                    raise GpuControlRefused("GPU stopping declaration postdates original control measurements")
                for tier in (("T0", "T1") if final_tier == "T1" else ("T0", "T1", "T2")):
                    request = replace(submission.request, tier=tier, anchor_tier=tier, device_state=device)
                    effect = None if tier == "T0" else reducer.reduce_blocks(request, submission.blocks)
                    outcome = dispatcher.dispatch(request, submission.window, effect=effect)
                    stages.append(outcome.durable_payload)
                    if outcome.verdict.status != api.STATUS_PASS:
                        return outcome
                return outcome
        runner = cr.ExecutedControlRunner(pipeline=Pipeline(), fixtures=fixtures,
            binding=binding, campaign_statistics=statistics)
        window_id = declaration["control_campaign_id"] + ":" + control_id
        window = _window(store, declaration, material, blocks, statistics, control_id, raw_ref)
        runner.open_window(window_id=window_id, window=window)
        try:
            observations[control_id] = runner.run_control(definition, controls.ControlRunContext(
                declaration["control_campaign_id"], "llama_gpu", "decode", "tiny_graph", window_id,
                final_tier, declaration["control_seeds"][control_id], anchor, resolution().declaration))
        finally:
            runner.close_window()
        evaluations[control_id] = stages
    return observations, evaluations


def _belief_receipt(declaration, result):
    """Prospective native rows, including failed setup; no read-side tuple."""
    if declaration.get("belief_capture_schema") != BELIEF_SCHEMA:
        return None
    observations = result["observations"]
    # A failed setup is an observed inability to run, not a failed scientific
    # control. Its row is a count of original native completed launches only.
    basis = {"declaration": result["declaration"], "material": result["material"],
        "window": result["window"], "error": result["error"], "observations": observations,
        "tier_evaluations": result["tier_evaluations"], "fixture_sha256": FIXTURE_SHA256,
        "source": declaration["source"], "frame": declaration["fixture"]["frame"],
        "held_components": declaration["holders"], "completed_launches": result["completed_launches"]}
    rows = []
    for control_id in (controls.CONTROL_POSITIVE, controls.CONTROL_HISTORICAL_WIN_REPLAY):
        observation = observations[control_id]
        native = {"control_id": control_id, "observation": observation, "basis": basis}
        row = {"measurement_id": "direct_gpu_" + control_id + "_ran",
            "metric": "autokernel_control_execution_observed", "value": float(observation["ran"]),
            "unit": "fraction", "metric_direction": "higher_better", "category": "BASELINE",
            "protocol_id": "", "reps": observation["abs_effect_count"] or None,
            "reps_basis": "observed:original GPU control paired blocks",
            "claim": f"Original GPU control {control_id} ran={observation['ran']}; native outcome retained",
            "native_verdict": observation["verdict_status"],
            "extra": {"producer_id": PRODUCER_ID, "producer_sha256": declaration["source"]["self"],
                "evidence_basis": native, "evidence_sha256": schemas.content_hash(native),
                "scope": "original control fixture only; no candidate gain or qualification"}}
        row["measurement_sha256"] = schemas.content_hash(row)
        rows.append(row)
    receipt = {"schema": BELIEF_SCHEMA, "status": "complete", "campaign_id": declaration["control_campaign_id"],
        "created_at": declaration["issued_at"], "ended_at": result["ended_at"],
        "producer": {"producer_id": PRODUCER_ID, "path": "scripts/kernel_rnd/autokernel/loop/direct_gpu_control.py",
            "sha256": declaration["source"]["self"]}, "native_basis": basis, "belief_measurements": rows}
    receipt["receipt_sha256"] = schemas.content_hash(receipt)
    return receipt


def run_or_reopen(*, store, held_claim, gpu_claim, campaign_id, window_index,
                  reference=None, deadline_monotonic_s=None, statistical=None, recovery_reference=None):
    """Return original resolution, two computed observations, and durable ref."""
    from . import archive, status, runtime_recovery
    from .measurement_capture import ArtifactStore
    if type(store) is not ArtifactStore or type(held_claim) is not HeldCpuClaim or type(gpu_claim) is not HeldCpuClaim:
        raise TypeError("GPU control requires the original store and both acquired contexts")
    if not isinstance(campaign_id, str) or not campaign_id.startswith("ak-") or type(window_index) is not int or window_index < 0:
        raise GpuControlRefused("GPU control original campaign/window identity required")
    declared_resolution = resolution()
    if reference is not None:
        result = _read(store, NAMESPACE, reference)
        declaration = _read(store, "direct-gpu-declaration", result["declaration"])
        if (declaration["parent_campaign_id"] != campaign_id or declaration["window_index"] != window_index
                or declaration["source"] != source_identity() or declaration["fixture"] != fixture()):
            raise GpuControlRefused("original GPU control source/frame/campaign differs; no relabeling")
        if result["error"] is None:
            material = _read(store, "direct-gpu-material", result["material"])
            observations, stages = _compose(store, declaration, material, _ref(StoredArtifact(**result["material"])))
        else:
            stages = {}
            observations = {control_id: controls.ControlObservation(control_id, False,
                could_not_run_reason=result["error"], evidence_ref=_ref(StoredArtifact(**result["declaration"])))
                for control_id in (controls.CONTROL_POSITIVE, controls.CONTROL_HISTORICAL_WIN_REPLAY)}
        if {key: row.to_dict() for key, row in observations.items()} != result["observations"] or stages != result["tier_evaluations"]:
            raise GpuControlRefused("original GPU control outcome does not rederive")
        return declared_resolution, observations, StoredArtifact(**reference)
    identity = schemas.content_hash({"campaign_id": campaign_id, "window_index": window_index})
    root = store.root / ("gpu-control-" + identity)
    # Follow only this original allocation's completed recovery join, never a
    # caller-supplied directory or a newest-file heuristic.
    for _ in range(64):
        if not (root / "replacement.json").exists():
            break
        replacement, _sha = serial_run._json(root / "replacement.json", limit=65536)
        runtime_recovery.reopen(replacement["recovery"])
        root = store.root / ("gpu-control-" + schemas.content_hash(replacement))
    else:
        raise GpuControlRefused("GPU control replacement chain exceeds finite capacity")
    result_path = root / "result.json"
    if result_path.exists():
        prior, _sha = serial_run._json(result_path, limit=4096)
        return run_or_reopen(store=store, held_claim=held_claim, gpu_claim=gpu_claim,
            campaign_id=campaign_id, window_index=window_index, reference=prior)
    declaration_path = root / "declaration.json"
    if declaration_path.exists():
        original, _sha = serial_run._json(declaration_path, limit=MAX_RAW_BYTES)
        declaration = _read(store, "direct-gpu-declaration", original)
        current_holders = {"cpu": runtime_recovery.holder_identity(held_claim),
                           "gpu": runtime_recovery.holder_identity(gpu_claim)}
        if declaration["holders"] != current_holders:
            from .loop import RunAborted
            previous, _sha = serial_run._json(root / "window.json", limit=4 * 1024 * 1024)
            if previous["pending"] is not None or recovery_reference is None:
                raise RunAborted("original GPU control child cleanup unresolved; no fresh window or fallback")
            binding = runtime_recovery.replacement(reference=recovery_reference,
                original_holder=declaration["holders"]["cpu"], new_holder=held_claim)
            old_gpu = runtime_recovery.reopen(recovery_reference).get("gpu_component")
            if (old_gpu is None or declaration["holders"]["gpu"] != {
                    "context_id": old_gpu["context_id"], "domain": old_gpu["domain"]}
                    or gpu_claim.observe()["status"] != "held" or gpu_claim._domain != held_claim._domain
                    or (gpu_claim._domain["boot_id"] == old_gpu["domain"]["boot_id"]
                        and gpu_claim._started_at < old_gpu["ended_at"])):
                raise RunAborted("GPU control recovery lacks released old and actual new component contexts")
            replacement = {**binding, "gpu_holder": current_holders["gpu"], "previous_declaration": original}
            next_root = store.root / ("gpu-control-" + schemas.content_hash(replacement))
            next_root.mkdir(exist_ok=True)
            next_declaration = {**declaration, "holders": current_holders, "replacement": replacement,
                "issued_at": datetime.now(timezone.utc).isoformat()}
            next_ref = store.write("direct-gpu-declaration", next_declaration)
            archive._retain_bytes(next_root / "declaration.json", json.dumps(next_ref.to_dict(),
                sort_keys=True, separators=(",", ":")).encode() + b"\n")
            archive._retain_bytes(root / "replacement.json", json.dumps(replacement,
                sort_keys=True, separators=(",", ":")).encode() + b"\n")
            return run_or_reopen(store=store, held_claim=held_claim, gpu_claim=gpu_claim,
                campaign_id=campaign_id, window_index=window_index, deadline_monotonic_s=deadline_monotonic_s,
                statistical=statistical, recovery_reference=recovery_reference)
        if declaration["source"] != source_identity() or declaration["fixture"] != fixture():
            raise GpuControlRefused("GPU control declaration source/frame changed")
    else:
        control_campaign = "ak-gpu-control-" + identity[:32]
        # These are prospective inputs, never the caller's measured outputs.
        supplied = None if statistical is None else replace(statistical,
            commitment=st.StoppingRuleCommitment.commit(statistical.stopping_rule,
                campaign_id=control_campaign, committed_at=datetime.now(timezone.utc).isoformat()))
        stats = rc.declare_statistics(store=store, campaign_id=control_campaign, epoch=identity, supplied=supplied)
        declaration = {"schema": SCHEMA, "parent_campaign_id": campaign_id, "window_index": window_index,
            "control_campaign_id": control_campaign, "fixture": fixture(), "source": source_identity(),
            "statistical": stats.to_dict(), "issued_at": datetime.now(timezone.utc).isoformat(),
            "belief_capture_schema": BELIEF_SCHEMA,
            "control_seeds": {control_id: controls.derive_control_seed(campaign_seed=stats.campaign_seed,
                control_id=control_id, epoch=window_index)
                for control_id in (controls.CONTROL_POSITIVE, controls.CONTROL_HISTORICAL_WIN_REPLAY)},
            "holders": {"cpu": runtime_recovery.holder_identity(held_claim),
                        "gpu": runtime_recovery.holder_identity(gpu_claim)},
            "cpu_footprint": {key: held_claim[key] for key in ("device_id", "cpu_list", "regions")}}
        original = store.write("direct-gpu-declaration", declaration).to_dict()
        root.mkdir(exist_ok=True)
        archive._retain_bytes(declaration_path, json.dumps(original, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    material_ref, error, stages = None, None, {}
    try:
        _resolution, claims = require_available(store=store, held_claim=held_claim, gpu_claim=gpu_claim)
        material = _collect(store, claims, declaration, root, deadline_monotonic_s)
        material_ref = store.write("direct-gpu-material", material)
        observations, stages = _compose(store, declaration, material, _ref(material_ref))
    except tp.ProcessEscaped:
        raise
    except (GpuControlRefused, rc.RuntimeLaunchBudgetExhausted, st.CalibrationFailed,
            tp.InstrumentCapabilityError, tp.AnchorCaptureIncomplete, OSError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        observations = {control_id: controls.ControlObservation(control_id, False,
            could_not_run_reason=error, evidence_ref=_ref(StoredArtifact(**original)))
            for control_id in (controls.CONTROL_POSITIVE, controls.CONTROL_HISTORICAL_WIN_REPLAY)}
    window = None
    if (root / "window.json").exists():
        body, _sha = serial_run._json(root / "window.json", limit=4 * 1024 * 1024)
        window = store.write("direct-gpu-window-checkpoint", body)
    result = {"schema": SCHEMA, "declaration": original,
        "material": None if material_ref is None else material_ref.to_dict(),
        "window": None if window is None else window.to_dict(), "error": error,
        "observations": {key: row.to_dict() for key, row in observations.items()},
        "tier_evaluations": stages, "ended_at": datetime.now(timezone.utc).isoformat(),
        "completed_launches": 0 if window is None else len(body["launches"])}
    try:
        receipt = _belief_receipt(declaration, result)
        result["belief_receipt"] = None if receipt is None else store.write("direct-gpu-control-beliefs", receipt).to_dict()
        result["belief_export_error"] = None
    except Exception as exc:
        result["belief_receipt"] = None
        result["belief_export_error"] = f"{type(exc).__name__}: {exc}"
    saved = store.write(NAMESPACE, result)
    # Setup interruption preserves completed valid prefixes and writes its own
    # immutable failed observation, but never seals the frame as complete.
    status.write_json(root, "last-attempt.json", saved.to_dict(), prefix=".gpu-control-")
    if error is None:
        archive._retain_bytes(result_path, json.dumps(saved.to_dict(), sort_keys=True, separators=(",", ":")).encode() + b"\n")
    return declared_resolution, observations, saved
