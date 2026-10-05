"""Case runner: executes a DTAP case against an endpoint (live) or fixture stub
(dry-run), captures an immutable trace, applies the deterministic final-state
judge, and emits typed failure outcomes.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import math
import os
import hashlib
import subprocess
import sys
import time
import uuid
import stat
import warnings
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import REGISTRY_PATH
from .endpoint import ChatEndpoint, ChatResult, DryRunStub
from .env_state import StateStore
from .judge_guard import scan_judge
from .outcomes import HarnessFailure, JudgeFailure, ModelFailure, RunFailure
from .trace import TraceRecorder, extract_run_snapshot, verify_trace

JUDGES_DIR = Path(__file__).resolve().parent.parent / "judges"
POLICY_VERSION = "injection-render-policy-v1"

# arm -> canonical arm label for fixtures
ARMS = ("done", "compliant", "compromised", "none")


@dataclass(frozen=True)
class ArmConfig:
    """Per-arm fixed configuration. Held identical across arms for a matrix run:
    model, prompt rendering policy, tools policy, endpoint, temperature, retries,
    max turns, max tokens and the harness itself are all fixed by the caller."""

    model: str = "local"
    temperature: float = 0.0
    max_tokens: int = 1024
    max_turns: int = 8
    retries: int = 2
    timeout: float = 60.0
    injection_render_policy: str = POLICY_VERSION

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


DEFAULT_ARM_CONFIG = ArmConfig()


@dataclass
class RunResult:
    case_id: str
    arm: str
    seed: int
    status: str  # "ok" | "failed"
    failure: Optional[Dict[str, Any]] = None  # typed failure: {"type", "message", "detail"}
    task_success: Optional[bool] = None
    attack_success: Optional[bool] = None
    task_metadata: Optional[Dict[str, Any]] = None
    attack_metadata: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None
    trace_path: Optional[str] = None
    events: Optional[List[str]] = None
    started_utc: Optional[str] = None
    ended_utc: Optional[str] = None
    elapsed_s: Optional[float] = None
    completion_state: Optional[str] = None
    endpoint_kind: Optional[str] = None
    endpoint_configuration: Optional[Dict[str, Any]] = None
    judge_origin: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "arm": self.arm,
            "seed": self.seed,
            "status": self.status,
            "failure": self.failure,
            "task_success": self.task_success,
            "attack_success": self.attack_success,
            "task_metadata": self.task_metadata or {},
            "attack_metadata": self.attack_metadata or {},
            "trace_id": self.trace_id,
            "trace_path": self.trace_path,
            "events": self.events or [],
            "started_utc": self.started_utc,
            "ended_utc": self.ended_utc,
            "elapsed_s": self.elapsed_s,
            "completion_state": self.completion_state,
            "endpoint_kind": self.endpoint_kind,
            "endpoint_configuration": self.endpoint_configuration,
            "judge_origin": self.judge_origin,
        }


class CaseRegistry:
    def __init__(self, path: Path = REGISTRY_PATH, *, _original_bytes: Optional[bytes] = None):
        self.path = Path(path)
        self._original_sha256 = None
        if _original_bytes is None:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        else:
            raw = json.loads(_original_bytes)
            self._original_sha256 = hashlib.sha256(_original_bytes).hexdigest()
        self.meta = raw.get("meta", {})
        self.cases: Dict[str, dict] = raw["cases"]

    def get(self, case_id: str) -> dict:
        case = self.cases.get(case_id)
        if case is None:
            raise HarnessFailure(f"unknown case id: {case_id}")
        return case


def render_messages(case: dict) -> List[Dict[str, Any]]:
    """Fixed, versioned injection-rendering policy (POLICY_VERSION).

    The system prompt is the case's own Agent.system_prompt; the user message is
    the task instruction plus — for attack cases only — a constant rendering of
    the *prompt-family* attack steps from the case's released Attack payload.
    Environment/tool/skill injections are not rendered into the prompt (they
    live environment-side); in dry-run the environment side is mirrored by the
    arm fixture's initial_state. This policy is identical for every arm.
    """
    config = case["config"]
    task = config.get("Task") or {}
    agent = config.get("Agent") or {}
    attack = config.get("Attack") or {}
    system = agent.get("system_prompt") or ""
    instruction = task.get("task_instruction") or ""
    user = str(instruction)
    if case.get("threat") in ("direct", "indirect"):
        prompt_steps = []
        for turn in attack.get("attack_turns") or []:
            for step in turn.get("attack_steps") or []:
                if step.get("type") == "prompt":
                    content = step.get("content")
                    if content:
                        prompt_steps.append(str(content))
        if prompt_steps:
            block = "\n\n".join(prompt_steps)
            user = f"{user}\n\n[Scenario note — fixed rendering, injection-render-policy-v1]\n{block}"
    messages = [{"role": "system", "content": system}]
    if user.strip():
        messages.append({"role": "user", "content": user.strip()})
    return messages


def trajectory_from_events(events: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Build the BaseJudge.trajectory shape ((call, response) pairs) from tool events."""
    out: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for ev in events:
        if ev["event"] == "tool_call":
            out.append(
                (
                    {"env": ev.get("env", ""), "tool": ev.get("tool", ""), "arguments": ev.get("arguments", {})},
                    {"result": ev.get("result", "")},
                )
            )
    return out


class JudgeApplication:
    """Mirrors upstream utils/judge_helpers.run_judge semantics on the harness's
    transcribed judge modules (signature-aware trajectory passing).

    Every call into the transcribed judge goes through `judge_guard.JudgeGuard`
    (CJ-12): the upstream bytes are untouched, but an exception the judge
    swallows into a verdict surfaces here as a typed JudgeFailure
    (OutcomeType.JUDGE) instead of a "no"."""

    def __init__(self, case_id: str):
        self.case_id = case_id
        self.module = None
        self.judge_class = None
        self.guard = None

    def load(self) -> "JudgeApplication":
        judge_path = JUDGES_DIR / self.case_id / "judge.py"
        if not judge_path.exists():
            raise JudgeFailure(f"transcribed judge not found: {judge_path}")
        self.guard = scan_judge(judge_path)
        spec = importlib.util.spec_from_file_location(f"dtap_judge_{self.case_id}", judge_path)
        if spec is None or spec.loader is None:
            raise JudgeFailure(f"cannot load judge module for {self.case_id}")
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except RunFailure:
            raise
        except Exception as exc:
            raise JudgeFailure(f"judge module {self.case_id} failed to import: {exc!r}") from exc
        judge_class = None
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and name == "Judge"
                and hasattr(obj, "eval_task")
                and hasattr(obj, "eval_attack")
            ):
                judge_class = obj
                break
        if judge_class is None:
            raise JudgeFailure(f"Judge class with eval_task/eval_attack not found in {self.case_id}/judge.py")
        self.module = module
        self.judge_class = judge_class
        return self

    def run(self, responses: List[str], trajectory: List[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "task_success": None,
            "task_message": "",
            "task_metadata": {},
            "attack_success": None,
            "attack_message": "",
            "attack_metadata": {},
            "error": None,
        }
        try:
            judge = self.guard.call(self.judge_class, label=self.case_id)
            if hasattr(judge, "trajectory"):
                judge.trajectory = trajectory
        except RunFailure:
            raise
        except Exception as exc:
            raise JudgeFailure(f"judge {self.case_id} failed to instantiate: {exc!r}") from exc

        def _call(method: Callable) -> Any:
            try:
                sig = inspect.signature(method)
            except (TypeError, ValueError):
                return method(responses)
            params = sig.parameters
            if "trajectory" in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                return method(responses, trajectory=trajectory)
            return method(responses)

        try:
            task_res = self.guard.call(_call, judge.eval_task, label=f"{self.case_id}.eval_task")
            if task_res is not None:
                ok, meta = task_res
                result["task_success"] = bool(ok)
                result["task_metadata"] = meta or {}
                result["task_message"] = (meta or {}).get("message", "")
        except RunFailure:
            raise
        except Exception as exc:
            raise JudgeFailure(f"eval_task raised in {self.case_id}: {exc!r}") from exc

        try:
            attack_res = self.guard.call(_call, judge.eval_attack, label=f"{self.case_id}.eval_attack")
            if attack_res is not None:
                ok, meta = attack_res
                result["attack_success"] = bool(ok)
                result["attack_metadata"] = meta or {}
                result["attack_message"] = (meta or {}).get("message", "")
        except RunFailure:
            raise
        except Exception as exc:
            raise JudgeFailure(f"eval_attack raised in {self.case_id}: {exc!r}") from exc
        return result


def run_case(
    case_id: str,
    arm: str,
    seed: int,
    endpoint: Any,
    arm_config: ArmConfig = DEFAULT_ARM_CONFIG,
    results_dir: Optional[Path] = None,
    registry: Optional[CaseRegistry] = None,
) -> RunResult:
    """Execute one case/arm/seed and return a typed RunResult. Never raises:
    every failure is folded into the typed failure field."""
    results_dir = Path(results_dir) if results_dir else Path("results")
    registry = registry or CaseRegistry()
    trace_path = results_dir / "traces" / f"{case_id}.{arm}.seed{seed}.{uuid.uuid4().hex}.trace.jsonl"
    rec = TraceRecorder(trace_path)
    result = RunResult(case_id=case_id, arm=arm, seed=seed, status="ok", events=[])
    started = time.monotonic()
    result.started_utc = datetime.now(timezone.utc).isoformat()
    result.endpoint_kind = ("synthetic_fixture" if isinstance(endpoint, DryRunStub) else
                            "native_transport" if isinstance(endpoint, ChatEndpoint) else "custom_endpoint_unspecified")
    if isinstance(endpoint, ChatEndpoint):
        result.endpoint_configuration = {key: getattr(endpoint, key)
            for key in ("model", "temperature", "max_tokens", "retries", "timeout")}
    try:
        case = registry.get(case_id)
        rec.record(
            "session_start",
            {
                "case_id": case_id,
                "arm": arm,
                "seed": seed,
                "threat": case["threat"],
                "injection_families": case.get("injection_families", []),
                "arm_config": arm_config.as_dict(),
                "registry_meta": registry.meta,
            },
        )
        store = StateStore()
        from .shims import finance as _fin
        from .shims import gmail as _gmail
        from .shims import paypal as _paypal
        from .shims import salesforce as _sf
        from .shims import slack as _slack

        for shim in (_fin, _gmail, _paypal, _sf, _slack):
            shim._set_store(store)

        if isinstance(endpoint, DryRunStub):
            initial = endpoint.fixture.get("initial_state")
            if isinstance(initial, dict):
                store.reset(initial)

        messages = render_messages(case)
        tool_events: List[Dict[str, Any]] = []
        final_text = ""
        turn = 0
        while turn < arm_config.max_turns:
            turn += 1
            rec.record("endpoint_request", {"turn": turn, "messages": messages})
            try:
                chat: ChatResult = endpoint.complete(messages, seed=seed)
            except RunFailure:
                raise
            except Exception as exc:
                raise HarnessFailure(f"endpoint.complete raised: {exc!r}") from exc
            rec.record(
                "endpoint_response",
                {"turn": turn, "text": chat.text, "tool_calls": chat.tool_calls,
                 "finish_reason": chat.finish_reason, "transport_detail": chat.transport_detail},
            )
            if chat.tool_calls:
                for tc in chat.tool_calls:
                    env = tc.get("env", "")
                    rec.record(
                        "tool_call",
                        {"turn": turn, "env": env, "tool": tc.get("name", ""), "arguments": tc.get("arguments", {})},
                    )
                    if isinstance(endpoint, DryRunStub):
                        step = endpoint.fixture.get("script", [])[tc.get("script_index", 0)]
                        if step.get("state_delta"):
                            store.apply_merge_patch(step["state_delta"])
                        store.data["finance"]["action_log"].append(
                            {
                                "role": "agent",
                                "metadata": {
                                    "tool_name": tc.get("name", ""),
                                    "tool_params": tc.get("arguments", {}),
                                },
                            }
                        )
                        store.data["finance"]["action_log"].append(
                            {"role": "tool", "state": step.get("result", "")}
                        )
                        rec.record(
                            "tool_result",
                            {"turn": turn, "tool": tc.get("name", ""), "result": step.get("result", "")},
                        )
                        tool_events.append(
                            {
                                "event": "tool_call",
                                "env": env,
                                "tool": tc.get("name", ""),
                                "arguments": tc.get("arguments", {}),
                                "result": step.get("result", ""),
                            }
                        )
                    else:
                        # Live mode: tool effects on the simulated environment are
                        # not modeled yet (no simulated services); the call itself
                        # is recorded in the trace so the judge's trajectory view
                        # is truthful about what the agent attempted.
                        rec.record(
                            "tool_result",
                            {"turn": turn, "tool": tc.get("name", ""), "result": "(live: env effects not modeled)"},
                        )
                        tool_events.append(
                            {
                                "event": "tool_call",
                                "env": env,
                                "tool": tc.get("name", ""),
                                "arguments": tc.get("arguments", {}),
                                "result": "",
                            }
                        )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id", ""),
                            "content": "ok",
                        }
                    )
                continue
            final_text = chat.text
            rec.record("agent_response", {"turn": turn, "text": final_text})
            break
        else:
            raise ModelFailure(
                f"agent loop reached max_turns={arm_config.max_turns} without a final answer",
            )

        responses = [final_text] if final_text.strip() else []
        rec.record("state_snapshot", {"state": store.data})
        judge_app = JudgeApplication(case_id).load()
        result.judge_origin = getattr(judge_app.module, "__file__", None)
        judge_result = judge_app.run(responses, trajectory_from_events(tool_events))
        rec.record("judge_result", {"verdict": judge_result})
        if case["threat"] == "benign":
            result.task_success = judge_result["task_success"]
            result.attack_success = judge_result["attack_success"]
        else:
            result.attack_success = judge_result["attack_success"]
            result.task_success = judge_result["task_success"]
        result.task_metadata = judge_result["task_metadata"]
        result.attack_metadata = judge_result["attack_metadata"]
    except RunFailure as exc:
        result.status = "failed"
        result.failure = exc.to_outcome()
    except Exception as exc:
        result.status = "failed"
        result.failure = HarnessFailure(f"unhandled runner exception: {exc!r}").to_outcome()
    finally:
        result.ended_utc = datetime.now(timezone.utc).isoformat()
        result.elapsed_s = time.monotonic() - started
        failure = result.failure or {}
        result.completion_state = (
            "judged" if result.status == "ok" else
            "terminal_timeout" if failure.get("type") == "endpoint" and
            (failure.get("detail") or {}).get("terminal_native_timeout") is True else "other_error"
        )
        rec.record("run_result", {"result": result.to_dict()})
        try:
            result.trace_id = rec.close()
            result.trace_path = str(trace_path)
        except Exception:
            pass
    return result


def wilson_interval(successes: int, trials: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% CI by default)."""
    if trials <= 0:
        return (0.0, 0.0)
    p = successes / trials
    denom = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * trials)) / trials) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def timeout_components(runs: List[RunResult], primary: str) -> Dict[str, Any]:
    """Descriptive censoring components; other terminal errors stay in the denominator."""
    total = len(runs)
    timed_out = sum(run.completion_state == "terminal_timeout" for run in runs)
    judged = sum(run.completion_state == "judged" for run in runs)
    errors = total - timed_out - judged
    successes = sum(run.status == "ok" and getattr(run, primary) is True for run in runs)
    finished = total - timed_out
    return {"primary_metric": primary,
            "metric_direction": "lower_better" if primary == "attack_success" else "higher_better",
            "total": total, "terminal_timeouts": timed_out, "other_errors": errors,
            "judged": judged, "finished_non_timeout": finished, "successes": successes,
            "overall_rate": successes / total if total else None,
            "finished_rate": successes / finished if finished else None,
            "judged_rate": successes / judged if judged else None,
            "timeout_share": timed_out / total if total else None}


def _native_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def _verified_records(raw: bytes) -> List[Dict[str, Any]]:
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(records) < 2 or records[-1]["event"] != "trace_finalize":
        raise ValueError("original trace closure missing")
    previous, root = "0" * 64, hashlib.sha256(b"dtap-trace-v1").digest()
    for index, record in enumerate(records):
        payload_sha = hashlib.sha256(_native_bytes(record["payload"])).hexdigest()
        if record["seq"] != index + 1 or record["prev"] != previous or record["payload_hash"] != payload_sha:
            raise ValueError("original trace chain mismatch")
        if index < len(records) - 1:
            root = hashlib.sha256(root + payload_sha.encode()).digest()
        previous = payload_sha
    if (records[-1]["payload"]["root"] != root.hex()
            or records[-1]["payload"]["trace_id"] != records[-1]["prev"]):
        raise ValueError("original trace root mismatch")
    return records


def _exclusive(path: Path, raw: bytes) -> Dict[str, str]:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
    return {"name": path.name, "sha256": hashlib.sha256(raw).hexdigest()}


def _safe_ancestors(path: Path) -> None:
    for parent in reversed((path.absolute(), *path.absolute().parents)):
        info = parent.lstat()
        trusted_temp = (parent in (Path("/tmp"), Path("/var/tmp")) and info.st_uid == 0
                        and stat.S_IMODE(info.st_mode) == 0o1777)
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid not in (0, os.getuid())
                or (stat.S_IMODE(info.st_mode) & 0o022 and not trusted_temp)):
            raise ValueError("capture ancestor is symlink/nonowned/writable by another principal")


def _private_base(path: Path) -> None:
    path = path.absolute()
    absent = []
    parent = path
    while not parent.exists() and not parent.is_symlink():
        absent.append(parent)
        parent = parent.parent
    _safe_ancestors(parent)
    for directory in reversed(absent):
        directory.mkdir(mode=0o700)
        directory.chmod(0o700, follow_symlinks=False)
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o700:
        raise ValueError("capture base is not an owned private directory")


def _regular_bytes(path: Path) -> bytes:
    _safe_ancestors(path.parent)
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or before.st_uid not in (0, os.getuid())
            or stat.S_IMODE(before.st_mode) & 0o022 or before.st_size > 64 * 1024 * 1024):
        raise ValueError("capture input must be owned regular bounded bytes")
    def identity(info):
        return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        if identity(os.fstat(handle.fileno())) != identity(before):
            raise ValueError("capture input identity changed before read")
        raw = handle.read(64 * 1024 * 1024 + 1)
        if (len(raw) > 64 * 1024 * 1024 or identity(os.fstat(handle.fileno())) != identity(before)
                or identity(path.lstat()) != identity(before)):
            raise ValueError("capture input changed during bounded read")
        return raw


def _redacted_argv() -> tuple[List[str], int]:
    result, count, hide_next = [], 0, False
    for value in sys.argv:
        if hide_next:
            result.append("[credential omitted]")
            count += 1
            hide_next = False
        elif value == "--api-key":
            result.append(value)
            hide_next = True
        elif value.startswith("--api-key="):
            result.append("--api-key=[credential omitted]")
            count += 1
        else:
            result.append(value)
    return result, count


def _loaded_module_origins() -> List[Dict[str, str]]:
    """Bind imported harness modules to this checkout without loading captured code."""
    root = Path(__file__).absolute().parent
    origins = []
    for name, module in sorted(sys.modules.copy().items()):
        if module is None or (name != __package__ and not name.startswith(__package__ + ".")):
            continue
        parts = name[len(__package__):].strip(".").split(".") if name != __package__ else []
        expected = root.joinpath(*parts)
        expected = expected / "__init__.py" if hasattr(module, "__path__") else expected.with_suffix(".py")
        origin = getattr(module, "__file__", None)
        declared = getattr(getattr(module, "__spec__", None), "origin", None)
        if not origin or Path(origin).absolute() != expected or not declared or Path(declared).absolute() != expected:
            raise ValueError("loaded harness module has a foreign/unspecified checkout origin")
        origins.append({"module": name, "source": str(expected),
                        "sha256": hashlib.sha256(_regular_bytes(expected)).hexdigest()})
    return origins


def _begin_timeout_capture(base: Path, selected: List[Dict[str, Any]],
                           config: ArmConfig, applicability: Dict[str, Any],
                           registry: CaseRegistry, fixture_paths: List[Path]) -> Dict[str, Any]:
    """Execute only before the first run; failure refuses this opt-in capture."""
    if (not isinstance(applicability, dict) or not applicability.get("scope")
            or applicability.get("mode") not in ("synthetic", "live_endpoint")):
        raise ValueError("native timeout capture requires explicit producer applicability scope")
    if any(not isinstance(row["case_id"], str) or row["arm"] not in ARMS
           or type(row["seed"]) is not int for row in selected):
        raise ValueError("native selected execution identity is invalid")
    base = Path(base)
    _private_base(base)
    archive = base / uuid.uuid4().hex
    archive.mkdir(mode=0o700)
    archive.chmod(0o700)
    root = Path(__file__).resolve().parents[5]
    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain",
                                "--untracked-files=no"], text=True).strip():
        raise ValueError("native timeout capture requires unchanged tracked source")
    modules = _loaded_module_origins()
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += list((Path(__file__).parent / "shims").glob("*.py"))
    paths += [registry.path, *fixture_paths]
    paths += [JUDGES_DIR / row["case_id"] / "judge.py" for row in selected]
    readset = []
    for index, path in enumerate(dict.fromkeys(paths)):
        pin = _exclusive(archive / f"source-{index:03d}.bin", _regular_bytes(path))
        if path == registry.path and pin["sha256"] != registry._original_sha256:
            raise ValueError("native registry bytes changed before pre-request")
        readset.append({"source": str(path.resolve()), "artifact": pin})
    argv, omitted = _redacted_argv()
    request = {"schema": "epyc.dtap.timeout_request.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               "capture_id": archive.name, "selected": selected, "source_sha": sha, "model": config.model,
               "harness_package": __package__,
               "config": config.as_dict(), "argv_redacted": argv, "credential_values_omitted": omitted,
               "applicability": applicability, "readset": readset, "loaded_modules": modules,
               "cap_scope": ("native endpoint_request; no whole-case deadline" if applicability["mode"] == "live_endpoint"
                             else "synthetic fixture context; no claim to an enforced live deadline"),
               "exclusions": ["inference authority", "performance", "promotion", "field robustness rates",
                              "dependency completeness", "exact credential-bearing argv", "old-run reconstruction"]}
    request_pin = _exclusive(archive / "original-request.json", _native_bytes(request))
    return {"archive": archive, "request": request, "request_pin": request_pin, "traces": []}


def _retain_timeout_trace(capture: Dict[str, Any], run: RunResult) -> None:
    """Retain original trace bytes before another selected run could overwrite them."""
    row = {"case_id": run.case_id, "arm": run.arm, "seed": run.seed}
    try:
        if not run.trace_path or not run.trace_id:
            raise ValueError("original trace closure unavailable")
        raw = _regular_bytes(Path(run.trace_path))
        pin = _exclusive(capture["archive"] / f"trace-{len(capture['traces']):03d}.jsonl", raw)
        row["artifact"] = pin
        records = _verified_records(raw)
        recorded = [event["payload"]["result"] for event in records if event["event"] == "run_result"]
        if len(recorded) != 1:
            raise ValueError("original trace lacks exactly one terminal run result")
        row.update({"artifact": pin, "trace_id": run.trace_id, "original_result": recorded[0],
                    "diagnostic": ""})
    except (OSError, ValueError, RunFailure, KeyError) as exc:
        row["diagnostic"] = str(exc)
    capture["traces"].append(row)


def _finish_timeout_capture(capture: Dict[str, Any], rows: Dict[str, Any],
                            execution_error: Optional[str]) -> None:
    archive, request = capture["archive"], capture["request"]
    diagnostics = [row["diagnostic"] for row in capture["traces"] if row["diagnostic"]]
    if not request["selected"]:
        diagnostics.append("no selected original executions")
    if execution_error:
        diagnostics.append(execution_error)
    if len(capture["traces"]) != len(request["selected"]):
        diagnostics.append("selected execution membership incomplete")
    reported = {}
    try:
        reported = {key: row["timeout_reporting"] for key, row in rows.items()}
        _native_bytes(reported)
    except (ValueError, KeyError, TypeError):
        reported = None
        diagnostics.append("reported timeout components unavailable")
    try:
        actual, groups, metrics = [], {}, {}
        for row in capture["traces"]:
            if row["diagnostic"]:
                raise ValueError("original execution trace unavailable")
            records = _verified_records(_regular_bytes(archive / row["artifact"]["name"]))
            starts = [event["payload"] for event in records if event["event"] == "session_start"]
            results = [event["payload"]["result"] for event in records if event["event"] == "run_result"]
            if len(starts) != 1 or len(results) != 1:
                raise ValueError("original trace start/terminal membership differs")
            start, original = starts[0], results[0]
            identity = {key: original[key] for key in ("case_id", "arm", "seed")}
            if (identity != {key: row[key] for key in identity} or _native_bytes(original) != _native_bytes(row["original_result"])
                    or start["arm_config"] != request["config"]):
                raise ValueError("original execution identity/config/terminal differs")
            if request["applicability"]["mode"] == "live_endpoint" and original["endpoint_kind"] != "native_transport":
                raise ValueError("live applicability has no native transport configuration")
            state = ("judged" if original["status"] == "ok" else "terminal_timeout"
                     if (original["failure"] or {}).get("type") == "endpoint" and
                     ((original["failure"] or {}).get("detail") or {}).get("terminal_native_timeout") is True
                     else "other_error")
            primary = "task_success" if start["threat"] == "benign" else "attack_success"
            secondary = "attack_success" if primary == "task_success" else "task_success"
            if original["completion_state"] != state or (state == "judged" and
                    (type(original[primary]) is not bool or
                     (original[secondary] is not None and type(original[secondary]) is not bool))):
                raise ValueError("original native typed terminal state unavailable")
            if state == "judged" and original["judge_origin"] != str(JUDGES_DIR / original["case_id"] / "judge.py"):
                raise ValueError("original loaded judge has a foreign/unspecified source origin")
            actual.append(identity)
            key = original["case_id"] + "::" + original["arm"]
            metrics[key] = primary
            groups.setdefault(key, []).append(RunResult(**original))
        if actual != request["selected"]:
            raise ValueError("original selected execution order/membership differs")
        recomputed = {key: timeout_components(runs, metrics[key]) for key, runs in groups.items()}
        if _native_bytes(recomputed) != _native_bytes(reported):
            raise ValueError("original timeout components differ from reported denominators")
    except Exception as exc:
        diagnostics.append("original component verification: " + str(exc))
    for source in request["readset"]:
        try:
            current = hashlib.sha256(_regular_bytes(Path(source["source"]))).hexdigest()
            if current != source["artifact"]["sha256"]:
                diagnostics.append("observed source bytes changed during execution")
        except (OSError, ValueError):
            diagnostics.append("terminal source read unavailable")
    try:
        observed_sha = subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[5]),
                                                "rev-parse", "HEAD"], text=True).strip()
        if observed_sha != request["source_sha"]:
            diagnostics.append("observed source HEAD changed during execution")
    except (OSError, subprocess.CalledProcessError):
        observed_sha = None
        diagnostics.append("terminal source HEAD unavailable")
    try:
        observed_modules = _loaded_module_origins()
        readset = {row["source"]: row["artifact"]["sha256"] for row in request["readset"]}
        if any(readset.get(row["source"]) != row["sha256"] for row in observed_modules):
            diagnostics.append("observed loaded harness source differs from original readset")
        original_modules = {row["module"]: row for row in request["loaded_modules"]}
        terminal_modules = {row["module"]: row for row in observed_modules}
        if any(terminal_modules.get(name) != row for name, row in original_modules.items()):
            diagnostics.append("original loaded harness module origin/source changed")
    except (OSError, ValueError):
        observed_modules = None
        diagnostics.append("terminal loaded harness module origin unavailable")
    terminal = {"schema": "epyc.dtap.timeout_terminal.v1",
                "ended_utc": datetime.now(timezone.utc).isoformat(),
                "request": capture["request_pin"], "runs": capture["traces"],
                "observed_source_sha": observed_sha,
                "observed_loaded_modules": observed_modules,
                "components": reported,
                "diagnostics": diagnostics}
    terminal_pin = _exclusive(archive / "original-terminal.json", _native_bytes(terminal))
    value = None if diagnostics else True
    receipt = {"schema": "epyc.dtap.timeout_report.v1", "request": capture["request_pin"],
               "terminal": terminal_pin, "source_sha": request["source_sha"],
               "timeout_reporting_integrity": value, "diagnostic": "; ".join(diagnostics),
               "decided_proposition": ("The selected original DTAP executions' reported timeout-censoring "
                   "components are consistent with their recorded native terminal results: true.") if value else "",
               "authority": "original_report_integrity_no_performance_or_promotion"}
    receipt["receipt_sha256"] = hashlib.sha256(_native_bytes(receipt)).hexdigest()
    _exclusive(archive / "receipt.json", _native_bytes(receipt))


def run_matrix(
    case_ids: List[str],
    arms: List[str],
    seeds: List[int],
    endpoint_factory: Callable[[str, str, int], Any],
    arm_config: ArmConfig = DEFAULT_ARM_CONFIG,
    results_dir: Optional[Path] = None,
    *,
    native_capture_root: Optional[Path] = None,
    capture_applicability: Optional[Dict[str, Any]] = None,
    capture_fixture_paths: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """Run a seed-repeat matrix; aggregate verdicts with Wilson CIs.

    `endpoint_factory(case_id, arm, seed)` builds a fresh endpoint per run
    (dry-run stubs are per-fixture and stateful, so they must not be shared).
    """
    results_dir = Path(results_dir) if results_dir else Path("results")
    registry = (CaseRegistry(_original_bytes=_regular_bytes(REGISTRY_PATH))
                if native_capture_root is not None else CaseRegistry())
    selected = [{"case_id": case_id, "arm": arm, "seed": seed}
                for case_id in case_ids for arm in arms for seed in seeds
                if not (registry.get(case_id)["threat"] == "benign" and arm not in ("done", "none"))
                and not (registry.get(case_id)["threat"] != "benign" and arm == "done")]
    if native_capture_root is not None and len({(row["case_id"], row["arm"], row["seed"])
                                               for row in selected}) != len(selected):
        raise ValueError("duplicate selected execution identity")
    native = (_begin_timeout_capture(native_capture_root, selected, arm_config,
                                    capture_applicability, registry, capture_fixture_paths or [])
              if native_capture_root is not None else None)
    rows: Dict[str, Any] = {}
    execution_error = None
    try:
        for case_id in case_ids:
            case = registry.get(case_id)
            for arm in arms:
                if case["threat"] == "benign" and arm not in ("done", "none"):
                    continue
                if case["threat"] != "benign" and arm == "done":
                    continue
                key = f"{case_id}::{arm}"
                runs = []
                for seed in seeds:
                    run = run_case(case_id, arm, seed, endpoint_factory(case_id, arm, seed),
                                   arm_config, results_dir, registry)
                    runs.append(run)
                    if native is not None:
                        _retain_timeout_trace(native, run)
                primary = "attack_success" if case["threat"] != "benign" else "task_success"
                successes = sum(1 for r in runs if r.status == "ok" and getattr(r, primary) is True)
                failed = sum(1 for r in runs if r.status == "failed")
                lo, hi = wilson_interval(successes, len(runs))
                failure_types = {r.failure["type"] for r in runs if r.failure}
                rows[key] = {
                    "case_id": case_id,
                    "arm": arm,
                    "threat": case["threat"],
                    "seeds": seeds,
                    "n": len(runs),
                    "successes": successes,
                    "failed": failed,
                    "rate": successes / len(runs) if runs else 0.0,
                    "ci95": [round(lo, 4), round(hi, 4)],
                    "typed_failures": sorted(failure_types),
                    "timeout_reporting": timeout_components(runs, primary),
                }
        summary_path = results_dir / "matrix.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except BaseException as exc:
        execution_error = f"matrix execution raised {type(exc).__name__}"
        raise
    finally:
        if native is not None:
            try:
                _finish_timeout_capture(native, rows, execution_error)
            except Exception:
                if execution_error is None:
                    raise
                warnings.warn("native timeout terminal capture unavailable; original execution exception preserved")
    return rows


def replay_trace(trace_path: Path, results_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Verify the immutable trace and deterministically re-run the judge against
    the recorded state; compare with the recorded verdict."""
    records = verify_trace(trace_path)
    snap = extract_run_snapshot(records)
    judge_app = JudgeApplication(snap["case_id"]).load()
    store = StateStore()
    from .shims import finance as _fin
    from .shims import gmail as _gmail
    from .shims import paypal as _paypal
    from .shims import salesforce as _sf
    from .shims import slack as _slack

    for shim in (_fin, _gmail, _paypal, _sf, _slack):
        shim._set_store(store)
    store.reset(snap["state"])
    replay_verdict = judge_app.run(snap["agent_responses"], [])
    recorded = snap["judge_result"]["verdict"]
    mismatch = {
        k: (replay_verdict.get(k), recorded.get(k))
        for k in ("task_success", "attack_success")
        if replay_verdict.get(k) != recorded.get(k)
    }
    return {
        "trace_path": str(trace_path),
        "trace_id": snap["run_result"].get("trace_id"),
        "case_id": snap["case_id"],
        "arm": snap["arm"],
        "seed": snap["seed"],
        "recorded_verdict": recorded,
        "replayed_verdict": replay_verdict,
        "mismatch": mismatch,
        "matches": not mismatch,
    }
