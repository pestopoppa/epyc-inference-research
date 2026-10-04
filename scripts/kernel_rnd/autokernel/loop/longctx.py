"""Long-context serving surface with prefill-once (audit C1 + C4, 2026-10-04).

WHY. Every AutoKernel instrument ran at a KV depth of ~300 tokens while production
decode happens at depth (Q38FN `:8074`: 67% of decode wall above 32k context). The
operator ruled (2026-10-04) that AutoKernel optimizes across every tracked dimension,
including decode and prefill at depth. Audit:
`/mnt/raid0/llm/tmp/ak-longctx-audit-20261004/REPORT.md` section 3.2 C1/C4.

OPT-IN. A target opts in with `--longctx-surface <spec.json>` (`Spec`, built by
`longctx_tools build-manifest`). Without the flag nothing here runs and every launch,
floor and keep is exactly as before.

THE SURFACE. One frozen long prompt per target (~64k tokens of real text). Its frozen
request (`request B`) is an ordinary `FrozenPromptManifest` v2 prompt -- prefix plus a
short probe, decode `n_predict` greedy -- so request digests, floors, the perf capture and
the node profile all consume it unchanged. The spec adds the prefix length and a fixed
4k-token tail. The long launch is the target's own launch with `-c` raised and
`--slot-save-path` added (`derive_launch`).

PREFILL ONCE. A fresh 64k prefill costs ~7 min (Q38FN) to ~27 min (DS41), so the anchor
prefills the prefix ONCE and saves the slot (`ensure_slot`), keyed by model, the anchor
arm's execution digest and the prefix digest. A new anchor regenerates it. Per launch
(`SurfaceLaunch.serve`, hooked into `serving._measure_once`):

  1. restore the slot; request A = prefix + tail, n_predict 1 -> `prompt_per_second`
     at depth (prefill at depth). Refused if its `prompt_n` exceeds the tail bound.
  2. restore again; request B = prefix + probe, decode n_predict -> `predicted_per_second`
     at depth, the launch's scalar. Refused if its `prompt_n` exceeds the probe bound.

Both requests only EXTEND the restored state, never roll it back: Q38FN's recurrent
layers cannot truncate without a checkpoint, so a request that shared less than the
whole restored prefix would silently re-prefill 64k tokens -- which is exactly what the
`prompt_n` bounds refuse. A failed restore is refused too.

RESTORE-VS-FRESH IDENTITY (DS41-B14: dsv4 caches mutate on state write). The first slot
generation per (model, prefix) also serves the probe greedily from the in-process prefix,
restores the saved file and serves it again: the continuations must be identical, or the
surface refuses forever (receipt `identity-*.json`; delete it after a fix).

GATES. Scalar = decode at depth, judged by the ordinary matched instrument and a
matched floor calibrated ONCE per target (24 pairs, `ensure_floor`; the frame excludes
the executable, so it survives anchor changes). Prefill at depth rides in the same
launches (each launch's residency record carries a `longctx` block) and is judged
against a prefill floor replayed from the same sealed calibration row (`verdict`).
`run.py` uses the surface (1) as the PRIMARY metric of attention-route candidates
(`attention_route`) and (2) as a no-regression gate on every other keep.

PLANNER (C4). `run.py` repeats the perf capture and the node profile on the long manifest
with the slot restored, puts the target depth on the target card and feeds a production
context histogram (`parse_server_logs`) as the workload weighting (`planner_context`).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
import os
import math
from pathlib import Path
import re
import statistics
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import urllib.error
import urllib.request

from . import serving
from . import native_server_response as server_response
from .loop import MeasurementFailed

SPEC_SCHEMA = "epyc.autokernel.longctx_surface.v1"
LAUNCH_SCHEMA = "epyc.autokernel.longctx_launch.v1"
IDENTITY_SCHEMA = "epyc.autokernel.longctx_identity.v1"
VERDICT_SCHEMA = "epyc.autokernel.longctx_verdict.v1"
HISTOGRAM_SCHEMA = "epyc.autokernel.context_histogram.v1"
FLAG = "--longctx-surface"
MODES = ("measure", "profile", "generate")
#: A route whose name says attention is judged on this surface first. Data-driven on the
#: route NAME so the FA routes (cpu_fa_schedule, cpu_fa_numerics, a DS41 attention-graph
#: route) qualify the moment `gates.py` admits them, with no edit here.
ATTENTION_ROUTE = re.compile(r"(^cpu_fa_|attn|attention|flash)")
RESTORE_TIMEOUT_S = 900
REQUEST_TIMEOUT_S = 3600
GENERATE_TIMEOUT_S = 4 * 3600
SLOTS_RETAINED = 2
CONTEXT_BUCKETS = (8192, 16384, 32768, 65536, 131072)


class LongCtxRefused(MeasurementFailed):
    """The long-context surface cannot produce an honest number; never a null result.
    A `MeasurementFailed`, so a refused candidate lands as `bench_failed`."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else _canonical(value)).hexdigest()


def spec_digest(body: Mapping[str, Any]) -> str:
    return _sha({key: value for key, value in body.items() if key != "digest"})


def _tokens(value: Any, label: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value or any(
            type(token) is not int or not 0 <= token < 2 ** 31 for token in value):
        raise LongCtxRefused(f"{label} must be a non-empty list of token IDs")
    return tuple(value)


def _positive(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise LongCtxRefused(f"{label} must be a positive integer")
    return value


# --------------------------------------------------------------------------- spec

@dataclass(frozen=True)
class Spec:
    """A validated long-context spec plus its frozen prompt manifest (request B)."""
    path: Path
    body: Mapping[str, Any]
    manifest: Any  # planned_serving.FrozenPromptManifest

    _FIELDS = frozenset({"schema", "target_id", "version", "prompt_manifest",
                         "prompt_manifest_digest", "prefix_tokens", "tail", "ctx",
                         "identity_tokens", "bounds", "production_logs", "source", "digest"})

    @classmethod
    def load(cls, path: Path | str) -> "Spec":
        from . import planned_serving
        path = Path(path)
        body = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(body, Mapping) or set(body) != cls._FIELDS \
                or body["schema"] != SPEC_SCHEMA:
            raise LongCtxRefused(f"{path}: expected a {SPEC_SCHEMA} document with fields "
                                 f"{sorted(cls._FIELDS)}")
        if body["digest"] != spec_digest(body):
            raise LongCtxRefused(f"{path}: spec digest mismatch")
        manifest = planned_serving.FrozenPromptManifest.from_dict(
            json.loads(Path(body["prompt_manifest"]).read_text(encoding="utf-8")))
        if manifest.digest != body["prompt_manifest_digest"]:
            raise LongCtxRefused(f"{path}: prompt manifest digest differs from the spec")
        spec = cls(path, body, manifest)
        spec._validate()
        return spec

    def _validate(self) -> None:
        if len(self.manifest.prompts) != 1:
            raise LongCtxRefused("a long-context manifest carries exactly one prompt")
        request = json.loads(self.manifest.prompts[0].body)
        prompt = _tokens(request.get("prompt"), "request B prompt")
        if request.get("cache_prompt") is not True:
            raise LongCtxRefused("request B must set cache_prompt=true: it reuses the "
                                 "restored prefix")
        if not (request.get("temperature") == 0 or request.get("top_k") == 1):
            raise LongCtxRefused("request B must be greedy (temperature 0 or top_k 1): the "
                                 "restore-vs-fresh and long identity gates require it")
        if request.get("ignore_eos") is not True:
            raise LongCtxRefused("request B must set ignore_eos=true: decode at depth is "
                                 "timed over exactly n_predict tokens")
        _positive(request.get("n_predict"), "n_predict")
        prefix = _positive(self.body["prefix_tokens"], "prefix_tokens")
        if not prefix < len(prompt):
            raise LongCtxRefused("request B must extend the prefix with a probe")
        _tokens(self.body["tail"], "tail")
        _positive(self.body["identity_tokens"], "identity_tokens")
        bounds = self.body["bounds"]
        if not isinstance(bounds, Mapping) or set(bounds) != {"a_prompt_n_max",
                                                              "b_prompt_n_max"}:
            raise LongCtxRefused("bounds must name a_prompt_n_max and b_prompt_n_max")
        if _positive(bounds["a_prompt_n_max"], "a bound") < len(self.tail) \
                or _positive(bounds["b_prompt_n_max"], "b bound") < len(self.probe):
            raise LongCtxRefused("a prompt_n bound is below the tokens the request must "
                                 "evaluate")
        need = prefix + max(len(self.tail) + 1,
                            len(self.probe) + max(request["n_predict"],
                                                  self.body["identity_tokens"]))
        if _positive(self.body["ctx"], "ctx") < need:
            raise LongCtxRefused(f"ctx {self.body['ctx']} is below the {need} tokens the "
                                 "surface occupies")
        logs = self.body["production_logs"]
        if not isinstance(logs, list) or not all(isinstance(p, str) and Path(p).is_absolute()
                                                 for p in logs):
            raise LongCtxRefused("production_logs must be a list of absolute paths")

    # -- identity and requests ------------------------------------------------------
    @property
    def digest(self) -> str:
        return self.body["digest"]

    @property
    def target_id(self) -> str:
        return self.body["target_id"]

    @property
    def request_b(self) -> tuple[str, bytes]:
        prompt = self.manifest.prompts[0]
        return prompt.prompt_id, prompt.body

    @property
    def _b(self) -> dict:
        return json.loads(self.request_b[1])

    @property
    def prefix(self) -> tuple[int, ...]:
        return tuple(self._b["prompt"][:self.body["prefix_tokens"]])

    @property
    def probe(self) -> tuple[int, ...]:
        return tuple(self._b["prompt"][self.body["prefix_tokens"]:])

    @property
    def tail(self) -> tuple[int, ...]:
        return tuple(self.body["tail"])

    @property
    def depth(self) -> int:
        return self.body["prefix_tokens"]

    @property
    def prefix_digest(self) -> str:
        return _sha(list(self.prefix))

    def _variant(self, **changes) -> bytes:
        from . import planned_serving
        return planned_serving._canonical({**self._b, **changes})

    @property
    def request_a(self) -> bytes:
        return self._variant(prompt=list(self.prefix + self.tail), n_predict=1,
                             return_tokens=False)

    @property
    def request_prefill(self) -> bytes:
        return self._variant(prompt=list(self.prefix), n_predict=1, return_tokens=False)

    @property
    def request_identity(self) -> bytes:
        return self._variant(n_predict=self.body["identity_tokens"], return_tokens=True)

    def requests(self, template: serving.Recipe) -> tuple[tuple[str, bytes], ...]:
        """The frozen request tuple, through the manifest's own workload check."""
        return self.manifest.requests((self.request_b[0],), template)

    def recipe_name(self, base: str) -> str:
        return f"{base}-lc{self.depth // 1024}k-{self.digest[:8]}"


# --------------------------------------------------------------------------- launch

def derive_launch(launch, spec: Spec, slot_dir: Path):
    """The target's own canonical launch at the spec's context, with a slot directory.

    Everything else -- binary, model, drafter, threads, topology, environment, port --
    is the target's. The template gets its own NAME (the spec digest is in it, so a
    different tail/bound is a different recipe hash and never reuses a floor) and ctx.
    """
    from . import resolved_recipe as rr
    if launch.backend != "cpu" or launch.template.np != 1:
        raise LongCtxRefused("the long-context surface serves one CPU slot (np=1)")
    command = list(launch.command_argv)
    if "--slot-save-path" in command:
        raise LongCtxRefused("target launch already declares --slot-save-path")
    if "-c" not in command:
        raise LongCtxRefused("target launch has no -c")
    command[command.index("-c") + 1] = str(spec.body["ctx"])
    command += ["--slot-save-path", str(Path(slot_dir).resolve())]
    template = replace(launch.template, name=spec.recipe_name(launch.template.name),
                       ctx=spec.body["ctx"])
    return rr.resolve_canonical_launch(
        template, build_dir=launch.build_dir, command_argv=command,
        topology_prefix=launch.topology_prefix, launch_environment=dict(launch.launch_env),
        artifact_identities={"model": launch.model.to_dict(),
                             "drafter": launch.drafter.to_dict() if launch.drafter else None,
                             "executable": launch.executable.to_dict(),
                             "dsos": [item.to_dict() for item in launch.dsos]},
        backend=launch.backend, environment_policy=launch.environment_policy,
        port=launch.port, runtime_binary_dir=launch.runtime_binary_dir,
        runtime_ld_paths=launch.runtime_ld_paths,
        provenance={**dict(launch.provenance), "longctx_spec": spec.digest,
                    "longctx_parent_snapshot": launch.snapshot_digest})


def _http_post(port: int, path: str, body: bytes, timeout: float) -> bytes:
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body,
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(server_response.MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", "replace")
        exc.close()
        raise LongCtxRefused(f"POST {path} -> HTTP {exc.code}: {detail}") from None
    if len(raw) > server_response.MAX_RESPONSE_BYTES:
        raise LongCtxRefused(f"POST {path} response exceeds the capture byte budget")
    return raw


def _timing(response: Mapping, key: str, kind=float):
    value = (response.get("timings") or {}).get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) \
            or not math.isfinite(float(value)) or value < 0:
        raise LongCtxRefused(f"response timings.{key} is missing or not a finite number")
    return kind(value)


@dataclass(frozen=True)
class SurfaceLaunch:
    """One launch's request protocol; `serving._measure_once` calls `serve` per round.

    `measure`: warmup round = restore + request A, measurement round = restore + B.
    `profile`: both rounds restore + B, so the perf capture's request bytes are the
    frozen ones. `generate`: the warmup round prefills the prefix, saves the slot and,
    with `identity_receipt`, runs the restore-vs-fresh greedy identity gate.
    """
    spec: Spec
    slot_filename: str
    mode: str = "measure"
    identity_receipt: str | None = None
    post: Callable[[int, str, bytes, float], bytes] = field(
        default=_http_post, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise LongCtxRefused(f"unknown long-context mode {self.mode!r}")
        if "/" in self.slot_filename or not self.slot_filename.endswith(".bin"):
            raise LongCtxRefused("slot filename must be a plain *.bin name")

    def with_mode(self, mode: str) -> "SurfaceLaunch":
        return replace(self, mode=mode, identity_receipt=None)

    def _completion(self, port: int, body: bytes, timeout: float) -> tuple[dict, bytes]:
        raw = self.post(port, "/completion", body, timeout)
        response = json.loads(raw)
        if not isinstance(response, Mapping) or response.get("stop") is not True:
            raise LongCtxRefused("completion lacks an explicit terminal stop")
        return dict(response), raw

    def _slot(self, port: int, action: str) -> dict:
        raw = self.post(port, f"/slots/0?action={action}",
                        _canonical({"filename": self.slot_filename}), RESTORE_TIMEOUT_S)
        response = json.loads(raw)
        if not isinstance(response, Mapping):
            raise LongCtxRefused(f"slot {action} returned no object")
        return dict(response)

    def _restore(self, port: int) -> dict:
        response = self._slot(port, "restore")
        restored = response.get("n_restored")
        depth = self.spec.depth
        # The cache may also hold the one token sampled by the prefill request.
        if type(restored) is not int or not depth <= restored <= depth + 1:
            raise LongCtxRefused(f"LONGCTX_REFUSED slot restore of {self.slot_filename} "
                                 f"restored {restored!r} tokens, expected the {depth}-token "
                                 "prefix")
        return {"n_restored": restored,
                "restore_ms": (response.get("timings") or {}).get("restore_ms")}

    def _bounded(self, response: Mapping, which: str) -> int:
        prompt_n = _timing(response, "prompt_n", int)
        bound = self.spec.body["bounds"][f"{which}_prompt_n_max"]
        if prompt_n > bound:
            raise LongCtxRefused(
                f"LONGCTX_REFUSED request {which.upper()} evaluated prompt_n={prompt_n} > "
                f"bound {bound}: the restored prefix was not reused (a hybrid model needs "
                "the request to extend the restored state exactly)")
        return prompt_n

    def _generate(self, port: int) -> dict:
        response, _raw = self._completion(port, self.spec.request_prefill, GENERATE_TIMEOUT_S)
        prompt_n = _timing(response, "prompt_n", int)
        if prompt_n != self.spec.depth:
            raise LongCtxRefused(f"slot generation evaluated {prompt_n} prompt tokens, not "
                                 f"the {self.spec.depth}-token prefix (was the cache warm?)")
        saved = self._slot(port, "save")
        facts = {"prefill_prompt_n": prompt_n,
                 "prefill_prompt_per_second": _timing(response, "prompt_per_second"),
                 "n_saved": saved.get("n_saved"), "n_written": saved.get("n_written")}
        if self.identity_receipt is not None:
            fresh, _ = self._completion(port, self.spec.request_identity, REQUEST_TIMEOUT_S)
            # Both completions must EXTEND the cached/restored prefix: a server that
            # silently re-prefilled would reproduce the same greedy tokens and pass
            # vacuously, proving nothing about the restored state.
            self._bounded(fresh, "b")
            self._restore(port)
            restored, _ = self._completion(port, self.spec.request_identity,
                                           REQUEST_TIMEOUT_S)
            self._bounded(restored, "b")
            want = self.spec.body["identity_tokens"]
            a, b = fresh.get("tokens"), restored.get("tokens")
            passed = (isinstance(a, list) and a == b and len(a) == want)
            receipt = {"schema": IDENTITY_SCHEMA, "passed": passed,
                       "slot": self.slot_filename, "prefix_digest": self.spec.prefix_digest,
                       "spec_digest": self.spec.digest, "identity_tokens": want,
                       "fresh_tokens": a, "restored_tokens": b,
                       "method": "in-process prefix vs restored slot file, same probe "
                                 "batch, greedy; must be token-identical",
                       "at": time.time()}
            target = Path(self.identity_receipt)
            partial = target.with_name(target.name + f".{os.getpid()}.part")
            partial.write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
            os.replace(partial, target)
            facts["identity"] = {"passed": passed, "receipt": self.identity_receipt}
            if not passed:
                raise LongCtxRefused("LONGCTX_REFUSED restore-vs-fresh greedy identity "
                                     f"FAILED (DS41-B14 risk); receipt {self.identity_receipt}")
        return facts

    def serve(self, port: int, index: int, phase: str, request: tuple[str, bytes], *,
              capture: bool = False) -> tuple:
        """The `_measure_once` per-slot result tuple: (tokens, rate, terminal, record, raw)."""
        prompt_id, body = request
        if body != self.spec.request_b[1]:
            raise LongCtxRefused("the long-context launch was handed foreign request bytes")
        request_kind = ("prefill" if self.mode == "generate" else
                        "A" if self.mode == "measure" and phase == "warmup" else "B")
        if self.mode == "generate" and phase != "warmup":
            request_kind = "none"
        sent = {"A": self.spec.request_a, "prefill": self.spec.request_prefill}.get(
            request_kind, body)
        record = {"phase": phase, "slot_index": index, "prompt_id": prompt_id,
                  "request_sha256": hashlib.sha256(sent).hexdigest(),
                  "predicted_n": None, "predicted_per_second": None, "terminal": False,
                  "error": None, "longctx": {"mode": self.mode, "request": request_kind,
                                             "slot": self.slot_filename}}
        facts = record["longctx"]
        result, raw = (0, 0.0, False), None
        started = ended = time.monotonic()
        try:
            if request_kind == "prefill":
                facts.update(self._generate(port))
                result = (0, 0.0, True)
            elif request_kind == "none":
                result = (0, 0.0, True)
            else:
                facts["restore"] = self._restore(port)
                started = time.monotonic()
                response, raw = self._completion(port, sent, REQUEST_TIMEOUT_S)
                ended = time.monotonic()
                facts["prompt_n"] = self._bounded(response, request_kind.lower())
                facts["prompt_per_second"] = _timing(response, "prompt_per_second")
                tokens = _timing(response, "predicted_n", int)
                want = self.spec._b["n_predict"] if request_kind == "B" else 1
                if tokens != want:
                    raise LongCtxRefused(f"LONGCTX_REFUSED request {request_kind} predicted "
                                         f"{tokens} tokens, not {want}: the rate is not over "
                                         "the frozen decode length")
                rate = _timing(response, "predicted_per_second")
                facts["predicted_per_second"] = rate
                record.update(predicted_n=tokens, predicted_per_second=rate, terminal=True)
                result = (tokens, rate if request_kind == "B" else 0.0, True)
        except Exception as exc:  # recorded; `_measure_once` refuses the launch on it
            ended = time.monotonic()
            record["error"] = f"{type(exc).__name__}: {exc}"
        captured = (server_response.RawServerResponse(
            phase, index, prompt_id, sent, raw, started, ended, record["error"])
            if capture else None)
        return (*result, record, captured)

    def launch_record(self, request_rows: Sequence[Mapping]) -> dict:
        """The per-launch facts a residency record carries (prefill at depth lives here)."""
        out = {"schema": LAUNCH_SCHEMA, "mode": self.mode, "slot": self.slot_filename,
               "spec_digest": self.spec.digest, "depth": self.spec.depth}
        for row in request_rows:
            facts = row.get("longctx") if isinstance(row, Mapping) else None
            if isinstance(facts, Mapping) and facts.get("request") in ("A", "B", "prefill"):
                out[facts["request"].lower()] = {key: value for key, value in facts.items()
                                                 if key not in {"mode", "slot", "request"}}
        return out


# --------------------------------------------------------------------------- verdict

def _prefill_rates(records: Sequence[Mapping]) -> list[float] | None:
    rates = []
    for record in records or ():
        value = ((record.get("longctx") or {}).get("a") or {}).get("prompt_per_second") \
            if isinstance(record, Mapping) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            return None
        rates.append(float(value))
    return rates or None


def prefill_floor_pct(floor_row: Mapping, pairs: int) -> float | None:
    """The prefill-at-depth floor, replayed from the SAME sealed calibration launches."""
    a = _prefill_rates(floor_row.get("anchor_residency"))
    c = _prefill_rates(floor_row.get("candidate_residency"))
    if a is None or c is None or len(a) != len(c):
        return None
    return serving._matched_summary(a, c, pairs)["floor_pct"]


def verdict(row: Mapping, floor_row: Mapping | None, pairs: int) -> dict:
    """Decode (the row's own matched verdict) and prefill at depth, against their floors.

    `passed` means neither dimension regressed decisively; an uncalibrated or
    incomplete reading never passes."""
    a = _prefill_rates(row.get("anchor_residency"))
    c = _prefill_rates(row.get("candidate_residency"))
    p_floor = prefill_floor_pct(floor_row, pairs) if floor_row else None
    p_effect = (statistics.median(c) / statistics.median(a) - 1.0) if a and c else None
    p_decisive = (None if p_floor is None or p_effect is None
                  else abs(p_effect) * 100.0 >= p_floor)
    decode_regressed = row.get("decisive") is True and row.get("effect", 0) < 0
    prefill_regressed = p_decisive is True and p_effect < 0
    out = {"schema": VERDICT_SCHEMA, "decode_effect": row.get("effect"),
           "decode_decisive": row.get("decisive"),
           "decode_floor_pct": row.get("noise_floor_pct"),
           "prefill_effect": p_effect, "prefill_decisive": p_decisive,
           "prefill_floor_pct": p_floor,
           "anchor_prefill_tok_s": statistics.median(a) if a else None,
           "candidate_prefill_tok_s": statistics.median(c) if c else None,
           "decode_regressed": decode_regressed, "prefill_regressed": prefill_regressed}
    if row.get("decisive") is None or p_decisive is None:
        out.update(passed=False, reason="long-context surface uncalibrated or incomplete")
    elif decode_regressed or prefill_regressed:
        out.update(passed=False, reason=(
            f"long-context regression: decode {row['effect'] * 100:+.3f}% "
            f"(floor {row.get('noise_floor_pct'):.3f}%), prefill {p_effect * 100:+.3f}% "
            f"(floor {p_floor:.3f}%)"))
    else:
        out.update(passed=True, reason=(
            f"long-context decode {row['effect'] * 100:+.3f}%, prefill "
            f"{p_effect * 100:+.3f}%: no decisive regression"))
    return out


def attention_route(hypothesis) -> str | None:
    """The attention route a hypothesis names, if any (primary-metric selection)."""
    from . import gates
    surface = str(getattr(hypothesis, "target_surface", "") or "")
    symbol = str(getattr(hypothesis, "target_symbol", "") or "")
    routes = list(gates.cpu_source_routes(surface, symbol)) + [
        route for route in gates.cpu_multi_file_routes(symbol) if surface in route.paths]
    return next((route.route for route in routes if ATTENTION_ROUTE.search(route.route)),
                None)


# --------------------------------------------------------------------------- surface

class Surface:
    """A target's long-context surface: slot cache, floor and A/B, under `<store>/longctx`."""

    def __init__(self, spec: Spec, *, store: Path | str):
        self.spec = spec
        # Per TARGET: two lanes on one model+prefix must never prune each other's slots.
        self.root = Path(store) / "longctx" / spec.target_id
        self.slot_dir = self.root / "slots"
        self.floor_store = self.root / "floors"

    @classmethod
    def load(cls, path: Path | str, *, store: Path | str, target_id: str | None) -> "Surface":
        spec = Spec.load(path)
        if target_id is not None and spec.target_id != target_id:
            raise LongCtxRefused(f"long-context spec is for {spec.target_id}, not {target_id}")
        return cls(spec, store=store)

    @property
    def manifest(self):
        return self.spec.manifest

    def launch_for(self, launch):
        return derive_launch(launch, self.spec, self.slot_dir)

    def slot_name(self, anchor_arm) -> str:
        return (f"{anchor_arm.model.sha256[:16]}-{anchor_arm.execution_digest[:16]}-"
                f"{self.spec.prefix_digest[:16]}.bin")

    def _identity_path(self, anchor_arm) -> Path:
        # Keyed like the slot itself: model, ANCHOR EXECUTION DIGEST and prefix. A new
        # anchor (or the same model under another launch) re-proves restore-vs-fresh;
        # DS41-B14 is a kernel/KV-layout defect, exactly what a new anchor can introduce.
        return self.slot_dir / f"identity-{self.slot_name(anchor_arm)[:-4]}.json"

    def _identity(self, anchor_arm) -> dict | None:
        path = self._identity_path(anchor_arm)
        if not path.is_file():
            return None
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise LongCtxRefused(f"{path} is not a readable identity receipt: {exc}") from exc
        if not isinstance(receipt, Mapping) or receipt.get("schema") != IDENTITY_SCHEMA:
            raise LongCtxRefused(f"{path} is not an identity receipt")
        expected = {"slot": self.slot_name(anchor_arm), "prefix_digest": self.spec.prefix_digest,
                    "spec_digest": self.spec.digest,
                    "identity_tokens": self.spec.body["identity_tokens"]}
        mismatch = {key: receipt.get(key) for key, want in expected.items()
                    if receipt.get(key) != want}
        if mismatch:
            raise LongCtxRefused(f"identity receipt {path} is for another slot/spec "
                                 f"({mismatch}); remove it to re-prove")
        if receipt.get("passed") is not True:
            raise LongCtxRefused(f"restore-vs-fresh identity FAILED earlier ({path}); the "
                                 "surface stays refused until the cause is fixed and the "
                                 "receipt removed")
        return receipt

    def ensure_slot(self, anchor_arm, *, measure: Callable | None = None) -> SurfaceLaunch:
        """The measuring launch for this anchor, prefilling and saving the slot if absent."""
        name = self.slot_name(anchor_arm)
        receipt = self._identity(anchor_arm)
        launch = SurfaceLaunch(self.spec, name)
        if receipt is not None and (self.slot_dir / name).is_file():
            return launch
        self.slot_dir.mkdir(parents=True, exist_ok=True)
        generate = replace(launch, mode="generate", identity_receipt=(
            None if receipt is not None else str(self._identity_path(anchor_arm))))
        template = anchor_arm.template
        try:
            (measure or serving._measure_once)(
                template, Path(anchor_arm.build_dir), anchor_arm.port,
                resolved_recipe=anchor_arm, frozen_requests=self.spec.requests(template),
                longctx=generate)
        except serving.ServerDied as exc:
            errors = [row.get("error") for row in
                      (getattr(exc, "record", None) or {}).get("requests", ())
                      if isinstance(row, Mapping) and row.get("error")]
            raise LongCtxRefused(f"slot generation failed: {exc}; {errors}") from exc
        if not (self.slot_dir / name).is_file():
            raise LongCtxRefused(f"slot generation reported success but {name} is absent")
        if self._identity(anchor_arm) is None:
            raise LongCtxRefused("slot generation ran without a passing identity receipt")
        self._retain(anchor_arm, keep=name)
        return launch

    def _retain(self, anchor_arm, *, keep: str) -> None:
        """Keep the newest slot files of this model+prefix; each is gigabytes."""
        pattern = f"{anchor_arm.model.sha256[:16]}-*-{self.spec.prefix_digest[:16]}.bin"
        stale = sorted((p for p in self.slot_dir.glob(pattern) if p.name != keep),
                       key=lambda p: p.stat().st_mtime, reverse=True)
        for path in stale[SLOTS_RETAINED - 1:]:
            path.unlink(missing_ok=True)

    def ensure_floor(self, anchor_arm, launch: SurfaceLaunch, *, pairs: int,
                     calibrate: Callable | None = None, log=print) -> serving.FloorReading:
        """The per-target matched floor; calibrated once (24 A/A pairs) when absent."""
        template = anchor_arm.template
        requests = self.spec.requests(template)
        mode = {"instrument": serving.MATCHED_INSTRUMENT, "pairs": pairs}
        reading = serving.load_floor(self.floor_store, template, frozen_requests=requests,
                                     **mode)
        if reading.floor_pct is not None:
            return reading
        log(f"longctx   calibrating the per-target floor for {template.name}: "
            f"{serving.MATCHED_CALIBRATION_PAIRS} matched A/A pairs (one-time)")
        row = (calibrate or serving.calibrate_floor)(
            template, Path(anchor_arm.build_dir), samples=serving.MATCHED_CALIBRATION_PAIRS,
            port=anchor_arm.port, resolved_recipe=anchor_arm, frozen_requests=requests,
            longctx=launch, **mode)
        if prefill_floor_pct(row, pairs) is None:
            raise LongCtxRefused("calibration launches carry no prefill-at-depth samples")
        serving.write_floor(self.floor_store, template, row, unit=serving.CALIBRATION_UNIT,
                            frozen_requests=requests, **mode)
        return serving.load_floor(self.floor_store, template, frozen_requests=requests,
                                  **mode)

    def compare(self, anchor_arm, candidate_arm, *, pairs: int, measure: Callable | None = None,
                compare: Callable | None = None, calibrate: Callable | None = None,
                log=print) -> dict:
        """Anchor vs candidate at depth: the matched serving row plus `row['longctx']`."""
        launch = self.ensure_slot(anchor_arm, measure=measure)
        reading = self.ensure_floor(anchor_arm, launch, pairs=pairs, calibrate=calibrate,
                                    log=log)
        template = anchor_arm.template
        row = dict((compare or serving.compare)(
            template, Path(anchor_arm.build_dir), Path(candidate_arm.build_dir),
            pairs=pairs, floor_pct=reading.gate_floor(effect_unit=serving.COMPARE_EFFECT_UNIT),
            floor_unit=serving.CALIBRATION_UNIT, port=anchor_arm.port,
            anchor_resolved_recipe=anchor_arm, candidate_resolved_recipe=candidate_arm,
            frozen_requests=self.spec.requests(template),
            floor_request_digest=reading.request_digest,
            instrument=serving.MATCHED_INSTRUMENT, floor_record=reading.row,
            longctx=launch))
        row["longctx"] = verdict(row, reading.row, pairs)
        return row

    def identity_targets(self, anchor_arm, candidate_arm) -> list[tuple]:
        """The `long_identity` route target (gates.check_model_identity_targets 5-tuple):
        both arms restore the ANCHOR's saved slot before every greedy request B, so the
        identity gate judges the attention path at depth and across repetitions."""
        launch = self.ensure_slot(anchor_arm)
        return [("longctx", anchor_arm, candidate_arm,
                 self.spec.requests(anchor_arm.template), launch._restore)]


def prefill_dimension_row(row: Mapping) -> dict:
    """Project a long-context A/B row's prefill-at-depth verdict into the serving A/B
    shape `surface_validation.classify` reads (G5 `prefill_at_depth`). An incomplete or
    uncalibrated prefill reading is an `error` row, which the keep-dimensions gate
    records as pending (fail closed)."""
    long_verdict = row.get("longctx") if isinstance(row, Mapping) else None
    if not isinstance(long_verdict, Mapping):
        return {"error": "long-context row carries no verdict"}
    effect, floor, decisive = (long_verdict.get("prefill_effect"),
                               long_verdict.get("prefill_floor_pct"),
                               long_verdict.get("prefill_decisive"))
    if (not isinstance(effect, (int, float)) or isinstance(effect, bool)
            or not isinstance(floor, (int, float)) or isinstance(floor, bool)
            or type(decisive) is not bool):
        return {"error": "prefill at depth uncalibrated or incomplete: "
                         + str(long_verdict.get("reason"))}
    return {"schema": "epyc.autokernel.serving_ab.v1", "dimension": "prefill_at_depth",
            "effect": float(effect), "effect_pct": float(effect) * 100.0,
            "noise_floor_pct": float(floor), "decisive": decisive,
            "effect_unit": row.get("effect_unit"), "floor_unit": row.get("floor_unit"),
            "anchor_tok_s": long_verdict.get("anchor_prefill_tok_s"),
            "candidate_tok_s": long_verdict.get("candidate_prefill_tok_s"),
            "derived_from": {key: row.get(key) for key in ("recipe_hash", "floor_sha256",
                                                             "pairs")}}


# --------------------------------------------------------------------------- C4

_PROMPT_EVAL = re.compile(r"task (\d+) \| prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
_EVAL = re.compile(r"task (\d+) \|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens")
_RELEASE = re.compile(r"task (\d+) \| stop processing: n_tokens = (\d+)")
_PORT = re.compile(r"llama-server-(\d+)\.log$")


def _bucket(ctx: int) -> str:
    low = 0
    for edge in CONTEXT_BUCKETS:
        if ctx < edge:
            return f"{low // 1024}k-{edge // 1024}k"
        low = edge
    return f">={low // 1024}k"


def parse_server_logs(paths: Iterable[Path | str]) -> dict:
    """Production context histogram from llama-server logs: per server, requests and
    decode/prefill wall by context-at-completion bucket. The planner's workload weight."""
    servers: dict[str, Any] = {}
    sources = []
    for path in sorted({str(Path(p)) for p in paths}):
        path = Path(path)
        if not path.is_file():
            continue
        sources.append({"path": str(path), "bytes": path.stat().st_size,
                        "mtime": path.stat().st_mtime})
        match = _PORT.search(path.name)
        server = match.group(1) if match else path.name
        pending: dict[str, dict] = {}
        tasks: list[dict] = []
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if (m := _PROMPT_EVAL.search(line)):
                    pending.setdefault(m.group(1), {}).update(
                        prompt_ms=float(m.group(2)), prompt_n=int(m.group(3)))
                elif (m := _EVAL.search(line)):
                    pending.setdefault(m.group(1), {}).update(
                        eval_ms=float(m.group(2)), eval_n=int(m.group(3)))
                elif (m := _RELEASE.search(line)):
                    task = pending.pop(m.group(1), {})
                    task["ctx"] = int(m.group(2))
                    tasks.append(task)
        if not tasks:
            continue
        buckets: dict[str, dict] = {}
        for task in tasks:
            row = buckets.setdefault(_bucket(task["ctx"]), {
                "requests": 0, "decode_tokens": 0, "decode_ms": 0.0,
                "prefill_tokens": 0, "prefill_ms": 0.0})
            row["requests"] += 1
            row["decode_tokens"] += task.get("eval_n", 0)
            row["decode_ms"] += task.get("eval_ms", 0.0)
            row["prefill_tokens"] += task.get("prompt_n", 0)
            row["prefill_ms"] += task.get("prompt_ms", 0.0)
        decode_ms = sum(row["decode_ms"] for row in buckets.values()) or 1.0
        prefill_ms = sum(row["prefill_ms"] for row in buckets.values()) or 1.0
        ctxs = sorted(task["ctx"] for task in tasks)
        order = [_bucket(edge - 1) for edge in CONTEXT_BUCKETS] + [_bucket(10 ** 9)]
        servers[server] = {
            "requests": len(tasks), "ctx_p50": ctxs[len(ctxs) // 2],
            "ctx_p90": ctxs[min(len(ctxs) - 1, int(0.9 * len(ctxs)))], "ctx_max": ctxs[-1],
            "decode_wall_s": round(decode_ms / 1000, 1),
            "prefill_wall_s": round(prefill_ms / 1000, 1),
            "buckets": [{"bucket": name, "requests": row["requests"],
                         "decode_wall_share": round(row["decode_ms"] / decode_ms, 4),
                         "prefill_wall_share": round(row["prefill_ms"] / prefill_ms, 4),
                         "decode_tok_s": (round(row["decode_tokens"] * 1000 / row["decode_ms"], 2)
                                          if row["decode_ms"] else None),
                         "prefill_tok_s": (round(row["prefill_tokens"] * 1000 / row["prefill_ms"], 2)
                                           if row["prefill_ms"] else None)}
                        for name in order if (row := buckets.get(name))]}
    return {"schema": HISTOGRAM_SCHEMA, "sources": sources, "servers": servers,
            "bucket_basis": "slot context (n_tokens) at request completion"}


def target_card(spec: Spec) -> dict:
    """The depth facts the planner's target card shows."""
    return {"depth_tokens": spec.depth, "tail_tokens": len(spec.tail), "ctx": spec.body["ctx"],
            "spec": str(spec.path)}


def _compact_cpu_profile(observation: Mapping | None, limit: int = 12) -> dict:
    observation = dict(observation or {})
    if observation.get("status") != "observed":
        return {"status": observation.get("status", "not_collected"),
                "reason": observation.get("reason")}
    return {"status": "observed", "record": observation.get("record"),
            "ranked_levers": [{"family": row.get("family"),
                               "sampled_period_fraction": row.get("sampled_period_fraction")}
                              for row in (observation.get("ranked_levers") or [])[:limit]],
            "hotspots": [{"symbol": row.get("symbol"), "dso": row.get("dso"),
                          "sampled_period_fraction": row.get("sampled_period_fraction")}
                         for row in (observation.get("hotspots") or [])[:limit]]}


def _compact_node_profile(observation: Mapping | None, limit: int = 12) -> dict:
    observation = dict(observation or {})
    if observation.get("status") != "observed":
        return {"status": observation.get("status", "not_collected"),
                "reason": observation.get("reason")}
    return {"status": "observed",
            "mechanism_shares": [{"family": row.get("family"),
                                  "wall_fraction": row.get("wall_fraction")}
                                 for row in (observation.get("mechanism_shares") or [])[:limit]],
            "ranked_op_shares": [{"op": row.get("op"), "wall_fraction": row.get("wall_fraction")}
                                 for row in (observation.get("ranked_op_shares") or [])[:limit]]}


def planner_context(spec: Spec, *, histogram: Mapping | None,
                    cpu_profile: Mapping | None, node_profile: Mapping | None) -> dict:
    """The `long_context` planner section: depth, how keeps are judged at depth, the
    production workload weighting and the profiles taken at depth."""
    return {"target_depth_tokens": spec.depth, "tail_tokens": len(spec.tail),
            "ctx": spec.body["ctx"],
            "how_judged": ("decode at depth (request B, restored prefix) is the PRIMARY metric "
                           "for attention-route candidates; decode and prefill at depth "
                           "(request A: +tail on the restored prefix) must not regress "
                           "decisively on every keep"),
            "workload_weighting": histogram or {"status": "no production log declared"},
            "cpu_profile_at_depth": _compact_cpu_profile(cpu_profile),
            "node_profile_at_depth": _compact_node_profile(node_profile)}


__all__ = ["ATTENTION_ROUTE", "FLAG", "HISTOGRAM_SCHEMA", "IDENTITY_SCHEMA", "LAUNCH_SCHEMA",
           "LongCtxRefused", "SPEC_SCHEMA", "Spec", "Surface", "SurfaceLaunch",
           "VERDICT_SCHEMA", "attention_route", "derive_launch", "parse_server_logs",
           "planner_context", "prefill_floor_pct", "spec_digest", "target_card", "verdict"]
