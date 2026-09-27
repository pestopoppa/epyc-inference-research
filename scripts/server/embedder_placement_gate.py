#!/usr/bin/env python3
"""Measurement gates G1/G2 for the UFH-12 Phase-0 embedder placement change.

Pre-registered in epyc-root artifacts/operator/stack-change-ufh12-phase0-20260926/
PACKAGE.md section 4. Run it TWICE on the live stack -- once BEFORE the apply
(`--label pre`, old placement) and once AFTER the embedder reload
(`--label post`) -- under a region claim over every CPU:

    scripts/region-lock run --cpu-list 0-191 -- \
      .venv/bin/python scripts/server/embedder_placement_gate.py --label pre \
        --out /mnt/raid0/llm/epyc-orchestrator/data/embedder_placement/pre.json

G1  frontdoor decode cost. For each frontdoor port, ABA-alternating pairs of
    (Q) pool idle and (S) pool saturated. One sample = one fixed-prompt
    /completion, temperature 0, fixed seed, `timings.predicted_per_second`
    (tok/s, higher = better). Reports median S/Q per port and the Q-vs-Q
    A/A spread (the noise floor, same unit).
G2  pool scaling. Embedding texts/s with 4 in flight on ONE port versus 4 in
    flight on EVERY pool port, frontdoor idle. Reports the ratio.

The same ~130-token text is used in both runs so it fits the OLD 256-token
slot as well as the new 512 one; the arms differ only in placement.

BELIEF CAPTURE (VB-UFH12-PLACEMENT, 2026-09-27). After the record is written, the gate writes
`<out-stem>.belief_measurements.jsonl` beside it through `embedder_placement_capture.py`: one
self-hashed row per gate x port x metric, with the serving identity (pid, argv, cpuset, binary
digest, mapped libggml, frontdoor /props build_info, topology hash) snapshotted before the first and
after the last timed sample. The snapshots sit outside every timed window, so the measurement is
unchanged. A refused capture never touches the record; the gate then exits 4 and says why.

This script only sends HTTP requests. It starts, stops and signals nothing,
and it refuses to run if a declared pool port is not answering /health.

LOAD MODES (UFH-12 REPL-EMB-1.4, 2026-09-26). `--load-mode raw` (the default)
is the Phase-0 method, unchanged: `per_port` threads per port POST straight to
each embedder, so no scheduler is involved. `--load-mode scheduler` offers the
same load (`per_port` x ports concurrent callers, the same text) THROUGH the
pooled client and its scheduler (src/embedding_pool), so the busy-frontdoor
neighbour cap from orchestration/embedding_pool_policy.yaml is exercised: while
a frontdoor instance decodes, embedders on its NUMA nodes hold at most
`neighbour_cap.max_in_flight` texts. The client's cache is bypassed so every
call reaches an embedder. Scheduler mode additionally records, per S sample,
the embedding throughput and the scheduler's grant counters, so the cap's
throughput cost is visible next to the decode ratio. The G1 re-measure with
the cap on is:

    scripts/region-lock run --cpu-list 0-191 -- \
      .venv/bin/python scripts/server/embedder_placement_gate.py --label post-cap \
        --load-mode scheduler \
        --out /mnt/raid0/llm/epyc-orchestrator/data/embedder_placement/post-cap-<date>.json

ARMS (UFH-12 A0/A3, 2026-09-27). `--label arm-baseline|arm-candidate --arm <name>` runs one arm
of a pre-registered ABA (runbook: /mnt/raid0/llm/tmp/next-window-runbook-20260927.md).
`--policy-override neighbour_cap.max_in_flight=0` (scheduler mode only) runs the gate against a
policy that differs from orchestration/embedding_pool_policy.yaml in exactly the named keys,
without touching that file; the effective policy is what the record stores. `--per-port-in-flight`
and `--load-pace-s` shape a low-duty point (e.g. 1 in flight per port, 1 s between requests).
Every run reads the embedders' live OpenMP env from /proc (read-only) before the first and after
the last timed sample, refuses if it changed, and `--expect-embedder-env KEY=VALUE` refuses to start
unless every embedder runs it; the readback and the embedder env override record
(scripts/server/embedder_env_override.py) ride in `params`, so the belief sidecar's provenance
carries the arm, the policy overrides and the env.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
import threading
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]

POOL_PORTS = [8090, 8091, 8092, 8093, 8094, 8095]
PER_PORT_IN_FLIGHT = 4
LOAD_WARMUP_S = 3.0
CAPTURE_EXIT = 4
FD_PROMPT = (
    "Write a detailed, step-by-step explanation of how a hash map handles collisions, "
    "covering separate chaining and open addressing, with the trade-offs of each."
)
EMB_TEXT = (
    "The orchestrator routes each request to a role, and every role is backed by a "
    "llama-server process pinned to a declared set of cores. Retrieval inside the REPL "
    "returns pointers into spill files and context bundles rather than summaries, so the "
    "model reads the exact lines it needs through get and peek. Lexical and dense scores "
    "are fused with reciprocal rank fusion, and when no embedder is available the search "
    "degrades to lexical results that are labelled as such. An index belongs to exactly "
    "one embedding model, and the scheduler only ever chooses among instances of it. "
)


def _post(url: str, body: dict, timeout: float = 600.0) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — loopback only
        return json.loads(resp.read())


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:  # noqa: S310
            return r.status == 200
    except OSError:
        return False


def _decode_tps(port: int, n_predict: int) -> float:
    out = _post(
        f"http://127.0.0.1:{port}/completion",
        {"prompt": FD_PROMPT, "n_predict": n_predict, "temperature": 0, "seed": 42,
         "cache_prompt": False},
    )
    return float(out["timings"]["predicted_per_second"])


class _Load:
    """`per_port` embedding requests in flight on each port until stopped. With `pace_s` > 0 each
    caller waits that long after every request (a low-duty point)."""

    def __init__(self, ports: list[int], per_port: int, *, pace_s: float = 0.0) -> None:
        self.ports, self.per_port = ports, per_port
        self.pace_s = pace_s
        self.stop = threading.Event()
        self.done = 0
        self.errors = 0
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def _worker(self, port: int) -> None:
        while not self.stop.is_set():
            try:
                _post(f"http://127.0.0.1:{port}/v1/embeddings", {"input": [EMB_TEXT]}, timeout=120)
                with self._lock:
                    self.done += 1
            except Exception:  # noqa: BLE001 — counted, reported, never swallowed silently
                with self._lock:
                    self.errors += 1
            if self.pace_s > 0:
                self.stop.wait(self.pace_s)

    def __enter__(self) -> "_Load":
        for port in self.ports:
            for _ in range(self.per_port):
                t = threading.Thread(target=self._worker, args=(port,), daemon=True)
                t.start()
                self._threads.append(t)
        time.sleep(LOAD_WARMUP_S)  # let every server reach steady state before sampling
        with self._lock:
            self.done = 0
            self.errors = 0
        self.t0 = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.monotonic() - self.t0
        self.stop.set()
        for t in self._threads:
            t.join(timeout=150)


def apply_policy_overrides(policy: Any, items: Iterable[str]) -> tuple[Any, dict[str, Any]]:
    """`section.key=value` overrides applied to a loaded policy through the policy's own validator
    (parse_policy), so an override can never produce a policy the file could not. Returns the
    effective policy and the parsed overrides. The policy FILE is never touched."""
    from src.embedding_pool.policy import parse_policy

    items = list(items or ())
    doc = {"version": 1, **policy.as_dict()}
    parsed: dict[str, Any] = {}
    for item in items:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"policy override {item!r} is not section.key=value")
        dotted, raw = item.split("=", 1)
        section, key = dotted.strip().split(".", 1)
        if section not in ("client", "neighbour_cap") or key not in doc[section]:
            raise ValueError(f"policy override {dotted!r} names no policy field")
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw  # a bare string
        doc[section][key] = value
        parsed[dotted.strip()] = value
    return parse_policy(doc), parsed


def _default_pool_client(ports: list[int], policy: Any = None) -> Any:
    """A pooled client + scheduler over `ports`, built from the repo's policy (or `policy`, the
    effective policy after overrides) and the stack's declared placement (scheduler mode only)."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from src.embedding_pool.client import PooledEmbeddingClient
    from src.embedding_pool.policy import load_policy
    from src.embedding_pool.scheduler import EmbeddingScheduler
    from src.embedding_pool.topology import live_topology

    policy = policy if policy is not None else load_policy()
    topology = live_topology(policy.neighbour_cap.guarded_roles).restricted_to(ports)
    return PooledEmbeddingClient(EmbeddingScheduler(topology, policy), policy=policy)


class _SchedulerLoad:
    """The same offered load as `_Load` (`per_port` x len(ports) concurrent callers, the
    same text), but every call goes through the pooled client and its scheduler, so the
    neighbour cap decides how many texts each embedder holds. Calls that find no
    headroom within `admission_wait_s` count as `deferred` (the lexical-now outcome),
    not as errors; anything else that fails is an error."""

    def __init__(self, ports: list[int], per_port: int, *,
                 client_factory: Callable[[list[int]], Any] | None = None,
                 admission_wait_s: float = 30.0, warmup_s: float = LOAD_WARMUP_S,
                 pace_s: float = 0.0) -> None:
        self.ports, self.per_port = ports, per_port
        self.pace_s = pace_s
        self.client_factory = client_factory or _default_pool_client
        self.admission_wait_s = admission_wait_s
        self.warmup_s = warmup_s
        self.stop = threading.Event()
        self.done = 0
        self.errors = 0
        self.deferred = 0
        self.error_samples: list[str] = []
        self.stats_start: dict = {}
        self.stats_end: dict = {}
        self._lock = threading.Lock()
        self._client: Any = None
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._crash: BaseException | None = None

    def _snapshot(self) -> dict:
        try:
            return self._client.scheduler.stats() if self._client is not None else {}
        except Exception:  # noqa: BLE001
            return {}

    async def _worker(self) -> None:
        from src.embedding_pool.client import EmbeddingUnavailable

        while not self.stop.is_set():
            try:
                await self._client.embed_many([EMB_TEXT], use_cache=False, cancel_event=self.stop,
                                              admission_wait_s=self.admission_wait_s, timeout_s=120.0)
                with self._lock:
                    self.done += 1
            except EmbeddingUnavailable as exc:
                with self._lock:
                    if exc.reason == "cancelled" and self.stop.is_set():
                        pass  # the sample ended; not a failure
                    elif exc.reason == "saturated":
                        self.deferred += 1
                    else:
                        self.errors += 1
                        if len(self.error_samples) < 5:
                            self.error_samples.append(str(exc))
            except Exception as exc:  # noqa: BLE001 — counted, reported, never swallowed silently
                with self._lock:
                    self.errors += 1
                    if len(self.error_samples) < 5:
                        self.error_samples.append(f"{type(exc).__name__}: {exc}")
            if self.pace_s > 0 and not self.stop.is_set():
                await asyncio.sleep(self.pace_s)

    async def _main(self) -> None:
        self._client = self.client_factory(self.ports)
        try:
            workers = [asyncio.ensure_future(self._worker())
                       for _ in range(len(self.ports) * self.per_port)]
            self._ready.set()
            while not self.stop.is_set():
                await asyncio.sleep(0.05)
            # every in-flight call finishes (or times out) before the load reports done
            await asyncio.gather(*workers, return_exceptions=True)
        finally:
            await self._client.aclose()

    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except BaseException as exc:  # noqa: BLE001 — surfaced by __exit__
            self._crash = exc
            self._ready.set()

    def __enter__(self) -> "_SchedulerLoad":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=60)
        if self._crash is not None:
            raise SystemExit(f"scheduler load failed to start: {self._crash!r}")
        time.sleep(self.warmup_s)  # let every server reach steady state before sampling
        with self._lock:
            self.done = 0
            self.errors = 0
            self.deferred = 0
            # The peaks describe the SAMPLE, not the warm-up: during warm-up the frontdoor is
            # idle, so every embedder is legitimately uncapped at its full slots, and a
            # lifetime peak reports that width forever (post-cap-20260927: peak 4 everywhere).
            reset = getattr(getattr(self._client, "scheduler", None), "reset_peaks", None)
            if reset is not None:
                reset()
            self.stats_start = self._snapshot()
        self.t0 = time.monotonic()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.monotonic() - self.t0
        with self._lock:
            self.stats_end = self._snapshot()
        self.stop.set()
        if self._thread is not None:
            self._thread.join(timeout=150)
        if self._crash is not None:
            raise SystemExit(f"scheduler load crashed: {self._crash!r}")

    def summary(self) -> dict:
        start = self.stats_start.get("counters", {})
        end = self.stats_end.get("counters", {})
        return {
            "texts": self.done,
            "seconds": self.elapsed,
            "texts_per_s": self.done / self.elapsed if self.elapsed else 0.0,
            "deferred": self.deferred,
            "errors": self.errors,
            "counters_delta": {k: end.get(k, 0) - start.get(k, 0) for k in sorted(set(end) | set(start))},
            "effective_caps_at_end": self.stats_end.get("effective_caps", {}),
            # peak_in_flight: raw in-flight high-water mark over the sample (includes texts
            # admitted before busy detection flipped). peak_in_flight_capped: the high-water
            # mark reached by grants made while capped — the cap-enforcement evidence.
            "peak_in_flight": self.stats_end.get("peak_in_flight", {}),
            "peak_in_flight_capped": self.stats_end.get("peak_in_flight_capped", {}),
            "busy_at_end": self.stats_end.get("busy", {}),
        }


def _load_for(mode: str, *, pace_s: float = 0.0, policy: Any = None,
              **scheduler_kwargs: Any) -> Callable[[list[int], int], Any]:
    if mode == "raw":
        if policy is not None:
            raise ValueError("a policy override needs --load-mode scheduler (raw load has no scheduler)")
        if not pace_s:
            return _Load
        return lambda ports, per_port: _Load(ports, per_port, pace_s=pace_s)
    if mode == "scheduler":
        if policy is not None:
            scheduler_kwargs = {**scheduler_kwargs,
                                "client_factory": lambda ports: _default_pool_client(ports, policy)}
        return lambda ports, per_port: _SchedulerLoad(ports, per_port, pace_s=pace_s, **scheduler_kwargs)
    raise ValueError(f"unknown load mode {mode!r}")


EMBEDDER_ENV_KEYS = ("OMP_WAIT_POLICY", "KMP_BLOCKTIME", "KMP_LIBRARY")


def embedder_env_readback(ports: list[int]) -> dict[str, Any]:
    """Read-only /proc readback of each pool embedder's OpenMP env (port -> {pid, env})."""
    from scripts.server.embedder_env_override import readback
    from scripts.server.embedder_placement_capture import _pids_for_ports

    pids = _pids_for_ports(ports)
    return readback({p: pids.get(p) for p in ports}, EMBEDDER_ENV_KEYS)


def env_expectation_problems(rb: dict[str, Any], expect: dict[str, str]) -> list[str]:
    problems = []
    for port, facts in sorted(rb.items()):
        if "error" in facts:
            problems.append(f":{port}: {facts['error']}")
            continue
        for key, want in sorted(expect.items()):
            if facts["env"].get(key) != want:
                problems.append(f":{port} pid {facts['pid']}: {key}={facts['env'].get(key)!r}, expected {want!r}")
    return problems


def g1(fd_ports: list[int], pairs: int, n_predict: int, load_factory: Callable[[list[int], int], Any] = _Load) -> dict:
    out: dict = {}
    for port in fd_ports:
        q, s = [], []
        s_loads: list[dict] = []
        for _ in range(pairs):
            q.append(_decode_tps(port, n_predict))
            with load_factory(POOL_PORTS, PER_PORT_IN_FLIGHT) as load:
                s.append(_decode_tps(port, n_predict))
            if load.errors:
                raise SystemExit(f"G1 :{port}: {load.errors} embedding errors under load; refuse")
            if hasattr(load, "summary"):
                s_loads.append(load.summary())
        q.append(_decode_tps(port, n_predict))  # closing A of the ABA
        qq = [abs(b - a) / a for a, b in zip(q, q[1:])]
        ratios = [si / ((qa + qb) / 2) for si, qa, qb in zip(s, q, q[1:])]
        out[str(port)] = {
            "unit": "decode tok/s (timings.predicted_per_second)",
            "q_samples": q, "s_samples": s,
            "s_over_q_median": statistics.median(ratios),
            "s_over_q_all": ratios,
            "aa_noise_floor_rel_median": statistics.median(qq),
        }
        if s_loads:
            out[str(port)]["s_embedding_load"] = s_loads
            out[str(port)]["s_embed_texts_per_s_median"] = statistics.median(
                x["texts_per_s"] for x in s_loads)
    return out


def g2(window_s: float, load_factory: Callable[[list[int], int], Any] = _Load) -> dict:
    res = {}
    for name, ports in (("one_port", POOL_PORTS[:1]), ("whole_pool", POOL_PORTS)):
        with load_factory(ports, PER_PORT_IN_FLIGHT) as load:
            time.sleep(window_s)
        if load.errors:
            raise SystemExit(f"G2 {name}: {load.errors} embedding errors; refuse")
        res[name] = {"texts": load.done, "seconds": load.elapsed, "texts_per_s": load.done / load.elapsed}
        if hasattr(load, "summary"):
            res[name]["load"] = load.summary()
    res["scaling_ratio"] = res["whole_pool"]["texts_per_s"] / res["one_port"]["texts_per_s"]
    return res


def _capture_window(fd_ports: list[int]) -> Any:
    from scripts.server.embedder_placement_capture import CaptureWindow

    return CaptureWindow(POOL_PORTS + fd_ports, fd_ports)


def _write_capture(out: Path, record: dict, window: Any) -> int:
    """The belief sidecar. Runs after the record is on disk; a refusal is loud, never silent."""
    from scripts.server.embedder_placement_capture import CaptureError, write_belief_measurements

    try:
        sidecar = write_belief_measurements(
            out, record, window=window, producer="epyc-orchestrator scripts/server/embedder_placement_gate.py",
            gate_path=Path(__file__).resolve())
    except (CaptureError, OSError, ValueError) as exc:
        print(f"belief capture REFUSED (record kept at {out}): {exc}", file=sys.stderr)
        return CAPTURE_EXIT
    print(f"belief capture: {sidecar}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--label", required=True,
                    choices=("pre", "post", "post-cap", "arm-baseline", "arm-candidate"))
    ap.add_argument("--arm", default=None,
                    help="arm name (required with arm-* labels), e.g. A0-cap0, A3-passive, BASE-cap1")
    ap.add_argument("--policy-override", action="append", default=[], metavar="SECTION.KEY=VALUE",
                    help="scheduler mode: run against the policy file with these fields replaced "
                         "(e.g. neighbour_cap.max_in_flight=0); the file is not modified")
    ap.add_argument("--per-port-in-flight", type=int, default=PER_PORT_IN_FLIGHT,
                    help="concurrent callers per pool port (default 4 = saturation)")
    ap.add_argument("--load-pace-s", type=float, default=0.0,
                    help="seconds each caller waits after every request (0 = back-to-back; "
                         ">0 = a low-duty point)")
    ap.add_argument("--expect-embedder-env", action="append", default=[], metavar="KEY=VALUE",
                    help="refuse to start unless every pool embedder's live env has KEY=VALUE")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--fd-ports", default="8070,8080,8180")
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--g2-window-s", type=float, default=60.0)
    ap.add_argument("--load-mode", choices=("raw", "scheduler"), default="raw",
                    help="raw = Phase-0 method (straight HTTP); scheduler = through the pooled "
                         "client + neighbour cap (REPL-EMB-1.4)")
    ap.add_argument("--sched-admission-wait-s", type=float, default=30.0,
                    help="scheduler mode: how long a load call waits for headroom before it "
                         "counts as deferred")
    args = ap.parse_args()
    if args.label.startswith("arm-") and not (args.arm and args.arm.strip()):
        raise SystemExit("refusing: an arm-* label needs --arm <name>")
    if args.per_port_in_flight < 1 or args.load_pace_s < 0:
        raise SystemExit("refusing: --per-port-in-flight >= 1 and --load-pace-s >= 0")
    fd_ports = [int(p) for p in args.fd_ports.split(",")]
    down = [p for p in POOL_PORTS + fd_ports if not _healthy(p)]
    if down:
        raise SystemExit(f"refusing: ports not healthy: {down}")
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from scripts.server.embedder_env_override import OverrideError, parse_overrides, read_record

    try:
        expect_env = parse_overrides(args.expect_embedder_env)
    except OverrideError as exc:
        raise SystemExit(f"refusing: {exc}") from exc
    env_before = embedder_env_readback(POOL_PORTS)
    problems = env_expectation_problems(env_before, expect_env)
    if problems:
        raise SystemExit("refusing: embedder env is not the arm's: " + "; ".join(problems))
    override_record = read_record()
    effective_policy, policy_overrides = None, {}
    if args.policy_override:
        if args.load_mode != "scheduler":
            raise SystemExit("refusing: --policy-override needs --load-mode scheduler")
        from src.embedding_pool.policy import load_policy

        try:
            effective_policy, policy_overrides = apply_policy_overrides(load_policy(), args.policy_override)
        except ValueError as exc:
            raise SystemExit(f"refusing: {exc}") from exc
    load_factory = _load_for(args.load_mode, pace_s=args.load_pace_s, policy=effective_policy,
                             admission_wait_s=args.sched_admission_wait_s)
    if args.per_port_in_flight != PER_PORT_IN_FLIGHT:
        # g1/g2 always ask for PER_PORT_IN_FLIGHT; a low-duty point substitutes its own width here,
        # so the measurement path (and its call shape) is the Phase-0 one.
        _base_factory, _width = load_factory, args.per_port_in_flight
        load_factory = lambda ports, _per_port: _base_factory(ports, _width)  # noqa: E731
    window = _capture_window(fd_ports)
    window.begin()  # serving-identity snapshot, before any timed sample
    record: dict = {
        "schema": "epyc.embedder_placement_gate.v1",
        "label": args.label,
        "load_mode": args.load_mode,
        "started_at": datetime.now(UTC).isoformat(),
        "params": {
            "fd_ports": fd_ports, "pool_ports": POOL_PORTS,
            "per_port_in_flight": args.per_port_in_flight, "load_warmup_s": LOAD_WARMUP_S,
            "pairs": args.pairs, "n_predict": args.n_predict, "g2_window_s": args.g2_window_s,
            "sched_admission_wait_s": args.sched_admission_wait_s,
            "load_pace_s": args.load_pace_s,
            "arm": args.arm,
            "policy_overrides": policy_overrides,
            "expect_embedder_env": expect_env,
            "embedder_env": env_before,
            "embedder_env_override_record": override_record,
            "fd_prompt_sha256": hashlib.sha256(FD_PROMPT.encode()).hexdigest(),
            "emb_text_sha256": hashlib.sha256(EMB_TEXT.encode()).hexdigest(),
        },
    }
    if args.load_mode == "scheduler":
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from src.embedding_pool.policy import load_policy
        from src.embedding_pool.topology import live_topology

        policy = effective_policy if effective_policy is not None else load_policy()
        record["scheduler_policy"] = policy.as_dict()
        record["pool_topology"] = live_topology(policy.neighbour_cap.guarded_roles).describe()
    t0 = datetime.now(UTC).isoformat()
    record["g2_pool_scaling"] = g2(args.g2_window_s, load_factory)
    t1 = datetime.now(UTC).isoformat()
    record["g1_frontdoor_decode"] = g1(fd_ports, args.pairs, args.n_predict, load_factory)
    t2 = datetime.now(UTC).isoformat()
    record["gate_windows"] = {"g2": {"started_at": t0, "finished_at": t1},
                              "g1": {"started_at": t1, "finished_at": t2}}
    window.mark("g2", t0, t1)
    window.mark("g1", t1, t2)
    record["finished_at"] = datetime.now(UTC).isoformat()
    window.finish()  # serving-identity snapshot, after the last timed sample
    env_after = embedder_env_readback(POOL_PORTS)
    if env_after != env_before:
        record["embedder_env_after"] = env_after
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"embedder env CHANGED during the run (record kept at {args.out}); no belief capture",
              file=sys.stderr)
        return CAPTURE_EXIT
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("label", "load_mode", "g2_pool_scaling")}, indent=2))
    if args.arm:
        print(f"arm {args.arm}: policy_overrides={policy_overrides} expect_env={expect_env} "
              f"override_record={(override_record or {}).get('experiment_id')}")
    for port, row in record["g1_frontdoor_decode"].items():
        print(f"G1 :{port} S/Q median {row['s_over_q_median']:.3f}  A/A floor {row['aa_noise_floor_rel_median']:.3%}")
    return _write_capture(args.out, record, window)


if __name__ == "__main__":
    raise SystemExit(main())
