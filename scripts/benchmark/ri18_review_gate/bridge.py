"""In-process bridge to the orchestrator's PRODUCTION review functions (RI-23b pattern).

The driver never re-implements the verdict prompt, its parsing, the revision or the gate: it
imports ``src.api.routes.chat_review`` from ``--code-root`` and calls
``_architect_verdict_with_status``, ``_fast_revise``, ``review_gate_score`` and
``_should_review`` directly. Process hygiene, as ``/mnt/raid0/llm/tmp/ri23b/ri23b_ab.py``:

* a PRIVATE inference tap (``INFERENCE_TAP_FILE`` / ``INFERENCE_TAP_EVENTS_FILE`` under the run
  directory) -- the live tap is never written;
* ``ORCHESTRATOR_RUNTIME_FLAGS_PATH`` at an absent file -- the production ``runtime_flags.json``
  posture is never read;
* features set in-process with ``set_features``: ``thinking_roles_chat_lane`` ON (asserted, as
  the live API runs it), ``content_cache`` OFF (a noise re-run must not be served from cache),
  ``model_fallback`` OFF (a failure shows as a failure, not as another role's answer);
* ``LLMPrimitives(mock_mode=False, server_urls={<only the role this segment needs>}, registry=
  RegistryLoader(validate_paths=False))`` -- a call to any other role fails loudly.

Device-seconds of a call are the ``v1_escalation._counters`` delta (``total_prompt_eval_ms`` +
``total_generation_ms``), exactly as ``v1_escalation._record_step`` computes a step's cost.

Stage 4 region claim (verified in the orchestrator source, 2026-09-30): ``LLMPrimitives.llm_call``
DOES take the per-call CPU region claim itself -- ``src/llm_primitives/inference.py``
``_real_call_single`` wraps every direct single-instance call in
``cpu_region_lock_for_instance(topology_role, idx)`` (resolved by
``topology_instance_for_port``) -- but ONLY when ``ORCHESTRATOR_PER_REGION_LOCKS=1``; the GLOBAL
cross-role layer (the one AutoKernel and bench holders contend on) needs
``ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT=1``. Both default OFF outside the API launcher
(``orchestrator_stack.py`` sets them for the API), so a plain script would take only the legacy
``heavy_model.lock`` and exclude nothing. The CPU segments therefore export both flags (and a
bounded ``ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_S``), pin ``worker_general`` to the single URL
:8070 (a direct ``CachingBackend``, so the claim is frontdoor instance 0 = q0-q3 + GLOBAL, the
same claim the API takes), and assert at segment start that the topology resolves :8070 to
``("frontdoor", 0)`` with non-empty regions -- otherwise the lock would silently be a no-op. No
outer claim is held: nesting ``region_claim`` around ``llm_call`` with the flag on would
self-deadlock on the in-process holder table. A ``CpuRegionLockTimeout``/``ContentionDenied``
(``_fast_revise`` swallows it and returns the original answer!) is caught by the call recorder
and turned into ``WindowLost``: the item is NOT recorded, the segment stops, it resumes next
window.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from . import fakes

GPU_URL = "http://localhost:8083"
CPU_URL = "http://localhost:8070"
REVIEWER_ROLE = "architect_critic"
REVISER_ROLE = "worker_general"
FRONTDOOR_ROLE = "frontdoor"
VFULL_QUESTION_CAP = 1500
CONTENTION_ERRORS = ("CpuRegionLockTimeout", "ContentionDenied")
CPU_LOCK_ENV = {
    "ORCHESTRATOR_PER_REGION_LOCKS": "1",
    "ORCHESTRATOR_CROSS_ROLE_DISJOINT_PLACEMENT": "1",
}


class WindowLost(RuntimeError):
    """The CPU region claim was denied mid-segment: stop, resume next window."""


class InstrumentFault(RuntimeError):
    """The measuring instrument itself is broken (wrong flag, dead embedder, no-op lock)."""


def git_state(path: str | Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    head = run("rev-parse", "HEAD") or None
    dirty = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": head, "dirty": bool(dirty), "dirty_files": dirty.splitlines()[:50]}


class CallRecorder:
    """Transparent proxy around primitives: remembers each ``llm_call``'s raw text or error."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: list[dict[str, Any]] = []

    def llm_call(self, prompt: str, *args: Any, **kwargs: Any) -> str:
        t0 = time.monotonic()
        entry: dict[str, Any] = {"role": kwargs.get("role"), "n_tokens": kwargs.get("n_tokens"),
                                 "kwargs": sorted(kwargs), "prompt_chars": len(prompt)}
        try:
            out = self._inner.llm_call(prompt, *args, **kwargs)
        except BaseException as exc:
            entry.update(error=f"{type(exc).__name__}: {exc}"[:500], error_type=type(exc).__name__,
                         wall_s=round(time.monotonic() - t0, 3))
            self.calls.append(entry)
            raise
        entry.update(raw=out, wall_s=round(time.monotonic() - t0, 3))
        self.calls.append(entry)
        return out

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _CheckedEmbedder:
    """Embedder proxy that refuses a degenerate vector (a dead embedder must fail loudly)."""

    def __init__(self, inner: Any, is_degenerate: Any) -> None:
        self._inner = inner
        self._is_degenerate = is_degenerate
        self.embeds = 0

    def embed_task_ir(self, task_ir: dict[str, Any]) -> Any:
        vec = self._inner.embed_task_ir(task_ir)
        reason = self._is_degenerate(vec)
        if reason:
            raise InstrumentFault(f"degenerate query embedding ({reason})")
        self.embeds += 1
        return vec

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class Bridge:
    """One segment's view of the orchestrator.

    ``mode``: ``real`` (live servers), ``dry`` (REAL orchestrator functions, fake primitives,
    fake retriever) or ``stub`` (no orchestrator import at all: ``fakes.StubChatReview``).
    ``device``: ``gpu`` (reviewer on :8083), ``cpu`` (reviser on :8070, region-lock env) or
    ``none`` (quality mark / gate only).
    """

    def __init__(self, *, mode: str, device: str, code_root: str | None, run_dir: Path,
                 segment_tag: str, gpu_url: str = GPU_URL, cpu_url: str = CPU_URL,
                 lock_timeout_s: float = 30.0, fake_primitives: Any = None) -> None:
        self.mode = mode
        self.device = device
        self.code_root = code_root
        self.gpu_url = gpu_url
        self.cpu_url = cpu_url
        self.header: dict[str, Any] = {"mode": mode, "device": device}
        if mode == "stub":
            self.code_root = None  # nothing imported, so no code commit to pin or drift
            self.cr: Any = fakes.StubChatReview()
            self._counters = fakes.stub_counters
            self._quality = fakes.stub_quality
            self.prim = fake_primitives or fakes.FakePrimitives()
            self.rec = CallRecorder(self.prim)
            self.header.update(stub=True, thinking_roles_chat_lane_enabled=True,
                               chat_template_kwargs_architect_critic=None,
                               reviewer_role=REVIEWER_ROLE, review_threshold=0.6)
            return
        if not code_root or not Path(code_root, "src").is_dir():
            raise InstrumentFault(f"--code-root {code_root!r} is not an orchestrator checkout")
        tap_dir = run_dir / "tap"
        tap_dir.mkdir(parents=True, exist_ok=True)
        os.environ["INFERENCE_TAP_FILE"] = str(tap_dir / f"{segment_tag}.tap.log")
        os.environ["INFERENCE_TAP_EVENTS_FILE"] = str(tap_dir / f"{segment_tag}.events.jsonl")
        os.environ["ORCHESTRATOR_RUNTIME_FLAGS_PATH"] = str(run_dir / "runtime_flags.absent.json")
        if device == "cpu":
            os.environ.update(CPU_LOCK_ENV)
            os.environ["ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_S"] = str(lock_timeout_s)
        else:
            for key in CPU_LOCK_ENV:
                os.environ.pop(key, None)
        if code_root not in sys.path:
            sys.path.insert(0, code_root)
        os.chdir(code_root)

        from src.api.routes import chat_review
        from src.api.routes.v1_escalation import _counters
        from src.classifiers import detect_output_quality_issue
        from src.features import get_features, set_features

        self.cr = chat_review
        self._counters = _counters
        self._quality = detect_output_quality_issue
        set_features(get_features(override={"thinking_roles_chat_lane": True,
                                            "content_cache": False, "model_fallback": False}))
        if mode == "dry":
            self.prim = fake_primitives or fakes.FakePrimitives()
        elif device in ("gpu", "cpu"):
            from src.llm_primitives import LLMPrimitives
            from src.registry_loader import RegistryLoader

            role, url = (REVIEWER_ROLE, gpu_url) if device == "gpu" else (REVISER_ROLE, cpu_url)
            self.prim = LLMPrimitives(mock_mode=False, server_urls={role: url},
                                      registry=RegistryLoader(validate_paths=False))
        else:
            self.prim = None
        self.rec = CallRecorder(self.prim) if self.prim is not None else None
        self.header.update(self._orch_header())
        self._assert_instrument()

    # ── header / instrument checks ───────────────────────────────────────
    def _orch_header(self) -> dict[str, Any]:
        from src.chat_completions_roles import thinking_roles_chat_lane_enabled
        from src.config import get_config
        from src.prompt_builders import review as review_prompts
        from src.registry.registry_loader import chat_template_kwargs_for_role
        from src.roles import resolve_reviewer_role

        head: dict[str, Any] = {
            "code_root": self.code_root,
            "code_root_git": git_state(self.code_root),
            "thinking_roles_chat_lane_enabled": bool(thinking_roles_chat_lane_enabled()),
            "chat_template_kwargs_architect_critic": chat_template_kwargs_for_role(REVIEWER_ROLE),
            "reviewer_role": str(resolve_reviewer_role()),
            "review_threshold": float(get_config().chat.review_low_q_threshold),
            "verdict_question_cap": getattr(review_prompts, "REVIEW_VERDICT_QUESTION_CAP", None),
            "verdict_answer_cap": getattr(review_prompts, "REVIEW_VERDICT_ANSWER_CAP", None),
            "has_review_gate_score": hasattr(self.cr, "review_gate_score"),
            "server_urls": getattr(self.prim, "server_urls", None) if self.mode == "real" else None,
        }
        if self.device == "cpu":
            from src.runtime.instance_topology import get_instance_regions, topology_instance_for_port

            port = int(self.cpu_url.rsplit(":", 1)[-1].split("/")[0])
            inst = topology_instance_for_port(port)
            regions = get_instance_regions().get(tuple(inst)) if inst else None
            head["region_claim"] = {
                "env": {k: os.environ.get(k) for k in (*CPU_LOCK_ENV,
                                                       "ORCHESTRATOR_INFERENCE_LOCK_TIMEOUT_S")},
                "port": port, "topology_instance": list(inst) if inst else None,
                "regions": sorted(regions) if regions else None,
            }
        return head

    def _assert_instrument(self) -> None:
        h = self.header
        if not h["thinking_roles_chat_lane_enabled"]:
            raise InstrumentFault("thinking_roles_chat_lane is not ON in-process")
        if h["reviewer_role"] != REVIEWER_ROLE:
            raise InstrumentFault(f"reviewer binding is {h['reviewer_role']!r}, not {REVIEWER_ROLE}")
        if not h["has_review_gate_score"]:
            raise InstrumentFault("chat_review.review_gate_score missing: code_root predates RI-18 C2")
        if self.device == "cpu" and self.mode == "real":
            claim = h["region_claim"]
            if claim["topology_instance"] != [FRONTDOOR_ROLE, 0] or not claim["regions"]:
                raise InstrumentFault(f":8070 region claim would be a no-op: {claim}")

    # ── calls ────────────────────────────────────────────────────────────
    def _call(self, fn: Any, *args: Any, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        before = self._counters(self.prim)
        n0 = len(self.rec.calls)
        t0 = time.monotonic()
        out = fn(*args, **kwargs)
        wall = time.monotonic() - t0
        after = self._counters(self.prim)
        calls = self.rec.calls[n0:]
        meta = {}
        getter = getattr(self.prim, "get_last_inference_meta", None)
        if callable(getter) and calls:
            meta = getter() or {}
        last = calls[-1] if calls else {}
        if last.get("error_type") in CONTENTION_ERRORS:
            raise WindowLost(last["error"])
        prompt_ms = max(0.0, after["prompt_ms"] - before["prompt_ms"])
        gen_ms = max(0.0, after["gen_ms"] - before["gen_ms"])
        n_calls = int(after["calls"] - before["calls"])
        info = {
            "wall_s": round(wall, 3),
            "llm_calls": n_calls,
            "prompt_ms": round(prompt_ms, 3),
            "gen_ms": round(gen_ms, 3),
            "device_seconds": round((prompt_ms + gen_ms) / 1000.0, 6) if n_calls > 0 else None,
            "tokens": int(max(0.0, after["tokens"] - before["tokens"])),
            "prompt_tokens": int(max(0.0, after["prompt_tokens"] - before["prompt_tokens"])),
            "completion_reason": meta.get("completion_reason") if isinstance(meta, dict) else None,
            "raw": last.get("raw"),
            "call_error": last.get("error"),
            "call_kwargs": last.get("kwargs"),
            "call_role": last.get("role"),
        }
        return out, info

    def verdict(self, question: str, answer: str, *, question_cap: int | None = None
                ) -> dict[str, Any]:
        """The production verdict; ``question_cap=None`` is the production call itself."""
        kwargs = {} if question_cap is None else {"question_cap": question_cap}
        (verdict_text, status), info = self._call(
            self.cr._architect_verdict_with_status, question, answer, self.rec, **kwargs)
        return {"status": status, "verdict": verdict_text, "question_cap": question_cap, **info}

    def revise(self, question: str, answer: str, corrections: str) -> dict[str, Any]:
        revised, info = self._call(self.cr._fast_revise, question, answer, corrections, self.rec)
        failed = info["call_error"] is not None or info["llm_calls"] <= 0
        return {"revised": revised, "changed": revised != answer, "revise_failed": failed, **info}

    def canaries(self) -> dict[str, Any]:
        right = self.verdict(fakes.CANARY_QUESTION, fakes.CANARY_RIGHT)
        wrong = self.verdict(fakes.CANARY_QUESTION, fakes.CANARY_WRONG)
        ok = right["status"] == "ok" and wrong["status"] == "wrong"
        return {"ok": ok,
                "right": {k: right[k] for k in ("status", "raw", "wall_s", "call_error")},
                "wrong": {k: wrong[k] for k in ("status", "raw", "wall_s", "call_error")}}

    def quality(self, answer: str) -> str | None:
        return self._quality(answer)

    # ── gate ─────────────────────────────────────────────────────────────
    def gate(self, state: Any, role: str, answer: str, question: str, task_id: str
             ) -> dict[str, Any]:
        t0 = time.monotonic()
        gs = self.cr.review_gate_score(state, role, answer)
        gq = self.cr.review_gate_score(state, role, answer, key_text=question)
        prod = self.cr._should_review(state, task_id, role, answer)
        return {"gate": _gate_dict(gs), "gate_question": _gate_dict(gq),
                "should_review": bool(prod), "gate_wall_s": round(time.monotonic() - t0, 3),
                "_gs": gs, "_gq": gq}

    def snapshot_state(self, store_dir: Path, retriever_kind: str) -> tuple[Any, dict[str, Any]]:
        """``state.hybrid_router.retriever`` over a store COPY, built as the API builds it."""
        if self.mode != "real":
            return fakes.fake_state(), {"retriever": "fake"}
        from orchestration.repl_memory.embedder import (
            EmbeddingConfig,
            TaskEmbedder,
            is_degenerate_embedding,
        )
        from orchestration.repl_memory.episodic_store import EpisodicStore
        from orchestration.repl_memory.retriever import (
            GraphEnhancedRetriever,
            RetrievalConfig,
            TwoPhaseRetriever,
        )

        store = EpisodicStore(db_path=Path(store_dir))
        faiss = getattr(store, "_embedding_store", None) or getattr(store, "embedding_store", None)
        n_vectors = getattr(getattr(faiss, "index", None), "ntotal", None)
        if n_vectors is not None and n_vectors <= 0:
            raise InstrumentFault(f"snapshot FAISS index at {store_dir} is empty")
        # Production builds TaskEmbedder() (hash fallback ON). Same servers, same vectors; the
        # only change is the failure mode: a dead embedder raises instead of hashing.
        embedder = _CheckedEmbedder(
            TaskEmbedder(EmbeddingConfig(use_fallback=False, allow_subprocess=False)),
            is_degenerate_embedding)
        rc = RetrievalConfig()
        if retriever_kind == "graph":
            from orchestration.repl_memory.failure_graph import FailureGraph
            from orchestration.repl_memory.hypothesis_graph import HypothesisGraph

            retriever = GraphEnhancedRetriever(
                store=store, embedder=embedder,
                failure_graph=FailureGraph(path=Path(store_dir) / "failure_graph"),
                hypothesis_graph=HypothesisGraph(path=Path(store_dir) / "hypothesis_graph"),
                config=rc)
        elif retriever_kind == "two_phase":
            retriever = TwoPhaseRetriever(store=store, embedder=embedder, config=rc)
        else:
            raise ValueError(retriever_kind)
        info = {"retriever": type(retriever).__name__, "faiss_ntotal": n_vectors,
                "retrieval_config": retrieval_config_dict(rc)}
        return SimpleNamespace(hybrid_router=SimpleNamespace(retriever=retriever)), info


def retrieval_config_dict(rc: Any) -> dict[str, Any]:
    return {k: getattr(rc, k, None) for k in ("semantic_k", "min_similarity", "min_q_value",
                                              "q_weight", "top_n")}


def _gate_dict(gs: Any) -> dict[str, Any]:
    thr = gs.threshold
    return {"avg_q": gs.avg_q, "n_results": gs.n_results, "n_role_rows": gs.n_role_rows,
            "skip_reason": gs.skip_reason, "threshold": None if thr != thr else thr,
            "answer_chars": gs.answer_chars}
