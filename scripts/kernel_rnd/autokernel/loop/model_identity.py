"""Whole-model output identity: the correctness gate for a placement/loader route.

`cpu_weight_placement` (gates.CPU_SOURCE_ROUTES, 2026-10-03, Fable CPU seed 1) admits
edits to the model loader's data-load bodies so a candidate can place weight bytes
(mbind of row quarters to NUMA nodes, madvise, first touch). No `test-backend-ops`
suite runs the loader, and placement changes no value, so the only meaningful gate is
the model itself: serve the campaign's frozen requests greedily from the ANCHOR and from
the CANDIDATE under the campaign launch (same argv, env, topology prefix and port; only
the build differs) and require byte-identical completions.

Requests are sent one at a time, so batch composition cannot differ between arms. If
the arms disagree, the anchor is served once more: an anchor that disagrees with itself
means this instrument cannot judge (`oracle_unavailable`), never that the patch is
wrong. Only an anchor that reproduces itself while the candidate differs is `wrong`.

This is a gate, not a measurement: nothing here is timed or compared for speed.

REPETITIONS (2026-10-04, `cpu_graph_sched` / `cpu_graph_optimize`): a scheduler that runs
two graph nodes concurrently can race, and a race that only two concurrent nodes produce
is invisible to a per-op suite. With `repeats > 1` the candidate serves every selected
request that many times (prompt cache off in BOTH arms, so every repetition recomputes
the whole prompt) and all repetitions must be byte-identical. A candidate that disagrees
with itself is `wrong` only when the anchor, served the same way, agrees with itself.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Sequence

#: How many frozen requests each arm serves. Two already exercise every weight the
#: graph reads (each token reads all of them); more only lengthens the gate.
DEFAULT_REQUESTS = 2
BOOT_TIMEOUT_S = 900       # a --no-mmap DS41 load is minutes, not seconds
REQUEST_TIMEOUT_S = 900
TEARDOWN_S = 180           # a --no-mmap server unmaps ~0.5 TB on SIGTERM


@dataclass(frozen=True)
class IdentityResult:
    status: str            # "pass" | "wrong" | "unavailable"
    reason: str
    detail: str = ""


def _completion_digest(raw: bytes) -> tuple[str, str, dict]:
    """(digest of the generated content and token ids, short preview, full record).

    The full record (`{"content": ..., "tokens": ...}`) is what a divergence receipt
    persists (`_row_record`/`_persist_divergence`); the digest/preview pair is what every
    existing caller already keyed its comparisons on and stays unchanged."""
    body = json.loads(raw)
    content = body.get("content")
    tokens = body.get("tokens")
    if not isinstance(content, str):
        raise ValueError("completion has no content string")
    tokens = tokens if isinstance(tokens, list) and tokens else None
    payload = {"content": content, **({"tokens": tokens} if tokens else {})}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return (hashlib.sha256(canonical).hexdigest(), content[:80],
            {"content": content, "tokens": tokens})


def _launch_shape(recipe) -> tuple:
    """The launch with its build directory abstracted: what must match across arms."""
    build = str(recipe.build_dir).rstrip("/")

    def strip(value: str) -> str:
        return str(value).replace(build, "<build>")

    return (tuple(strip(item) for item in recipe.command_argv),
            tuple(recipe.topology_prefix), recipe.port,
            tuple(sorted((key, strip(value)) for key, value in recipe.launch_env)))


def serve(recipe, requests: Sequence[tuple[str, bytes]], *,
          boot_timeout_s: int = BOOT_TIMEOUT_S,
          request_timeout_s: int = REQUEST_TIMEOUT_S,
          prepare: Callable[[int], object] | None = None) -> list[tuple[str, str, str]]:
    """Launch `recipe`'s llama-server, serve `requests` sequentially, always tear down.

    `prepare(port)`, when given, runs before EVERY request (the long-context surface
    restores its saved slot there, so each completion extends the same prefix).
    Returns [(prompt_id, digest, preview, full)], `full` = {"content", "tokens"} (the
    divergence receipt's raw material). Raises on any launch/request failure."""
    recipe.validate_launch(recipe.template, recipe.build_dir, recipe.port)
    server = subprocess.Popen(list(recipe.argv), env=dict(recipe.launch_env),
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        deadline = time.monotonic() + boot_timeout_s
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"server exited {server.returncode} during load")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{recipe.port}/health",
                                            timeout=2):
                    break
            except (urllib.error.URLError, OSError, ValueError):
                if time.monotonic() > deadline:
                    raise RuntimeError("server not healthy within the boot timeout")
                time.sleep(2)
        out = []
        for prompt_id, body in requests:
            if prepare is not None:
                prepare(recipe.port)
            request = urllib.request.Request(
                f"http://127.0.0.1:{recipe.port}/completion", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=request_timeout_s) as response:
                raw = response.read()
            digest, preview, full = _completion_digest(raw)
            out.append((prompt_id, digest, preview, full))
        return out
    finally:
        if server.poll() is None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                server.wait(TEARDOWN_S)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(server.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                server.wait(10)


def model_architecture(recipe) -> str | None:
    """`general.architecture` of the GGUF a recipe serves (None when unreadable)."""
    model = getattr(getattr(recipe, "model", None), "path", None) or \
        getattr(getattr(recipe, "template", None), "model", None)
    if not model:
        return None
    try:
        from ..controller.workload_contract import read_census
        return read_census(model).architecture
    except Exception:  # an unreadable model never satisfies an architecture requirement
        return None


def _uncached(requests: Sequence[tuple[str, bytes]]) -> tuple[tuple[str, bytes], ...]:
    """The requests with the prompt cache off, so a repetition recomputes the prompt."""
    out = []
    for prompt_id, body in requests:
        sampler = json.loads(body)
        sampler["cache_prompt"] = False
        out.append((prompt_id, json.dumps(sampler, sort_keys=True).encode()))
    return tuple(out)


def _inconsistent(rows: list, n: int) -> list[int]:
    """Indices of requests whose repeated completions (rows = n-strided) differ."""
    return [i for i in range(n) if len({rows[r * n + i][1] for r in range(len(rows) // n)}) > 1]


def _row_record(row: tuple) -> dict:
    """A persistable projection of one `serve()` row; `full` is absent for a test double
    that returns bare `(prompt_id, digest, preview)` rows, never a hard failure."""
    full = row[3] if len(row) > 3 and isinstance(row[3], dict) else {}
    return {"prompt_id": row[0], "digest": row[1],
           "content": full.get("content"), "tokens": full.get("tokens")}


def _divergence(row_a: tuple, row_b: tuple) -> dict:
    """The first index at which two rows' completions disagree: token ids when both rows
    carry them, else the raw content characters. `index: None` with equal records means
    the rows carry no comparable material (a test double with no `full`)."""
    ra, rb = _row_record(row_a), _row_record(row_b)
    ta, tb = ra.get("tokens"), rb.get("tokens")
    if isinstance(ta, list) and isinstance(tb, list) and ta and tb:
        level, seq_a, seq_b = "token", ta, tb
    else:
        level, seq_a, seq_b = "char", ra.get("content") or "", rb.get("content") or ""
    index = next((i for i, (x, y) in enumerate(zip(seq_a, seq_b)) if x != y), None)
    if index is None and len(seq_a) != len(seq_b):
        index = min(len(seq_a), len(seq_b))
    return {"level": level, "index": index}


def _persist_divergence(record_dir, *, kind: str, **rows) -> str:
    """Persist the full per-repeat records behind an inconsistent identity verdict so a
    future refusal can be localized instead of re-run blind (no divergence position was
    previously kept anywhere, and `IdentityResult.detail` is truncated to 600 chars by
    `gates.check_model_identity_targets`). Returns the written file's path."""
    record_dir = Path(record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "epyc.autokernel.model_identity_divergence.v1", "kind": kind, **rows}
    blob = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
    path = record_dir / f"identity-divergence-{kind}-{digest}.json"
    path.write_text(blob, encoding="utf-8")
    return str(path)


def check(*, anchor_recipe, candidate_recipe, requests: Sequence[tuple[str, bytes]],
          n_requests: int = DEFAULT_REQUESTS,
          window: Callable[[], object] | None = None,
          serve_fn: Callable[..., list] = serve, repeats: int = 1,
          prepare: Callable[[int], object] | None = None,
          record_dir: "Path | str | None" = None) -> IdentityResult:
    """Anchor vs candidate greedy completions on the first `n_requests` frozen requests.

    `window` is the caller's CPU measurement window (a context-manager factory), so the
    two model loads never overlap another CPU measurement on this host. `repeats > 1`
    adds the repetition-identity race detector (module docstring). `prepare(port)` runs
    before every request on both arms; it resets the server state itself (a slot
    restore), so repetitions keep the prompt cache instead of `_uncached`.

    `record_dir`, when given, persists the full per-repeat token ids and content behind
    any non-`pass` verdict (`_persist_divergence`) and puts the written file's path first
    in `detail`, so a future refusal can be localized instead of re-run blind. Absent
    (the default) nothing is written and every verdict is exactly as before."""
    if not requests:
        return IdentityResult("unavailable", "no frozen requests to serve")
    if anchor_recipe.backend != "cpu" or candidate_recipe.backend != "cpu":
        return IdentityResult("unavailable", "model identity gate requires CPU launches")
    if _launch_shape(anchor_recipe) != _launch_shape(candidate_recipe):
        return IdentityResult("unavailable", "anchor and candidate launches differ beyond "
                              "the build; the comparison would not isolate the patch")
    selected = tuple(requests[:max(1, n_requests)])
    for _prompt_id, body in selected:
        try:
            sampler = json.loads(body)
        except (ValueError, TypeError):
            return IdentityResult("unavailable", "frozen request is not JSON")
        if not (sampler.get("temperature") == 0 or sampler.get("top_k") == 1):
            return IdentityResult("unavailable", "frozen request is not greedy "
                                  "(temperature 0 or top_k 1); identity cannot be required")
    guard = window if window is not None else nullcontext
    repeats = max(1, int(repeats))
    if repeats > 1 and prepare is None:
        selected = _uncached(selected)
    n = len(selected)
    if prepare is not None:
        base_serve = serve_fn

        def serve_fn(recipe, requests):  # noqa: E306
            return base_serve(recipe, requests, prepare=prepare)
    try:
        with guard():
            anchor = serve_fn(anchor_recipe, selected)
            candidate_all = serve_fn(candidate_recipe, selected * repeats)
    except Exception as exc:  # a launch/request failure is never a correctness verdict
        return IdentityResult("unavailable", f"model identity serving failed: "
                              f"{type(exc).__name__}: {exc}")
    candidate = candidate_all[:n]
    racy = _inconsistent(candidate_all, n) if repeats > 1 else []
    if racy:
        try:
            with guard():
                anchor_all = serve_fn(anchor_recipe, selected * repeats)
        except Exception as exc:
            return IdentityResult("unavailable", f"anchor repetition serve failed: "
                                  f"{type(exc).__name__}: {exc}")
        if _inconsistent(anchor_all, n) or [row[1] for row in anchor_all[:n]] != \
                [row[1] for row in anchor]:
            detail: dict = {"anchor_first": [_row_record(r) for r in anchor],
                            "anchor_repeats": [_row_record(r) for r in anchor_all],
                            "candidate_repeats": [_row_record(r) for r in candidate_all]}
            if record_dir is not None:
                detail = {"record": _persist_divergence(
                    record_dir, kind="anchor_self_inconsistent", **detail), **detail}
            return IdentityResult("unavailable", "the anchor's own repeated greedy "
                                  "completions differ; this instrument cannot judge a race",
                                  json.dumps(detail))
        detail = {"candidate_repeats": [_row_record(r) for r in candidate_all],
                  "anchor": [_row_record(r) for r in anchor]}
        if record_dir is not None:
            detail = {"record": _persist_divergence(record_dir, kind="race", **detail),
                      **detail}
        return IdentityResult("wrong", f"{len(racy)} of {n} request(s) gave different greedy "
                              f"completions across {repeats} candidate repetitions while the "
                              "anchor reproduces itself (a scheduling race)",
                              json.dumps(detail))
    differing = [(a, c) for a, c in zip(anchor, candidate) if a[1] != c[1]]
    if not differing:
        return IdentityResult("pass", f"{len(selected)} greedy completion(s) byte-identical "
                              "to the anchor under the campaign launch"
                              + (f", each reproduced {repeats}x by the candidate"
                                 if repeats > 1 else ""),
                              json.dumps([row[:2] for row in candidate]))
    try:
        with guard():
            again = serve_fn(anchor_recipe, selected)
    except Exception as exc:
        return IdentityResult("unavailable", f"anchor re-serve failed: {type(exc).__name__}: {exc}")
    if [row[1] for row in again] != [row[1] for row in anchor]:
        detail = {"first": [_row_record(r) for r in anchor], "second": [_row_record(r) for r in again]}
        if record_dir is not None:
            detail = {"record": _persist_divergence(record_dir, kind="anchor_unstable",
                                                     **detail), **detail}
        return IdentityResult("unavailable", "the anchor's own greedy completions differ "
                              "between two launches; this instrument cannot judge identity",
                              json.dumps(detail))
    (a, c), = differing[:1]
    divergence = _divergence(a, c)
    detail = {"divergence": divergence, "anchor": _row_record(a), "candidate": _row_record(c)}
    if record_dir is not None:
        detail = {"record": _persist_divergence(
            record_dir, kind="wrong", divergence=divergence,
            anchor=[_row_record(r) for r in anchor],
            candidate=[_row_record(r) for r in candidate]), **detail}
    return IdentityResult("wrong", f"{len(differing)} of {len(selected)} greedy completion(s) "
                          f"differ from the reproducible anchor (first: {a[0]}, "
                          f"{divergence['level']} divergence at index {divergence['index']})",
                          json.dumps(detail))


__all__ = ["IdentityResult", "check", "serve", "model_architecture", "DEFAULT_REQUESTS"]
