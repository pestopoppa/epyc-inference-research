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


def _completion_digest(raw: bytes) -> tuple[str, str]:
    """(digest of the generated content and token ids, short preview)."""
    body = json.loads(raw)
    content = body.get("content")
    tokens = body.get("tokens")
    if not isinstance(content, str):
        raise ValueError("completion has no content string")
    payload = {"content": content,
               **({"tokens": tokens} if isinstance(tokens, list) and tokens else {})}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest(), content[:80]


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
          request_timeout_s: int = REQUEST_TIMEOUT_S) -> list[tuple[str, str, str]]:
    """Launch `recipe`'s llama-server, serve `requests` sequentially, always tear down.

    Returns [(prompt_id, digest, preview)]. Raises on any launch/request failure."""
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
            request = urllib.request.Request(
                f"http://127.0.0.1:{recipe.port}/completion", data=body,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=request_timeout_s) as response:
                raw = response.read()
            digest, preview = _completion_digest(raw)
            out.append((prompt_id, digest, preview))
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


def check(*, anchor_recipe, candidate_recipe, requests: Sequence[tuple[str, bytes]],
          n_requests: int = DEFAULT_REQUESTS,
          window: Callable[[], object] | None = None,
          serve_fn: Callable[..., list] = serve) -> IdentityResult:
    """Anchor vs candidate greedy completions on the first `n_requests` frozen requests.

    `window` is the caller's CPU measurement window (a context-manager factory), so the
    two model loads never overlap another CPU measurement on this host."""
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
    try:
        with guard():
            anchor = serve_fn(anchor_recipe, selected)
            candidate = serve_fn(candidate_recipe, selected)
    except Exception as exc:  # a launch/request failure is never a correctness verdict
        return IdentityResult("unavailable", f"model identity serving failed: "
                              f"{type(exc).__name__}: {exc}")
    differing = [(a, c) for a, c in zip(anchor, candidate) if a[1] != c[1]]
    if not differing:
        return IdentityResult("pass", f"{len(selected)} greedy completion(s) byte-identical "
                              "to the anchor under the campaign launch",
                              json.dumps([row[:2] for row in candidate]))
    try:
        with guard():
            again = serve_fn(anchor_recipe, selected)
    except Exception as exc:
        return IdentityResult("unavailable", f"anchor re-serve failed: {type(exc).__name__}: {exc}")
    if [row[1] for row in again] != [row[1] for row in anchor]:
        return IdentityResult("unavailable", "the anchor's own greedy completions differ "
                              "between two launches; this instrument cannot judge identity",
                              json.dumps({"first": anchor, "second": again}))
    (a, c), = differing[:1]
    return IdentityResult("wrong", f"{len(differing)} of {len(selected)} greedy completion(s) "
                          f"differ from the reproducible anchor (first: {a[0]})",
                          json.dumps({"anchor": a, "candidate": c}))


__all__ = ["IdentityResult", "check", "serve", "DEFAULT_REQUESTS"]
