#!/usr/bin/env python3
"""DS41-C95 driver, run 3 (run3.py). A copy of run2.py (run2.py/proxy2.py keep serving the live F12 run
untouched). Every run-2 arm behaves as before unless a run-3 option is given. Run 3 adds:
  * UFH14-A3 prefix warming + staggered starts: `--warm-prefix` runs calls in start groups of
    --concurrency; each member's proxy3 holds its first tool-bearing request, the group coordinator
    sends one warm copy per member (that member's exact body, add_generation_prompt=false,
    max tokens 1, pinned to its own idle slot, serially), then releases the members in reverse
    warm order with --stagger-s between them. `--canon-paths` (default on) aliases each call's
    workspace root to one fixed string on the wire, so concurrent calls on one context send
    byte-identical prompts. Cold baseline: `--group-starts` without --warm-prefix.
    Measured: per-call first-request cache_n/prompt_n (result.json -> first_request), the group
    record results/_groups/<gid>.json, and `group-report` (cold-prefill tokens/seconds, decode
    tok/s of turns overlapped by a neighbour's prefill vs clean).
  * UFH14-A4 prefix-preserving compaction (`--keep-prefix-compaction`, proxy3) and D3 task pinning
    (`--pin-task`). Measured: wire_timing -> compactions (re-prefill tokens per compaction).
  * Sandbox (`--sandbox landlock`, default): every proxied harness runs under sandbox3.py
    (Landlock; the live campaign store, shared git object stores, governance repos, other calls'
    results and the shared /tmp are unreadable; the call's lane, bundle, TMPDIR and result dir
    stay writable). scan_contamination stays as the backstop; sandbox denials are counted.
  * Arm `cxa`: codex inline with the F1 transport settings (1 h stream idle timeout, window =
    slot ctx, compaction 150k or --compact-limit) but NOT solo, for the A3/A4 A/Bs.
  * `--label X` stores a variant under results/<ctx>/<arm>+X/ (graded as its own arm).
  * `dry-run`: prints the exact proxy/harness/sandbox commands and the warm-request and
    compaction-rewrite bodies, replayed offline from recorded wire bodies. Sends nothing.

Run-2 docstring follows.
DS41-C95 driver, run 2 (run2.py). A copy of run.py, which keeps serving run 1 untouched. It adds:
  * arms `cxv` (27B x codex on the VARIABLE prompt and bundle) and `orv` (27B via the orchestrator
    REPL, orchestrator-variable context mode, INF-78 OAB-7);
  * the run-1 crash fix: proxy2.py closes its gzip on SIGTERM, a tolerant gzip reader, and
    post-processing that can never skip result.json;
  * `finalize`, which writes result.json for run-1 calls whose post-processing crashed;
  * `--after-pid PID`, which waits for a running driver to exit before starting, and prints
    "RUN DONE" when its last call ends (launch_arms2.sh prints the final "ALL ARMS DONE").

Original run.py docstring follows.
DS41-C95 driver: replay frozen DS41 planner contexts through three agent harnesses with the
same model (Qwen3.8-27B Q8_0 on :8083) to separate HARNESS from MODEL.

Arms (see README.md for the exact settings of each):
  oc   27B x opencode, the as-ran lane-0 planner seat (plain+trim-instr+trim-tools+lane-guard+
       ak-check+ctx88k+out32000+concise+reason-drop+budget4500s+ctx-variable), variable prompt,
       900 s salvage turn -- built with the harness's own config code (snapshot @ 56dfdccf).
  cx   27B x codex exec (the harness gpt-6.1-sol runs under in the live loop): inline prompt,
       no lane block / concise rule (actors.py codex path), isolated CODEX_HOME, 90112 window.
  hm   27B x NousResearch Hermes agent (MIT): variable prompt, --ignore-rules, 90112 window.
  sol  OPTIONAL reference: gpt-6.1-sol x codex @low with the operator's CODEX_HOME (no GPU;
       egresses the DS41 prompt to OpenAI -- approved for this reference arm only).
All 27B arms go through a per-call wire proxy (proxy.py) that logs every request and strips
reasoning-effort fields so every arm runs at the template default the as-ran 27B got.

Subcommands:
  plan    print the call list and the GPU-hour estimate
  run     execute (resumable; results/<ctx>/<arm>/r<k>/result.json; STOP file = no new calls)
  smoke   one trivial tool-call prompt through one arm (wiring proof; no DS41 context)
  status  summarize results/

Constraints honoured: at most --concurrency (default 2, max 2) concurrent 27B sessions; every
process this driver starts is in its own process group and is ended by PID; nothing outside
/mnt/raid0/llm/tmp/ds41-c95 is written (the 27B server is only called, never touched).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

BASE = Path(os.environ.get("DS41_C95_BASE", "/mnt/raid0/llm/tmp/ds41-c95"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sandbox3  # noqa: E402  (run 3: Landlock policy)
import proxy3  # noqa: E402  (run 3: request transforms, replayed offline by dry-run)
import copy  # noqa: E402

RESULTS = BASE / "results"
WORK = BASE / "work"
CONTEXTS = BASE / "contexts"


def _pinned_aklib_source(name: str):
    """Load only the run's historical snapshot, never an installed/cached aklib."""
    if name not in ("actor_opencode_config", "actor_context"):
        raise ValueError(name)
    source = BASE / "harness_snapshot" / "aklib" / f"{name}.py"
    if not source.is_file():
        raise RuntimeError(f"SOURCE_MISSING: pinned C95 aklib snapshot required: {source}")
    module_name = f"_ds41_c95_pinned_{name}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"SOURCE_MISSING: cannot load pinned C95 aklib snapshot: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


UPSTREAM = "http://127.0.0.1:8083"
OPENCODE = "/usr/local/share/npm-global/bin/opencode"
CODEX = os.environ.get(
    "C95_CODEX_BIN", "/home/node/.codex/packages/app-server-daemon/current/bin/codex"
)
CODEX_HOME_27B = BASE / "codex-home"
HERMES_RUN = BASE / "hermes" / "hermes_run.sh"
MODEL_OC = "qwen-gpu/qwen3.8-27b"
MODEL_CX = "Qwen3.8-27B-Q8_0"

PLANNER_BUDGET_S = 4500  # as-ran planner_budget_s
SALVAGE_BUDGET_S = 900  # as-ran planner_salvage_s
HARD_TIMEOUT_S = 7200  # as-ran hard per-call timeout
WINDOW = 90112  # as-ran limit.context; the same nominal window for every arm
STOP_GRACE_S = 15

# as-ran PLANNER_SALVAGE_MESSAGE (actors.py @ 56dfdccf)
SALVAGE_MESSAGE = (
    "Your time for this proposal is up. Stop investigating now: do not call any tool "
    "(no reads, searches, greps or commands). Using only what you have already found in "
    "this session, reply immediately with the ONE proposal JSON object the task asked "
    "for -- the same fields: mechanism_id (starting with akm-), statement, falsifier, "
    "target_surface and target_symbol -- and nothing else: no summary of your work, no "
    "prose before or after the object. If what you found supports no honest, feasible "
    "hypothesis, reply instead with a JSON object whose only key is abstain, giving the "
    "specific reason."
)
SMOKE_PROMPT = "Run the shell command `echo C95_TOOL_OK` and then reply with exactly its output and nothing else.\n"

KEEP_COMMITS = ["053c3bd82", "7acd34a8c", "1ef72c228", "6f8e232aa", "ad8979943", "9b148baab"]
#: Access to the FUTURE: the live campaign root (every prompt path was relocated away from it,
#: so the prompts contain it 0 times) and git/cd into the live experimental repo (whose refs
#: hold every post-anchor keep). The prompts DO name the repo's 00d118d44 baseline binary
#: (unchanged since 09-25), so a bare path mention is not a needle.
LIVE_NEEDLES = ["/mnt/raid0/llm/autokernel/campaigns", "state-run10i"]
LIVE_PATTERNS = [
    r"(?:git\s+(?:-C\s+)?|cd\s+|--git-dir[= ])/mnt/raid0/llm/llama\.cpp-experimental-fastload"
]
HYP_FIELDS = ("mechanism_id", "statement", "falsifier", "target_surface", "target_symbol")

ARMS = {
    "oc": {
        "gpu": True,
        "prompt": "prompt.variable.txt",
        "desc": "27B x opencode 1.18.31, as-ran lane-0 planner seat",
    },
    "cx": {
        "gpu": True,
        "prompt": "prompt.inline.txt",
        "desc": "27B x codex exec (gpt-6.1-sol's harness)",
    },
    "hm": {"gpu": True, "prompt": "prompt.variable.txt", "desc": "27B x NousResearch Hermes agent"},
    "sol": {
        "gpu": False,
        "prompt": "prompt.inline.txt",
        "desc": "gpt-6.1-sol x codex @low (reference, no GPU)",
    },
    "cxv": {
        "gpu": True,
        "prompt": "prompt.variable.txt",
        "desc": "27B x codex exec on the VARIABLE prompt + bundle (same text as oc/hm)",
    },
    "orv": {
        "gpu": True,
        "proxy": False,
        "prompt": "prompt.inline.txt",
        "desc": "27B architect + frontdoor workers via orchestrator (REPL /chat, orch:architect_critic, "
        "orchestrator-variable bundle) -- NOT a pure 27B arm",
    },
    # DS41-C95 F12 follow-up (operator-approved 2026-10-01; diagnosis-27b-report.md §7)
    "cxf1": {
        "gpu": True,
        "prompt": "prompt.inline.txt",
        "f2": False,
        "solo": True,
        "desc": "27B x codex inline, F1 serving fixes: solo, stream idle timeout 1 h, "
        "window = server slot ctx (<=196608), auto-compact 150k",
    },
    "cxf12": {
        "gpu": True,
        "prompt": "prompt.inline.txt",
        "f2": True,
        "solo": True,
        "desc": "27B x codex inline, F1 + F2 answer protocol: JSON-first prompt rule, forced answer "
        "turn at 65% of budget (no resume-compaction), refine turn, 8k per-turn thinking cap",
    },
    # run 3 (UFH14-A3/A4 A/Bs): F1 transport without the solo requirement
    "cxa": {
        "gpu": True,
        "prompt": "prompt.inline.txt",
        "f2": False,
        "solo": False,
        "desc": "27B x codex inline, F1 transport (1 h idle timeout, window = slot ctx, compaction "
        "150k or --compact-limit), not solo -- run-3 A3/A4 A/B arm",
    },
    "orsv": {
        "gpu": True,
        "proxy": False,
        "prompt": "prompt.inline.txt",
        "desc": "27B architect + frontdoor workers via orchestrator, + UFH-12 context.search "
        "(ChatRequest.context_search=true) -- NOT a pure 27B arm",
    },
}
DEFAULT_CONTEXTS = ["C1", "C2", "C4", "C5"]
#: run 3: the one workspace-root string every call shows the model (proxy3 --alias REAL=CANON_ROOT).
#: Same length as a real work/<8 hex> root; never created on disk.
CANON_ROOT = str(WORK / "ws000000")
GROUPS = RESULTS / "_groups"
CAPTURE_TIMEOUT_S = 600
_tl = threading.local()  # per-call run-3 settings (each call runs in its own thread)
EST_CALL_MIN = 45  # as-ran 27B planner median 41 min (C83 §5), rounded up
EST_CONCURRENCY_SLOWDOWN = 1.35

_print_lock = threading.Lock()
_git_lock = threading.Lock()  # concurrent `git worktree add/remove` on one scratch clone


def log(msg: str) -> None:
    with _print_lock:
        print(time.strftime("%H:%M:%S ") + msg, flush=True)


# ---------------------------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------------------------
def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def extract_answer(text: str) -> dict | None:
    """The LAST JSON object in `text` that is a hypothesis or an abstention (string-aware:
    json.raw_decode at every '{', so braces inside code or strings cannot derail it -- the
    C83 §5.1 `_extract_json` defect)."""
    dec = json.JSONDecoder()
    found = None
    i = text.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(text, i)
        except ValueError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict) and ("mechanism_id" in obj or "abstain" in obj):
            found = obj
        i = text.find("{", i + 1)
    return found


def classify(obj: dict | None) -> str:
    if obj is None:
        return "no_answer"
    if "abstain" in obj and not obj.get("mechanism_id"):
        return "abstain"
    if all(isinstance(obj.get(k), str) and obj.get(k).strip() for k in HYP_FIELDS):
        return "hypothesis"
    return "incomplete"


def start_proxy(
    log_dir: Path, policy: str = "strip", extra: list[str] | None = None
) -> tuple[subprocess.Popen, int]:
    log_dir.mkdir(parents=True, exist_ok=True)
    out = open(log_dir / "proxy.stdout", "w")
    p = subprocess.Popen(
        [
            sys.executable,
            str(BASE / "proxy3.py"),
            "--port",
            "0",
            "--log",
            str(log_dir),
            "--upstream",
            UPSTREAM,
            "--effort-policy",
            policy,
            *(extra or []),
        ],
        stdout=out,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    for _ in range(100):
        time.sleep(0.1)
        txt = (log_dir / "proxy.stdout").read_text()
        m = re.search(r"PORT (\d+)", txt)
        if m:
            return p, int(m.group(1))
        if p.poll() is not None:
            break
    end_group(p)
    raise RuntimeError(f"proxy did not start: {(log_dir / 'proxy.stdout').read_text()[-500:]}")


def end_group(p: subprocess.Popen, grace: float = STOP_GRACE_S) -> int | None:
    """TERM then KILL the process group of a child THIS driver started; verify it is gone."""
    if p.poll() is not None:
        return p.returncode
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        return p.poll()
    try:
        p.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        p.wait(timeout=30)
    return p.returncode


def run_budgeted(
    argv: list[str],
    *,
    cwd: Path,
    env: dict,
    stdin_text: str | None,
    out: Path,
    err: Path,
    budget_s: float,
) -> dict:
    """Run one harness process (own process group) under a wall budget."""
    t0 = time.time()
    wrap = getattr(_tl, "wrap", None)
    if wrap:
        argv = [*wrap, *argv]
    with open(out, "w") as fo, open(err, "w") as fe:
        p = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=env,
            stdout=fo,
            stderr=fe,
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
        if stdin_text is not None:

            def feed():
                try:
                    p.stdin.write(stdin_text)
                    p.stdin.close()
                except (BrokenPipeError, OSError):
                    pass

            threading.Thread(target=feed, daemon=True).start()
        exhausted = False
        while p.poll() is None:
            if time.time() - t0 > budget_s:
                exhausted = True
                end_group(p)
                break
            if (BASE / "STOP_NOW").exists():
                end_group(p)
                break
            time.sleep(1.0)
    return {"rc": p.returncode, "wall_s": round(time.time() - t0, 1), "budget_exhausted": exhausted}


def wire_metrics(log_dir: Path) -> dict:
    rows = []
    wf = log_dir / "wire.jsonl"
    if wf.exists():
        for line in wf.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    model_rows = [
        r
        for r in rows
        if str(r.get("path", "")).rstrip("/").endswith(("chat/completions", "responses"))
    ]

    def ptok(r):
        u = r.get("usage") or {}
        return u.get("prompt_tokens", u.get("input_tokens"))

    def ctok(r):
        u = r.get("usage") or {}
        return u.get("completion_tokens", u.get("output_tokens"))

    tool_rows = [r for r in model_rows if (r.get("n_tools") or 0) > 0 and ptok(r)]
    drops = 0
    for a, b in zip(tool_rows, tool_rows[1:]):
        if ptok(b) < 0.6 * ptok(a) and ptok(a) > 8000:
            drops += 1
    t = [r.get("timings") or {} for r in model_rows]
    return {
        "requests": len(model_rows),
        "requests_with_tools": len(tool_rows),
        "errors": sum(1 for r in model_rows if r.get("error") or (r.get("status") or 200) >= 400),
        "prompt_tokens_sum": sum(ptok(r) or 0 for r in model_rows),
        "prefill_tokens_sum": sum(x.get("prompt_n") or 0 for x in t),
        "generated_tokens_sum": sum(ctok(r) or 0 for r in model_rows),
        "context_max_tokens": max((ptok(r) or 0 for r in model_rows), default=0),
        "context_first_tokens": ptok(tool_rows[0]) if tool_rows else None,
        "compactions_inferred": drops,
        "model_wall_s": round(sum(r.get("wall_s") or 0 for r in model_rows), 1),
        "reasoning_fields_sent": sorted(
            {
                json.dumps(r.get("sent_reasoning"), sort_keys=True)
                for r in model_rows
                if r.get("sent_reasoning")
            }
        ),
        "max_tokens_sent": sorted({str(r.get("max_tokens")) for r in model_rows}),
        "tool_names": sorted({n for r in model_rows for n in (r.get("tool_names") or []) if n}),
    }


def read_gz_tolerant(p: Path) -> str:
    """Decompress as much of a gzip file as exists (run-1's proxy.py was killed before writing
    the end-of-stream marker; gzip.open().read() raises EOFError on such a file)."""
    import zlib

    raw = p.read_bytes()
    out, d = [], zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        out.append(d.decompress(raw))
    except zlib.error:
        pass
    return b"".join(out).decode("utf-8", errors="replace")


def scan_contamination(paths: list[Path], anchor: str) -> dict:
    """Needles that would show the planner saw the FUTURE (post-anchor keep commits, the live
    campaign store / experimental repo). The prompts themselves contain none of these."""
    future = (
        KEEP_COMMITS[KEEP_COMMITS.index(anchor[:9]) + 1 :]
        if anchor[:9] in KEEP_COMMITS
        else KEEP_COMMITS
    )
    needles = LIVE_NEEDLES + future
    hits = {}
    for p in paths:
        if not p.exists() or p.is_dir():
            continue
        try:
            data = read_gz_tolerant(p) if p.suffix == ".gz" else p.read_text(errors="replace")
        except OSError:
            continue
        for n in needles:
            c = data.count(n)
            if c:
                hits[n] = hits.get(n, 0) + c
        for pat in LIVE_PATTERNS:
            c = len(re.findall(pat, data))
            if c:
                hits[pat] = hits.get(pat, 0) + c
    return hits


def crosscheck_contamination(paths: list[Path], anchor: str) -> dict:
    """Classify each needle occurrence against the command's denial result.

    A command string alone proves only an attempted access. A successful tool output
    containing the needle without a denial merits review; a denied tool result
    is an attempted access. Retain ambiguous raw transcript lines for review.
    """
    future = (
        KEEP_COMMITS[KEEP_COMMITS.index(anchor[:9]) + 1 :]
        if anchor[:9] in KEEP_COMMITS
        else KEEP_COMMITS
    )
    needles = LIVE_NEEDLES + future
    detail = {
        n: {"denied_attempts": 0, "output_without_denial": 0, "unresolved": 0, "samples": []}
        for n in needles
    }
    for p in paths:
        if not p.exists() or p.is_dir():
            continue
        try:
            data = read_gz_tolerant(p) if p.suffix == ".gz" else p.read_text(errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(data.splitlines(), 1):
            try:
                obj = json.loads(line)
            except ValueError:
                obj = None
            item = obj.get("item", {}) if isinstance(obj, dict) else {}
            if not isinstance(item, dict):
                item = {}
            command = str(item.get("command") or "")
            output = str(item.get("aggregated_output") or "")
            denial = bool(re.search(r"Permission denied|Operation not permitted", output or line))
            for needle in needles:
                count = line.count(needle)
                if not count:
                    continue
                row = detail[needle]
                if output and needle in output and not denial and item.get("exit_code") == 0:
                    category = "output_without_denial"
                elif denial and needle in command:
                    category = "denied_attempts"
                else:
                    category = "unresolved"
                row[category] += count
                if len(row["samples"]) < 5:
                    row["samples"].append({"file": str(p), "line": lineno, "category": category})
    return {
        k: v
        for k, v in detail.items()
        if sum(v[x] for x in ("denied_attempts", "output_without_denial", "unresolved"))
    }


# ---------------------------------------------------------------------------------------------
# per-call workspace: an isolated worktree of the scratch clone + a relocated bundle
# ---------------------------------------------------------------------------------------------
class Workspace:
    def __init__(self, meta: dict):
        self.meta = meta
        self.root = WORK / uuid.uuid4().hex[:8]
        self.lane = self.root / "lane"
        self.bundle = self.root / "actor-context" / "planner-bundle"
        self.tmp = self.root / "tmp"
        self.scratch_clone = Path(meta["lane"])
        self.codex_home = self.root / "codex-home"

    def __enter__(self):
        self.root.mkdir(parents=True)
        self.tmp.mkdir()
        if getattr(_tl, "codex_home_per_call", False):
            # run 3: a fresh CODEX_HOME per call (same config, skills and installation id as the shared
            # one, but no other call's sessions/memories/state databases to read)
            self.codex_home.mkdir()
            for f in ("config.toml", "installation_id", ".sandbox_migration"):
                if (CODEX_HOME_27B / f).exists():
                    shutil.copy2(CODEX_HOME_27B / f, self.codex_home / f)
            if (CODEX_HOME_27B / "skills").is_dir():
                shutil.copytree(
                    CODEX_HOME_27B / "skills", self.codex_home / "skills", symlinks=True
                )
        with _git_lock:
            subprocess.run(
                [
                    "nice",
                    "-n",
                    "19",
                    "git",
                    "-C",
                    str(self.scratch_clone),
                    "worktree",
                    "add",
                    "-q",
                    "--detach",
                    str(self.lane),
                    self.meta["anchor_commit"],
                ],
                check=True,
            )
        src_bundle = CONTEXTS / self.meta["id"] / "bundle"
        shutil.copytree(src_bundle, self.bundle)
        for f in self.bundle.rglob("*"):
            if f.is_file():
                f.write_text(self.relocate(f.read_text()))
        return self

    def relocate(self, text: str) -> str:
        return text.replace(str(CONTEXTS / self.meta["id"] / "bundle"), str(self.bundle)).replace(
            str(self.scratch_clone), str(self.lane)
        )

    def prompt(self, name: str) -> str:
        return self.relocate((CONTEXTS / self.meta["id"] / name).read_text())

    def lane_changes(self) -> list[str]:
        r = subprocess.run(
            ["git", "-C", str(self.lane), "status", "--porcelain"], text=True, capture_output=True
        )
        return [ln for ln in r.stdout.splitlines() if ln.strip()][:50]

    def __exit__(self, *exc):
        with _git_lock:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.scratch_clone),
                    "worktree",
                    "remove",
                    "--force",
                    str(self.lane),
                ],
                capture_output=True,
            )
        shutil.rmtree(self.root, ignore_errors=True)
        return False


# ---------------------------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------------------------
def base_env(ws_tmp: Path) -> dict:
    env = {k: v for k, v in os.environ.items()}
    env.update({"TMPDIR": str(ws_tmp), "TMP": str(ws_tmp), "TEMP": str(ws_tmp)})
    return env


def arm_oc(ws: Workspace, out: Path, prompt: str, port: int, budget: float, salvage: float) -> dict:
    seat_config = _pinned_aklib_source("actor_opencode_config")
    cfg = seat_config.build_plain_config(
        role="planner",
        lane=ws.lane,
        build_dir=ws.meta["anchor_build_path"],
        model=MODEL_OC,
        trim_instructions=True,
        trim_tools=True,
        lane_guard=True,
        author_sandbox=True,
        context_limit=WINDOW,
        output_limit=32000,
        thinking="default",
        reasoning_history="drop",
    )
    cfg["provider"]["qwen-gpu"]["options"] = {
        "baseURL": f"http://127.0.0.1:{port}/v1"
    }  # transport only
    seat = out / "seat.opencode.json"
    seat.write_text(json.dumps(cfg, indent=2) + "\n")
    xdg = ws.root / "xdg-data"
    xdg.mkdir(exist_ok=True)
    env = base_env(ws.tmp)
    env.update(seat_config.TRIM_ENV)
    env.update({"OPENCODE_CONFIG": str(seat), "XDG_DATA_HOME": str(xdg)})
    argv = [OPENCODE, "run", "--auto", "--dir", str(ws.lane), "-m", MODEL_OC, "--variant", "high"]
    (out / "command.txt").write_text(" ".join(argv) + "  < prompt (stdin)\n")
    res = run_budgeted(
        argv,
        cwd=ws.lane,
        env=env,
        stdin_text=prompt,
        out=out / "stdout.txt",
        err=out / "stderr.txt",
        budget_s=budget,
    )

    def sessions() -> list[dict]:
        r = subprocess.run(
            [OPENCODE, "session", "list", "--format", "json", "-n", "50"],
            cwd=str(ws.lane),
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
        )
        try:
            return json.loads(r.stdout)
        except ValueError:
            return []

    sess = sessions()
    sid = None
    if sess:
        sess.sort(
            key=lambda s: (s.get("time") or {}).get("created", s.get("created", 0)), reverse=True
        )
        sid = sess[0].get("id")
    res["session_id"] = sid
    final_src = out / "stdout.txt"
    if res["budget_exhausted"] and salvage and sid:
        sres = run_budgeted(
            argv + ["--session", sid],
            cwd=ws.lane,
            env=env,
            stdin_text=SALVAGE_MESSAGE,
            out=out / "salvage.stdout.txt",
            err=out / "salvage.stderr.txt",
            budget_s=salvage,
        )
        res["salvage"] = sres
        final_src = out / "salvage.stdout.txt"
    if sid:
        with open(out / "export.json", "w") as fh:
            subprocess.run(
                [OPENCODE, "export", sid],
                cwd=str(ws.lane),
                env=env,
                stdout=fh,
                stderr=subprocess.DEVNULL,
                timeout=300,
            )
    native = {}
    final_text = final_src.read_text(errors="replace") if final_src.exists() else ""
    try:
        d = json.loads((out / "export.json").read_text())
        msgs = d.get("messages", [])
        native = {
            "steps": sum(
                1
                for m in msgs
                if m["info"]["role"] == "assistant" and m["info"].get("mode") != "compaction"
            ),
            "tool_calls": sum(1 for m in msgs for p in m["parts"] if p["type"] == "tool"),
            "compactions": sum(1 for m in msgs for p in m["parts"] if p["type"] == "compaction"),
            "tokens": d.get("info", {}).get("tokens"),
        }
        last = [
            m
            for m in msgs
            if m["info"]["role"] == "assistant" and m["info"].get("mode") != "compaction"
        ]
        if last:
            txt = "".join(p.get("text", "") for p in last[-1]["parts"] if p["type"] == "text")
            if txt.strip():
                final_text = txt
    except (OSError, ValueError, KeyError):
        pass
    res["native"] = native
    res["final_text"] = final_text
    res["transcripts"] = [
        str(out / "export.json"),
        str(out / "stdout.txt"),
        str(out / "stderr.txt"),
    ]
    return res


def arm_codex(
    ws: Workspace,
    out: Path,
    prompt: str,
    port: int | None,
    budget: float,
    salvage: float,
    *,
    sandbox: str,
    model: str,
    reference: bool,
) -> dict:
    env = base_env(ws.tmp)
    common = ["--skip-git-repo-check", "-s", sandbox, "-c", 'web_search="disabled"']
    if reference:
        common += ["-m", model, "-c", 'model_reasoning_effort="low"']
    else:
        env["CODEX_HOME"] = str(
            ws.codex_home if getattr(_tl, "codex_home_per_call", False) else CODEX_HOME_27B
        )
        env.pop("OPENAI_API_KEY", None)
        common += [
            "-m",
            model,
            "-c",
            'model_reasoning_effort="low"',
            "-c",
            f'model_providers.c95local.base_url="http://127.0.0.1:{port}/v1"',
        ]
    argv = [CODEX, "exec", *common, "-C", str(ws.lane), "--json", "-o", str(out / "last.txt"), "-"]
    (out / "command.txt").write_text(" ".join(argv) + "  < prompt (stdin)\n")
    res = run_budgeted(
        argv,
        cwd=ws.lane,
        env=env,
        stdin_text=prompt,
        out=out / "events.jsonl",
        err=out / "stderr.txt",
        budget_s=budget,
    )
    tid = None
    try:
        first = (out / "events.jsonl").read_text().splitlines()[0]
        tid = json.loads(first).get("thread_id")
    except (OSError, IndexError, ValueError):
        pass
    res["thread_id"] = tid
    if res["budget_exhausted"] and salvage and tid:
        sargv = [
            CODEX,
            "exec",
            *common,
            "-C",
            str(ws.lane),
            "--json",
            "-o",
            str(out / "last.txt"),
            "resume",
            tid,
            "-",
        ]
        res["salvage"] = run_budgeted(
            sargv,
            cwd=ws.lane,
            env=env,
            stdin_text=SALVAGE_MESSAGE,
            out=out / "salvage.events.jsonl",
            err=out / "salvage.stderr.txt",
            budget_s=salvage,
        )
    items: dict[str, int] = {}
    usage = []
    for f in ("events.jsonl", "salvage.events.jsonl"):
        p = out / f
        if not p.exists():
            continue
        for line in p.read_text(errors="replace").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("type") == "item.completed":
                t = (e.get("item") or {}).get("type")
                items[t] = items.get(t, 0) + 1
            if e.get("type") == "turn.completed":
                usage.append(e.get("usage"))
            if "compact" in json.dumps(e).lower() and e.get("type") != "item.completed":
                items["_compaction_events"] = items.get("_compaction_events", 0) + 1
    res["native"] = {
        "item_types": items,
        "tool_calls": items.get("command_execution", 0) + items.get("mcp_tool_call", 0),
        "turn_usage": usage,
    }
    res["final_text"] = (
        (out / "last.txt").read_text(errors="replace") if (out / "last.txt").exists() else ""
    )
    res["transcripts"] = [
        str(out / "events.jsonl"),
        str(out / "salvage.events.jsonl"),
        str(out / "stderr.txt"),
    ]
    return res


def arm_hm(ws: Workspace, out: Path, prompt: str, port: int, budget: float, salvage: float) -> dict:
    pfile = out / "prompt.txt"
    pfile.write_text(prompt)
    env = base_env(ws.tmp)
    env.update(
        {
            "HERMES_BASE_URL": f"http://127.0.0.1:{port}/v1",
            "HERMES_CTX_LEN": str(WINDOW),
            "HERMES_COMPRESSION_THRESHOLD": "0.90",
            "HERMES_MAX_TURNS": "400",
            "HERMES_RUN_BUDGET": str(int(budget)),
            "HERMES_OUTER_TIMEOUT": str(int(budget + salvage)),
            "HERMES_IGNORE_RULES": "1",
            "HERMES_TOOLSETS": "terminal,file",
        }
    )
    hout = out / "hermes"
    argv = [str(HERMES_RUN), str(pfile), str(ws.lane), str(hout)]
    (out / "command.txt").write_text(" ".join(argv) + "\n")
    res = run_budgeted(
        argv,
        cwd=ws.lane,
        env=env,
        stdin_text=None,
        out=out / "wrapper.stdout.txt",
        err=out / "wrapper.stderr.txt",
        budget_s=budget + salvage + 120,
    )
    try:
        s = json.loads((hout / "summary.json").read_text())
    except (OSError, ValueError):
        s = {}
    res["native"] = {
        k: s.get(k)
        for k in (
            "rc",
            "result_exit_code",
            "error",
            "tokens_main_loop",
            "tool_calls_stream",
            "tool_calls_by_name",
            "api_calls_transcript",
            "sessions_in_lineage",
            "compaction",
        )
    }
    res["final_text"] = (
        (hout / "final_text.txt").read_text(errors="replace")
        if (hout / "final_text.txt").exists()
        else ""
    )
    res["transcripts"] = [
        str(hout / "session.jsonl"),
        str(hout / "events.jsonl"),
        str(hout / "stderr.txt"),
    ]
    return res


# ---------------------------------------------------------------------------------------------
# F12 follow-up arms (cxf1, cxf12): codex inline with the serving fixes (F1) and the answer
# protocol (F2). See diagnosis-27b-report.md §6-7.
# ---------------------------------------------------------------------------------------------
F1_IDLE_TIMEOUT_MS = 3_600_000  # codex default ~300 s killed every >300 s silent prefill
F1_COMPACT = 150_000
F1_WINDOW_MAX = 196_608
NO_COMPACT = 1_000_000_000  # forced/salvage resumes: never compact before the turn
F2_FORCE_FRAC = 0.65
F2_PHASE_CAP_S = 900
F2_THINK_BUDGET = 8000
F2_THINK_MESSAGE = (
    "\n\n[Thinking budget for this turn is used up. Stop deliberating now and act on your "
    "best current conclusion: call one tool, or write the proposal JSON.]\n"
)
F2_JSON_FIRST = (
    "\n\n## Answer protocol for this run\n"
    "As soon as you have ONE concrete candidate, write the complete proposal JSON object (all five fields: "
    "mechanism_id, statement, falsifier, target_surface, target_symbol) in a visible message BEFORE you verify "
    "it further; you may call tools in the same response and keep investigating. If later evidence changes "
    "your choice, write an updated complete JSON object. Your final message must end with the final JSON "
    "object.\n"
)
F2_FORCED_MESSAGE = (
    "Checkpoint: 65% of your time budget is used. Stop investigating now and do not call any tool. Using "
    "only what you have already found in this session, reply immediately with the ONE proposal JSON object "
    "the task asked for -- mechanism_id (starting with akm-), statement, falsifier, target_surface and "
    "target_symbol -- and nothing else. If what you found supports no honest, feasible hypothesis, reply "
    "instead with a JSON object whose only key is abstain, giving the specific reason. You will get the "
    "remaining time afterwards to verify and refine it."
)
F2_REFINE_MESSAGE = (
    "Your provisional proposal above is recorded. You have about {minutes} minutes left. You may now use "
    "tools to verify its premises and refine it, or switch to a better candidate if the evidence demands "
    "it. End with the final complete JSON object (the same five fields), or repeat the provisional one "
    "unchanged if it stands."
)
SMOKE_RESUME_MESSAGE = "Reply with exactly C95_RESUME_OK and nothing else. Do not call any tool."
DFLASH2_LIVE = BASE / "DFLASH2_LIVE"


def server_drafter() -> dict:
    """Which drafter :8083 runs right now (read-only: GET /slots + ps args of the --port 8083 process)."""
    out: dict = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        import urllib.request

        with urllib.request.urlopen(UPSTREAM + "/slots", timeout=10) as r:
            sl = json.loads(r.read())
        out["speculative_types"] = sorted(
            {str((s.get("params") or {}).get("speculative.types")) for s in sl}
        )
        out["speculative_n_max"] = sorted(
            {(s.get("params") or {}).get("speculative.n_max") for s in sl}, key=str
        )
        out["n_ctx_slot"] = min(int(s.get("n_ctx") or 0) for s in sl)
        out["n_slots"] = len(sl)
        out["slots_processing"] = [s.get("id") for s in sl if s.get("is_processing")]
    except Exception as exc:  # noqa: BLE001
        out["slots_error"] = str(exc)[:200]
    try:
        ps = subprocess.run(["ps", "-eo", "pid,lstart,args"], capture_output=True, text=True).stdout
        rows = [ln for ln in ps.splitlines() if "llama-server" in ln and "--port 8083" in ln]
        out["server_ps"] = [ln.strip()[:1200] for ln in rows][:2]
    except Exception as exc:  # noqa: BLE001
        out["ps_error"] = str(exc)[:200]
    types = " ".join(out.get("speculative_types") or [])
    argv = " ".join(out.get("server_ps") or [])
    if "draft-dflash" in types or "dflash" in argv.lower():
        out["drafter"] = "dflash2" if "dflash2" in argv.lower() else "dflash"
    elif "draft-mtp" in types or "draft-mtp" in argv:
        out["drafter"] = "mtp"
    else:
        out["drafter"] = "unknown"
    if DFLASH2_LIVE.exists():
        out["dflash2_live_file"] = DFLASH2_LIVE.read_text(errors="replace")[:2000]
    return out


def wire_timing(log_dir: Path, out_json: Path | None = None) -> dict:
    """The diagnosis breakdown (report §1.1) from wire.jsonl: prefill / decode / side / aborted /
    gaps, decode rate, cache hit, context growth, draft acceptance; plus a per-request table."""
    rows = []
    wf = log_dir / "wire.jsonl"
    for line in wf.read_text().splitlines() if wf.exists() else []:
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    posts = sorted([r for r in rows if r.get("method") == "POST"], key=lambda r: r["t0"])
    if not posts:
        return {"requests": 0}
    t_start = min(r["t0"] for r in rows)
    t_end = max(r.get("t1", r["t0"]) for r in rows)
    agg = dict(main_prefill_s=0.0, main_decode_s=0.0, side_s=0.0, aborted_s=0.0, queue_s=0.0)
    gen = cache = pf = dn = da = 0
    tps, ctxs, table = [], [], []
    for r in posts:
        t = r.get("timings") or {}
        side = (r.get("n_tools") or 0) == 0
        ok = bool(t) and not r.get("client_gone")
        rec = {
            "seq": r["seq"],
            "t_rel": round(r["t0"] - t_start, 1),
            "wall_s": r.get("wall_s"),
            "side": side,
            "status": r.get("status"),
            "client_gone": r.get("client_gone"),
            "finish": r.get("finish_reason"),
            "injected": bool(r.get("injected")),
        }
        if t:
            rec.update(
                {
                    k: t.get(k)
                    for k in ("cache_n", "prompt_n", "predicted_n", "draft_n", "draft_n_accepted")
                }
            )
            rec["prefill_tps"] = round(t.get("prompt_per_second") or 0, 1)
            rec["decode_tps"] = round(t.get("predicted_per_second") or 0, 2)
            rec["ctx"] = (t.get("cache_n") or 0) + (t.get("prompt_n") or 0)
        table.append(rec)
        if ok:
            gen += t.get("predicted_n") or 0
            cache += t.get("cache_n") or 0
            pf += t.get("prompt_n") or 0
            dn += t.get("draft_n") or 0
            da += t.get("draft_n_accepted") or 0
            pms, dms = (t.get("prompt_ms") or 0) / 1000, (t.get("predicted_ms") or 0) / 1000
            if side:
                agg["side_s"] += r.get("wall_s") or 0
            else:
                agg["main_prefill_s"] += pms
                agg["main_decode_s"] += dms
                agg["queue_s"] += max(0.0, (r.get("wall_s") or 0) - pms - dms)
                ctxs.append(rec["ctx"])
                if (t.get("predicted_n") or 0) >= 1000:
                    tps.append(t.get("predicted_per_second") or 0)
        else:
            agg["aborted_s"] += r.get("wall_s") or 0
    busy, cur = 0.0, None
    for a, b in sorted((r["t0"], r.get("t1", r["t0"])) for r in posts):
        if cur is None:
            cur = [a, b]
        elif a <= cur[1]:
            cur[1] = max(cur[1], b)
        else:
            busy += cur[1] - cur[0]
            cur = [a, b]
    busy += cur[1] - cur[0]
    tps.sort()
    big = [x for x in table if (x.get("predicted_n") or 0) >= 5000 and not x["side"]]
    summ = {
        **{k: round(v, 1) for k, v in agg.items()},
        "span_s": round(t_end - t_start, 1),
        "gaps_s": round((t_end - t_start) - busy, 1),
        "requests": len(posts),
        "generated_tokens": gen,
        "median_decode_tps_ge1k": round(tps[len(tps) // 2], 2) if tps else None,
        "cache_hit": round(cache / (cache + pf), 3) if cache + pf else None,
        "ctx_first": ctxs[0] if ctxs else None,
        "ctx_max": max(ctxs) if ctxs else None,
        "draft_acceptance": round(da / dn, 3) if dn else None,
        "turns_ge5k": [(x["seq"], x["predicted_n"], x["wall_s"]) for x in big],
        "aborted_requests": sum(
            1 for x in table if not x.get("predicted_n") and x.get("status") is not None
        ),
    }
    if out_json is not None:
        out_json.write_text(json.dumps({"summary": summ, "requests": table}, indent=1) + "\n")
    return summ


def _codex_events_native(paths: list[Path]) -> dict:
    items: dict[str, int] = {}
    usage = []
    for p in paths:
        if not p.exists():
            continue
        for line in p.read_text(errors="replace").splitlines():
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e.get("type") == "item.completed":
                t = (e.get("item") or {}).get("type")
                items[t] = items.get(t, 0) + 1
            if e.get("type") == "turn.completed":
                usage.append(e.get("usage"))
            if e.get("type") == "error" or (
                e.get("type") == "item.completed" and (e.get("item") or {}).get("type") == "error"
            ):
                msg = e.get("message") or (e.get("item") or {}).get("message") or ""
                if "Reconnecting" in msg:
                    items["_reconnects"] = items.get("_reconnects", 0) + 1
    return {
        "item_types": items,
        "tool_calls": items.get("command_execution", 0) + items.get("mcp_tool_call", 0),
        "turn_usage": usage,
    }


def arm_codex_f(
    ws: Workspace,
    out: Path,
    prompt: str,
    port: int,
    budget: float,
    salvage: float,
    *,
    sandbox: str,
    f2: bool,
    smoke: bool = False,
) -> dict:
    """cxf1 / cxf12. F1: the codex cx arm with the serving fixes. F2 adds the answer protocol:
    main turn until 65% of the budget (or its natural end), a forced-answer turn (no tools, no
    resume-compaction), a refine turn for the rest of the budget, and the as-ran salvage turn
    (also without resume-compaction) if the refine turn exhausts the budget. The final answer is
    the last valid JSON across phases (salvage > refine > forced > main)."""
    t_call = time.time()
    drafter = server_drafter()
    n_ctx = int(drafter.get("n_ctx_slot") or F1_WINDOW_MAX)
    window = min(F1_WINDOW_MAX, n_ctx)
    compact = int(getattr(_tl, "compact_limit", 0) or 0) or min(F1_COMPACT, int(window * 0.76))
    env = base_env(ws.tmp)
    env["CODEX_HOME"] = str(
        ws.codex_home if getattr(_tl, "codex_home_per_call", False) else CODEX_HOME_27B
    )
    env.pop("OPENAI_API_KEY", None)
    base = [
        "--skip-git-repo-check",
        "-s",
        sandbox,
        "-c",
        'web_search="disabled"',
        "-m",
        MODEL_CX,
        "-c",
        'model_reasoning_effort="low"',
        "-c",
        f'model_providers.c95local.base_url="http://127.0.0.1:{port}/v1"',
        "-c",
        f"model_providers.c95local.stream_idle_timeout_ms={F1_IDLE_TIMEOUT_MS}",
        "-c",
        f"model_context_window={window}",
    ]

    def common(compact_limit: int) -> list[str]:
        return base + ["-c", f"model_auto_compact_token_limit={compact_limit}"]

    phases: list[dict] = []
    cmds: list[str] = []
    transcripts: list[Path] = []

    def run_phase(
        name: str, stdin_text: str, budget_s: float, *, compact_limit: int, tid: str | None
    ) -> dict:
        ev = out / ("events.jsonl" if name == "main" else f"events.{name}.jsonl")
        last = out / ("last.txt" if name == "main" else f"last.{name}.txt")
        argv = [
            CODEX,
            "exec",
            *common(compact_limit),
            "-C",
            str(ws.lane),
            "--json",
            "-o",
            str(last),
        ]
        argv += ["resume", tid, "-"] if tid else ["-"]
        cmds.append(f"[{name}] " + " ".join(argv) + "  < stdin")
        r = run_budgeted(
            argv,
            cwd=ws.lane,
            env=env,
            stdin_text=stdin_text,
            out=ev,
            err=out / ("stderr.txt" if name == "main" else f"stderr.{name}.txt"),
            budget_s=max(1.0, budget_s),
        )
        txt = last.read_text(errors="replace") if last.exists() else ""
        ans = extract_answer(txt)
        rec = {
            "phase": name,
            "t_start_rel": round(time.time() - t_call - r["wall_s"], 1),
            **r,
            "budget_s": round(budget_s, 1),
            "compact_limit": compact_limit,
            "answer_status": classify(ans),
            "mechanism_id": (ans or {}).get("mechanism_id")
            or ("ABSTAIN" if ans and "abstain" in ans else None),
            "final_text_chars": len(txt),
        }
        phases.append(rec)
        transcripts.extend([ev, out / ("stderr.txt" if name == "main" else f"stderr.{name}.txt")])
        rec["_text"], rec["_answer"] = txt, ans
        return rec

    main_budget = budget * F2_FORCE_FRAC if (f2 and not smoke) else budget
    main = run_phase(
        "main",
        (prompt + F2_JSON_FIRST) if (f2 and not smoke) else prompt,
        main_budget,
        compact_limit=compact,
        tid=None,
    )
    tid = None
    try:
        tid = json.loads((out / "events.jsonl").read_text().splitlines()[0]).get("thread_id")
    except (OSError, IndexError, ValueError):
        pass
    res: dict = {
        "thread_id": tid,
        "drafter": drafter,
        "codex_window": window,
        "codex_auto_compact": compact,
        "stream_idle_timeout_ms": F1_IDLE_TIMEOUT_MS,
        "budget_exhausted": main["budget_exhausted"],
    }
    if smoke:
        if f2 and tid:  # prove the resume path (no compaction) and the thinking-cap injection
            run_phase("resume", SMOKE_RESUME_MESSAGE, 300, compact_limit=NO_COMPACT, tid=tid)
    elif not f2:
        if main["budget_exhausted"] and salvage and tid:
            s = run_phase("salvage", SALVAGE_MESSAGE, salvage, compact_limit=compact, tid=tid)
            res["salvage"] = {k: s[k] for k in ("rc", "wall_s", "budget_exhausted")}
    elif tid:
        elapsed = lambda: time.time() - t_call  # noqa: E731
        if main["budget_exhausted"] or main["answer_status"] not in ("hypothesis", "abstain"):
            run_phase(
                "forced",
                F2_FORCED_MESSAGE,
                min(F2_PHASE_CAP_S, max(60.0, budget - elapsed())),
                compact_limit=NO_COMPACT,
                tid=tid,
            )
        left = budget - elapsed()
        if left > 120:
            rf = run_phase(
                "refine",
                F2_REFINE_MESSAGE.format(minutes=int(left // 60)),
                left,
                compact_limit=compact,
                tid=tid,
            )
            if rf["budget_exhausted"] and salvage:
                s = run_phase(
                    "salvage", SALVAGE_MESSAGE, salvage, compact_limit=NO_COMPACT, tid=tid
                )
                res["salvage"] = {k: s[k] for k in ("rc", "wall_s", "budget_exhausted")}
        res["budget_exhausted"] = any(p["budget_exhausted"] for p in phases)
    # last complete answer (hypothesis/abstain) wins; an incomplete one is used only when no phase completed
    chosen = None
    for p in phases:
        if p["_answer"] is not None and classify(p["_answer"]) in ("hypothesis", "abstain"):
            chosen = p
    if chosen is None:
        for p in phases:
            if p["_answer"] is not None and classify(p["_answer"]) == "incomplete":
                chosen = p
    if smoke:
        res["final_text"] = main["_text"]
        res["smoke_resume_text"] = phases[1]["_text"] if len(phases) > 1 else None
    else:
        res["final_text"] = (chosen or phases[-1])["_text"]
    res["answer_phase"] = chosen["phase"] if chosen else None
    res["phase_answers"] = {p["phase"]: p["_answer"] for p in phases}
    for p in phases:
        p.pop("_text", None)
        p.pop("_answer", None)
    res["phases"] = phases
    (out / "command.txt").write_text("\n".join(cmds) + "\n")
    res["native"] = _codex_events_native(
        [out / "events.jsonl"]
        + [out / f"events.{p['phase']}.jsonl" for p in phases if p["phase"] != "main"]
    )
    res["transcripts"] = [str(p) for p in transcripts]
    return res


# ---------------------------------------------------------------------------------------------
# orv: the orchestrator REPL, driven exactly as the loop's OrchestratorBackend.argv drives it
# (research actor_orchestrator.py, HEAD == 56dfdccf for that file): the orchestrator repo's
# stdlib CLI under `python -I`, prompt on stdin, the bundle as --context-bundle.
# ---------------------------------------------------------------------------------------------
ORCH_ROOT = os.environ.get("AK_ORCHESTRATOR_ROOT") or "/mnt/raid0/llm/epyc-orchestrator"
ORCH_URL = (os.environ.get("AK_ORCHESTRATOR_URL") or "http://127.0.0.1:8000").rstrip("/")
ORCH_PYTHON = "/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python"  # actors.ORCHESTRATOR_PYTHON
ORCH_ROLE = "architect_critic"  # orch:architect_critic -> the 27B on :8083 (launch_manifest)
ORCH_TURN_CAP = 50  # actor_orchestrator.TURN_CAP
ORCH_SERVER_TIMEOUT_CAP = 3600  # ChatRequest.timeout_s le=3600 (the loop would send 1710)
SERVER_LOG_8083 = Path("/mnt/raid0/llm/epyc-orchestrator/logs/llama-server-8083.log")
TAP_EVENTS = Path("/mnt/raid0/llm/tmp/inference_tap_events.jsonl")
ORCH_SMOKE_TIMEOUT_S = int(os.environ.get("C95_ORCH_SMOKE_TIMEOUT", "3600"))


def tap_window(offset: int, rid: str, out: Path) -> dict:
    """Every backend call the orchestrator made FOR this request (tap events whose request_id
    is ours or a child `rid:xxxx`, or whose parent_request_id is ours), summarised per
    (role, port, topology_role). Frontdoor (:8070) rows are the worker delegations that make
    this arm '27B architect + frontdoor workers', not a pure 27B arm."""
    rows, per = [], {}
    try:
        with open(TAP_EVENTS, "rb") as fh:
            fh.seek(offset)
            for raw in fh:
                if rid.encode() not in raw:
                    continue
                try:
                    e = json.loads(raw)
                except ValueError:
                    continue
                if e.get("event") == "chunk":
                    continue
                r, par = str(e.get("request_id") or ""), str(e.get("parent_request_id") or "")
                if not (r == rid or r.startswith(rid + ":") or par == rid):
                    continue
                e.pop("prompt", None)
                rows.append(e)
                key = f"{e.get('role')}@{e.get('port')}"
                s = per.setdefault(
                    key, {"calls": 0, "tokens": 0, "gen_s": 0.0, "errors": 0, "topology_roles": []}
                )
                if e.get("topology_role") and e["topology_role"] not in s["topology_roles"]:
                    s["topology_roles"].append(e["topology_role"])
                if e.get("event") == "start":
                    s["calls"] += 1
                if e.get("event") == "timings":
                    s["tokens"] += int(float(e.get("tokens") or 0))
                    s["gen_s"] = round(s["gen_s"] + float(e.get("gen_ms") or 0) / 1000, 1)
                if e.get("event") in ("error", "failed") or e.get("error"):
                    s["errors"] += 1
    except OSError as exc:
        return {"error": str(exc)}
    with open(out, "w") as fh:
        for e in rows:
            fh.write(json.dumps(e) + "\n")
    fd = {k: v for k, v in per.items() if "@8070" in k}
    return {
        "request_id": rid,
        "events": len(rows),
        "per_backend": per,
        "frontdoor_8070_calls": sum(v["calls"] for v in fd.values()),
        "frontdoor_8070_tokens": sum(v["tokens"] for v in fd.values()),
        "frontdoor_roles": sorted(fd),
    }


#: actors.HYPOTHESIS_SCHEMA @56dfdccf, wrapped exactly as actor_orchestrator.wire_schema does.
HYPOTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        **{n: {"type": "string"} for n in HYP_FIELDS},
        "relies_on_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": list(HYP_FIELDS),
    "additionalProperties": False,
}
ABSTAIN_BRANCH = {
    "type": "object",
    "properties": {"abstain": {"type": "string"}},
    "required": ["abstain"],
    "additionalProperties": False,
}
WIRE_SCHEMA = {"anyOf": [HYPOTHESIS_SCHEMA, ABSTAIN_BRANCH]}


def slots_8083() -> list | None:
    try:
        import urllib.request

        with urllib.request.urlopen(UPSTREAM + "/slots", timeout=10) as r:
            return [
                {k: s.get(k) for k in ("id", "id_task", "is_processing")}
                for s in json.loads(r.read())
            ]
    except Exception:  # noqa: BLE001
        return None


def server_log_window(offset: int, out: Path) -> dict:
    """The :8083 log written during the call (all clients: this call shares the server)."""
    try:
        with open(SERVER_LOG_8083, "rb") as fh:
            fh.seek(offset)
            data = fh.read()
    except OSError as exc:
        return {"error": str(exc)}
    out.write_bytes(data)
    text = data.decode("utf-8", errors="replace")
    return {
        "bytes": len(data),
        "tasks_launched": len(re.findall(r"launch_slot_|slot launch", text)),
        "tasks_released": text.count("stop processing"),
    }


#: orsv: the UFH-12 branch CLI (stdlib-only, run under -I like the loop's CLI); the SERVER must be
#: a deployed build that honours ChatRequest.context_search (gate: orsv_deployed + the smoke).
ORSV_CLI = os.environ.get(
    "C95_ORSV_CLI", "/mnt/raid0/llm/worktrees/orch-ufh12/scripts/autokernel_actor_cli.py"
)
ORSV_HINT = (
    "context.search('<what you need>') returns pointers; read them with "
    "context.get(section, offset=start, max_chars=end-start)"
)


def orsv_deployed() -> dict:
    """Read-only pre-check (no request): is the LIVE API a build with ChatRequest.context_search?
    True only if the deployed repo's requests.py declares the field AND the uvicorn process on
    :8000 started after that file last changed. The definitive gate is the orsv smoke
    (context_search_acknowledged and search.modes.hybrid > 0)."""
    req = Path(ORCH_ROOT) / "src/api/models/requests.py"
    out: dict = {"requests_py_has_field": "context_search" in req.read_text(errors="replace")}
    try:
        ps = subprocess.run(["ps", "-eo", "pid,lstart,args"], capture_output=True, text=True).stdout
        rows = [ln for ln in ps.splitlines() if "uvicorn" in ln and "--port 8000" in ln]
        if rows:
            fields = rows[0].split()
            started = time.mktime(time.strptime(" ".join(fields[1:6]), "%a %b %d %H:%M:%S %Y"))
            out.update(
                {
                    "api_pid": int(fields[0]),
                    "api_started": " ".join(fields[1:6]),
                    "api_started_after_field": started > req.stat().st_mtime,
                }
            )
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    # uvicorn --workers N can be reloaded by restarting workers under an old master, so the
    # master's start time is informational only; the orsv SMOKE is the definitive gate.
    out["deployed"] = bool(out.get("requests_py_has_field"))
    return out


def orch_prompt_and_bundle(ws: "Workspace", *, search: bool = False) -> tuple[str, dict]:
    """The loop's orchestrator-variable prompt: actors.propose with mode orchestrator-variable
    builds _HYPOTHESIS_TASK around OrchestratorBundle.index (no lane block and no concise rule:
    both are opencode-only), with lane = the worktree because lane_guard was on."""
    actor_context = _pinned_aklib_source("actor_context")
    inline = ws.prompt("prompt.inline.txt")
    ctx = ws.relocate((CONTEXTS / ws.meta["id"] / "context.full.txt").read_text())
    body = ctx.rstrip("\n")
    assert inline.count(body) == 1, "inline prompt does not embed the context exactly once"
    bundle = actor_context.orchestrator_bundle(ctx, role="planner", lane=ws.lane)
    index = bundle.index
    if search:  # orsv: the one-line hint, after the index's opening paragraph
        head, sep, rest = index.partition("\n\n")
        index = head + "\n" + ORSV_HINT + sep + rest
        bundle.index = index
    prompt = inline.replace(body, index, 1)
    bundle.seal(prompt)  # actors._sealed: role None
    return prompt, bundle.payload


def arm_orv(
    ws: "Workspace",
    out: Path,
    prompt: str,
    payload: dict | None,
    budget: float,
    *,
    search: bool = False,
) -> dict:
    schema_f = out / "schema.json"
    schema_f.write_text(json.dumps(WIRE_SCHEMA, sort_keys=True))
    prov = out / "provenance.json"
    cli = ORSV_CLI if search else f"{ORCH_ROOT}/scripts/autokernel_actor_cli.py"
    argv = [
        ORCH_PYTHON,
        "-I",
        cli,
        "--root",
        str(ws.lane),
        "--read-only",
        "--schema",
        str(schema_f),
    ]
    if payload is not None:
        bf = out / "context-bundle.json"
        bf.write_text(json.dumps(payload, ensure_ascii=False))
        argv += ["--context-bundle", str(bf)]
        if search:
            argv += ["--context-search"]
    server_timeout = int(min(ORCH_SERVER_TIMEOUT_CAP, budget))
    rid = f"c95-{out.parts[-3]}-{out.parts[-2]}-{uuid.uuid4().hex[:10]}"
    argv += [
        "--url",
        ORCH_URL,
        "--role",
        ORCH_ROLE,
        "--max-turns",
        str(ORCH_TURN_CAP),
        "--timeout-s",
        str(server_timeout),
        "--provenance-out",
        str(prov),
        "--request-id",
        rid,
    ]
    try:
        tap_off = TAP_EVENTS.stat().st_size
    except OSError:
        tap_off = 0
    (out / "command.txt").write_text(" ".join(argv) + "  < prompt (stdin)\n")
    env = base_env(ws.tmp)
    before = slots_8083()
    try:
        off = SERVER_LOG_8083.stat().st_size
    except OSError:
        off = 0
    res = run_budgeted(
        argv,
        cwd=ws.lane,
        env=env,
        stdin_text=prompt,
        out=out / "stdout.txt",
        err=out / "stderr.txt",
        budget_s=server_timeout + 300,
    )
    res["slots_8083_before"], res["slots_8083_after"] = before, slots_8083()
    res["server_8083_window"] = server_log_window(off, out / "server-8083-window.log")
    time.sleep(2)  # let the tap writer flush the last events
    res["delegations"] = tap_window(tap_off, rid, out / "tap-events.jsonl")
    try:
        sc = json.loads(prov.read_text())
    except (OSError, ValueError):
        sc = {}
    resp = sc.get("response") if isinstance(sc.get("response"), dict) else sc
    pulls = resp.get("context_pulls") if isinstance(resp, dict) else None
    keep = (
        "routed_to",
        "role_history",
        "routing_strategy",
        "turns",
        "mode",
        "tokens_generated",
        "tokens_used",
        "tools_used",
        "tools_called",
        "tool_output_tokens",
        "compaction_triggered",
        "compaction_tokens_saved",
        "elapsed_seconds",
        "prompt_eval_ms",
        "generation_ms",
        "predicted_tps",
        "error_code",
        "error_detail",
        "task_root",
    )
    res["native"] = {
        "provenance_keys": sorted(sc.keys()) if isinstance(sc, dict) else None,
        **({k: resp.get(k) for k in keep} if isinstance(resp, dict) else {}),
        "context_pulls": (
            {
                "offered": pulls.get("offered"),
                "totals": pulls.get("totals"),
                "print_cap_bytes": pulls.get("print_cap_bytes"),
                "sections": {
                    n: {k: r.get(k) for k in ("pulls", "bytes_pulled", "unique_bytes", "coverage")}
                    for n, r in (pulls.get("sections") or {}).items()
                    if isinstance(r, dict)
                },
            }
            if isinstance(pulls, dict)
            else None
        ),
        "effort": "orchestrator architect_critic chat lane: chat_template_kwargs "
        "{enable_thinking: true, reasoning_effort: medium} (RI-23, flag "
        "thinking_roles_chat_lane); reasoning server-split into reasoning_content",
        "server_timeout_s": server_timeout,
        "max_turns": ORCH_TURN_CAP,
    }
    if search:
        s = (pulls or {}).get("search") if isinstance(pulls, dict) else None
        res["native"]["context_search_acknowledged"] = sc.get("context_search_acknowledged")
        res["native"]["context_search"] = (
            {
                "enabled": s.get("enabled"),
                "calls": s.get("calls"),
                "modes": s.get("modes"),
                "embedder_model": (s.get("index") or {}).get("embedder_model"),
                "index": s.get("index"),
                "log_len": len(s.get("log") or []),
                "log_dropped": s.get("log_dropped"),
            }
            if isinstance(s, dict)
            else None
        )
        if isinstance(s, dict):
            (out / "context-search-log.json").write_text(json.dumps(s.get("log"), indent=1))
    res["final_text"] = (out / "stdout.txt").read_text(errors="replace")
    res["transcripts"] = [str(out / "stdout.txt"), str(out / "stderr.txt"), str(prov)]
    return res


# ---------------------------------------------------------------------------------------------
# one call
# ---------------------------------------------------------------------------------------------
def call_dir(cid: str, arm: str, rep: int) -> Path:
    """`arm` may carry a run-3 variant label: "cxa+warm" -> results/<cid>/cxa+warm/r<rep>."""
    return RESULTS / cid / arm / f"r{rep}"


def require_fresh_call_slot(d: Path) -> None:
    """Refuse a partial/reused call before prompt, policy, or wire files change."""
    if d.exists() and any(d.iterdir()):
        raise FileExistsError(f"call directory already has artifacts: {d}")


def base_arm(arm: str) -> str:
    return arm.split("+", 1)[0]


def labelled(arm: str, args) -> str:
    lab = getattr(args, "label", None)
    return f"{arm}+{lab}" if lab else arm


def acquire(d: Path) -> bool:
    d.mkdir(parents=True, exist_ok=True)
    lock = d / ".lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            pid = int(lock.read_text().strip() or 0)
            os.kill(pid, 0)
            return False
        except (ValueError, ProcessLookupError):
            lock.unlink(missing_ok=True)
            return acquire(d)
        except PermissionError:
            return False


def _source_sha(path: Path) -> str | None:
    return __import__("hashlib").sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def do_call(
    meta: dict,
    arm: str,
    rep: int,
    args,
    *,
    smoke: bool = False,
    group: dict | None = None,
    start_delay_s: float = 0.0,
) -> dict:
    cid = meta["id"]
    arm_dir = labelled(arm, args)
    d = call_dir("smoke" if smoke else cid, arm_dir, rep)
    if (d / "result.json").exists():
        return json.loads((d / "result.json").read_text())
    try:
        require_fresh_call_slot(d)
    except FileExistsError as exc:
        return {"status": "duplicate_call_refused", "error": str(exc)}
    if not acquire(d):
        return {"status": "locked"}
    if start_delay_s > 0:  # run 3: cold-leg stagger inside a start group
        time.sleep(start_delay_s)
    _tl.codex_home_per_call = bool(getattr(args, "codex_home_per_call", True))
    _tl.compact_limit = int(getattr(args, "compact_limit", 0) or 0)
    _tl.wrap = None
    budget = args.budget if not smoke else 600
    salvage = args.salvage if not smoke else 0
    t0 = time.time()
    proxy = None
    result: dict = {
        "context": cid,
        "arm": arm_dir,
        "arm_base": arm,
        "rep": rep,
        "smoke": smoke,
        "arm_desc": ARMS[arm]["desc"],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "budget_s": budget,
        "salvage_budget_s": salvage,
        "run3": run3_config(args, arm, group),
    }
    result["sandbox_probe"] = bool(smoke and getattr(args, "sandbox_probe", False))
    try:
        with Workspace(meta) as ws:
            payload = None
            if arm in ("orv", "orsv") and not smoke:
                prompt, payload = orch_prompt_and_bundle(ws, search=(arm == "orsv"))
                (d / "prompt.txt").write_text(prompt)
            elif smoke and arm in ("cxv", "orv", "orsv"):
                prompt, payload = smoke_prompt_for(arm, ws)
            else:
                prompt = SMOKE_PROMPT if smoke else ws.prompt(ARMS[arm]["prompt"])
                if smoke and getattr(args, "sandbox_probe", False):
                    denied = "/mnt/raid0/llm/autokernel/campaigns/ak-ds41-cpu-decode-20260923/store/anchor-gen-007/provenance.json"
                    command = (
                        f"cat {denied} 2>&1; touch {ws.root}/.a6-probe && "
                        f"test -f {ws.root}/.a6-probe && echo WORKSPACE_OK; "
                        f"touch {d}/.a6-probe && test -f {d}/.a6-probe && echo RESULT_OK"
                    )
                    prompt = (
                        "Run exactly this shell command, then reply C95_TOOL_OK only if "
                        "it prints Permission denied, WORKSPACE_OK and RESULT_OK: `"
                        + command
                        + "`\n"
                    )
            (d / "prompt.sha256").write_text(
                __import__("hashlib").sha256(prompt.encode()).hexdigest() + "\n"
            )
            result.update(
                {
                    "prompt_file": ARMS[arm]["prompt"],
                    "prompt_chars": len(prompt),
                    "workspace": str(ws.root),
                }
            )
            port = None
            proxied = ARMS[arm]["gpu"] and ARMS[arm].get("proxy", True)
            if proxied and getattr(args, "sandbox", "landlock") == "landlock":
                _tl.wrap, result["sandbox"] = sandbox_wrap(meta, ws, d, prompt, args)
            provenance = {
                "schema": "ds41-c95.preexposure.v1",
                "purpose": "harness-call",
                "run_id": uuid.uuid4().hex,
                "call_locator": str(d),
                "started_at_utc": result["started_at"],
                "wall_clock_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "pid": os.getpid(),
                "uid": os.getuid(),
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "context_id": cid,
                "arm": arm_dir,
                "rep": rep,
                "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
                "context_meta_sha256": _source_sha(CONTEXTS / cid / "meta.json"),
                "policy_sha256": _source_sha(Path(result["sandbox"]["policy"]))
                if result.get("sandbox")
                else None,
                "source_sha256": {
                    "run3": _source_sha(Path(__file__)),
                    "sandbox3": _source_sha(Path(sandbox3.__file__)),
                    "proxy3": _source_sha(Path(proxy3.__file__)),
                },
                "grade_source_sha256": _source_sha(BASE / "grade.py"),
                "declared_model": (
                    MODEL_OC
                    if arm == "oc"
                    else MODEL_CX
                    if arm in ("cx", "cxv", "cxf1", "cxf12", "cxa")
                    else None
                ),
                "declared_endpoint": UPSTREAM if proxied else None,
                "verified_backend_identity": None,
                "verified_model_identity": None,
                "grade": None,
            }
            result["preexposure_provenance"] = sandbox3.emit_preexposure(d, provenance)
            if proxied:
                extra = (
                    ["--think-budget", str(F2_THINK_BUDGET), "--think-message", F2_THINK_MESSAGE]
                    if ARMS[arm].get("f2")
                    else []
                )
                extra += proxy3_flags(args, ws, group)
                proxy, port = start_proxy(d / "wire", policy=args.effort_policy, extra=extra)
            if arm in ("cx", "cxv", "cxf1", "cxf12", "cxa", "sol"):
                result["codex_bin"] = os.path.realpath(CODEX)
            log(f"start {cid}/{arm}/r{rep} (prompt {len(prompt)} chars, port {port})")
            if arm == "oc":
                res = arm_oc(ws, d, prompt, port, budget, salvage)
            elif arm == "cx":
                res = arm_codex(
                    ws,
                    d,
                    prompt,
                    port,
                    budget,
                    salvage,
                    sandbox=args.codex_sandbox,
                    model=MODEL_CX,
                    reference=False,
                )
            elif arm == "sol":
                res = arm_codex(
                    ws,
                    d,
                    prompt,
                    None,
                    budget,
                    salvage,
                    sandbox=args.codex_sandbox,
                    model="gpt-6.1-sol",
                    reference=True,
                )
            elif arm == "hm":
                res = arm_hm(ws, d, prompt, port, budget, salvage)
            elif arm == "cxv":
                res = arm_codex(
                    ws,
                    d,
                    prompt,
                    port,
                    budget,
                    salvage,
                    sandbox=args.codex_sandbox,
                    model=MODEL_CX,
                    reference=False,
                )
            elif arm in ("cxf1", "cxf12", "cxa"):
                before = slots_8083()
                try:
                    off = SERVER_LOG_8083.stat().st_size
                except OSError:
                    off = 0
                res = arm_codex_f(
                    ws,
                    d,
                    prompt,
                    port,
                    budget,
                    salvage,
                    sandbox=args.codex_sandbox,
                    f2=bool(ARMS[arm].get("f2")),
                    smoke=smoke,
                )
                res["slots_8083_before"], res["slots_8083_after"] = before, slots_8083()
                res["server_8083_window"] = server_log_window(off, d / "server-8083-window.log")
            elif arm in ("orv", "orsv"):
                res = arm_orv(
                    ws,
                    d,
                    prompt,
                    payload,
                    budget if not smoke else ORCH_SMOKE_TIMEOUT_S,
                    search=(arm == "orsv"),
                )
            else:
                raise ValueError(arm)
            result["lane_changes"] = ws.lane_changes()
    except Exception as exc:  # noqa: BLE001
        res = {"error": f"{type(exc).__name__}: {exc}"[:500], "final_text": "", "transcripts": []}
    finally:
        _tl.wrap = None
        if proxy is not None:
            end_group(proxy, grace=5)
    return finish_call(d, meta, arm, rep, res, result, t0, smoke=smoke)


def smoke_prompt_for(arm: str, ws: "Workspace") -> tuple[str, dict | None]:
    """cxv: the reply needs a file of the relocated bundle (codex must reach the bundle).
    orv: a 2-section dummy bundle; the reply needs context.get() (REPL `context` + pulls)."""
    if arm == "cxv":
        return (
            f"Run the shell command `head -c 34 {ws.bundle}/INDEX.md` and then reply with exactly "
            "its output and nothing else.\n",
            None,
        )
    payload = {
        "schema": "epyc.orchestrator.context_bundle.v1",
        "sections": [
            {
                "name": "preamble",
                "text": "smoke bundle\n",
                "kind": "text",
                "inline": True,
                "description": "the task line",
            },
            {
                "name": "inbox",
                "text": "C95_TOOL_OK\n",
                "kind": "text",
                "inline": False,
                "description": "operator suggestions",
            },
        ],
    }
    if arm == "orsv":
        return (
            "First call context.search('smoke marker') once. Then reply with the JSON object "
            '{"abstain": "<X>"} where <X> is the exact text of context.get(\'inbox\') with surrounding '
            "whitespace removed. Read it with context.get; do not guess.\n",
            payload,
        )
    return (
        'Reply with the JSON object {"abstain": "<X>"} where <X> is the exact text of '
        "context.get('inbox') with surrounding whitespace removed. Read it with context.get; "
        "do not guess.\n",
        payload,
    )


def finish_call(
    d: Path,
    meta: dict,
    arm: str,
    rep: int,
    res: dict,
    result: dict,
    t0: float,
    *,
    smoke: bool = False,
) -> dict:
    """Post-processing that ALWAYS ends in result.json (run 1 lost every result to one
    unreadable gzip): each derived field is computed under its own guard."""
    cid = meta["id"]
    arm = base_arm(arm)
    final_text = res.pop("final_text", "")
    (d / "final_text.txt").write_text(final_text)
    answer = extract_answer(final_text)
    status = classify(answer) if not res.get("error") else "error"
    if smoke and arm == "orv":
        status = (
            "smoke_ok"
            if (answer or {}).get("abstain", "").strip() == "C95_TOOL_OK"
            else "smoke_fail"
        )
    elif smoke and arm == "orsv":
        cs = (res.get("native") or {}).get("context_search") or {}
        gate = (
            bool((res.get("native") or {}).get("context_search_acknowledged"))
            and int((cs.get("modes") or {}).get("hybrid") or 0) > 0
        )
        result["orsv_gate"] = {
            "acknowledged": (res.get("native") or {}).get("context_search_acknowledged"),
            "modes": cs.get("modes"),
            "embedder_model": cs.get("embedder_model"),
            "passed": gate,
        }
        status = (
            "smoke_ok"
            if gate and (answer or {}).get("abstain", "").strip() == "C95_TOOL_OK"
            else "smoke_fail"
        )
    elif smoke and arm == "cxf12":
        inj = False
        try:
            inj = any(
                json.loads(wire_line).get("injected")
                for wire_line in (d / "wire" / "wire.jsonl").read_text().splitlines()
            )
        except (OSError, ValueError):
            pass
        rtxt = (res.get("smoke_resume_text") or "").strip().strip("`").strip()
        result["smoke_checks"] = {
            "main": final_text.strip().strip("`").strip(),
            "resume": rtxt,
            "think_budget_injected": inj,
        }
        status = (
            "smoke_ok"
            if final_text.strip().strip("`").strip() == "C95_TOOL_OK"
            and rtxt == "C95_RESUME_OK"
            and inj
            else "smoke_fail"
        )
    elif smoke and arm == "cxv":
        status = (
            "smoke_ok"
            if final_text.strip().strip("`").strip().startswith("## Context bundle")
            else "smoke_fail"
        )
    elif smoke:
        # the reply must BE the command output (a model explaining why it could not run the
        # command also names the string)
        status = (
            "smoke_ok" if final_text.strip().strip("`").strip() == "C95_TOOL_OK" else "smoke_fail"
        )
    transcripts = [Path(p) for p in res.pop("transcripts", [])] + [
        d / "wire" / "wire-bodies.jsonl.gz"
    ]
    result.update(res)
    if result.get("sandbox_probe"):
        outputs = []
        for p in transcripts:
            if not p.exists() or p.suffix != ".jsonl":
                continue
            for line in p.read_text(errors="replace").splitlines():
                try:
                    item = json.loads(line).get("item", {})
                except ValueError:
                    continue
                if item.get("type") == "command_execution" and item.get("status") == "completed":
                    outputs.append(str(item.get("aggregated_output") or ""))
        tool_output = "\n".join(outputs)
        checks = {
            "denial_logged": "Permission denied" in tool_output,
            "workspace_confirmed": "WORKSPACE_OK" in tool_output,
            "result_confirmed": "RESULT_OK" in tool_output and (d / ".a6-probe").exists(),
            "wire_proxy_recorded": (d / "wire" / "wire.jsonl").exists(),
        }
        result["sandbox_probe_checks"] = checks
        if not all(checks.values()):
            status = "smoke_fail"

    def guarded(fn, *a):
        try:
            return fn(*a)
        except Exception as exc:  # noqa: BLE001
            return {"postprocess_error": f"{type(exc).__name__}: {exc}"[:300]}

    result.update(
        {
            "status": status,
            "answer": answer,
            "wall_s": round(time.time() - t0, 1) if t0 else result.get("wall_s"),
            "wire": (
                guarded(wire_metrics, d / "wire")
                if ARMS[arm]["gpu"] and ARMS[arm].get("proxy", True)
                else None
            ),
            **(
                {"wire_timing": guarded(wire_timing, d / "wire", d / "wire" / "timing.json")}
                if arm in ("cxf1", "cxf12", "cxa")
                else {}
            ),
            **(
                {
                    "first_request": guarded(first_request, d / "wire"),
                    "compactions": guarded(compaction_table, d / "wire"),
                }
                if ARMS[arm]["gpu"] and ARMS[arm].get("proxy", True)
                else {}
            ),
            "contamination": {}
            if smoke
            else guarded(scan_contamination, transcripts, meta["anchor_commit"]),
            "sandbox_denials": guarded(scan_denials, transcripts)
            if result.get("sandbox")
            else None,
            "contamination_crosscheck": {}
            if smoke
            else guarded(crosscheck_contamination, transcripts, meta["anchor_commit"]),
            "finished_at": result.get("finished_at")
            or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    )
    atomic_write(d / "result.json", json.dumps(result, indent=2) + "\n")
    (d / ".lock").unlink(missing_ok=True)
    log(
        f"done  {cid}/{result.get('arm', arm)}/r{rep}: {status} in {result.get('wall_s') or 0:.0f}s "
        f"(req {((result.get('wire') or {}).get('requests'))}, contamination {result['contamination'] or '-'})"
    )
    return result


# ---------------------------------------------------------------------------------------------
# planning
# ---------------------------------------------------------------------------------------------
def load_meta(cid: str) -> dict:
    return json.loads((CONTEXTS / cid / "meta.json").read_text())


def build_calls(args) -> list[tuple[str, str, int]]:
    ctxs = args.contexts.split(",")
    arms = args.arms.split(",")
    r2 = [c for c in (args.r2 or "").split(",") if c]
    calls = []
    for rep in range(1, args.repeats + 1):
        for c in ctxs:
            for a in arms:
                calls.append((c, a, rep))
    if args.repeats < 2:
        for c in r2:
            for a in arms:
                calls.append((c, a, 2))
    return calls


def estimate(calls) -> dict:
    gpu = [c for c in calls if ARMS[c[1]]["gpu"]]
    n = len(gpu)
    typical_h = n * EST_CALL_MIN * EST_CONCURRENCY_SLOWDOWN / 2 / 60
    worst_h = n * (PLANNER_BUDGET_S + SALVAGE_BUDGET_S) / 3600 / 2
    return {
        "gpu_calls": n,
        "ref_calls": len(calls) - n,
        "est_gpu_wall_h_typical": round(typical_h, 1),
        "est_gpu_wall_h_worst": round(worst_h, 1),
    }


def cmd_plan(args) -> int:
    calls = build_calls(args)
    for c in calls:
        m = load_meta(c[0])
        done = (call_dir(c[0], labelled(c[1], args), c[2]) / "result.json").exists()
        print(
            f"{c[0]}/{c[1]}/r{c[2]}  anchor={m['anchor_build']} present={m['anchor_build_present']}"
            f"{'  DONE' if done else ''}"
        )
    print(json.dumps(estimate(calls), indent=2))
    return 0


def preflight(args, calls) -> list[str]:
    problems = []
    for cid in sorted({c[0] for c in calls}):
        m = load_meta(cid)
        if not (Path(m["anchor_build_path"]) / "bin" / "libggml-cpu.so.0.16.0").exists():
            problems.append(
                f"{cid}: anchor build {m['anchor_build']} missing -> run prep_anchor.sh "
                "(CPU build, DS41 paused) or drop the context"
            )
    arms = {c[1] for c in calls}
    if arms & {"cxf1", "cxf12"} and args.concurrency != 1:
        problems.append("cxf1/cxf12 are SOLO arms (F1): run them with --concurrency 1")
    if getattr(args, "warm_prefix", False) and getattr(args, "canon_paths", True) is False:
        problems.append(
            "--warm-prefix without --canon-paths cannot share a prefix: each call's workspace path "
            "differs from the first path occurrence (~37k chars into a codex prompt)"
        )
    if getattr(args, "warm_prefix", False) and arms & {"cx", "cxv"}:
        problems.append(
            "--warm-prefix holds each first request through the serial warm copies (minutes); cx/cxv "
            "keep codex's ~300 s stream idle timeout and would retry into a cold duplicate -- use cxa"
        )
    if arms & {"cx", "cxv", "sol", "cxf1", "cxf12", "cxa"} and not args.codex_sandbox:
        problems.append(
            "codex arms need --codex-sandbox: codex's sandboxed modes cannot run commands in "
            "this container (bwrap EPERM); the live loop uses danger-full-access. Operator decision."
        )
    if args.concurrency > 2:
        problems.append("--concurrency is capped at 2 concurrent 27B sessions")
    return problems


def cmd_run(args, *, smoke_arm: str | None = None) -> int:
    if smoke_arm:
        meta = load_meta(args.smoke_context)
        r = do_call(meta, smoke_arm, int(time.time()), args, smoke=True)
        print(
            json.dumps(
                {k: r.get(k) for k in ("status", "wall_s", "rc", "wire", "native", "error")},
                indent=2,
            )
        )
        return 0 if r.get("status") == "smoke_ok" else 1
    if args.after_pid:
        log(f"waiting for pid {args.after_pid} (the running driver) to exit before starting")
        while pid_alive(args.after_pid):
            time.sleep(30)
        log(f"pid {args.after_pid} has exited; finalizing its crashed calls")
        cmd_finalize(args)
    calls = build_calls(args)
    problems = preflight(args, calls)
    if problems:
        for p in problems:
            print("PREFLIGHT:", p, file=sys.stderr)
        return 2
    est = estimate(calls)
    log(
        f"plan: {len(calls)} calls {est}; deadline {args.deadline_hours} h; concurrency {args.concurrency}"
    )
    deadline = time.time() + args.deadline_hours * 3600
    gpu_calls = [c for c in calls if ARMS[c[1]]["gpu"]]
    ref_calls = [c for c in calls if not ARMS[c[1]]["gpu"]]
    metas = {cid: load_meta(cid) for cid in {c[0] for c in calls}}

    def worker(c):
        if (BASE / "STOP").exists() or time.time() > deadline:
            log(f"not started (STOP or deadline): {c}")
            return {"status": "not_started"}
        return do_call(metas[c[0]], c[1], c[2], args)

    if getattr(args, "group_starts", False) or getattr(args, "warm_prefix", False):
        run_groups(args, gpu_calls, metas, deadline)
        log(f"RUN DONE ({len(calls)} calls in start groups); :8083 is free of C95 sessions")
        return cmd_status(args)
    with (
        cf.ThreadPoolExecutor(max_workers=args.concurrency) as gpu_pool,
        cf.ThreadPoolExecutor(max_workers=2) as ref_pool,
    ):
        futs = [gpu_pool.submit(worker, c) for c in gpu_calls] + [
            ref_pool.submit(worker, c) for c in ref_calls
        ]
        for f in cf.as_completed(futs):
            try:
                f.result()
            except Exception as exc:  # noqa: BLE001 -- one call's failure never ends the run
                log(f"call raised: {type(exc).__name__}: {exc}")
    log(
        f"RUN DONE ({len(calls)} calls: {','.join(sorted({c[1] for c in calls}))} on "
        f"{','.join(sorted({c[0] for c in calls}))}); :8083 is free of C95 sessions"
    )
    return cmd_status(args)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def recover_res(d: Path, arm: str) -> dict:
    """Rebuild a crashed run-1 call's harness record from the files it left behind."""
    res: dict = {
        "recovered_by": "run2.py finalize",
        "recovery_note": "run-1 post-processing crashed (EOFError: proxy.py's wire-bodies.jsonl.gz "
        "had no end-of-stream marker); harness outputs are intact, lane_changes "
        "and the process rc were not recorded",
    }
    res["budget_exhausted"] = any(
        (d / f).exists() for f in ("salvage.stdout.txt", "salvage.events.jsonl")
    )
    res["final_text"] = (d / "final_text.txt").read_text(errors="replace")
    if arm == "oc":
        try:
            dd = json.loads((d / "export.json").read_text())
            msgs = dd.get("messages", [])
            res["session_id"] = dd.get("info", {}).get("id")
            res["native"] = {
                "steps": sum(
                    1
                    for m in msgs
                    if m["info"]["role"] == "assistant" and m["info"].get("mode") != "compaction"
                ),
                "tool_calls": sum(1 for m in msgs for q in m["parts"] if q["type"] == "tool"),
                "compactions": sum(
                    1 for m in msgs for q in m["parts"] if q["type"] == "compaction"
                ),
                "tokens": dd.get("info", {}).get("tokens"),
            }
        except (OSError, ValueError, KeyError) as exc:
            res["native"] = {"error": str(exc)}
        res["transcripts"] = [str(d / "export.json"), str(d / "stdout.txt"), str(d / "stderr.txt")]
    elif arm in ("cx", "cxv", "sol", "cxf1", "cxf12"):
        items: dict[str, int] = {}
        usage = []
        for f in ("events.jsonl", "salvage.events.jsonl"):
            if not (d / f).exists():
                continue
            for line in (d / f).read_text(errors="replace").splitlines():
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("type") == "thread.started" and f == "events.jsonl":
                    res["thread_id"] = e.get("thread_id")
                if e.get("type") == "item.completed":
                    t = (e.get("item") or {}).get("type")
                    items[t] = items.get(t, 0) + 1
                if e.get("type") == "turn.completed":
                    usage.append(e.get("usage"))
        res["native"] = {
            "item_types": items,
            "tool_calls": items.get("command_execution", 0) + items.get("mcp_tool_call", 0),
            "turn_usage": usage,
        }
        res["transcripts"] = [
            str(d / "events.jsonl"),
            str(d / "salvage.events.jsonl"),
            str(d / "stderr.txt"),
        ]
    elif arm == "hm":
        try:
            s = json.loads((d / "hermes" / "summary.json").read_text())
        except (OSError, ValueError):
            s = {}
        res["rc"] = s.get("rc")
        res["native"] = {
            k: s.get(k)
            for k in (
                "rc",
                "result_exit_code",
                "error",
                "tokens_main_loop",
                "tool_calls_stream",
                "tool_calls_by_name",
                "api_calls_transcript",
                "sessions_in_lineage",
                "compaction",
            )
        }
        h = d / "hermes"
        res["transcripts"] = [
            str(h / "session.jsonl"),
            str(h / "events.jsonl"),
            str(h / "stderr.txt"),
        ]
    return res


def cmd_finalize(args) -> int:
    """result.json for every call that FINISHED (final_text.txt present) but has none, unless a
    live driver still holds it and has not written final_text yet (then it is in flight)."""
    n = 0
    for ft in sorted(RESULTS.glob("C*/*/r*/final_text.txt")):
        d = ft.parent
        if (d / "result.json").exists():
            continue
        cid, arm, rep = d.parts[-3], base_arm(d.parts[-2]), int(d.parts[-1][1:])
        meta = load_meta(cid)
        t_start = (d / "prompt.sha256").stat().st_mtime if (d / "prompt.sha256").exists() else None
        t_end = ft.stat().st_mtime
        result = {
            "context": cid,
            "arm": arm,
            "rep": rep,
            "smoke": False,
            "arm_desc": ARMS[arm]["desc"],
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start))
            if t_start
            else None,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_end)),
            "wall_s": round(t_end - t_start, 1) if t_start else None,
            "budget_s": PLANNER_BUDGET_S,
            "salvage_budget_s": SALVAGE_BUDGET_S,
            "prompt_file": ARMS[arm]["prompt"],
        }
        finish_call(d, meta, arm, rep, recover_res(d, arm), result, 0)
        n += 1
    log(f"finalize: wrote {n} result.json file(s)")
    return 0


def cmd_status(args) -> int:
    rows = []
    for rj in sorted(RESULTS.glob("*/*/r*/result.json")):
        r = json.loads(rj.read_text())
        if r.get("smoke"):
            continue
        w = r.get("wire") or {}
        a = r.get("answer") or {}
        rows.append(
            (
                r["context"],
                r["arm"],
                r["rep"],
                r["status"],
                int(r.get("wall_s", 0)),
                w.get("requests"),
                w.get("generated_tokens_sum"),
                w.get("compactions_inferred"),
                (r.get("native") or {}).get("compactions"),
                bool(r.get("contamination")),
                (a.get("mechanism_id") or ("ABSTAIN" if "abstain" in a else ""))[:48],
            )
        )
    print(
        "ctx arm rep status wall_s req gen_tok compact(wire) compact(native) contaminated mechanism"
    )
    for row in rows:
        print(" ".join(str(x) for x in row))
    return 0


# =============================================================================================
# run 3 additions
# =============================================================================================
def run3_config(args, arm: str, group: dict | None) -> dict:
    return {
        "label": getattr(args, "label", None),
        "canon_paths": getattr(args, "canon_paths", True),
        "canon_root": CANON_ROOT if getattr(args, "canon_paths", True) else None,
        "warm_prefix": bool(group and group.get("warm")),
        "group": (group or {}).get("gid"),
        "stagger_s": (group or {}).get("stagger_s"),
        "keep_prefix_compaction": getattr(args, "keep_prefix_compaction", False),
        "pin_task": getattr(args, "pin_task", False),
        "compact_limit": getattr(args, "compact_limit", 0),
        "sandbox": getattr(args, "sandbox", "landlock"),
        "sandbox_tmp": getattr(args, "sandbox_tmp", "deny"),
        "codex_home_per_call": getattr(args, "codex_home_per_call", True),
    }


def proxy3_flags(args, ws, group: dict | None) -> list[str]:
    flags: list[str] = []
    if getattr(args, "canon_paths", True):
        flags += ["--alias", f"{ws.root}={CANON_ROOT}"]
    if group and group.get("warm"):
        flags += ["--hold-first", "--hold-timeout-s", str(CAPTURE_TIMEOUT_S + 1800)]
    if getattr(args, "keep_prefix_compaction", False):
        flags.append("--keep-prefix-compaction")
    if getattr(args, "pin_task", False):
        flags.append("--pin-task")
    return flags


def sandbox_wrap(meta: dict, ws, d: Path, prompt: str, args) -> tuple[list[str], dict]:
    """The Landlock launcher prefix for this call's harness processes, plus its record."""
    pol = sandbox3.c95_policy(
        cutoff=sandbox3.iso_epoch(meta["asran_started"]),
        ws_root=ws.root,
        call_dir=d,
        scratch_clone=ws.scratch_clone,
        context_id=meta["id"],
        prompt_text=prompt,
        tmp=getattr(args, "sandbox_tmp", "deny"),
    )
    pf, rf = d / "sandbox-policy.json", d / "sandbox-report.json"
    pf.write_text(json.dumps(pol, indent=1) + "\n")
    wrap = [
        sys.executable,
        str(Path(sandbox3.__file__)),
        "exec",
        "--policy",
        str(pf),
        "--report",
        str(rf),
        "--",
    ]
    return wrap, {
        "mode": "landlock",
        "policy": str(pf),
        "report": str(rf),
        "cutoff": meta["asran_started"],
        "tmp": getattr(args, "sandbox_tmp", "deny"),
    }


def _wire_rows(log_dir: Path) -> list[dict]:
    rows = []
    wf = log_dir / "wire.jsonl"
    for line in wf.read_text().splitlines() if wf.exists() else []:
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return sorted(
        [r for r in rows if r.get("method") == "POST" and proxy3.is_model_path(r.get("path", ""))],
        key=lambda r: r["seq"],
    )


def _tim(r: dict) -> dict:
    t = r.get("timings") or {}
    return {
        "cache_n": t.get("cache_n"),
        "prompt_n": t.get("prompt_n"),
        "prompt_s": round((t.get("prompt_ms") or 0) / 1000, 1) if t else None,
        "predicted_n": t.get("predicted_n"),
        "decode_tps": round(t.get("predicted_per_second") or 0, 2) if t else None,
    }


def first_request(log_dir: Path) -> dict:
    """UFH14-A3 measure: the call's first tool-bearing request -- how much of its prompt came from
    the server's cache (cache_n) and how much was prefilled cold (prompt_n)."""
    for r in _wire_rows(log_dir):
        if (r.get("n_tools") or 0) > 0 and "compaction" not in r:
            t = r.get("timings") or {}
            held = r.get("held_s") or 0.0
            out = {
                "seq": r["seq"],
                "status": r.get("status"),
                "wall_s": r.get("wall_s"),
                "held_s": held,
                "hold_released_by": r.get("hold_released_by"),
                "aliased_req": r.get("aliased_req"),
                **_tim(r),
            }
            if t:
                c, p = t.get("cache_n") or 0, t.get("prompt_n") or 0
                out["cache_hit"] = round(c / (c + p), 4) if c + p else None
                out["ttft_after_release_s"] = round(
                    (r.get("wall_s") or 0) - held - (t.get("predicted_ms") or 0) / 1000, 1
                )
            return out
    return {}


def compaction_table(log_dir: Path) -> dict:
    """UFH14-A4 measure: every compaction request (re-prefill tokens = its prompt_n) and the first
    tool-bearing request after it; plus D3 task-pin events."""
    rows = _wire_rows(log_dir)
    comp, pins = [], []
    for i, r in enumerate(rows):
        if r.get("task_pin"):
            pins.append({"seq": r["seq"], **r["task_pin"], **_tim(r)})
        if "compaction" not in r:
            continue
        c = r["compaction"]
        nxt = next(
            (x for x in rows[i + 1 :] if (x.get("n_tools") or 0) > 0 and "compaction" not in x),
            None,
        )
        comp.append(
            {
                "seq": r["seq"],
                "detected_by": c.get("detected_by"),
                "rewritten": c.get("rewritten"),
                "prefix_items_shared": c.get("prefix_items_shared"),
                "n_tools_sent": c.get("n_tools_sent"),
                "n_tools_upstream": r.get("n_tools_upstream", c.get("n_tools_sent")),
                "status": r.get("status"),
                "client_gone": r.get("client_gone"),
                "wall_s": r.get("wall_s"),
                "summary_tool_markup": r.get("content_tool_markup", 0),
                **_tim(r),
                "next_request": ({"seq": nxt["seq"], **_tim(nxt)} if nxt else None),
            }
        )
    done = [c for c in comp if c["prompt_n"] is not None]
    return {
        "n": len(comp),
        "rewritten": sum(1 for c in comp if c["rewritten"]),
        "reprefill_tokens_sum": sum(c["prompt_n"] or 0 for c in done),
        "reprefill_s_sum": round(sum(c["prompt_s"] or 0 for c in done), 1),
        "unfinished": len(comp) - len(done),
        "rows": comp,
        "task_pins": pins,
    }


def scan_denials(paths: list[Path]) -> dict:
    """How often the sandbox actually stopped the agent (Permission denied on a denied root)."""
    roots = sandbox3.HARD_DENY + ["/tmp", "/mnt/raid0/llm/llama.cpp-experimental"]
    n, sample = 0, []
    for p in paths:
        if not p.exists() or p.is_dir():
            continue
        try:
            data = read_gz_tolerant(p) if p.suffix == ".gz" else p.read_text(errors="replace")
        except OSError:
            continue
        for m in re.finditer(
            r"[^\n\"]{0,160}(?:Permission denied|Operation not permitted)[^\n\"]{0,40}", data
        ):
            frag = m.group(0)
            if any(r in frag for r in roots):
                n += 1
                if len(sample) < 8 and frag not in sample:
                    sample.append(frag[-200:])
    return {"denied_hits": n, "sample": sample}


# ---------------------------------------------------------------------------------------------
# UFH14-A3: start groups, warm copies, staggered release
# ---------------------------------------------------------------------------------------------
def warm_body(body: dict, path: str, id_slot: int | None) -> dict:
    """The warm copy of a captured first request. The rendered prompt of this body is a STRICT
    prefix of the member's own prompt: same messages/tools/template kwargs, add_generation_prompt
    false (the epyc-qwen3x template appends '<|im_start|>assistant\\n<think>\\n' only under
    add_generation_prompt, and llama-server's responses->chat conversion keeps unknown fields).
    A strict prefix is required on this hybrid (Gated-DeltaNet) model: reuse at n_past == the slot's
    full length needs no context checkpoint, reuse of an IDENTICAL prompt does. One sampled token
    is never decoded into the KV, so the slot holds exactly the prefix. The timings land in the warm
    proxy's wire.jsonl (results/_groups/<gid>/warm-wire)."""
    b = copy.deepcopy(body)
    b["add_generation_prompt"] = False
    b["stream"] = True  # same server code path as the member; the SSE carries the timings

    if str(path).rstrip("/").endswith("responses"):
        b["max_output_tokens"] = 1
        b.pop("max_tokens", None)
    else:
        b["max_tokens"] = 1
        b.pop("max_output_tokens", None)
    b.pop("max_completion_tokens", None)
    if id_slot is not None:
        b["id_slot"] = id_slot
    return b


def rendered_view(body: dict) -> dict:
    """The fields that reach the chat template (ids, client metadata and cache keys never do)."""
    keep = (
        "instructions",
        "messages",
        "input",
        "tools",
        "chat_template_kwargs",
        "reasoning_format",
    )
    return proxy3._strip_ids({k: body.get(k) for k in keep if k in body})


def _sha(obj) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def http_post_json(
    port: int, path: str, body: dict, timeout: float = 3600
) -> tuple[int, dict | None, float]:
    import http.client

    t0 = time.time()
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        c.request(
            "POST",
            path,
            body=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        r = c.getresponse()
        data = r.read()
        try:
            js = json.loads(data)
        except ValueError:
            js = {"raw": data[:500].decode(errors="replace")}
        return r.status, js, round(time.time() - t0, 2)
    finally:
        c.close()


def form_groups(calls: list, size: int) -> list[list]:
    groups: list[list] = []
    for c in calls:
        if groups and len(groups[-1]) < size and groups[-1][0][0] == c[0]:
            groups[-1].append(c)
        else:
            groups.append([c])
    return groups


def coordinate_warm(group: dict, members: list, args, stagger: float) -> dict:
    t0 = group["t_start"]
    dirs = [call_dir(c, labelled(a, args), r) for c, a, r in members]
    rec: dict = {"captured": [], "warms": [], "released": []}
    captured: dict[int, dict] = {}
    t_cap_end = time.time() + CAPTURE_TIMEOUT_S
    while time.time() < t_cap_end and not (BASE / "STOP_NOW").exists():
        for i, d in enumerate(dirs):
            f = d / "wire" / "first-body.json"
            if i not in captured and f.exists() and f.stat().st_mtime >= t0 - 1:
                try:
                    captured[i] = json.loads(f.read_text())
                except ValueError:
                    pass
        if all(i in captured or (d / "result.json").exists() for i, d in enumerate(dirs)):
            break
        time.sleep(1.0)
    rec["capture_wait_s"] = round(time.time() - t0, 1)
    rec["captured"] = sorted(captured)
    views = {i: rendered_view(v["body"]) for i, v in captured.items()}
    rec["rendered_sha"] = {i: _sha(v) for i, v in views.items()}
    rec["bodies_identical"] = len(set(rec["rendered_sha"].values())) <= 1
    order = sorted(captured)
    gdir = GROUPS / group["gid"]
    gdir.mkdir(parents=True, exist_ok=True)
    if order:
        wp, wport = start_proxy(gdir / "warm-wire", policy=args.effort_policy)
        try:
            slots = slots_8083() or []
            rec["slots_before"] = slots
            idle = [s["id"] for s in slots if not s.get("is_processing")]
            for k, i in enumerate(order):
                sid = idle[k] if k < len(idle) else None
                body = warm_body(captured[i]["body"], captured[i]["path"], sid)
                (gdir / f"warm-{i}.json").write_text(json.dumps(body))
                try:
                    st, js, wall = http_post_json(wport, captured[i]["path"], body)
                except Exception as exc:  # noqa: BLE001
                    st, js, wall = None, {"error": str(exc)[:300]}, None
                wrow = (_wire_rows(gdir / "warm-wire") or [{}])[-1]
                w = {
                    "member": i,
                    "id_slot": sid,
                    "status": st,
                    "wall_s": wall,
                    "timings": wrow.get("timings"),
                    "error": (js or {}).get("error") or wrow.get("error"),
                }
                rec["warms"].append(w)
                log(
                    f"warm {group['gid']} member {i} slot {sid}: HTTP {st} in {wall}s "
                    f"timings={json.dumps(w['timings'])[:160]}"
                )
            rec["slots_after_warm"] = slots_8083()
        finally:
            end_group(wp, grace=5)
    release = list(reversed(order)) + [i for i in range(len(dirs)) if i not in captured]
    for j, i in enumerate(release):
        (dirs[i] / "wire").mkdir(parents=True, exist_ok=True)
        (dirs[i] / "wire" / "RELEASE").touch()
        rec["released"].append({"member": i, "t": time.time(), "captured": i in captured})
        if j < len(release) - 1 and stagger > 0:
            time.sleep(stagger)
    return rec


def run_groups(args, gpu_calls: list, metas: dict, deadline: float) -> None:
    warm = bool(getattr(args, "warm_prefix", False))
    stagger = args.stagger_s if args.stagger_s is not None else (3.0 if warm else 0.0)
    GROUPS.mkdir(parents=True, exist_ok=True)
    for gi, members in enumerate(form_groups(gpu_calls, args.concurrency)):
        if (BASE / "STOP").exists() or time.time() > deadline:
            log(f"group not started (STOP or deadline): {members}")
            break
        gid = (
            f"{getattr(args, 'label', None) or 'nolabel'}-{members[0][0]}-"
            f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-g{gi}"
        )
        pending = [
            m
            for m in members
            if not (call_dir(m[0], labelled(m[1], args), m[2]) / "result.json").exists()
        ]
        group = {
            "gid": gid,
            "label": getattr(args, "label", None),
            "warm": warm,
            "stagger_s": stagger,
            "t_start": time.time(),
            "concurrency": args.concurrency,
            "members": [
                {
                    "context": c,
                    "arm": labelled(a, args),
                    "rep": r,
                    "dir": str(call_dir(c, labelled(a, args), r)),
                }
                for c, a, r in pending
            ],
        }
        log(f"group {gid}: {len(pending)} member(s), warm={warm}, stagger={stagger}s")
        if not pending:
            continue
        with cf.ThreadPoolExecutor(max_workers=len(pending) + 1) as pool:
            futs = [
                pool.submit(
                    do_call,
                    metas[c],
                    a,
                    r,
                    args,
                    group=group,
                    start_delay_s=0.0 if warm else k * stagger,
                )
                for k, (c, a, r) in enumerate(pending)
            ]
            coord = pool.submit(coordinate_warm, group, pending, args, stagger) if warm else None
            for f in cf.as_completed(futs):
                try:
                    f.result()
                except Exception as exc:  # noqa: BLE001
                    log(f"call raised: {type(exc).__name__}: {exc}")
            if coord is not None:
                try:
                    group["coordinator"] = coord.result()
                except Exception as exc:  # noqa: BLE001
                    group["coordinator"] = {"error": f"{type(exc).__name__}: {exc}"}
        group["t_end"] = time.time()
        atomic_write(GROUPS / f"{gid}.json", json.dumps(group, indent=1, default=str) + "\n")
        log(f"group {gid} done in {group['t_end'] - group['t_start']:.0f}s")


def _windows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        t = r.get("timings") or {}
        if not t or "t1" not in r:
            continue
        dm, pm = (t.get("predicted_ms") or 0) / 1000, (t.get("prompt_ms") or 0) / 1000
        out.append(
            {
                "seq": r.get("seq"),
                "prefill": (r["t1"] - dm - pm, r["t1"] - dm),
                "decode": (r["t1"] - dm, r["t1"]),
                "prompt_n": t.get("prompt_n") or 0,
                "predicted_n": t.get("predicted_n") or 0,
                "tps": t.get("predicted_per_second") or 0,
            }
        )
    return out


def group_metrics(g: dict) -> dict:
    mem = []
    wins: list[list[dict]] = []
    for m in g["members"]:
        wd = Path(m["dir"]) / "wire"
        rows = _wire_rows(wd)
        wins.append(_windows(rows))
        mem.append(
            {"call": f"{m['context']}/{m['arm']}/r{m['rep']}", "first_request": first_request(wd)}
        )
    warm_rows = _wire_rows(GROUPS / g["gid"] / "warm-wire")
    warm_w = _windows(warm_rows)
    cold_tok = sum((x["first_request"].get("prompt_n") or 0) for x in mem) + sum(
        w["prompt_n"] for w in warm_w
    )
    cold_s = sum((x["first_request"].get("prompt_s") or 0) for x in mem) + sum(
        w["prefill"][1] - w["prefill"][0] for w in warm_w
    )
    stalled, clean, stalled_rows = [], [], []
    for i, ws_ in enumerate(wins):
        others = [
            w["prefill"] for j, o in enumerate(wins) if j != i for w in o if w["prompt_n"] >= 512
        ]
        others += [w["prefill"] for w in warm_w if w["prompt_n"] >= 512]
        for w in ws_:
            a, b = w["decode"]
            if w["predicted_n"] < 200 or b - a <= 0 or not w["tps"]:
                continue
            ov = sum(max(0.0, min(b, y) - max(a, x)) for x, y in others)
            (stalled if ov / (b - a) >= 0.10 else clean).append(w["tps"])
            if ov / (b - a) >= 0.10:
                stalled_rows.append((w["predicted_n"], w["tps"]))
    med = lambda xs: round(sorted(xs)[len(xs) // 2], 2) if xs else None  # noqa: E731
    mc = med(clean)
    stall_s = round(sum(n / t - n / mc for n, t in stalled_rows if t and mc), 1) if mc else None
    return {
        "gid": g["gid"],
        "label": g.get("label"),
        "warm": g.get("warm"),
        "stagger_s": g.get("stagger_s"),
        "members": mem,
        "warm_requests": [
            {
                "seq": w["seq"],
                "prompt_n": w["prompt_n"],
                "prefill_s": round(w["prefill"][1] - w["prefill"][0], 1),
            }
            for w in warm_w
        ],
        "cold_prefill_tokens": cold_tok,
        "cold_prefill_s": round(cold_s, 1),
        "decode_tps_median_neighbour_prefill": med(stalled),
        "n_turns_stalled": len(stalled),
        "decode_tps_median_clean": mc,
        "n_turns_clean": len(clean),
        "est_stall_s": stall_s,
        "bodies_identical": (g.get("coordinator") or {}).get("bodies_identical"),
    }


def cmd_group_report(args) -> int:
    out = []
    for gf in sorted(GROUPS.glob("*.json")):
        try:
            g = json.loads(gf.read_text())
        except ValueError:
            continue
        if "members" not in g or (args.label and g.get("label") != args.label):
            continue
        out.append(group_metrics(g))
    for m in out:
        print(
            f"== {m['gid']} warm={m['warm']} stagger={m['stagger_s']} identical_prompts={m['bodies_identical']}"
        )
        for x in m["members"]:
            f = x["first_request"]
            print(
                f"   {x['call']}: first req cache_n={f.get('cache_n')} prompt_n={f.get('prompt_n')} "
                f"prefill={f.get('prompt_s')}s held={f.get('held_s')}s ttft_after_release={f.get('ttft_after_release_s')}s"
            )
        for w in m["warm_requests"]:
            print(f"   warm seq {w['seq']}: prompt_n={w['prompt_n']} prefill={w['prefill_s']}s")
        print(
            f"   cold prefill: {m['cold_prefill_tokens']} tok / {m['cold_prefill_s']} s; decode tok/s with a neighbour "
            f"prefilling {m['decode_tps_median_neighbour_prefill']} (n={m['n_turns_stalled']}) vs clean "
            f"{m['decode_tps_median_clean']} (n={m['n_turns_clean']}); est. stall {m['est_stall_s']} s"
        )
    GROUPS.mkdir(parents=True, exist_ok=True)
    atomic_write(GROUPS / f"report-{args.label or 'all'}.json", json.dumps(out, indent=1) + "\n")
    return 0


# ---------------------------------------------------------------------------------------------
# dry-run: exact commands + warm and compaction-rewrite bodies, replayed offline. Sends nothing.
# ---------------------------------------------------------------------------------------------
def _elide(obj, n: int = 160):
    if isinstance(obj, str):
        if len(obj) <= n:
            return obj
        return f"<{len(obj)} chars sha={_sha(obj)} head={obj[:70]!r} ... tail={obj[-50:]!r}>"
    if isinstance(obj, list):
        if len(obj) > 12 and all(isinstance(x, dict) for x in obj):
            return (
                [_elide(x, n) for x in obj[:3]]
                + [f"<... {len(obj) - 5} more ...>"]
                + [_elide(x, n) for x in obj[-2:]]
            )
        return [_elide(x, n) for x in obj]
    if isinstance(obj, dict):
        return {k: _elide(v, n) for k, v in obj.items()}
    return obj


def _recorded_bodies(call: Path) -> dict[int, dict]:
    txt = read_gz_tolerant(call / "wire" / "wire-bodies.jsonl.gz")
    out = {}
    for line in txt.splitlines():
        try:
            o = json.loads(line)
            out[o["seq"]] = o
        except (ValueError, KeyError):
            pass
    return out


_WS_RE = re.compile(r"/mnt/raid0/llm/tmp/ds41-c95/work/[0-9a-f]{8}")


class _DryWorkspace(Workspace):
    def __init__(self, meta: dict, root: Path):
        super().__init__(meta)
        self.root = root / "work" / uuid.uuid4().hex[:8]
        self.lane = self.root / "lane"
        self.bundle = self.root / "actor-context" / "planner-bundle"
        self.tmp = self.root / "tmp"
        self.codex_home = self.root / "codex-home"
        for p in (self.lane, self.bundle, self.tmp, self.codex_home):
            p.mkdir(parents=True, exist_ok=True)


def _replay(
    call: Path, label: str, *, keep_prefix: bool, pin_task: bool, show_seq: set | None = None
) -> None:
    rows = {r["seq"]: r for r in _wire_rows(call / "wire")}
    bodies = _recorded_bodies(call)
    real = None
    for o in bodies.values():
        m = _WS_RE.search(json.dumps(o["body"]))
        if m:
            real = m.group(0)
            break
    xf = proxy3.Transformer([(real, CANON_ROOT)] if real else [], keep_prefix, pin_task)
    print(f"\n## replay {call.relative_to(RESULTS)} ({label}); alias {real} -> {CANON_ROOT}")
    prev_tool = None
    pins_shown = 0
    for seq in sorted(bodies):
        o = bodies[seq]
        if not proxy3.is_model_path(o["path"]):
            continue
        before = o["body"]
        after, rec = xf.transform(before, o["path"])
        t = (rows.get(seq) or {}).get("timings") or {}
        if "task_pin" in rec and "compaction" not in rec:
            pins_shown += 1
            if pins_shown > 2:
                if before.get("tools") or []:
                    prev_tool = (seq, after)
                continue
        if "compaction" in rec or "task_pin" in rec:
            c = rec.get("compaction") or {}
            print(f"seq {seq}: {json.dumps({k: v for k, v in rec.items() if k != 'aliased_req'})}")
            print(
                f"   observed on the wire (unrewritten run): cache_n={t.get('cache_n')} prompt_n={t.get('prompt_n')} "
                f"prompt_ms={t.get('prompt_ms')} predicted_n={t.get('predicted_n')}"
            )
            if c.get("rewritten") and prev_tool is not None:
                pt = (rows.get(prev_tool[0]) or {}).get("timings") or {}
                kv_prev = (
                    (pt.get("cache_n") or 0)
                    + (pt.get("prompt_n") or 0)
                    + (pt.get("predicted_n") or 1)
                    - 1
                )
                pb = prev_tool[1]
                print(
                    f"   before: tools={len(before.get('tools') or [])} tool_choice={before.get('tool_choice')!r} "
                    f"parallel_tool_calls={before.get('parallel_tool_calls')!r}"
                )
                print(
                    f"   after : tools={len(after.get('tools') or [])} tool_choice={after.get('tool_choice')!r} "
                    f"parallel_tool_calls={after.get('parallel_tool_calls')!r}"
                )
                print(
                    f"   prefix check vs seq {prev_tool[0]} (last tool-bearing request): tools identical="
                    f"{after.get('tools') == pb.get('tools')}, instructions identical="
                    f"{after.get('instructions') == pb.get('instructions')}, first "
                    f"{c.get('prefix_items_shared')} input items identical (ids ignored)"
                )
                print(
                    f"   -> the slot that served seq {prev_tool[0]} holds ~{kv_prev} tokens of this prefix: expected "
                    f"cache_n ~{kv_prev} instead of the observed cold re-prefill of {t.get('prompt_n')} tokens"
                )
                if show_seq is None or seq in show_seq:
                    print("   rewritten body (elided):")
                    print(
                        "   "
                        + json.dumps(
                            _elide(
                                {
                                    k: after[k]
                                    for k in (
                                        "tools",
                                        "tool_choice",
                                        "parallel_tool_calls",
                                        "instructions",
                                        "stream",
                                    )
                                    if k in after
                                }
                            )
                        )[:2400]
                    )
                    print(
                        f"   input: {len(after.get('input') or [])} items; last = "
                        f"{json.dumps(_elide((after.get('input') or [{}])[-1]))[:400]}"
                    )
        if (before.get("tools") or []) and "compaction" not in rec:
            prev_tool = (seq, after)
    print(f"   (task_pin fired on {pins_shown} tool-bearing request(s) in total; first two shown)")


def cmd_dry_run(args) -> int:
    print(
        "# run3 dry-run: NOTHING is sent, no proxy/harness/server is started; a scratch dir is created and removed"
    )
    calls = build_calls(args)
    gpu_calls = [c for c in calls if ARMS[c[1]]["gpu"]]
    warm = bool(args.warm_prefix)
    groups = (
        form_groups(gpu_calls, args.concurrency)
        if (warm or args.group_starts)
        else [[c] for c in gpu_calls]
    )
    stagger = args.stagger_s if args.stagger_s is not None else (3.0 if warm else 0.0)
    print(
        f"calls: {len(gpu_calls)}; start groups: {[[f'{c}/{labelled(a, args)}/r{r}' for c, a, r in g] for g in groups]}"
    )
    print(
        f"options: {json.dumps(run3_config(args, gpu_calls[0][1] if gpu_calls else '', {'warm': warm, 'stagger_s': stagger}))}"
    )
    tmp = BASE / "_dryrun" / uuid.uuid4().hex[:8]
    saved = {k: globals()[k] for k in ("run_budgeted", "server_drafter")}
    try:
        print("\n## 1. per-member commands (first group)")
        for cid, arm, rep in groups[0] if groups else []:
            meta = load_meta(cid)
            d = tmp / "results" / cid / labelled(arm, args) / f"r{rep}"
            (d / "wire").mkdir(parents=True)
            ws = _DryWorkspace(meta, tmp)
            prompt = ws.prompt(ARMS[arm]["prompt"])
            extra = (
                ["--think-budget", str(F2_THINK_BUDGET), "--think-message", "<F2_THINK_MESSAGE>"]
                if ARMS[arm].get("f2")
                else []
            )
            extra += proxy3_flags(args, ws, {"warm": warm})
            print(f"\n### {cid}/{labelled(arm, args)}/r{rep}")
            print(
                "proxy :",
                " ".join(
                    [
                        sys.executable,
                        str(BASE / "proxy3.py"),
                        "--port",
                        "0",
                        "--log",
                        str(call_dir(cid, labelled(arm, args), rep) / "wire"),
                        "--upstream",
                        UPSTREAM,
                        "--effort-policy",
                        args.effort_policy,
                        *extra,
                    ]
                ).replace(str(tmp / "work"), str(WORK)),
            )
            _tl.codex_home_per_call = args.codex_home_per_call
            _tl.compact_limit = args.compact_limit
            _tl.wrap = None
            if args.sandbox == "landlock":
                _tl.wrap, sb = sandbox_wrap(meta, ws, d, prompt, args)
                summ = sandbox3.summarize(
                    sandbox3.compute(json.loads(Path(sb["policy"]).read_text()))
                )
                print(
                    f"sandbox: landlock ABI {summ['landlock_abi']}, {summ['n_rules']} rules, cutoff {sb['cutoff']}; "
                    f"denied: {summ['hard_and_tree_denied']}; soft-denied (mtime >= cutoff): {summ['soft_denied_n']}; "
                    f"allow overrides: {summ['allow']}"
                )
            recorded: list = []

            def rec_run(argv, *, cwd, env, stdin_text, out, err, budget_s):
                w = getattr(_tl, "wrap", None)
                recorded.append(([*(w or []), *argv], env.get("CODEX_HOME"), budget_s))
                return {"rc": 0, "wall_s": 0.0, "budget_exhausted": False}

            globals()["run_budgeted"] = rec_run
            globals()["server_drafter"] = lambda: {
                "n_ctx_slot": F1_WINDOW_MAX,
                "drafter": "dry-run: GET /slots not sent",
            }
            if arm in ("cxf1", "cxf12", "cxa"):
                arm_codex_f(
                    ws,
                    d,
                    prompt,
                    0,
                    args.budget,
                    args.salvage,
                    sandbox=args.codex_sandbox or "<--codex-sandbox>",
                    f2=bool(ARMS[arm].get("f2")),
                )
            elif arm in ("cx", "cxv"):
                arm_codex(
                    ws,
                    d,
                    prompt,
                    0,
                    args.budget,
                    args.salvage,
                    sandbox=args.codex_sandbox or "<--codex-sandbox>",
                    model=MODEL_CX,
                    reference=False,
                )
            else:
                print(
                    f"harness: (the {arm} argv is built by arm_{arm} at run time; it gets the same sandbox prefix)"
                )
            for argv, ch, bud in recorded[:1]:
                ch = str(ch).replace(str(tmp / "work"), str(WORK))
                print(
                    f"harness (budget {bud:.0f}s, CODEX_HOME={ch}):\n  "
                    + " ".join(argv)
                    .replace(str(tmp / "work"), str(WORK))
                    .replace(str(tmp / "results"), str(RESULTS))
                    + "  < prompt (stdin)"
                )
                print("  (the proxy port 0 above is the per-call proxy's port at run time)")
        globals().update(saved)

        print(
            "\n## 2. UFH14-A3 warm request (from recorded first requests of this context; aliased like proxy3 does)"
        )
        cid0 = groups[0][0][0] if groups else "C2"
        fam = (
            ["cxa", "cxf1", "cxf12", "cx"]
            if ARMS[groups[0][0][1]]["prompt"] == "prompt.inline.txt"
            else ["cxv"]
        )
        srcs = [
            p
            for a in fam
            for p in sorted((RESULTS / cid0).glob(f"{a}*/r*"))
            if (p / "wire" / "wire-bodies.jsonl.gz").exists()
        ]
        caps = []
        for src in srcs:
            bodies = _recorded_bodies(src)
            first = next(
                (
                    bodies[s]
                    for s in sorted(bodies)
                    if proxy3.is_model_path(bodies[s]["path"])
                    and (bodies[s]["body"].get("tools") or [])
                ),
                None,
            )
            if not first:
                continue
            m = _WS_RE.search(json.dumps(first["body"]))
            body = (
                proxy3.replace_all(first["body"], [(m.group(0), CANON_ROOT)])
                if m
                else first["body"]
            )
            caps.append((src, first["path"], body))
            if len(caps) == max(2, args.concurrency):
                break
        for src, path, body in caps:
            print(
                f"captured {src.relative_to(RESULTS)} seq-first: rendered sha {_sha(rendered_view(body))} "
                f"({len(json.dumps(body))} bytes)"
            )
        if len(caps) >= 2:
            same = rendered_view(caps[0][2]) == rendered_view(caps[1][2])
            print(
                f"rendered views identical after aliasing: {same}  <- the condition for a shared prefix"
            )
        if caps:
            print(
                f"\nGET {UPSTREAM}/slots                      # read-only: pick idle slot ids s0, s1, ..."
            )
            for k, (src, path, body) in enumerate(caps):
                wb = warm_body(body, path, id_slot=k)
                print(
                    f"POST http://127.0.0.1:<warm-proxy-port>{path}   # warm copy for member {k}, serial, "
                    f"pinned id_slot=<idle slot {k}> (shown as {k})"
                )
                if k == 0:
                    diff = {
                        kk: wb.get(kk)
                        for kk in (
                            "add_generation_prompt",
                            "max_output_tokens",
                            "max_tokens",
                            "id_slot",
                        )
                        if kk in wb and wb.get(kk) != body.get(kk)
                    }
                    print(f"  fields changed vs the captured body: {json.dumps(diff)}")
                    print("  body (elided): " + json.dumps(_elide(wb))[:3800])
            order = list(reversed(range(len(caps))))
            print(
                f"then touch <member wire>/RELEASE in order {order} with {stagger}s between "
                "(the last-warmed copy is still in its slot; earlier copies were moved to --cache-ram "
                "by the idle-slot save at the next launch, and are loaded from there)"
            )

        print(
            "\n## 3. UFH14-A4 compaction rewrite + D3 task pin, replayed on recorded run-1 wire bodies"
        )
        for call, lab in (
            (RESULTS / "C4" / "cx" / "r1", "codex: full-history compaction with tools=[]"),
            (RESULTS / "C4" / "oc" / "r1", "opencode: separate summarizer, task dropped"),
        ):
            if (call / "wire" / "wire-bodies.jsonl.gz").exists():
                _replay(call, lab, keep_prefix=True, pin_task=True)
    finally:
        globals().update(saved)
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
            try:
                (BASE / "_dryrun").rmdir()
            except OSError:
                pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "cmd",
        choices=(
            "plan",
            "run",
            "smoke",
            "status",
            "finalize",
            "orsv-check",
            "dry-run",
            "group-report",
        ),
    )
    ap.add_argument(
        "--after-pid",
        type=int,
        default=0,
        help="run: wait until this (running driver) pid exits, finalize its calls, then start",
    )
    ap.add_argument("--contexts", default=",".join(DEFAULT_CONTEXTS))
    ap.add_argument("--arms", default="oc,cx,hm")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--r2", default="C1", help="contexts that get a 2nd repeat when --repeats 1")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--budget", type=float, default=PLANNER_BUDGET_S)
    ap.add_argument("--salvage", type=float, default=SALVAGE_BUDGET_S)
    ap.add_argument(
        "--deadline-hours",
        type=float,
        default=6.5,
        help="no NEW call starts after this; running calls finish (max +1.5 h)",
    )
    ap.add_argument("--effort-policy", choices=("strip", "passthrough"), default="strip")
    ap.add_argument(
        "--codex-sandbox",
        choices=("read-only", "workspace-write", "danger-full-access"),
        default=None,
        help="REQUIRED for cx/sol arms; operator decision (see README)",
    )
    ap.add_argument("--arm", help="smoke: which arm")
    ap.add_argument(
        "--smoke-context", default="C4", help="smoke: whose lane to run in (no prompt used)"
    )
    ap.add_argument(
        "--sandbox-probe",
        action="store_true",
        help="smoke: plant a live campaign read and require own workspace/result writes",
    )
    # ---- run 3
    ap.add_argument("--label", default=None, help="variant label: results/<ctx>/<arm>+<label>/r<k>")
    ap.add_argument(
        "--warm-prefix",
        action="store_true",
        help="UFH14-A3: start groups, hold first requests, serial per-member warm copies, staggered release",
    )
    ap.add_argument(
        "--group-starts",
        action="store_true",
        help="start calls in groups of --concurrency (the cold A3 baseline when used without --warm-prefix)",
    )
    ap.add_argument(
        "--stagger-s",
        type=float,
        default=None,
        help="seconds between member releases (warm; default 3) or member starts (cold; default 0)",
    )
    ap.add_argument("--canon-paths", dest="canon_paths", action="store_true", default=True)
    ap.add_argument("--no-canon-paths", dest="canon_paths", action="store_false")
    ap.add_argument("--keep-prefix-compaction", action="store_true", help="UFH14-A4 (proxy3)")
    ap.add_argument(
        "--pin-task",
        action="store_true",
        help="D3: restore a dropped/truncated task message (proxy3)",
    )
    ap.add_argument(
        "--compact-limit",
        type=int,
        default=0,
        help="codex model_auto_compact_token_limit for cxa/cxf* (0 = arm default)",
    )
    ap.add_argument("--sandbox", choices=("landlock", "none"), default="landlock")
    ap.add_argument(
        "--sandbox-tmp",
        choices=("deny", "allow"),
        default="deny",
        help="the shared /tmp (prior calls left champion disassembly there); TMPDIR stays private",
    )
    ap.add_argument(
        "--codex-home-per-call", dest="codex_home_per_call", action="store_true", default=True
    )
    ap.add_argument("--shared-codex-home", dest="codex_home_per_call", action="store_false")
    ap.add_argument("--keep", action="store_true", help="dry-run: keep its scratch dir")
    args = ap.parse_args()
    if args.cmd == "dry-run":
        return cmd_dry_run(args)
    if args.cmd == "group-report":
        return cmd_group_report(args)
    if args.cmd == "plan":
        return cmd_plan(args)
    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd == "finalize":
        return cmd_finalize(args)
    if args.cmd == "orsv-check":
        if (BASE / "HOLD_LAUNCHER").exists() and os.environ.get("C95_NO_HOLD") != "1":
            # launch_arms2.sh (already running when orv had to be re-planned) calls this right
            # after its cxv run. Holding here defers its ALL ARMS DONE + grading until
            # rerun_orv.sh has finished (it touches RERUN_DONE). Release by hand: touch RERUN_DONE.
            log("launch_arms2 HELD: orv/orsv were moved to rerun_orv.sh; waiting for RERUN_DONE")
            t_end = time.time() + 12 * 3600
            while not (BASE / "RERUN_DONE").exists() and time.time() < t_end:
                time.sleep(60)
            log(
                "launch_arms2 released: orv/orsv were run (or skipped) by rerun_orv.sh -- "
                "the 'orsv skipped' line that follows refers to this launcher only"
            )
            return 1
        r = orsv_deployed()
        print(json.dumps(r))
        return 0 if r["deployed"] else 1
    if args.cmd == "smoke":
        if args.sandbox_probe and args.sandbox != "landlock":
            print("--sandbox-probe requires --sandbox landlock", file=sys.stderr)
            return 2
        if args.arm in ("cx", "cxv", "sol", "cxf1", "cxf12") and not args.codex_sandbox:
            print("smoke cx/sol needs --codex-sandbox", file=sys.stderr)
            return 2
        return cmd_run(args, smoke_arm=args.arm)
    return cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
