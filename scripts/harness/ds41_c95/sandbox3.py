#!/usr/bin/env python3
"""DS41-C95 run 3: Landlock filesystem sandbox for the planner harness (sandbox3.py).

WHY LANDLOCK (probed 2026-10-03 in this container):
  * mount namespaces are unavailable: `unshare -Ur true` -> "write failed /proc/self/uid_map:
    Operation not permitted". The host sets kernel.apparmor_restrict_unprivileged_userns=1, so
    every unprivileged user namespace lands in the AppArmor profile `unprivileged_userns
    (enforce)` with CapEff=0: no uid_map, no setgroups, no mount. bwrap is not installed, and
    codex's own bwrap sandbox fails for the same reason (README prerequisite 3). A tmpfs over
    the campaign dir is therefore impossible without operator/root changes.
  * a separate uid needs root (no sudo here).
  * Landlock needs no privilege and no namespace: `landlock_create_ruleset(NULL,0,VERSION)`
    returns ABI 6 here. It is inherited by every child (codex -> bash -> tools), it cannot be
    lifted by the sandboxed process, and network access is left untouched (the harness still
    reaches its wire proxy on 127.0.0.1).

POLICY MODEL (Landlock is allow-list only, so "deny X" = "allow every sibling on the path to X"):
  * hard_deny: never readable, never writable (live campaign store, shared git object stores,
    the governance repos and agent memories that describe post-anchor keeps, other C95 calls'
    results/grading/answer keys, the shared /tmp that earlier calls littered with champion
    disassembly, ...). Children of a hard-denied directory are not enumerated.
  * soft_deny_roots: children of these directories whose mtime is at/after the context's as-ran
    start ("cutoff") are denied -- a written-after-the-anchor heuristic for scratch areas.
  * tree_deny: whole trees denied (experimental llama.cpp worktrees) that a prompt-referenced
    path may re-open read-only, but only if everything at that path predates the cutoff.
  * allow: explicit per-call overrides ({"path", "mode": ro|rw}) inside denied areas: the
    call's own workspace (lane worktree, bundle, TMPDIR, per-call CODEX_HOME), its result dir,
    the scratch clone's .git, the relocated campaign copy, the harness code, the codex binary.
  * prompt_paths: paths named by the call's own prompt; they re-open soft/tree denials only.
  Everything else keeps full access, so the harness and its tools behave as before.

KNOWN LIMITS: listing an ANCESTOR of a denied path fails (e.g. `ls /mnt/raid0/llm`, `ls /`),
because Landlock grants directory listing beneath a rule, never for one directory alone;
traversal, stat and getcwd are not restricted. A new top-level entry created after the
ruleset is applied is not reachable. /proc is readable (other processes' argv).

Subcommands:
  exec --policy P.json [--report R.json] -- ARGV...   apply the policy, then execvp ARGV
  explain --policy P.json                             print the computed rule summary
  selftest                                            harmless proof: denied vs allowed accesses
Stdlib only.
"""

from __future__ import annotations

import argparse
import ctypes
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

BASE = Path(os.environ.get("DS41_C95_BASE", "/mnt/raid0/llm/tmp/ds41-c95"))


def emit_preexposure(directory: Path, record: dict) -> dict:
    """Publish one immutable-by-contract JSON blob before any model exposure.

    The final name is created with an exclusive hard link, so an existing record
    cannot be overwritten. A later result may cite the returned digest only.
    """
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / "preexposure-provenance.json"
    blob = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    tmp = directory / (".preexposure-" + uuid.uuid4().hex + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(blob)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp, final)
        dirfd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    finally:
        tmp.unlink(missing_ok=True)
    return {"path": str(final), "sha256": hashlib.sha256(blob).hexdigest()}


def claim_fresh_output(parent: Path, label: str) -> Path:
    """Claim a new call directory before touching any prior call artifacts."""
    parent.mkdir(parents=True, exist_ok=True)
    out = parent / label
    out.mkdir(exist_ok=False)
    return out


# --------------------------------------------------------------------------------------------
# Landlock ABI (x86_64 syscall numbers; uapi/linux/landlock.h)
# --------------------------------------------------------------------------------------------
SYS_CREATE, SYS_ADD_RULE, SYS_RESTRICT = 444, 445, 446
CREATE_RULESET_VERSION = 1
RULE_PATH_BENEATH = 1
PR_SET_NO_NEW_PRIVS = 38

EXECUTE, WRITE_FILE, READ_FILE, READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
REMOVE_DIR, REMOVE_FILE, MAKE_CHAR, MAKE_DIR = 1 << 4, 1 << 5, 1 << 6, 1 << 7
MAKE_REG, MAKE_SOCK, MAKE_FIFO, MAKE_BLOCK, MAKE_SYM = 1 << 8, 1 << 9, 1 << 10, 1 << 11, 1 << 12
REFER, TRUNCATE, IOCTL_DEV = 1 << 13, 1 << 14, 1 << 15


def fs_rights_for_abi(abi: int) -> int:
    r = (1 << 13) - 1  # ABI 1: bits 0..12
    if abi >= 2:
        r |= REFER
    if abi >= 3:
        r |= TRUNCATE
    if abi >= 5:
        r |= IOCTL_DEV
    return r


FILE_ONLY = EXECUTE | WRITE_FILE | READ_FILE | TRUNCATE | IOCTL_DEV
RO = EXECUTE | READ_FILE | READ_DIR


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]  # net/scoped left unhandled (8-byte attr)


class _PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


_libc = ctypes.CDLL(None, use_errno=True)
_libc.syscall.restype = ctypes.c_long


def landlock_abi() -> int:
    v = _libc.syscall(SYS_CREATE, None, ctypes.c_size_t(0), ctypes.c_uint32(CREATE_RULESET_VERSION))
    return int(v) if v > 0 else 0


# --------------------------------------------------------------------------------------------
# policy -> rules
# --------------------------------------------------------------------------------------------
def _real(p: str) -> str:
    return os.path.realpath(os.path.expanduser(p))


def _under(p: str, root: str) -> bool:
    return p == root or p.startswith(root.rstrip("/") + "/")


def _mtime(p: str) -> float:
    try:
        return os.lstat(p).st_mtime
    except OSError:
        return 0.0


def compute(policy: dict) -> dict:
    """Return {"rules": [(path, rights)], "denied": [...], "soft_denied": [...], "notes": [...]}"""
    abi = landlock_abi()
    allr = fs_rights_for_abi(abi or 6)
    cutoff = float(policy.get("cutoff") or 0)
    hard = {_real(p) for p in policy.get("hard_deny", [])}
    # glob-style tree denials (e.g. /mnt/raid0/llm/llama.cpp-experimental*)
    trees: set[str] = set()
    for pat in policy.get("tree_deny", []):
        d = os.path.dirname(pat)
        try:
            for e in os.scandir(d):
                if fnmatch.fnmatch(e.path, pat) and not e.is_symlink():
                    trees.add(_real(e.path))
        except OSError:
            pass
    notes: list[str] = []
    soft: set[str] = set()
    for sr in policy.get("soft_deny_roots", []):
        root, exempt = _real(sr["dir"]), set(sr.get("exempt", []))
        try:
            for e in os.scandir(root):
                if e.name in exempt or e.is_symlink():
                    continue
                if cutoff and _mtime(e.path) >= cutoff:
                    soft.add(_real(e.path))
        except OSError as exc:
            notes.append(f"soft root {root}: {exc}")
    # explicit overrides
    allow: list[tuple[str, int]] = []
    for a in policy.get("allow", []):
        p = _real(a["path"])
        if os.path.exists(p):
            allow.append((p, allr if a.get("mode") == "rw" else RO))
        else:
            notes.append(f"allow path missing: {p}")
    # prompt-referenced paths re-open soft/tree denials (never hard ones)
    for raw in policy.get("prompt_paths", []):
        p = _real(raw)
        if not os.path.exists(p) or any(_under(p, h) for h in hard):
            continue
        in_tree = any(_under(p, t) for t in trees)
        in_soft = any(_under(p, s) for s in soft)
        if not (in_tree or in_soft):
            continue
        if os.path.isdir(p):
            try:
                ents = list(os.scandir(p))
            except OSError:
                ents = []
            if all(_mtime(e.path) < cutoff for e in ents) and _mtime(p) < cutoff:
                allow.append((p, RO))
            else:
                n_old = 0
                for e in ents:
                    if e.is_file(follow_symlinks=False) and _mtime(e.path) < cutoff:
                        allow.append((_real(e.path), RO))
                        n_old += 1
                notes.append(
                    f"prompt dir {p} has post-cutoff entries: re-opened {n_old} older files only"
                )
        elif _mtime(p) < cutoff:
            # a named binary needs its sibling libraries: re-open the parent directory too when
            # it and everything in it predates the cutoff (e.g. build-cpu/bin of 09-25)
            parent = os.path.dirname(p)
            try:
                pents = list(os.scandir(parent))
            except OSError:
                pents = []
            if (
                in_tree
                and pents
                and _mtime(parent) < cutoff
                and all(_mtime(e.path) < cutoff for e in pents)
            ):
                allow.append((parent, RO))
            else:
                allow.append((p, RO))
        else:
            notes.append(f"prompt path {p} modified after cutoff: stays denied")
    deny = hard | trees | soft

    rules: list[tuple[str, int]] = []
    deny_list = sorted(deny)

    def has_denied_below(p: str) -> bool:
        pre = p.rstrip("/") + "/"
        return any(d.startswith(pre) for d in deny_list)

    def visit(p: str) -> None:
        if p in deny:
            return
        if not has_denied_below(p):
            rules.append((p, allr))
            return
        try:
            ents = list(os.scandir(p))
        except OSError as exc:
            notes.append(f"scandir {p}: {exc}")
            return
        for e in ents:
            if e.is_symlink():
                continue  # reached through its target, which is judged on its own path
            visit(e.path)

    visit("/")
    rules.extend(allow)
    return {
        "abi": abi,
        "rules": rules,
        "denied": sorted(hard | trees),
        "soft_denied": sorted(soft),
        "allow": allow,
        "notes": notes,
    }


def summarize(c: dict) -> dict:
    return {
        "landlock_abi": c["abi"],
        "n_rules": len(c["rules"]),
        "hard_and_tree_denied": c["denied"],
        "soft_denied_n": len(c["soft_denied"]),
        "soft_denied_sample": c["soft_denied"][:25],
        "allow": [(p, "rw" if r & WRITE_FILE else "ro") for p, r in c["allow"]],
        "notes": c["notes"][:40],
    }


def apply(c: dict) -> None:
    abi = c["abi"]
    if abi <= 0:
        raise OSError("Landlock unavailable")
    allr = fs_rights_for_abi(abi)
    attr = _RulesetAttr(allr)
    rfd = _libc.syscall(
        SYS_CREATE, ctypes.byref(attr), ctypes.c_size_t(ctypes.sizeof(attr)), ctypes.c_uint32(0)
    )
    if rfd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    try:
        for path, rights in c["rules"]:
            try:
                fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
            except OSError:
                continue
            try:
                isdir = os.path.isdir(f"/proc/self/fd/{fd}")
                pb = _PathBeneath((rights & allr) if isdir else (rights & allr & FILE_ONLY), fd)
                if (
                    _libc.syscall(
                        SYS_ADD_RULE,
                        ctypes.c_int(rfd),
                        ctypes.c_int(RULE_PATH_BENEATH),
                        ctypes.byref(pb),
                        ctypes.c_uint32(0),
                    )
                    < 0
                ):
                    err = ctypes.get_errno()
                    raise OSError(err, f"landlock_add_rule {path}: {os.strerror(err)}")
            finally:
                os.close(fd)
        if _libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(NO_NEW_PRIVS)")
        if _libc.syscall(SYS_RESTRICT, ctypes.c_int(rfd), ctypes.c_uint32(0)) < 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    finally:
        os.close(rfd)


# --------------------------------------------------------------------------------------------
# the C95 policy (run3 builds one per call)
# --------------------------------------------------------------------------------------------
HARD_DENY = [
    "/mnt/raid0/llm/autokernel/campaigns",  # the LIVE campaign store (F12 C2/cxf1/r2 browsed it)
    "/mnt/raid0/llm/llama.cpp/.git",  # shared refs+objects of every llama.cpp worktree (all keeps)
    "/mnt/raid0/llm/worktrees",  # other sessions' worktrees (post-anchor sources)
    "/mnt/raid0/llm/epyc-root",
    "/mnt/raid0/llm/epyc-orchestrator",
    "/mnt/raid0/llm/epyc-inference-research",
    "/mnt/raid0/llm/epyc-web",
    "/workspace",  # = epyc-root (handoffs/progress name every keep)
    "/mnt/raid0/llm/claude",
    "/mnt/raid0/llm/claude-backups",
    "/home/node/.claude",
    "/home/node/.codex",  # operator codex: live-loop sessions/memories (binary re-allowed)
    "/home/node/.local/share/opencode",  # shared 10 GB opencode.db (live-loop transcripts)
    str(BASE),  # README/results/grading keys/diagnosis (per-call parts re-allowed)
]
TREE_DENY = ["/mnt/raid0/llm/llama.cpp-experimental*"]  # live/experimental source trees
SOFT_ROOTS = [
    {"dir": "/mnt/raid0/llm", "exempt": ["models", "tmp", "autokernel", "kernels"]},
    {"dir": "/mnt/raid0/llm/tmp", "exempt": []},
    {"dir": "/mnt/raid0/llm/autokernel", "exempt": []},
    {"dir": "/mnt/raid0/llm/autokernel/loop-memory", "exempt": []},
]
CODEX_PACKAGES = "/home/node/.codex/packages"
HOOKS_DIR = "/workspace/scripts/hooks"  # codex-home's PreToolUse containment hook


def iso_epoch(ts: str) -> float:
    import calendar

    return float(calendar.timegm(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")))


def c95_policy(
    *,
    cutoff: float,
    ws_root: Path,
    call_dir: Path,
    scratch_clone: Path,
    context_id: str,
    prompt_text: str = "",
    tmp: str = "deny",
    extra_allow: list[dict] | None = None,
) -> dict:
    import re

    meta = json.loads((BASE / "contexts" / context_id / "meta.json").read_text())
    anchor_build = meta["anchor_build"]
    if not re.fullmatch(r"anchor-gen-[0-9]{3}", anchor_build):
        raise ValueError(f"invalid anchor build: {anchor_build!r}")
    hard = list(HARD_DENY) + (["/tmp", "/var/tmp"] if tmp == "deny" else [])
    allow = [
        {"path": str(ws_root), "mode": "rw"},  # lane worktree, bundle, TMPDIR, CODEX_HOME
        {"path": str(call_dir), "mode": "rw"},  # codex -o last.txt, hermes OUT_DIR
        {"path": str(scratch_clone), "mode": "ro"},
        {"path": str(scratch_clone / ".git"), "mode": "rw"},  # worktree metadata, objects
        # The relocated copy contains later anchors too (C2's 006 sits beside 007).
        # Grant only this context's anchor and the two immutable input records.
        {"path": str(BASE / "ctx-campaign" / "store" / anchor_build), "mode": "ro"},
        {
            "path": str(BASE / "ctx-campaign" / "inputs" / "ds41-cpu-t48-dspark.recipe.json"),
            "mode": "ro",
        },
        {
            "path": str(BASE / "ctx-campaign" / "inputs" / "ds41-decode.prompt-manifest.json"),
            "mode": "ro",
        },
        {"path": str(BASE / "contexts" / context_id), "mode": "ro"},
        {"path": str(BASE / "hermes"), "mode": "ro"},
        {"path": CODEX_PACKAGES, "mode": "ro"},
        {"path": HOOKS_DIR, "mode": "ro"},
        *(extra_allow or []),
    ]
    paths = sorted(
        {
            m.rstrip(".,:;")
            for m in re.findall(
                r"/(?:mnt|home|workspace|usr|opt)/[^\s`'\"<>()\[\]{},;|*]+", prompt_text
            )
        }
    )
    return {
        "version": 1,
        "cutoff": cutoff,
        "hard_deny": hard,
        "tree_deny": TREE_DENY,
        "soft_deny_roots": SOFT_ROOTS,
        "allow": allow,
        "prompt_paths": paths,
    }


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def cmd_exec(a) -> int:
    policy = json.loads(Path(a.policy).read_text())
    c = compute(policy)
    if a.report:
        Path(a.report).write_text(
            json.dumps({"applied_at": time.time(), "pid": os.getpid(), **summarize(c)}, indent=1)
            + "\n"
        )
    apply(c)
    argv = a.argv[1:] if a.argv and a.argv[0] == "--" else a.argv
    os.execvp(argv[0], argv)
    return 127


def cmd_explain(a) -> int:
    c = compute(json.loads(Path(a.policy).read_text()))
    print(json.dumps(summarize(c), indent=1))
    return 0


def cmd_selftest(a) -> int:
    """Harmless proof on a throw-away call layout under BASE/sbxtest-<id> (removed afterwards):
    run a battery of shell probes under the C95 policy and print expected vs observed."""
    tid = uuid.uuid4().hex[:8]
    root = BASE / f"sbxtest-{tid}"
    ws, call = root / "ws", root / "call"
    for p in (ws / "tmp", call):
        p.mkdir(parents=True)
    lane = BASE / "lanes" / "053c3bd82"  # read-only use of a real scratch clone
    meta = json.loads((BASE / "contexts" / "C4" / "meta.json").read_text())
    cutoff = iso_epoch(meta["asran_started"])
    prompt = (BASE / "contexts" / "C4" / "prompt.inline.txt").read_text()
    pol = c95_policy(
        cutoff=cutoff,
        ws_root=ws,
        call_dir=call,
        scratch_clone=lane,
        context_id="C4",
        prompt_text=prompt,
    )
    pf = root / "policy.json"
    pf.write_text(json.dumps(pol, indent=1))
    store = "/mnt/raid0/llm/autokernel/campaigns/ak-ds41-cpu-decode-20260923/store"
    leak_file = next(
        (e.path for e in os.scandir(f"{store}/anchor-gen-007") if e.is_file()), "/nonexistent"
    )
    probes = [  # (expect_ok, label, shell)
        (False, "ls live campaign dir", "ls /mnt/raid0/llm/autokernel/campaigns"),
        (False, "ls live store anchor-gen-007 (the F12 leak)", f"ls {store}/anchor-gen-007"),
        (False, "read a live-store file", f"head -c 64 {leak_file} | wc -c"),
        (
            False,
            "git log in production .git (shared refs)",
            "git -C /mnt/raid0/llm/llama.cpp log -1 --oneline",
        ),
        (
            False,
            "read experimental tree source (future HEAD)",
            "head -c 64 /mnt/raid0/llm/llama.cpp-experimental-fastload-ds41-20260925/ggml/src/ggml-cpu/llamafile/sgemm.cpp",
        ),
        (False, "read C95 README (answer table)", f"head -c 64 {BASE}/README.md"),
        (False, "list C95 grading keys", f"ls {BASE}/grading"),
        (False, "read epyc-root handoffs via /workspace", "ls /workspace/handoffs/active"),
        (False, "ls shared /tmp (prior calls' champion asm)", "ls /tmp"),
        (False, "write shared /tmp", f"touch /tmp/c95-sbx-{tid} && rm -f /tmp/c95-sbx-{tid}"),
        (False, "read operator codex sessions", "ls /home/node/.codex/sessions"),
        (True, "ls own lane (scratch clone, ro)", f"ls {lane} | head -3"),
        (True, "git log in own lane", f"git -C {lane} log -1 --format=%h"),
        (True, "read own context copy", f"head -c 40 {BASE}/contexts/C4/meta.json"),
        (True, "read relocated campaign copy", f"ls {BASE}/ctx-campaign/store | head -3"),
        (
            True,
            "write+compile in own workspace",
            f"cd {ws} && printf 'int main(){{return 0;}}\\n' > t.c && cc -O0 -o t t.c && ./t && rm t t.c",
        ),
        (True, "write TMPDIR", f"touch {ws}/tmp/x && rm {ws}/tmp/x"),
        (True, "write own result dir", f"touch {call}/last.txt"),
        (
            True,
            "prompt-named baseline binary (pre-cutoff)",
            "head -c 4 /mnt/raid0/llm/llama.cpp-experimental-fastload-ds41-20260925/build-cpu/bin/llama-server | od -c | head -1",
        ),
        (
            True,
            "codex binary readable+executable bit",
            "test -x /home/node/.codex/packages/app-server-daemon/current/bin/codex && echo ok",
        ),
        (
            True,
            "containment hook readable",
            f"head -c 20 {HOOKS_DIR}/codex_filesystem_containment.py",
        ),
        (True, "models dir readable", "ls /mnt/raid0/llm/models | head -2"),
        (True, "python3 runs", "python3 -c 'import json,ssl; print(1)'"),
    ]
    rows, ok_all = [], True
    env = {**os.environ, "TMPDIR": str(ws / "tmp")}
    for exp, label, sh in probes:
        sh = "set -o pipefail; " + sh
        # control: the same probe WITHOUT the sandbox must succeed, or a denial proves nothing
        ctl = subprocess.run(
            ["bash", "-c", sh], capture_output=True, text=True, timeout=60, env=env, cwd=str(ws)
        )
        r = subprocess.run(
            [sys.executable, __file__, "exec", "--policy", str(pf), "--", "bash", "-c", sh],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            cwd=str(ws),
        )
        got = r.returncode == 0
        good = (got == exp) and ctl.returncode == 0
        ok_all &= good
        detail = (r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or [""])[-1][:100]
        verdict = "PASS" if good else ("VACUOUS" if ctl.returncode != 0 else "FAIL")
        rows.append(
            (
                verdict,
                ("allowed" if exp else "denied "),
                label,
                ctl.returncode,
                r.returncode,
                detail,
            )
        )
    print(f"landlock ABI {landlock_abi()}; policy {pf}")
    for row in rows:
        print(
            "%-7s expect %s  ctl_rc=%s sbx_rc=%-3s %-46s | %s"
            % (row[0], row[1], row[3], row[4], row[2], row[5])
        )
    c = compute(pol)
    print(
        json.dumps({k: v for k, v in summarize(c).items() if k != "soft_denied_sample"}, indent=1)[
            :3000
        ]
    )
    if not a.keep:
        shutil.rmtree(root, ignore_errors=True)
    print("SELFTEST", "OK" if ok_all else "FAILED")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("exec")
    e.add_argument("--policy", required=True)
    e.add_argument("--report")
    e.add_argument("argv", nargs=argparse.REMAINDER)
    x = sub.add_parser("explain")
    x.add_argument("--policy", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    return {"exec": cmd_exec, "explain": cmd_explain, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
