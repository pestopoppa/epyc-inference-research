#!/usr/bin/env python3
"""R23-64 CPU decode thread re-sweep (t48 / t64 / t96): plan, argv, env and proofs.

ONE source of truth for everything the shell scripts need, so no flag is retyped in bash:

  plan            the rotated launch order (round x arm), one line per launch
  serve-files     the exact argv + env for ONE launch, written as <prefix>.argv / <prefix>.env
  knob-proof      `strings -a` proof of which GGML_* knobs the build really contains
  flag-check      dry-run every flag form against the real llama-server (loads no model)
  artifacts       presence + size of the trunk shards and the MTP head
  files           the files to evict from the page cache before a load
  ram-gib         the private-RAM figure to hand to ram_admit.sh
  meta            write a small JSON record of a launch

DESIGN (handoffs/active/autokernel-rebuild-program.md, R23-64): 3 arms x 6 launches, UNIT = LAUNCH.
-t is a launch-time server flag, so the admissible floor is a LAUNCH floor (FLOOR-UNIT-1).
Everything except -t (and, for the optional 4th arm, the taskset mask) is the champion recipe,
imported from scripts/lib/qwen38_flash_next_recipe.py -- "the recipe is DATA, import the
constants, do not retype them".

THE KNOB RULE (CLAUDE.md, "Process Management"): a null from a knob that is not in the binary is
not evidence about the knob. `knob-proof` therefore runs `strings -a` over the build and refuses
unless every knob the recipe exports is compiled in, and unless GGML_KNOB_FILE (the HARNESS-1
runtime knob page, absent from the 141e4bcf5 champion) is ABSENT -- which is what makes NO_KNOBS=1
(skip knobs.py) the correct mode rather than a shortcut.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

HERE = Path(__file__).resolve().parent
_LIB = HERE.parents[1] / "lib"            # scripts/lib
if str(_LIB) not in sys.path:
    sys.path.insert(0, str(_LIB))
import qwen38_flash_next_recipe as R      # noqa: E402  (the codified recipe; do not retype it)

# ---------------------------------------------------------------------------
# The champion this sweep is pinned to. Both are checked at launch time, never assumed.
# ---------------------------------------------------------------------------
CHAMP_BINDIR = "/mnt/raid0/llm/kernels/builds/cpu-20261023-141e4bcf5-lbh/bin"
EXPECT_VERSION = "10412 (141e4bcf5)"
PORT = 18499                    # same scratch port the INF-70 sessions used
CTX = R.CONTEXT                 # 8192: the context the recipe was measured at
FULL_CPUS = "0-95"              # R.SERVE_PREFIX[2]
MASKED_CPUS = "0-47"            # the optional 6-CCD arm
CONTROL_ARM = "t48"
MASKED_ARM = "t48m"
ROUNDS_DEFAULT = 6

# Knobs the recipe exports and the library each one lives in on the 141e4bcf5 build
# (strings-proven 2026-10-23; knob-proof re-proves it per run and does not trust this table).
REQUIRED_KNOBS = ("GGML_IQK", "GGML_FUSED_DECODE_OFF", "GGML_FA_SPLIT_KV", "GGML_NOHUGEPAGE_PROCESS")
FORBIDDEN_KNOBS_WHEN_NO_KNOBS = ("GGML_KNOB_FILE",)

# RAM model (GiB). Observed overhead beyond the GGUF bytes for the previous trunk at -c 8192:
# Rss 95.3 GiB against 94.23 GiB of files = +1.07 GiB (retest1 PB_CM1.smaps / evict.1).
KV_KIB_PER_TOKEN = 24 + 9       # registry serving_shape: measured f16 KV + observed MTP-head KV
RAM_OVERHEAD_GIB = 1.5


@dataclass(frozen=True)
class Arm:
    name: str
    threads: int
    cpus: str


ARMS = (
    Arm("t48", 48, FULL_CPUS),
    Arm("t64", 64, FULL_CPUS),
    Arm("t96", 96, FULL_CPUS),
)
MASKED = Arm(MASKED_ARM, 48, MASKED_CPUS)


def arm_set(include_masked: bool = False) -> list[Arm]:
    return list(ARMS) + ([MASKED] if include_masked else [])


def arm_by_name(name: str) -> Arm:
    for a in arm_set(True):
        if a.name == name:
            return a
    raise KeyError(f"unknown arm {name!r}")


def rotated_order(round_no: int, include_masked: bool = False) -> list[Arm]:
    """Cyclic rotation by round: no arm owns a fixed slot, so slow host drift cannot pose as an
    arm effect. Three arms give exactly 48 64 96 / 64 96 48 / 96 48 64 (round 1, 2, 3; then repeat)."""
    if round_no < 1:
        raise ValueError("rounds are 1-based")
    base = arm_set(include_masked)
    k = (round_no - 1) % len(base)
    return base[k:] + base[:k]


def label(arm: str, round_no: int) -> str:
    return f"{arm}_r{round_no}"


LABEL_RE = re.compile(r"^(?P<arm>t\d+m?)_r(?P<round>\d+)$")


def parse_label(lab: str) -> tuple[str, int]:
    m = LABEL_RE.match(lab)
    if not m:
        raise ValueError(f"not a sweep launch label: {lab!r}")
    return m.group("arm"), int(m.group("round"))


@dataclass(frozen=True)
class Launch:
    round_no: int
    arm: Arm

    @property
    def label(self) -> str:
        return label(self.arm.name, self.round_no)


def plan(rounds: int = ROUNDS_DEFAULT, include_masked: bool = False) -> list[Launch]:
    return [Launch(r, a) for r in range(1, rounds + 1) for a in rotated_order(r, include_masked)]


# ---------------------------------------------------------------------------
# argv / env: the recipe, with ONLY -t (and the optional taskset mask) varying per arm.
# ---------------------------------------------------------------------------
def _replace_value(cmd: list[str], flag: str, value: str, start: int = len(R.SERVE_PREFIX)) -> None:
    """Replace the value after the FIRST `flag` at or beyond `start`. The default start skips the
    taskset/numactl prefix, which itself contains a `-c` (the cpu mask) -- a bare cmd.index("-c")
    would overwrite the mask with the context size."""
    i = cmd.index(flag, start)
    cmd[i + 1] = value


def serve_argv(bindir: str, threads: int, cpus: str = FULL_CPUS, port: int = PORT, ctx: int = CTX,
               trunk: str = R.TRUNK_GGUF, head: str = R.MTP_HEAD_GGUF,
               ubatch: Optional[int] = None) -> list[str]:
    """taskset -c <cpus> numactl --interleave=all <bindir>/llama-server <recipe flags> with -t <threads>.

    Built by the recipe's own build_serve_command(), then ONLY these fields are edited: -t, the
    taskset mask, -c, the draft-model path, and an optional -ub. Every validator the recipe
    ships (MTP present, KV f16, no known-bad flag form, canonical prefix) is run on the result.
    """
    if threads < 1:
        raise ValueError("threads must be >= 1")
    cmd = R.build_serve_command(bindir=bindir, model=trunk, mtp=True, port=port)
    cmd[2] = cpus                                   # SERVE_PREFIX = taskset -c <cpus> numactl --interleave=all
    _replace_value(cmd, "-t", str(threads))
    _replace_value(cmd, "-c", str(ctx))
    _replace_value(cmd, "-md", head)
    if ubatch is not None:
        cmd += ["-ub", str(ubatch)]
    if cmd[:2] != ["taskset", "-c"] or cmd[3:5] != ["numactl", "--interleave=all"]:
        raise R.RecipeViolation(f"serve prefix is not taskset -c <cpus> numactl --interleave=all: {cmd[:5]}")
    R.assert_no_bad_flag_forms(cmd)
    R.assert_mtp_present(cmd)
    R.assert_kv_f16(cmd)
    return cmd


def serve_env(bindir: str) -> dict[str, str]:
    """env -i environment: OMP stack + champion GGML env + THE BUILD'S OWN LD_LIBRARY_PATH only.

    Three ggml generations live on this host; a binary that inherits another tree's libs runs
    silently wrong, so LD_LIBRARY_PATH is exactly <bindir> and nothing is inherited."""
    env = R.build_serve_env(bindir=bindir, base_env={})
    env["PATH"] = "/usr/bin:/bin"
    env["HOME"] = "/tmp"
    R.assert_canonical_env(env)
    if "GGML_KNOB_FILE" in env:
        raise R.RecipeViolation("GGML_KNOB_FILE must not be set: the champion build has no knob page")
    return env


def model_files(trunk: str = R.TRUNK_GGUF, head: str = R.MTP_HEAD_GGUF) -> list[str]:
    """Every file a load reads: all shards of a split trunk (shard 1 is metadata only) + the head."""
    m = re.match(r"^(?P<stem>.*)-(?P<i>\d{5})-of-(?P<n>\d{5})\.gguf$", trunk)
    if m:
        n = int(m.group("n"))
        shards = [f"{m.group('stem')}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]
    else:
        shards = [trunk]
    return shards + [head]


def ram_gib(ctx: int = CTX) -> int:
    """Private RAM a launch needs, rounded UP to a whole GiB. --no-mmap makes a private copy of
    every shard, so this is the file bytes + the KV for the context + the observed overhead."""
    files_b = sum(b for _p, b, _s in R.TRUNK_SHARDS) + R.MTP_HEAD_BYTES
    kv_b = KV_KIB_PER_TOKEN * 1024 * ctx
    return math.ceil((files_b + kv_b) / 2**30 + RAM_OVERHEAD_GIB)


# ---------------------------------------------------------------------------
# Proofs
# ---------------------------------------------------------------------------
def _binary_files(bindir: str) -> list[Path]:
    out = []
    for p in sorted(Path(bindir).iterdir()):
        if p.is_file() and not p.is_symlink() and (".so" in p.name or p.name == "llama-server"):
            out.append(p)
    return out


def knob_proof(bindir: str, no_knobs: bool = True) -> dict:
    """`strings -a` over every shared object and llama-server. {knob: {file: count}} plus verdicts.

    A knob counts as PRESENT when any string contains it. Raises R.RecipeViolation on a
    required knob that is absent, or on GGML_KNOB_FILE present while no_knobs is True."""
    files = _binary_files(bindir)
    if not files:
        raise R.RecipeViolation(f"no libraries or llama-server under {bindir}")
    wanted = list(REQUIRED_KNOBS) + list(FORBIDDEN_KNOBS_WHEN_NO_KNOBS)
    found: dict[str, dict[str, int]] = {k: {} for k in wanted}
    for f in files:
        txt = subprocess.run(["strings", "-a", str(f)], capture_output=True, text=True, check=True).stdout
        for k in wanted:
            n = txt.count(k)
            if n:
                found[k][f.name] = n
    missing = [k for k in REQUIRED_KNOBS if not found[k]]
    forbidden = [k for k in FORBIDDEN_KNOBS_WHEN_NO_KNOBS if found[k]] if no_knobs else []
    proof = {"bindir": str(bindir), "no_knobs": no_knobs, "found": found,
             "missing_required": missing, "forbidden_present": forbidden,
             "ok": not missing and not forbidden, "ts": time.strftime("%FT%TZ", time.gmtime())}
    if missing:
        raise R.RecipeViolation(f"required knob(s) {missing} are in NO library of {bindir}: a null from them proves nothing")
    if forbidden:
        raise R.RecipeViolation(f"{forbidden} present in {bindir}: this is a knob-page build, NO_KNOBS=1 is wrong for it")
    return proof


def flag_check(bindir: str) -> None:
    """Dry-run every recipe flag form against the real binary with a nonexistent model."""
    os.environ["LD_LIBRARY_PATH"] = bindir
    R.assert_flag_forms_exist(str(Path(bindir) / "llama-server"))
    R.assert_flag_forms_exist(str(Path(bindir) / "llama-server"), [["-ub", "2048"]])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _write_lines(path: str, lines: Iterable[str]) -> None:
    for ln in lines:
        if "\n" in ln:
            raise ValueError("argv/env entries may not contain newlines")
    Path(path).write_text("".join(ln + "\n" for ln in lines))


def _main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="sweep_lib", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", help="rotated launch order")
    p.add_argument("--rounds", type=int, default=ROUNDS_DEFAULT)
    p.add_argument("--masked-arm", action="store_true", help="add the optional t48 masked to cpus 0-47")

    p = sub.add_parser("serve-files", help="write <prefix>.argv and <prefix>.env for one launch")
    p.add_argument("--bindir", default=CHAMP_BINDIR)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--cpus", default=FULL_CPUS)
    p.add_argument("--port", type=int, default=PORT)
    p.add_argument("--ctx", type=int, default=CTX)
    p.add_argument("--trunk", default=R.TRUNK_GGUF)
    p.add_argument("--head", default=R.MTP_HEAD_GGUF)
    p.add_argument("--ubatch", type=int, default=None)
    p.add_argument("--out-prefix", required=True)

    p = sub.add_parser("knob-proof")
    p.add_argument("--bindir", default=CHAMP_BINDIR)
    p.add_argument("--out", required=True)
    p.add_argument("--no-knobs", dest="no_knobs", action="store_true", default=True)
    p.add_argument("--allow-knob-page", dest="no_knobs", action="store_false")

    p = sub.add_parser("flag-check")
    p.add_argument("--bindir", default=CHAMP_BINDIR)

    sub.add_parser("artifacts", help="trunk shards + head present at the pinned size")

    p = sub.add_parser("files", help="files to evict (one per line)")
    p.add_argument("--trunk", default=R.TRUNK_GGUF)
    p.add_argument("--head", default=R.MTP_HEAD_GGUF)

    p = sub.add_parser("ram-gib")
    p.add_argument("--ctx", type=int, default=CTX)

    p = sub.add_parser("meta", help="write a JSON record: key=value ...")
    p.add_argument("--out", required=True)
    p.add_argument("kv", nargs="*")

    a = ap.parse_args(argv[1:])
    try:
        if a.cmd == "plan":
            for ln in plan(a.rounds, a.masked_arm):
                print(f"{ln.round_no}\t{ln.label}\t{ln.arm.name}\t{ln.arm.threads}\t{ln.arm.cpus}")
        elif a.cmd == "serve-files":
            _write_lines(a.out_prefix + ".argv", serve_argv(a.bindir, a.threads, a.cpus, a.port, a.ctx, a.trunk, a.head, a.ubatch))
            _write_lines(a.out_prefix + ".env", [f"{k}={v}" for k, v in sorted(serve_env(a.bindir).items())])
        elif a.cmd == "knob-proof":
            proof = knob_proof(a.bindir, a.no_knobs)
            Path(a.out).write_text(json.dumps(proof, indent=1, sort_keys=True) + "\n")
            print("KNOB PROOF OK " + json.dumps({k: sorted(v) for k, v in proof["found"].items()}))
        elif a.cmd == "flag-check":
            flag_check(a.bindir)
            print("FLAG FORMS OK")
        elif a.cmd == "artifacts":
            R.assert_artifacts_exist()
            print("ARTIFACTS OK")
        elif a.cmd == "files":
            print("\n".join(model_files(a.trunk, a.head)))
        elif a.cmd == "ram-gib":
            print(ram_gib(a.ctx))
        elif a.cmd == "meta":
            rec = dict(kv.split("=", 1) for kv in a.kv)
            Path(a.out).write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    except (R.RecipeViolation, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, OSError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
