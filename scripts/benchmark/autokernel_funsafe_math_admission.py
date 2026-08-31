#!/usr/bin/env python3
"""Greedy-parity + speed harness for the -funsafe-math-optimizations removal.

    python3 scripts/benchmark/autokernel_funsafe_math_admission.py \
        --admission-tree /mnt/raid0/llm/tmp/ak-admission-funsafe-20260831 \
        --out artifacts/funsafe-math-admission

DO NOT RUN while AutoKernel run 21 is live: this holds the mi210_0 claim, builds
twice at -j64 and measures on the device. Execution belongs to the run-21->22
boundary, which the operator gates. (The claim itself refuses a second holder, so
running early fails loudly rather than corrupting the live run.)

WHAT IT ANSWERS -- the operator's conditional, made falsifiable. The ruling on
upstream #26696 (commit e79e4bf66) was "the 2% decode hit is worth it if it
increases quality as stated", where "as stated" is upstream's RDNA3.5 evidence:
`-fassociative-math` reassociating FP reductions until greedy argmax flips at
temperature 0. Nothing in our record establishes gfx90a behaves the same, so:

  (a) PARITY: N fixed-seed greedy generations on flag-on and flag-off builds of
      the SAME champion commit. Any argmax divergence between the two builds
      means the flag is distorting our outputs on our silicon -- the quality
      claim is DEMONSTRATED here and the removal buys real correctness.
      Bit-identical streams mean the gain is UNDEMONSTRATED on gfx90a at these
      shapes; the operator decides with that fact (the ~2% then buys only
      upstream parity).
  (b) COST: the standard alternating 20-pair A/B on the loop's calibrated tg128
      surface, flag-on as anchor, flag-off as candidate -- the measured decode
      price on our machine, not the fork's.

Both arms are built from the admission branch's own geometry: flag-off is the
admission commit (`ak/admission/remove-funsafe-math-20260831`), flag-on is its
parent (the champion base it was cut from), so the ONLY difference between the
binaries is the one-line CMake removal. The harness refuses to proceed if the
diff between the two commits touches anything but ggml/src/ggml-hip/CMakeLists.txt.

Non-promotable screening: the keep/decline decision is the operator's (CH-7
manual admission), not this script's. It reports; it does not commit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernel_rnd"))

from autokernel.controller import build_recipe                      # noqa: E402
from autokernel.loop import bench, claim, gates, residency          # noqa: E402
from autokernel.loop.run import noise_floor_pct                     # noqa: E402

#: Fixed greedy probe set. Deliberately mixed regimes: prose, code, arithmetic and
#: repetition -- reassociation-sensitive reductions (attention softmax, RMS norm)
#: see different value distributions under each.
PROMPTS = (
    "Explain, step by step, why the sky appears blue during the day.",
    "Write a C function that reverses a singly linked list in place.",
    "Compute 47 * 89 - 1234 / 2, showing every intermediate step.",
    "List the first twelve prime numbers, one per line.",
    "Translate to French: 'The measurement is only as good as its instrument.'",
    "Summarize the plot of a heist film in exactly three sentences.",
    "Repeat the word 'benchmark' twenty times, separated by commas.",
    "What is the derivative of x^3 * ln(x)? Show the product rule explicitly.",
)
GREEDY_SEED = 42
GEN_TOKENS = 128


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                          text=True, timeout=600).stdout.strip()


def divergence(flag_on: str, flag_off: str) -> dict | None:
    """First point where two greedy streams part ways, or None if bit-identical.

    Character-level index plus the count of matching whitespace tokens before the
    split -- a token count is what a reader compares against `-n`.
    """
    if flag_on == flag_off:
        return None
    limit = min(len(flag_on), len(flag_off))
    at = next((i for i in range(limit) if flag_on[i] != flag_off[i]), limit)
    return {"char_index": at,
            "tokens_before_split": len(flag_on[:at].split()),
            "flag_on_continuation": flag_on[at:at + 80],
            "flag_off_continuation": flag_off[at:at + 80]}


def greedy_once(binary: Path, model: Path, prompt: str, *, n_tokens: int,
                seed: int, timeout_s: int = 600) -> str:
    """One deterministic greedy generation. Same pinning and loader env as the
    loop's own benchmark invocations, so the parity probe runs the binary it
    measures (`ldd` cannot prove HIP residency; the loader env at least removes
    the wrong-ggml-generation failure mode)."""
    argv = ["taskset", "-c", bench.CPU_LIST, "numactl", "--interleave=all",
            str(binary), "-m", str(model), "-p", prompt, "--temp", "0",
            "--seed", str(seed), "-n", str(n_tokens), "-no-cnv", "--simple-io",
            "--no-display-prompt", "-ngl", "99", "-fa", "1"]
    done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s,
                          env=residency.loader_env(binary))
    if done.returncode != 0:
        raise bench.BenchFailed(
            f"llama-cli rc={done.returncode}: {done.stderr[-400:]}")
    return done.stdout


def verify_one_line_geometry(tree: Path, admission_commit: str) -> str:
    """The two arms may differ by NOTHING but the CMake line. Returns the parent
    (flag-on) commit; raises if the branch carries anything else."""
    parent = _git(tree, "rev-parse", f"{admission_commit}^")
    touched = _git(tree, "diff", "--name-only", parent, admission_commit).splitlines()
    if touched != ["ggml/src/ggml-hip/CMakeLists.txt"]:
        raise SystemExit(f"admission branch touches {touched}; expected exactly "
                         f"ggml/src/ggml-hip/CMakeLists.txt — refusing to measure "
                         f"a confounded pair")
    return parent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--admission-tree", type=Path,
                        default=Path("/mnt/raid0/llm/tmp/ak-admission-funsafe-20260831"),
                        help="worktree holding the admission branch")
    parser.add_argument("--admission-ref",
                        default="ak/admission/remove-funsafe-math-20260831")
    parser.add_argument("--model", type=Path,
                        default=Path("/mnt/raid0/llm/models/"
                                     "DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf"))
    parser.add_argument("--pairs", type=int, default=20,
                        help="A/B pairs; 20 is the calibrated-floor row (1.188%% "
                             "on tg128) an expected ~2%% effect needs")
    parser.add_argument("--gen-tokens", type=int, default=GEN_TOKENS)
    parser.add_argument("--seed", type=int, default=GREEDY_SEED)
    parser.add_argument("--build-root", type=Path,
                        default=Path("/mnt/raid0/llm/tmp/funsafe-admission-builds"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    admission = _git(args.admission_tree, "rev-parse", args.admission_ref)
    flag_on_commit = verify_one_line_geometry(args.admission_tree, admission)
    print(f"pair      flag-on {flag_on_commit[:12]} (champion base)  "
          f"flag-off {admission[:12]} ({args.admission_ref})")

    floor = noise_floor_pct("tg128", args.pairs)
    recipe = build_recipe.HOUSE_GPU_RECIPE
    arms = {}
    for name, commit in (("flag_on", flag_on_commit), ("flag_off", admission)):
        src = args.build_root / f"src-{name}"
        if not src.exists():
            subprocess.run(["git", "-C", str(args.admission_tree), "worktree",
                            "add", "--detach", str(src), commit], check=True)
        build = args.build_root / f"build-{name}"
        # Serial builds, inside nothing: no claim is needed to compile, and holding
        # the device across 2x -j64 builds is run 9's idle-while-claimed defect.
        # llama-cli is IN the target list because greedy_once shells it. The first
        # boundary run (2026-08-31 22:04Z) built only gates.compiles' default
        # targets (llama-bench, test-backend-ops), passed the oracle, then died at
        # the parity stage on a binary that was never built -- rc=1, no merge, and
        # the whole step read red. The harness had never run end-to-end; this line
        # is what end-to-end would have caught.
        verdict = gates.compiles(src, build, cmake_defines=recipe.cmake_defines(),
                                 targets=("llama-bench", "test-backend-ops", "llama-cli"),
                                 jobs=64, cpu_list="96-183")
        if not verdict.passed:
            raise SystemExit(f"{name} build failed: {verdict.reason}")
        oracle = gates.op_correctness(build)
        if not oracle.passed:
            raise SystemExit(f"{name} failed test-backend-ops: {oracle.reason}")
        arms[name] = build
        print(f"build     {name} at {commit[:12]}: compiled, oracle passed")

    report = {"schema": "epyc.autokernel.funsafe_math_admission.v1",
              "authority": "non_promotable_screening_operator_decides",
              "flag_on_commit": flag_on_commit, "flag_off_commit": admission,
              "model": args.model.name, "seed": args.seed,
              "gen_tokens": args.gen_tokens, "prompts": len(PROMPTS),
              "parity": [], "recorded_at":
                  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with claim.hold() as receipt:
        print(f"claim     held on {receipt['device_id']}")
        diverged = 0
        for index, prompt in enumerate(PROMPTS):
            streams = {name: greedy_once(build / "bin" / "llama-cli", args.model,
                                         prompt, n_tokens=args.gen_tokens,
                                         seed=args.seed)
                       for name, build in arms.items()}
            split = divergence(streams["flag_on"], streams["flag_off"])
            diverged += split is not None
            report["parity"].append({"prompt_index": index, "prompt": prompt,
                                     "diverged": split is not None,
                                     "divergence": split})
            print(f"greedy    prompt {index}: "
                  f"{'DIVERGED at token ~' + str(split['tokens_before_split'])
                     if split else 'bit-identical'}")

        comparison = bench.compare(
            bench.Arm("flag_on", arms["flag_on"] / "bin" / "llama-bench"),
            bench.Arm("flag_off", arms["flag_off"] / "bin" / "llama-bench"),
            args.model, pp=0, tg=128, pairs=args.pairs, noise_floor_pct=floor)
        report["ab"] = comparison.to_dict()

    report["divergent_prompts"] = diverged
    report["verdict_hint"] = (
        "argmax DIVERGENCE on gfx90a: the flag was distorting outputs; the "
        "operator's quality condition is DEMONSTRATED on our silicon" if diverged
        else "NO divergence at these shapes: the quality gain is UNDEMONSTRATED "
             "on gfx90a; the operator decides whether upstream parity alone is "
             "worth the measured cost")
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "funsafe-math-admission.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nspeed     flag-off vs flag-on: {comparison.effect * 100:+.3f}% "
          f"(floor {floor:.3f}%, decisive={comparison.decisive})")
    print(f"parity    {diverged}/{len(PROMPTS)} prompts diverged")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
