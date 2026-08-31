#!/usr/bin/env python3
"""Refuse a CI run whose suites silently got smaller.

The autokernel guards workflow went 43 consecutive runs without executing a single
assertion, because `actions/setup-python` ships no pytest and the step died during
startup. That specific hole is closed by installing pytest and proving it importable
in its own named step. This guard closes the SHAPE of the hole, which is wider:

    a suite that runs FEWER assertions than it used to reports exactly the same
    green as one that runs all of them.

pytest already refuses the total collapse -- collecting nothing exits 5, so a path
that stops matching anything turns the step red. It does NOT refuse a PARTIAL
collapse. Rename one file, narrow one path filter, move the loop package one
directory, and the step collects 18 tests instead of 196, passes, and reports the
same green tick. Nothing in the log distinguishes the two, and the two-day silence
this project just lived through is what that looks like from the outside.

So the counts are declared here, as floors. They are tripwires in the same sense as
`LOOP_LOC_BUDGET`: not arithmetic anyone needs, but a number that cannot move
without a person editing this file and saying why in the commit. Adding tests must
never turn CI red, so the comparison is `>=`, never `==`.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# suite name -> (floor, paths). The floor is the count observed when the suite was
# declared; raising it is a deliberate edit, and lowering it must be argued for.
SUITE_FLOORS: dict[str, tuple[int, tuple[str, ...]]] = {
    # 109 -> 133 on 2026-08-31: `P-AK-SEARCH-1-A3` epoch-scoped ranking. 22 new tests in
    # `controller/test_experiments.py`, over a fixture of 200 real rows lifted from the
    # live loop-memory store, including the byte-identity of the default recall path
    # against digests captured from the pre-amendment function. The floor sat two under
    # the observed count before this edit; it is now level with it.
    # 133 -> 144 on 2026-08-31 (run-22 prep): the -funsafe-math-optimizations CH-7
    # admission harness. 11 new tests in
    # `scripts/benchmark/test_autokernel_funsafe_math_admission.py` -- the greedy
    # divergence detector, the one-line-geometry refusal (a confounded pair must not
    # be measured) and the argparse wiring that names the prepared admission branch.
    "the decision arithmetic": (
        144,
        (
            "scripts/benchmark/test_screen_effect_estimator.py",
            "scripts/benchmark/test_autokernel_funsafe_math_admission.py",
            "scripts/kernel_rnd/autokernel/controller/test_workload_contract.py",
            "scripts/kernel_rnd/autokernel/controller/test_build_recipe.py",
            "scripts/kernel_rnd/autokernel/controller/test_gate_parameters.py",
            "scripts/kernel_rnd/autokernel/controller/test_experiments.py",
            "scripts/kernel_rnd/autokernel/controller/test_gpu_utilization.py",
            "scripts/kernel_rnd/autokernel/controller/test_kernel_hotspots.py",
            "scripts/kernel_rnd/autokernel/test_check_regrowth_guards.py",
            "scripts/kernel_rnd/autokernel/test_check_suite_floors.py",
        ),
    ),
    # 196 -> 222 on 2026-08-30: the promotion A/A guard (`loop/anchor.py`) and the
    # rebuild-instead-of-move promotion contract. 26 new tests in `loop/test_anchor.py`,
    # mutation-tested 18 of 18 in both directions.
    # 222 -> 239 on 2026-08-31: `P-AK-SEARCH-1-A3` at the loop boundary. 17 new tests in
    # `loop/test_ranking.py` -- the opt-in defaults, and the conformance property that a
    # cross-epoch magnitude cannot reach the keep/null arithmetic. Mutation-tested 43 of
    # 43 in both directions, every one of the 39 new assertions killed by at least one.
    # 239 -> 271 on 2026-08-31: the champion-vs-production headline emitter
    # (`loop/production.py`), published at every champion advance so the dashboard's
    # cumulative number stops being a hand measurement. 32 new tests in
    # `loop/test_production.py`, over two fixtures lifted verbatim off disk -- the
    # live bundle and the real 20-pair sample vectors. Mutation-tested 41 of 41 in
    # both directions; 63 of the 67 new assertions are the first failure under one.
    # 271 -> 304 on 2026-08-31: the single-champion startup refusal
    # (`loop/champion.py`, 25 tests in `loop/test_champion.py` reconstructing the
    # actual incident geometry in temp git repos, refusal-before-claim ordering
    # asserted) and the live-resolved frozen-production baseline
    # (`loop/production.py`, 8 new tests including a real promotion driven through a
    # temp frozen tree). Both mutation-tested in both directions.
    # 304 -> 309 on 2026-08-31: the sequential CLI path deletion. FOUR tests deleted,
    # each because its subject moved into the pool rather than because it was
    # inconvenient: the two `loop.run` breaker tests (the consecutive-error breaker
    # moved to `pipeline.run_pool`, re-pinned there by five `ThePoolBreaker` tests
    # covering pool-wide counting, reset-on-success and the never-trip statuses) and
    # the two `TheTreeIsResetBeforeEachIteration` tests (the reset hook left the
    # test-only `loop.run` seam; the property lives in `pool.reset_to_champion` at
    # the top of every lane, which the pool's staleness and containment suites
    # cannot pass without). NINE added: the five above plus four drain-tier tests
    # (`AStopAbandonsFormationAtTheNextStageBoundary`) pinning that a stop abandons
    # formation before the next actor call while a lane in the serialized tail
    # still completes. Net +5, floor level with the observed count.
    # 309 -> 338 on 2026-08-31: the run-22 loop-surface extension. 29 new tests:
    # 18 in `loop/test_bench.py` (uncalibrated-refuses-decisive both directions --
    # a fake floor must yield decisive=None, the historical pp512 defect; the dec-b*
    # surface table and its `-b/-ub` argv, verified against llama-bench's own
    # source; floor_rows never defaults; bootstrap_floor reproduces the enforced D8
    # tg128 k=5 row 2.422 byte-for-byte from the archived SETTLED samples), 3 in
    # `loop/test_loop.py` (a +10%% raw effect on an uncalibrated surface records
    # measured_null/UNDECIDABLE and never draws commit) and 8 in the new
    # `loop/test_calibration.py` (run-level floor discipline, the commit path's own
    # refuse_uncalibrated_keep against a doctored decisive=True, the
    # --calibrate-surface dispatch to the one D8 campaign home, and the
    # write-calibration round trip that turns refusal into a real floor).
    "the loop package": (338, ("scripts/kernel_rnd/autokernel/loop/",)),
}

_COLLECTED = re.compile(r"(\d+) tests? collected")


def collected(paths: tuple[str, ...], *, root: Path = REPO_ROOT) -> int:
    """Number of tests pytest collects for `paths`.

    Raises rather than returning 0 when collection itself fails. A collection error
    laundered into a plausible zero is the same defect this file exists to refuse:
    it would read as "the suite shrank" when the truth is "the suite did not load".
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", *paths],
        cwd=root, capture_output=True, text=True,
    )
    match = _COLLECTED.search(proc.stdout)
    # The return code is checked FIRST and separately, because pytest prints a
    # perfectly plausible "93 tests collected" on the same line as ", 1 error" when
    # one module in the selection fails to import. Matching the regex alone reads
    # that as a healthy 93 and silently drops the broken module's tests from the
    # count -- an error laundered into a number, which is the family of defect this
    # whole file exists to refuse. Found the first time this guard ran against
    # itself, which is the argument for making a guard's own suite its subject.
    if proc.returncode != 0 or match is None:
        raise RuntimeError(
            f"pytest could not collect cleanly (exit {proc.returncode}).\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    return int(match.group(1))


def main(argv: list[str] | None = None) -> int:
    violations = 0
    for name, (floor, paths) in SUITE_FLOORS.items():
        try:
            count = collected(paths)
        except RuntimeError as exc:
            print(f"  ERROR  {name}: {exc}")
            violations += 1
            continue
        verdict = "ok" if count >= floor else "SHRANK"
        print(f"  {verdict:>6}  {name}: {count} collected, floor {floor}")
        if count < floor:
            violations += 1
    print(f"\n{violations} suite floor violation(s)")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
