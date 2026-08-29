#!/usr/bin/env python3
"""Retract the refusals produced by a correctness gate that never ran.

    python3 scripts/benchmark/autokernel_retract_false_oracle.py \
        --store /mnt/raid0/llm/autokernel/loop-memory [--apply]

WHAT HAPPENED
-------------
`gates.op_correctness` invoked `test-backend-ops ... --suite-seed <n>`. That flag does
not exist in this llama.cpp tree, so the binary printed its usage text and exited 1,
and the gate turned that into `MUL_MAT failed on ROCm0` -- for every candidate, in
every run. Proven against the anchor build, which passes 1139/1139: the exact gate
command exits 1 there, and exits 0 with the flag removed. The oracle never ran once.

Every refusal carrying that string is therefore a HARNESS FAULT recorded as a
scientific result. Left in place they are worse than absent, because the critic reads
prior refusals and cites them -- run 9 rejections already quote "Already measured and
rejected: ... MUL_MAT failed on ROCm0", so a fabricated verdict was actively steering
new hypotheses away from correct work.

WHY RETRACT RATHER THAN DELETE
------------------------------
The attempt happened; the mechanism was proposed; that is true and worth keeping. What
is false is the REASON. Deleting the row would also delete the evidence that the loop
tried this mechanism, and a future planner would re-derive it blind -- the exact
blindness this substrate exists to remove. So the row survives and its verdict is
annulled in place, with the retraction visible to anything that reads it.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sqlite3
import sys

FALSE_VERDICT = "MUL_MAT failed on ROCm0"
RETRACTION = (
    "[RETRACTED 2026-08-29 — this verdict was never measured. The correctness gate "
    "invoked test-backend-ops with an unsupported --suite-seed flag; the binary "
    "printed usage and exited 1, and the gate reported it as a correctness failure. "
    "Proven on the anchor, which passes 1139/1139. Treat this mechanism as UNTESTED, "
    "not as refuted, and do not cite it as a prior rejection.] Original text: ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--apply", action="store_true",
                        help="without this, report what would change and exit")
    args = parser.parse_args(argv)

    db = args.store / "experiments.db"
    if not db.is_file():
        print(f"REFUSED: no store at {db}", file=sys.stderr)
        return 1

    connection = sqlite3.connect(db)
    affected = list(connection.execute(
        "SELECT attempt_id, mechanism_id, status FROM experiments "
        "WHERE refusal_reason LIKE ? AND refusal_reason NOT LIKE ?",
        (f"%{FALSE_VERDICT}%", "%[RETRACTED%")))
    total = list(connection.execute("SELECT COUNT(*) FROM experiments"))[0][0]

    print(f"store      {db}")
    print(f"rows       {len(affected)} of {total} carry the fabricated verdict")
    for _, mechanism, status in affected:
        print(f"             {str(mechanism):34} {status}")
    if not affected:
        print("nothing to retract")
        return 0
    if not args.apply:
        print("\nDRY RUN — pass --apply to annul these verdicts in place.")
        return 0

    backup = db.with_suffix(".db.pre-retraction")
    shutil.copyfile(db, backup)
    print(f"\nbackup     {backup}")

    connection.execute(
        "UPDATE experiments SET refusal_reason = ? || refusal_reason "
        "WHERE refusal_reason LIKE ? AND refusal_reason NOT LIKE ?",
        (RETRACTION, f"%{FALSE_VERDICT}%", "%[RETRACTED%"))
    connection.commit()

    remaining = list(connection.execute(
        "SELECT COUNT(*) FROM experiments WHERE refusal_reason LIKE ? "
        "AND refusal_reason NOT LIKE ?",
        (f"%{FALSE_VERDICT}%", "%[RETRACTED%")))[0][0]
    print(f"retracted  {len(affected)} row(s); {remaining} unannulled remain")
    if remaining:
        print("REFUSED: some rows were not annulled", file=sys.stderr)
        return 1

    # Regenerate the human-readable view from the repaired store.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernel_rnd"))
    from autokernel.controller import experiments as store_module   # noqa: E402
    with store_module.ExperimentStore(args.store) as store:
        store.write_markdown(epoch="0" * 64)
    print(f"rewrote    {args.store / 'experiments.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
