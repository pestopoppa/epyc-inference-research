#!/usr/bin/env python3
"""Apply OAB-31 budget exceptions only after explicit operator authorization.

PENDING: this script is prepared for review and has not been run with --apply.
The provided reference must identify the actual operator approval, not this draft.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_SHA = "f6369b6a22e609e8db46b9b291f9a4eea6323311fdf3199fa70aadae4d6ac19d"
MANIFEST_SHA = "58c0e80411e2fbdb3e3e60efec94b1f9a884f68afa626f52369f066ac9b9f127"
BASE = "94b1dd64c163d915eb1347b118f57e44e981bf78"
MANIFEST = ROOT / "scripts/kernel_rnd/autokernel/loop_code_budget_baseline.json"
EVIDENCE = Path(__file__).with_name("budget-decision-evidence.json")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval-reference", required=True,
                        help="reference identifying the actual operator approval")
    parser.add_argument("--apply", action="store_true", required=True)
    args = parser.parse_args()
    ref = args.approval_reference.strip()
    if (len(ref) < 8 or any(term in ref.casefold() for term in
                            ("pending", "placeholder", "not approved", "unapproved"))):
        parser.error("an explicit actual operator approval reference is required")
    if sha(EVIDENCE) != EVIDENCE_SHA or sha(MANIFEST) != MANIFEST_SHA:
        raise SystemExit("review evidence or frozen manifest changed; refusing")
    if subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", BASE, "HEAD"]).returncode:
        raise SystemExit("reviewed source base is not an ancestor of HEAD; refusing")
    evidence = json.loads(EVIDENCE.read_text())
    expected = evidence["sha256_by_changed_file"]
    for name, digest in expected.items():
        if sha(ROOT / name) != digest:
            raise SystemExit(f"reviewed file changed: {name}; refusing")
    scope = "scripts/kernel_rnd/autokernel"
    changed = set(git("diff", "--name-only", "HEAD", "--", scope).splitlines())
    changed.update(git("ls-files", "--others", "--exclude-standard", "--", scope).splitlines())
    if changed != set(expected):
        raise SystemExit("owned source/test change set changed; refusing")
    staged = git("diff", "--cached", "--name-only", "--", scope).splitlines()
    if staged:
        raise SystemExit("source or test files are staged; refusing")
    raw = json.loads(MANIFEST.read_text())
    if raw["exceptions"]:
        raise SystemExit("manifest exceptions changed; refusing")
    for row in evidence["files"]:
        if row["oab_delta"] > 0 and row["current_overage"] > 0:
            raw["exceptions"][row["file"]] = {"ceiling": row["current"], "approval": ref}
    encoded = (json.dumps(raw, indent=2) + "\n").encode()
    fd, tmp = tempfile.mkstemp(prefix=".oab31-budget-", dir=MANIFEST.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, MANIFEST)
        parent = os.open(MANIFEST.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print("Applied six reviewed exceptions; existing baseline debt remains.")


if __name__ == "__main__":
    main()
