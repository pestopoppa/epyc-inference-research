#!/bin/bash
# ratify_v7_runner_repin_20260916.sh — re-seal the two runner pins that no longer match
# scripts/benchmark/v7_quality_gate_runner.py (operator decision 2026-09-16: RE-SEAL).
#
#   Review (default, writes nothing):  bash scripts/operator/ratify_v7_runner_repin_20260916.sh
#   Apply + receipts (no commit):      bash scripts/operator/ratify_v7_runner_repin_20260916.sh --apply
#   Operator's single command:         bash scripts/operator/run_v7_runner_repin_20260916.sh --operator <name>
#
# THE PINS (research repo, this checkout)
#   6dea92dd…  scripts/benchmark/dflash2_followups.py EXPECTED["runner_sha256"]
#              + the literal in scripts/benchmark/test_dflash2_followups.py
#   79721927…  artifacts/p3-shadow-bakeoff-20260728/manifest/p3_bakeoff_manifest.json
#              (capture.runner.sha256, duties.coder.suites.livecodebench_hard.scorer.runner_sha256)
#              + its .sha256 sidecar
#   Both move to the current runner, 20a97fbd…. The edit is exactly
#   artifacts/operator/v7-runner-repin-20260916.patch: 5 hash substitutions in 4 files.
#
# WHY IT IS SAFE (evidence: data/v7-runner-repin-20260916/, README.md there)
#   6dea92dd -> 20a97fbd is da06b371 (2026-08-26). It adds an opt-in --belief-category flag,
#     default off, that attaches belief rows at finalize. With the flag absent, the request
#     bytes are identical and the outputs are identical once wall-clock keys are removed
#     (6/6 stub-server scenarios).
#   79721927 -> 6dea92dd is baf36757 (2026-08-19). The operator decision text does not name it,
#     but re-sealing 79721927 crosses it. Request bytes and scoring are identical. It ADDS the
#     output keys effective_request (per row) and meta.sampling_fields_are_requested_not_effective.
#     No P3 bake-off capture exists under the manifest, so no banked output is re-labelled.
#   A negative control shows the comparison detects both a one-field request change and a
#   scoring change.
#
# WHAT IT DOES NOT DO. It edits no banked capture or sealed replay bundle: those hashes record
# history. It leaves replay_tc_nothink_v4.py and laguna_q4_cpu_bench_runner.py on 79721927 (see
# the evidence README). It changes no code path, starts no server and takes no compute. The
# validations it runs use a localhost stub, never a model.
#
# RECEIPTS: THE REPO SPLIT. The pins live in THIS repo, so every write lands here:
#   artifacts/operator/ratify_v7_runner_repin_20260916.json          decision receipt
#   artifacts/operator/ratify_v7_runner_repin_20260916.receipt.json  MEASUREMENT.md §5 consolidated receipt
# The section-5 tool and the constitution live in epyc-root and are only READ, from
# EPYC_ROOT (default /mnt/raid0/llm/epyc-root). The consolidated receipt is emitted with
# --repo-root EPYC_ROOT, so the protocol anchor is checked against MEASUREMENT.md. The research
# state files are passed by absolute path, and --evidence-repo is this checkout. No keyed
# index is written under epyc-root artifacts/operator/receipts/: that namespace belongs to
# bus-token --attest signers, this gate carries no bus token, and a second-repo write would
# split the apply across two commits. The decision receipt is this gate's spent marker
# (double-signing is refused).
#
# PINS. Any mismatch is refused. If a target has moved, REGENERATE the bundle; never force it.
#
# IDEMPOTENT. When all four targets sit at their post-state pins and the decision receipt exists,
# a re-run prints ALREADY RATIFIED and exits 0 without writing. Every other partial state is
# refused and left for resolution by hand.
#
# ROOT defaults to the research checkout this script lives in. LIVE is the path the P3 manifest
# pins the runner at. Both can be overridden (tests).
set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ROOT="${ROOT:-$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)}"
EPYC_ROOT="${EPYC_ROOT:-/mnt/raid0/llm/epyc-root}"
LIVE="${LIVE:-/mnt/raid0/llm/epyc-inference-research}"
RECEIPT_TOOL="$EPYC_ROOT/scripts/operator/ratification_receipt.py"

GATE_ID="RATIFY-V7-RUNNER-REPIN-20260916"
PATCH_REL="artifacts/operator/v7-runner-repin-20260916.patch"
EVID_REL="data/v7-runner-repin-20260916"
RECEIPT_REL="artifacts/operator/ratify_v7_runner_repin_20260916.json"
CONSOLIDATED_REL="artifacts/operator/ratify_v7_runner_repin_20260916.receipt.json"
PATCH="$ROOT/$PATCH_REL"
RECEIPT="$ROOT/$RECEIPT_REL"
CONSOLIDATED="$ROOT/$CONSOLIDATED_REL"

MANIFEST_REL="artifacts/p3-shadow-bakeoff-20260728/manifest/p3_bakeoff_manifest.json"
TARGETS=(
  "scripts/benchmark/dflash2_followups.py"
  "scripts/benchmark/test_dflash2_followups.py"
  "$MANIFEST_REL"
  "$MANIFEST_REL.sha256"
)
PRE=(
  "9dc48324ce54184a2bb826dbcba9bc4300e248af05514de0141b2660a0930586"
  "610d27dea09eabafe427f479a8739d3be5e6a7e37f87bacbd4bbc6b511561dc6"
  "3bb31a7a5fb38dfc90d463855db7af02cd1b8843ab72ded59b261e543ff4dbea"
  "041e898e32dbdaf446fb56f2276f641d686c9e3f7c5ee48696685eab623e9311"
)
POST=(
  "9f8baf829140232e5c6a34c48b9a3c33dd68a55c95cfbb0447da62f018d39c0d"
  "d4d54b724c5700a67863b0532214b8f91ef5296d455bc0a86c6784855860ea71"
  "0ce1f2c3a1911b068486a37841de80e8f241e50c40017acaeec3d01bc1cb14cf"
  "3b0ab2da74a85d4dc9a4704223c9db0cd339ef4f81834a5b7a7e25fa3a8aeffb"
)
PATCH_SHA256="8008dc986e9139094f97b69f99df8110ad11ede82b24ec3be1ed4d05c83cc79d"

OLD_DFLASH2_RUNNER="6dea92dd9e374f79691f5df502fa11035ffd484906754f20190a4189111ae7dc"
OLD_P3_RUNNER="79721927e95293d070aba294bf422a24b1182dde07310d461d9e3ddaf6c84b0e"
NEW_RUNNER="20a97fbd4aecc6f3887299243362d5315b76f819aa1cbc4b8fd92f33b307b45a"
RUNNER_REL="scripts/benchmark/v7_quality_gate_runner.py"
# The runner versions, as git blobs, and the commits that moved between them.
BLOB_OLD_P3="511f921c8abd347b32563cc87d407fc0764d8f8a"
BLOB_OLD_DFLASH2="5167f5702dcf218ca500efb3cb98d5a0e07e10ca"
BLOB_NEW="b1c2773881858e4750edbc022c93af6f650a64b9"
COMMITS=(da06b371 baf36757)

MODE="dry-run"
case "${1:-}" in
  ""|--dry-run) MODE="dry-run" ;;
  --apply)      MODE="apply" ;;
  *) echo "usage: $0 [--dry-run | --apply]   (default: --dry-run, writes nothing)" >&2; exit 64 ;;
esac
[ $# -le 1 ] || { echo "usage: $0 [--dry-run | --apply]" >&2; exit 64; }

say() { printf '%s\n' "$*"; }
die() { printf 'REFUSING: %s\n' "$*" >&2; exit 65; }
sha() { sha256sum "$1" | awk '{print $1}'; }

command -v python3   >/dev/null || die "python3 not on PATH"
command -v sha256sum >/dev/null || die "sha256sum not on PATH"
command -v git       >/dev/null || die "git not on PATH"
for t in "${TARGETS[@]}"; do [ -f "$ROOT/$t" ] || die "$t not found under ROOT=$ROOT"; done
[ -f "$RECEIPT_TOOL" ] || die "section-5 receipt tool missing at $RECEIPT_TOOL (set EPYC_ROOT)"
[ -f "$EPYC_ROOT/MEASUREMENT.md" ] || die "constitution missing at $EPYC_ROOT/MEASUREMENT.md (set EPYC_ROOT)"

# ---------------------------------------------------------------- patch pin
[ -f "$PATCH" ] || die "patch not found at $PATCH"
got="$(sha "$PATCH")"
[ "$got" = "$PATCH_SHA256" ] || die "patch hash mismatch: expected $PATCH_SHA256, found $got. The text you reviewed is not the text that would be applied."

# ---------------------------------------------------------------- idempotence / partial states
n_pre=0; n_post=0; n_other=0; state_lines=()
for i in "${!TARGETS[@]}"; do
  now="$(sha "$ROOT/${TARGETS[$i]}")"
  if   [ "$now" = "${POST[$i]}" ]; then n_post=$((n_post+1)); state_lines+=("post   ${TARGETS[$i]}")
  elif [ "$now" = "${PRE[$i]}"  ]; then n_pre=$((n_pre+1));   state_lines+=("pre    ${TARGETS[$i]}")
  else n_other=$((n_other+1)); state_lines+=("DRIFT  ${TARGETS[$i]} ($now)"); fi
done
if [ "$n_post" -eq "${#TARGETS[@]}" ]; then
  if [ -f "$RECEIPT" ] && [ -f "$CONSOLIDATED" ]; then
    say "ALREADY RATIFIED: all ${#TARGETS[@]} targets are at their post-state pins and $RECEIPT_REL and $CONSOLIDATED_REL exist. Nothing to do."
    exit 0
  fi
  die "half-applied state: all targets are re-pinned, but the decision receipt ($([ -f "$RECEIPT" ] && echo present || echo MISSING)) or the consolidated receipt ($([ -f "$CONSOLIDATED" ] && echo present || echo MISSING)) is not. Resolve by hand."
fi
if [ "$n_other" -ne 0 ] || [ "$n_post" -ne 0 ]; then
  printf '  %s\n' "${state_lines[@]}" >&2
  die "targets are neither all at their pre-state nor all at their post-state pins (pre=$n_pre post=$n_post drift=$n_other). A target moved or was hand-edited: regenerate the bundle."
fi
[ -f "$RECEIPT" ]      && die "$RECEIPT_REL already exists, but the targets are not re-pinned. This gate is spent; double-signing is refused."
[ -f "$CONSOLIDATED" ] && die "$CONSOLIDATED_REL already exists. Resolve by hand."

# ---------------------------------------------------------------- preflight
say "== preflight (ROOT=$ROOT, EPYC_ROOT=$EPYC_ROOT, mode=$MODE) =="
fail=0
ok()  { printf '  ok    %-52s %s\n' "$1" "${2:-}"; }
bad() { printf '  FAIL  %-52s %s\n' "$1" "${2:-}"; fail=1; }
ok "patch hash" "(${PATCH_SHA256:0:12})"
for i in "${!TARGETS[@]}"; do ok "pre-state ${TARGETS[$i]##*/}" "(${PRE[$i]:0:12})"; done

# The runner this re-seal points at must BE the runner, both in this checkout and at the absolute
# path the P3 manifest pins. Otherwise the new seal would be false on the day it is written.
[ "$(sha "$ROOT/$RUNNER_REL")" = "$NEW_RUNNER" ] && ok "runner in ROOT is 20a97fbd" \
  || bad "runner in ROOT is 20a97fbd" "found $(sha "$ROOT/$RUNNER_REL")"
if [ -f "$LIVE/$RUNNER_REL" ]; then
  [ "$(sha "$LIVE/$RUNNER_REL")" = "$NEW_RUNNER" ] && ok "runner at manifest-pinned path is 20a97fbd" \
    || bad "runner at manifest-pinned path is 20a97fbd" "found $(sha "$LIVE/$RUNNER_REL") at $LIVE/$RUNNER_REL"
else
  bad "runner at manifest-pinned path exists" "$LIVE/$RUNNER_REL"
fi
# The version chain the evidence reasons about must be real history in this checkout.
if git -C "$ROOT" rev-parse -q --verify "HEAD:$RUNNER_REL" >/dev/null 2>&1 \
   && [ "$(git -C "$ROOT" rev-parse "HEAD:$RUNNER_REL")" = "$BLOB_NEW" ]; then
  ok "HEAD runner blob is $BLOB_NEW" "(${BLOB_NEW:0:8})"
else
  bad "HEAD runner blob is $BLOB_NEW"
fi
for b in "$BLOB_OLD_P3:$OLD_P3_RUNNER" "$BLOB_OLD_DFLASH2:$OLD_DFLASH2_RUNNER"; do
  blob="${b%%:*}"; want="${b##*:}"
  have="$(git -C "$ROOT" cat-file blob "$blob" 2>/dev/null | sha256sum | awk '{print $1}')"
  [ "$have" = "$want" ] && ok "sealed blob ${blob:0:8} hashes to ${want:0:8}" \
    || bad "sealed blob ${blob:0:8} hashes to ${want:0:8}" "found ${have:0:12}"
done
for c in "${COMMITS[@]}"; do
  git -C "$ROOT" merge-base --is-ancestor "$c" HEAD 2>/dev/null && ok "commit $c is an ancestor of HEAD" \
    || bad "commit $c is an ancestor of HEAD"
done

# Evidence is pinned by its SHA256SUMS; every listed file must verify.
if [ -f "$ROOT/$EVID_REL/SHA256SUMS" ] && (cd "$ROOT/$EVID_REL" && sha256sum --quiet -c SHA256SUMS >/dev/null 2>&1); then
  ok "evidence SHA256SUMS verifies" "($(wc -l < "$ROOT/$EVID_REL/SHA256SUMS") files)"
else
  bad "evidence SHA256SUMS verifies" "$EVID_REL"
fi
verdict="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$ROOT/$EVID_REL/equivalence_summary.json" 2>/dev/null || echo UNREADABLE)"
[ "$verdict" = "EQUIVALENT" ] && ok "banked equivalence verdict" "($verdict)" || bad "banked equivalence verdict" "$verdict"

grep -qF -- '- **Trust boundary**: this file, its `protocols/` annexes, the era registry, the eval tower, and' \
  "$EPYC_ROOT/MEASUREMENT.md" && ok "MEASUREMENT.md §5 trust-boundary anchor" || bad "MEASUREMENT.md §5 trust-boundary anchor"

# The targets and receipts must carry no unrelated edits, or a commit would sweep a peer
# session's hunk into this re-seal (shared-clone hazard).
dirty="$(git -C "$ROOT" status --porcelain -- "${TARGETS[@]}" "$RUNNER_REL" "$RECEIPT_REL" "$CONSOLIDATED_REL" 2>/dev/null || echo 'GIT-STATUS-FAILED')"
[ -z "$dirty" ] && ok "targets clean in git" || { bad "targets clean in git"; printf '%s\n' "$dirty"; }

if git -C "$ROOT" apply --check "$PATCH" 2>/dev/null; then ok "patch applies cleanly (no fuzz)"; else bad "patch applies cleanly (no fuzz)"; fi

[ "$fail" -eq 0 ] || die "preflight failed: $ROOT is not in the state this bundle was prepared against (research origin/main e4576171 + sub/v7-repin-20260916). Moved target or runner: regenerate the bundle. Nothing written."
say "  preflight clean"

if [ "$MODE" = "dry-run" ]; then
  say ""
  say "== equivalence evidence ($EVID_REL) =="
  python3 - "$ROOT/$EVID_REL" <<'PYEOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
s = json.loads((d / "equivalence_summary.json").read_text())
print(f"  verdict: {s['verdict']}")
for name, e in s["scenarios"].items():
    print(f"  {name}  (current: {e['current']['n_requests']} requests, {e['current']['n_rows']} rows)")
    for v, r in e["vs"].items():
        print(f"      vs {v}: requests_identical={r['requests_byte_identical']} "
              f"outputs_identical={r['outputs_identical_after_volatile_strip']} "
              f"residual={', '.join(r['residual_paths']) or 'none'}")
nc = json.loads((d / "negative_control.json").read_text())["results"]
print("  negative control: " + ", ".join(f"{k} detected={v['detected']}" for k, v in nc.items()))
for line in (d / "offline_tests.tsv").read_text().splitlines()[1:]:
    v, suite, rc, summary, failed = (line.split("\t") + [""])[:5]
    print(f"  offline {v} {suite:<32} exit={rc} {summary} {failed}".rstrip())
PYEOF
  say ""
  say "== re-pin diff (${PATCH_REL}) =="
  cat "$PATCH"
  say ""
  say "DRY RUN: would apply the diff above (post-state pins ${POST[0]:0:12} ${POST[1]:0:12} ${POST[2]:0:12} ${POST[3]:0:12}),"
  say "re-run the validations, then write $RECEIPT_REL and $CONSOLIDATED_REL. Nothing written."
  say "Apply with: ROOT=$ROOT EPYC_ROOT=$EPYC_ROOT bash $SCRIPT_PATH --apply"
  exit 0
fi

# ---------------------------------------------------------------- apply
RATIFIED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TMPD="$(mktemp -d)"
trap 'rm -rf "$TMPD"' EXIT
for i in "${!TARGETS[@]}"; do cp "$ROOT/${TARGETS[$i]}" "$TMPD/target.$i.bak"; done
PRESNAP="$TMPD/pre.json"
restore() {
  for i in "${!TARGETS[@]}"; do cp "$TMPD/target.$i.bak" "$ROOT/${TARGETS[$i]}"; done
  rm -f "$RECEIPT"
}

say ""
say "== apply =="
state_args=()
for t in "${TARGETS[@]}"; do state_args+=(--state "$ROOT/$t"); done
python3 "$RECEIPT_TOOL" capture --repo-root "$EPYC_ROOT" "${state_args[@]}" --out "$PRESNAP" \
  || die "could not snapshot the pre-state; nothing written"

# Working tree only, never --index. Staging is done by the wrapper or by hand, after reading the diff.
git -C "$ROOT" apply "$PATCH" || { restore; die "git apply failed; targets restored. Nothing changed."; }

# ---------------------------------------------------------------- postflight
say ""
say "== postflight =="
pf=0
for i in "${!TARGETS[@]}"; do
  now="$(sha "$ROOT/${TARGETS[$i]}")"
  if [ "$now" = "${POST[$i]}" ]; then printf '  ok    post-state %-40s (%s)\n' "${TARGETS[$i]##*/}" "${now:0:12}"
  else printf '  FAIL  post-state %-40s expected %s found %s\n' "${TARGETS[$i]##*/}" "${POST[$i]}" "$now"; pf=1; fi
done
# Structural: exactly the old hashes left, exactly the new one arrived, the sidecar matches.
if python3 - "$ROOT" "$MANIFEST_REL" "$OLD_DFLASH2_RUNNER" "$OLD_P3_RUNNER" "$NEW_RUNNER" <<'PYEOF'
import json, hashlib, sys
from pathlib import Path
root, mrel, old_d, old_p, new = sys.argv[1:6]
root = Path(root)
ok = True
m = json.loads((root / mrel).read_text())
if m["capture"]["runner"]["sha256"] != new: print("  FAIL  manifest capture.runner.sha256"); ok = False
lcb = m["duties"]["coder"]["suites"]["livecodebench_hard"]["scorer"]["runner_sha256"]
if lcb != new: print("  FAIL  manifest livecodebench_hard scorer.runner_sha256"); ok = False
side = (root / (mrel + ".sha256")).read_text().split()[0]
if side != hashlib.sha256((root / mrel).read_bytes()).hexdigest(): print("  FAIL  manifest sidecar"); ok = False
for rel in ("scripts/benchmark/dflash2_followups.py", "scripts/benchmark/test_dflash2_followups.py"):
    t = (root / rel).read_text()
    if old_d in t or t.count(new) != 1: print(f"  FAIL  {rel} pin"); ok = False
if old_p in (root / mrel).read_text(): print("  FAIL  manifest still carries 79721927"); ok = False
if ok: print("  ok    both manifest fields, sidecar and dflash2 pins carry 20a97fbd; no old hash left")
sys.exit(0 if ok else 1)
PYEOF
then :; else pf=1; fi
if [ "$pf" -ne 0 ]; then restore; die "postflight failed: targets restored from backup. Nothing changed."; fi
say "  postflight clean"

# ---------------------------------------------------------------- consolidated receipt (MEASUREMENT.md §5)
say ""
say "== section-5 consolidated receipt (validations re-run now; zero inference) =="
mkdir -p "$ROOT/artifacts/operator"
ev_args=()
for f in README.md SHA256SUMS equivalence_summary.json negative_control.json offline_tests.tsv \
         runner_79721927_to_6dea92dd.diff runner_6dea92dd_to_20a97fbd.diff \
         equivalence_harness.py offline_tests.sh; do
  ev_args+=(--evidence "$EVID_REL/$f")
done
q() { printf '%q' "$1"; }
V_TESTS="cd $(q "$ROOT") && python3 -m pytest -q -p no:cacheprovider scripts/benchmark/test_dflash2_followups.py scripts/benchmark/test_p3_bakeoff.py"
V_TESTS_SUBDIR="cd $(q "$ROOT/scripts/benchmark") && python3 -m pytest -q -p no:cacheprovider test_p3_bakeoff.py"
V_EQUIV="cd $(q "$ROOT") && python3 $EVID_REL/equivalence_harness.py --repo . --out $(q "$TMPD/equiv")"
V_NEG="cd $(q "$ROOT") && python3 $EVID_REL/equivalence_harness.py --repo . --out $(q "$TMPD/neg") --negative-control"
receipt_rc=0
python3 "$RECEIPT_TOOL" emit \
  --repo-root "$EPYC_ROOT" \
  --evidence-repo "$ROOT" \
  --pre "$PRESNAP" \
  --protocol-id "EVAL-TOWER-INSTRUMENT-PIN" \
  --anchor '- **Trust boundary**: this file, its `protocols/` annexes, the era registry, the eval tower, and' \
  --ratification-id "$GATE_ID" \
  --script "$SCRIPT_PATH" \
  --operator "${RATIFY_OPERATOR:-${USER:-unknown}}" \
  "${ev_args[@]}" \
  --validation "$V_TESTS" \
  --validation "$V_TESTS_SUBDIR" \
  --validation "$V_EQUIV" \
  --validation "$V_NEG" \
  --out "$CONSOLIDATED" || receipt_rc=$?
if [ "$receipt_rc" -ne 0 ]; then
  refused="${CONSOLIDATED%.receipt.json}.refused-$(date -u +%Y%m%dT%H%M%SZ).receipt.json"
  [ -f "$CONSOLIDATED" ] && mv "$CONSOLIDATED" "$refused"
  restore
  die "consolidated receipt returned $receipt_rc (1 REFUSED, 2 COULD-NOT-CHECK). Re-pin rolled back; the receipt is kept at ${refused#$ROOT/} for reading."
fi

# ---------------------------------------------------------------- decision receipt
research_head="$(git -C "$ROOT" rev-parse HEAD)"
if ! python3 - "$RECEIPT" "$RATIFIED_AT" "$GATE_ID" "$CONSOLIDATED_REL" \
     "${RATIFY_OPERATOR:-${USER:-unknown}}" "$PATCH_SHA256" "$research_head" "$ROOT/$EVID_REL" \
     "$OLD_DFLASH2_RUNNER" "$OLD_P3_RUNNER" "$NEW_RUNNER" \
     "${TARGETS[*]}" "${PRE[*]}" "${POST[*]}" <<'PYEOF'
import json, sys
from pathlib import Path
(receipt, ts, gate, consolidated_rel, operator, patch_sha, head, evid,
 old_d, old_p, new, targets, pre, post) = sys.argv[1:15]
evid = Path(evid)
summary = json.loads((evid / "equivalence_summary.json").read_text())
neg = json.loads((evid / "negative_control.json").read_text())
sums = {line.split()[1]: line.split()[0] for line in (evid / "SHA256SUMS").read_text().splitlines() if line.strip()}
doc = {
  "schema": "epyc.instrument_repin.v1",
  "decision": "V7-RUNNER-RESEAL",
  "gate_id": gate,
  "status": "ratified",
  "ratified_at": ts,
  "operator": operator,
  "operator_decision": "2026-09-16: RE-SEAL the two sealed pins that no longer match scripts/benchmark/v7_quality_gate_runner.py",
  "research_head": head,
  "instrument": {"path": "scripts/benchmark/v7_quality_gate_runner.py", "sha256_now": new},
  "pins": [
    {"old": old_d, "new": new, "consumer": "scripts/benchmark/dflash2_followups.py EXPECTED.runner_sha256",
     "test": "scripts/benchmark/test_dflash2_followups.py::test_exact_runner_bytes_are_unchanged",
     "commits_crossed": ["da06b371"]},
    {"old": old_p, "new": new,
     "consumer": "artifacts/p3-shadow-bakeoff-20260728/manifest/p3_bakeoff_manifest.json capture.runner.sha256 + duties.coder.suites.livecodebench_hard.scorer.runner_sha256 (+ .sha256 sidecar)",
     "test": "scripts/benchmark/test_p3_bakeoff.py::test_real_manifest_verifies",
     "commits_crossed": ["baf36757", "da06b371"]},
  ],
  "justification": {
    "da06b371": "2026-08-26 purely additive: opt-in --belief-category (default None) and --belief-config (default ''); finalize block guarded by `if args.belief_category is not None`; absent flag -> return 0 as before",
    "baf36757": "2026-08-19 NOT named in the operator decision but crossed by the 79721927 re-seal: request bytes and scoring unchanged; ADDS output keys per_question.effective_request and meta.sampling_fields_are_requested_not_effective. No P3 bake-off capture exists under the manifest, so no banked output is re-labelled",
  },
  "equivalence_evidence": {
    "directory": "data/v7-runner-repin-20260916",
    "sha256sums": sums,
    "verdict": summary["verdict"],
    "scenarios": {n: {v: {"requests_byte_identical": r["requests_byte_identical"],
                          "outputs_identical_after_volatile_strip": r["outputs_identical_after_volatile_strip"],
                          "residual_paths": r["residual_paths"]}
                      for v, r in e["vs"].items()}
                  for n, e in summary["scenarios"].items()},
    "negative_control": {k: v["detected"] for k, v in neg["results"].items()},
    "offline_suites": (evid / "offline_tests.tsv").read_text().splitlines(),
    "method": "zero inference: each runner version (git blob) run as a subprocess against a deterministic 127.0.0.1 stub; validations re-run at apply time are in the consolidated receipt",
  },
  "not_repinned": {
    "banked captures and sealed bundles (artifacts/**, data/kernel-v9-candidate/**)": "history; record the runner that produced them",
    "scripts/benchmark/replay_tc_nothink_v4.py": "validates banked rows against 79721927; correct as-is",
    "scripts/benchmark/laguna_q4_cpu_bench_runner.py": "pins the completed Laguna Q4 CPU campaign evaluator; out of this decision's scope",
  },
  "critic_tasks_v1": "NOT committed: embeds 120 LiveCodeBench problem statements (third-party, unclear redistribution licence) and banked responses whose sources are deliberately untracked; stays banked, hash-pinned d97692840da2. Test failure was a cwd-relative locator defect, fixed by resolving relative manifest pins against RESEARCH_ROOT",
  "patch": {"path": "artifacts/operator/v7-runner-repin-20260916.patch", "sha256": patch_sha},
  "state": {t: {"sha256_before": a, "sha256_after": b}
            for t, a, b in zip(targets.split(), pre.split(), post.split())},
  "consolidated_receipt": consolidated_rel,
  "applied_by": "scripts/operator/ratify_v7_runner_repin_20260916.sh",
}
with open(receipt, "x", encoding="utf-8") as fh:
    json.dump(doc, fh, indent=2); fh.write("\n")
print(f"  decision receipt: {receipt}")
PYEOF
then
  restore; rm -f "$CONSOLIDATED"
  die "could not write the decision receipt. Re-pin rolled back."
fi

COMMIT_PATHS=("${TARGETS[@]}" "$RECEIPT_REL" "$CONSOLIDATED_REL")
say ""
say "APPLIED, NOT STAGED, NOT COMMITTED. Review, then commit exactly these ${#COMMIT_PATHS[@]} paths"
say "(preflight verified they were clean, so adding them whole takes no peer hunk):"
say ""
say "    git -C $ROOT add -- ${COMMIT_PATHS[*]}"
say "    git -C $ROOT diff --cached --stat"
say "    git -C $ROOT commit -m 'RATIFIED: re-seal v7 runner pins 6dea92dd/79721927 -> 20a97fbd (2026-09-16)' -- ${COMMIT_PATHS[*]}"
