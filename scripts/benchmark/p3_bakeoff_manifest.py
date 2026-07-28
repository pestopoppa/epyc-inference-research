#!/usr/bin/env python3
"""Build/verify the PINNED Phase-3 shadow bake-off manifest (P3-1).

Zero-inference.  The manifest is the pairing-discipline anchor: it records
the exact question ids, per-duty prompts (by sha256-pinned file), sampling
parameters, arm/model identities, and scorer identities for the stock-27B
vs Fable-Fusion bake-off on the GPU shadow lane, plus the A4 sequential
control arm.  Every later run (operator windows) replays THESE files; a
hash mismatch fails closed.

Task sets (constraint 1 -- reuse, do not resample):
- coder/swebench_oracle: artifacts/architect-code-eval-20260724/questions_swebench_oracle.json (n=40)
- coder/livecodebench_hard: .../questions_livecodebench_hard.json (n=53)
- FG-1 "hard-core" tag: the 14 SWE instances unsolved by all six FG-1 arms,
  extracted from fg1_results.json -> swe40.unsolved_by_all_six.
- cocritic: the pinned critic task file built by p3_bakeoff_critic_build.py.

Usage:
    p3_bakeoff_manifest.py build --critic-tasks <critic_tasks.json> \
        --output <manifest.json> [--model-hashes <sha256sum-output>]
    p3_bakeoff_manifest.py verify --manifest <manifest.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p3_bakeoff_common import (  # noqa: E402
    CRITIC_SUITE,
    MANIFEST_SCHEMA_VERSION,
    paired_mde,
    sha256_file,
    sha256_text,
    write_json,
)

RESEARCH_ROOT = Path("/mnt/raid0/llm/epyc-inference-research")
CODE_EVAL_ART = RESEARCH_ROOT / "artifacts/architect-code-eval-20260724"
FG1_RESULTS = (
    RESEARCH_ROOT
    / "artifacts/architect-27b-finetunes-v8-20260726"
    / "fg1-fine-grain-replay-20260727/fg1_results.json"
)
RUNNER = RESEARCH_ROOT / "scripts/benchmark/v7_quality_gate_runner.py"
SR_CONVERTER = CODE_EVAL_ART / "convert_sr_to_patch.py"
CRITIC_SCORER = RESEARCH_ROOT / "scripts/benchmark/p3_bakeoff_critic_score.py"

# Production sampling discipline (matches the 2026-07-24 architect code-eval
# arms and production temp+seed policy; no-think protocol).
SAMPLING = {
    "endpoint": "chat",
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "seed": 42,
    "enable_thinking": False,
    "repeats": 1,
    "concurrency": 1,
}

# Arms.  Tenant identities per gpu-shadow-lane.md section 5 (stock sha256 is
# the lane-spec pinned value, re-verifiable via --model-hashes).  MTP OFF is
# the lane default (D6): both 27B arms use non-MTP GGUFs.  The A4 control
# arm runs SEQUENTIALLY in bench windows (co-residency impossible: 37.8GB).
ARMS = {
    "stock27b": {
        "label": "stock-27B",
        "model_path": "/mnt/raid0/llm/models/Qwen_Qwen3.6-27B-Q8_0.gguf",
        "expected_bytes": 28665067072,
        "sha256": "5927dc06c2b19f732fb6e2a6546dff4c130b552f2ab5f91feb3daafe43897b2a",
        "serving": "gpu_shadow_lane port 18100 (shadow role coder_escalation_shadow)",
        "role_in_bakeoff": "incumbent-architecture tenant candidate",
    },
    "ff27b": {
        "label": "Fable-Fusion-27B",
        "model_path": (
            "/mnt/raid0/llm/models/Qwen3.6-27B-Fable-Fusion-711-GGUF/"
            "Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-Q8_0.gguf"
        ),
        "expected_bytes": 29787701792,
        "sha256": None,  # filled from --model-hashes when available
        "serving": "gpu_shadow_lane port 18100 (tenant swap, State-B' choreography)",
        "role_in_bakeoff": "challenger tenant (D7: retained as bake-off alternative)",
    },
    "a4_control": {
        "label": "A4-35B-A3B (control)",
        "model_path": "/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf",
        "expected_bytes": 37801097504,
        "sha256": None,  # filled from --model-hashes when available
        "serving": (
            "sequential GPU bench window only (37.8GB; co-residency with the "
            "phase-2 tenant set impossible) -- P3-1"
        ),
        "role_in_bakeoff": "production coder_escalation incumbent (A4), control arm",
    },
}


def load_hard_core(fg1_path: Path, swe_ids: set[str]) -> list[str]:
    """Extract the FG-1 discriminating hard-core instance ids (fail-closed)."""
    fg1 = json.loads(fg1_path.read_text())
    ids = fg1["swe40"]["unsolved_by_all_six"]
    if not isinstance(ids, list) or not ids:
        raise ValueError("fg1_results.json swe40.unsolved_by_all_six missing/empty")
    unknown = sorted(set(ids) - swe_ids)
    if unknown:
        raise ValueError(f"hard-core ids not in SWE manifest: {unknown}")
    return sorted(ids)


def pin_file(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def build_manifest(
    critic_tasks: Path,
    model_hashes: dict[str, str],
    *,
    created_utc: str | None = None,
    swe_questions: Path | None = None,
    lcb_questions: Path | None = None,
    fg1_results: Path | None = None,
) -> dict:
    swe_questions = swe_questions or (CODE_EVAL_ART / "questions_swebench_oracle.json")
    lcb_questions = lcb_questions or (CODE_EVAL_ART / "questions_livecodebench_hard.json")
    fg1_results = fg1_results or FG1_RESULTS

    swe_items = json.loads(swe_questions.read_text())
    lcb_items = json.loads(lcb_questions.read_text())
    swe_ids = [q["id"] for q in swe_items]
    lcb_ids = [q["id"] for q in lcb_items]
    if len(set(swe_ids)) != len(swe_ids) or len(set(lcb_ids)) != len(lcb_ids):
        raise ValueError("duplicate question ids in a pinned question file")
    hard_core = load_hard_core(fg1_results, set(swe_ids))

    critic_payload = json.loads(critic_tasks.read_text())
    critic_rows = critic_payload["suites"][CRITIC_SUITE]
    critic_prev = critic_payload.get("prevalence", {})

    arms = {}
    for key, arm in ARMS.items():
        entry = dict(arm)
        path = Path(entry["model_path"])
        if path.exists():
            size = path.stat().st_size
            if size != entry["expected_bytes"]:
                raise ValueError(
                    f"{key}: on-disk size {size} != expected {entry['expected_bytes']}"
                )
        if entry["sha256"] is None:
            entry["sha256"] = model_hashes.get(str(path))
        arms[key] = entry

    runner_sha = sha256_file(RUNNER)
    n_swe, n_lcb, n_crit = len(swe_ids), len(lcb_ids), len(critic_rows)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_utc": created_utc or datetime.now(timezone.utc).isoformat(),
        "program": {
            "authority": "epyc-root/handoffs/active/gpu-serving-tie-in-program.md (P3-1, D1-D10)",
            "lane_spec": "epyc-orchestrator/docs/gpu-shadow-lane.md",
            "spec": "docs/design/p3-shadow-bakeoff-spec.md",
        },
        "invariants": {
            "eval_path_only": "All traffic is eval-path, forced-role; NEVER live /chat (D3).",
            "not_authorized": (
                "This bake-off authorizes NO lineup change. The registry stays "
                "frozen; coder_escalation stays A4-bound until P3-3 operator "
                "three-gates sign-off (D3). Results feed the P3-2 tenancy "
                "decision package only."
            ),
            "servers": "The harness NEVER launches servers; lane is brought up via the operator-gated Steps 0-7 choreography.",
        },
        "arms": arms,
        "duties": {
            "coder": {
                "description": (
                    "Escalation-shaped SWE-flavored tasks at production sampling "
                    "(no-think). Two pinned suites, scored separately."
                ),
                "suites": {
                    "swebench_oracle": {
                        "questions_file": pin_file(swe_questions),
                        "n": n_swe,
                        "ids_sha256": sha256_text(json.dumps(swe_ids)),
                        "max_tokens": 3072,
                        "request_timeout_s": 3600,
                        "sampling": SAMPLING,
                        "scorer": {
                            "id": "swebench-verified-harness+convert_sr_to_patch",
                            "converter": pin_file(SR_CONVERTER),
                            "harness": (
                                ".venv-swebench swebench harness via "
                                "swebench_cpuset_adapter (cpuset 112-119); "
                                "report resolved_ids is authoritative"
                            ),
                            "deferred": True,
                        },
                        "hard_core_tag": {
                            "source": pin_file(fg1_results),
                            "extraction": "swe40.unsolved_by_all_six (14/40 unsolved by all six FG-1 arms)",
                            "ids": hard_core,
                            "analysis": "descriptive subset only, never a gate",
                        },
                    },
                    "livecodebench_hard": {
                        "questions_file": pin_file(lcb_questions),
                        "n": n_lcb,
                        "ids_sha256": sha256_text(json.dumps(lcb_ids)),
                        "max_tokens": 4096,
                        "request_timeout_s": 1800,
                        "sampling": SAMPLING,
                        "scorer": {
                            "id": "answer_scoring.code_execution (executable pass@1, scored at capture; offline re-scorable from captures)",
                            "runner_sha256": runner_sha,
                            "deferred": False,
                        },
                    },
                },
            },
            "cocritic": {
                "description": (
                    "Typed-verdict review duty: [candidate solution + typed "
                    "review request] -> ReviewDecision-shaped verdict, scored "
                    "against executable-oracle gold labels."
                ),
                "suite": CRITIC_SUITE,
                "tasks_file": pin_file(critic_tasks),
                "n": n_crit,
                "prevalence": critic_prev,
                "max_tokens": 1024,
                "request_timeout_s": 1800,
                "sampling": SAMPLING,
                "decision_vocabulary": (
                    "epyc-orchestrator/orchestration/review_decision.schema.json "
                    "(approve/reject/reject_to_empty/request_changes/"
                    "request_evidence/abstain/escalate)"
                ),
                "scorer": {
                    "id": "p3_bakeoff_critic_score.v1",
                    "source": pin_file(CRITIC_SCORER),
                    "deferred": True,
                },
            },
        },
        "capture": {
            "runner": {"path": str(RUNNER), "sha256": runner_sha},
            "capture_schema_version": "v7_quality_gate_capture.v4",
            "per_question_incremental": True,
        },
        "statistical_plan": {
            "primary": "paired per-question comparison per duty+suite; exact two-sided McNemar on discordant pairs",
            "n": {"swebench_oracle": n_swe, "livecodebench_hard": n_lcb, CRITIC_SUITE: n_crit},
            "mde_note": {
                "swebench_oracle": (
                    f"n={n_swe}: at an assumed discordant rate of 0.20 "
                    f"(FG-1 measured FF-vs-stock 8/40), MDE ~= "
                    f"{paired_mde(n_swe, 0.20):.2f} accuracy difference "
                    "(alpha=.05, power=.8). Small quality gaps are NOT resolvable "
                    "at this n; quality ties are decided on token-efficiency + "
                    "latency in the P3-2 package."
                ),
                "livecodebench_hard": (
                    f"n={n_lcb}: at discordant rate 0.25, MDE ~= "
                    f"{paired_mde(n_lcb, 0.25):.2f}."
                ),
                CRITIC_SUITE: (
                    f"n={n_crit}: at discordant rate 0.25, MDE ~= "
                    f"{paired_mde(max(n_crit, 1), 0.25):.2f} on verdict accuracy; "
                    "FA/FR reported with Cohen's kappa + prevalence disclosure "
                    "(intake-876), abstention estimand declared."
                ),
            },
            "secondary": (
                "token economics (median completion tokens, tokens/solved) as "
                "paired per-question observations; decode telemetry is "
                "observation-grade only"
            ),
        },
    }
    return manifest


def verify_manifest(manifest: dict) -> list[str]:
    """Re-hash every pinned source; return a list of failures (empty = ok)."""
    failures: list[str] = []

    def check(pin: dict, label: str) -> None:
        path = Path(pin["path"])
        if not path.exists():
            failures.append(f"{label}: missing {path}")
            return
        actual = sha256_file(path)
        if actual != pin["sha256"]:
            failures.append(f"{label}: sha256 mismatch for {path} "
                            f"(pinned {pin['sha256'][:12]}, actual {actual[:12]})")

    duties = manifest["duties"]
    for suite, spec in duties["coder"]["suites"].items():
        check(spec["questions_file"], f"coder/{suite}/questions")
        if "hard_core_tag" in spec:
            check(spec["hard_core_tag"]["source"], f"coder/{suite}/hard_core_source")
        if spec["scorer"].get("converter"):
            check(spec["scorer"]["converter"], f"coder/{suite}/converter")
    check(duties["cocritic"]["tasks_file"], "cocritic/tasks")
    check(duties["cocritic"]["scorer"]["source"], "cocritic/scorer")
    check(manifest["capture"]["runner"], "capture/runner")
    for key, arm in manifest["arms"].items():
        path = Path(arm["model_path"])
        if not path.exists():
            failures.append(f"arm {key}: model missing {path}")
        elif path.stat().st_size != arm["expected_bytes"]:
            failures.append(f"arm {key}: model size mismatch")
    return failures


def parse_model_hashes(path: Path | None) -> dict[str, str]:
    """Parse ``sha256sum`` output into {abs_path: sha256}."""
    if path is None:
        return {}
    hashes: dict[str, str] = {}
    for line in path.read_text().splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2 and len(parts[0]) == 64:
            hashes[parts[1].strip()] = parts[0]
    return hashes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--critic-tasks", type=Path, required=True)
    b.add_argument("--output", type=Path, required=True)
    b.add_argument("--model-hashes", type=Path, default=None,
                   help="sha256sum output covering FF/A4 model files")
    b.add_argument("--created-utc", default=None,
                   help="Override timestamp (determinism testing)")
    v = sub.add_parser("verify")
    v.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()

    if args.cmd == "build":
        manifest = build_manifest(
            args.critic_tasks,
            parse_model_hashes(args.model_hashes),
            created_utc=args.created_utc,
        )
        digest = write_json(args.output, manifest, sort_keys=False)
        missing = [k for k, a in manifest["arms"].items() if not a["sha256"]]
        if missing:
            print(f"[manifest] WARNING: arms without model sha256: {missing} "
                  "(pass --model-hashes; required before execution)",
                  file=sys.stderr)
        print(f"[manifest] written {args.output} sha256={digest[:16]}...")
        return 0

    manifest = json.loads(args.manifest.read_text())
    failures = verify_manifest(manifest)
    if failures:
        for f in failures:
            print(f"[manifest] FAIL {f}", file=sys.stderr)
        return 1
    print("[manifest] verify OK: all pinned sources match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
