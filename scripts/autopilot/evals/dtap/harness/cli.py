"""CLI for the DTAP disposable runner.

Examples (from the dtap/ directory):
  python3 -m harness list-cases
  python3 -m harness attest
  python3 -m harness run --case finance-benign-trade-execution-001 --arm done --stub
  python3 -m harness run --case finance-indirect-action-reversal-002 --arm compromised --stub --out /tmp/dtap-results
  python3 -m harness matrix --case finance-indirect-action-reversal-002 --arms compliant compromised --seeds 5 --stub
  python3 -m harness run --case finance-benign-trade-execution-001 --arm done \
      --endpoint http://localhost:8080/v1 --model qwen3-32b --temperature 0.0
  python3 -m harness replay --trace results/traces/<case>.<arm>.seed0.trace.jsonl

Dry-run (--stub) performs zero inference. Live mode (--endpoint) executes the
case prompt against an OpenAI-compatible local endpoint; environment tool
effects are recorded but not applied (no simulated services), so live judges
observe the recorded trajectory and an empty environment state.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .endpoint import ChatEndpoint, DryRunStub
from .outcomes import ALL_OUTCOME_TYPES
from .runner import (
    ARMS,
    ArmConfig,
    CaseRegistry,
    replay_trace,
    run_case,
    run_matrix,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DTAP bounded-subset disposable runner (EVL-46)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list-cases", help="list imported subset cases")

    run = sub.add_parser("run", help="run one case/arm")
    run.add_argument("--case", required=True)
    run.add_argument("--arm", default=None, help=f"arm label: {', '.join(ARMS)}")
    run.add_argument("--stub", action="store_true", help="dry-run with fixture stub (zero inference)")
    run.add_argument("--endpoint", default=None, help="OpenAI-compatible base URL, e.g. http://localhost:8080/v1")
    run.add_argument("--api-key", default="none")
    run.add_argument("--model", default="local")
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--max-tokens", type=int, default=1024)
    run.add_argument("--max-turns", type=int, default=8)
    run.add_argument("--retries", type=int, default=2)
    run.add_argument("--timeout", type=float, default=60.0)
    run.add_argument(
        "--tool-contract", choices=["dtap-native-tools-v1"], default=None,
        help="opt in to the pinned public native tool schemas (default: legacy message-only request)",
    )
    run.add_argument(
        "--endpoint-mode", choices=["openai-compatible", "orchestrator-client"],
        default="openai-compatible",
        help="explicit request mode; never inferred from the endpoint URL",
    )
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--seeds", type=int, default=None, help="repeat N seeds (implies matrix)")
    run.add_argument("--out", default="results")

    mat = sub.add_parser("matrix", help="repeated seeds with Wilson CIs")
    mat.add_argument("--case", required=True)
    mat.add_argument("--arms", nargs="+", default=None)
    mat.add_argument(
        "--compare-arm",
        action="append",
        default=None,
        metavar="BASELINE:CANDIDATE",
        help="explicit paired comparison for failure_ledger.json (repeatable; no inferred pairs)",
    )
    mat.add_argument("--seeds", type=int, default=5)
    mat.add_argument("--stub", action="store_true")
    mat.add_argument("--endpoint", default=None)
    mat.add_argument("--api-key", default="none")
    mat.add_argument("--model", default="local")
    mat.add_argument("--temperature", type=float, default=0.0)
    mat.add_argument("--max-tokens", type=int, default=1024)
    mat.add_argument("--max-turns", type=int, default=8)
    mat.add_argument("--retries", type=int, default=2)
    mat.add_argument("--timeout", type=float, default=60.0)
    mat.add_argument(
        "--tool-contract", choices=["dtap-native-tools-v1"], default=None,
        help="opt in to the pinned public native tool schemas (default: legacy message-only request)",
    )
    mat.add_argument(
        "--endpoint-mode", choices=["openai-compatible", "orchestrator-client"],
        default="openai-compatible",
        help="explicit request mode; never inferred from the endpoint URL",
    )
    mat.add_argument("--out", default="results")

    rp = sub.add_parser("replay", help="verify + deterministically replay a trace")
    rp.add_argument("--trace", required=True)

    sub.add_parser("check-outcomes", help="assert the typed-outcome set is exact")

    att = sub.add_parser(
        "attest",
        help="verify the judge attestation (upstream bytes + wrapper identity)",
    )
    att.add_argument(
        "--update",
        action="store_true",
        help="re-attest the WRAPPER-side manifest facts (never the upstream digests)",
    )
    return p


def _endpoint_factory_from(args: argparse.Namespace, arm_config: ArmConfig):
    """Return a Callable[[case_id, arm, seed], endpoint] honoring --stub/--endpoint."""
    if getattr(args, "stub", False):
        from .env_state import StateStore

        def _stub_factory(case_id: str, arm: str, seed: int) -> DryRunStub:
            fixture = StateStore().load_fixture(case_id, arm)
            return DryRunStub(fixture, seed=seed)

        return _stub_factory
    if getattr(args, "endpoint", None):
        endpoint = ChatEndpoint(
            args.endpoint,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            retries=args.retries,
            timeout=args.timeout,
        )
        return lambda _case_id, _arm, _seed: endpoint
    raise SystemExit("live mode needs --endpoint (or pass --stub for a zero-inference dry run)")


def _arm_config_from(args: argparse.Namespace) -> ArmConfig:
    return ArmConfig(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_turns=args.max_turns,
        retries=args.retries,
        timeout=args.timeout,
    )


def _default_arm(case: dict) -> str:
    return "done" if case["threat"] == "benign" else "compliant"


def _comparison_pairs_from(values: list[str] | None) -> list[tuple[str, str]] | None:
    if values is None:
        return None
    pairs = []
    seen = set()
    for value in values:
        if not isinstance(value, str) or value.count(":") != 1:
            raise SystemExit("--compare-arm must be BASELINE:CANDIDATE")
        baseline, candidate = value.split(":", 1)
        if not baseline or not candidate:
            raise SystemExit("--compare-arm must name two nonempty arms")
        if baseline == candidate:
            raise SystemExit("--compare-arm baseline and candidate must differ")
        pair = (baseline, candidate)
        if pair in seen:
            raise SystemExit(f"duplicate --compare-arm: {value}")
        seen.add(pair)
        pairs.append(pair)
    return pairs


def _tool_contract_from(args: argparse.Namespace, registry: CaseRegistry):
    selected = getattr(args, "tool_contract", None)
    if selected is None:
        if getattr(args, "endpoint_mode", "openai-compatible") == "orchestrator-client":
            raise SystemExit("--endpoint-mode orchestrator-client requires --tool-contract dtap-native-tools-v1")
        return None
    from .tool_contract import load_native_tool_contract

    return load_native_tool_contract(registry.cases, registry_path=registry.path)


def main(argv: list | None = None) -> int:
    args = _build_parser().parse_args(argv)
    registry = CaseRegistry()
    native_tool_contract = _tool_contract_from(args, registry)

    if args.command == "list-cases":
        for case_id in sorted(registry.cases):
            c = registry.cases[case_id]
            print(
                f"{case_id:60s} {c['threat']:8s} {c['domain']:8s} "
                f"fams={','.join(c['injection_families']) or '-'}"
            )
        return 0

    if args.command == "check-outcomes":
        missing = {"model", "parser", "tool", "endpoint", "harness", "judge", "infrastructure", "overflow"} - ALL_OUTCOME_TYPES
        extra = ALL_OUTCOME_TYPES - {"model", "parser", "tool", "endpoint", "harness", "judge", "infrastructure", "overflow"}
        print(f"typed outcome set: {sorted(ALL_OUTCOME_TYPES)}")
        print(f"missing: {sorted(missing) or 'none'}; extra: {sorted(extra) or 'none'}")
        return 0 if not missing and not extra else 1

    if args.command == "attest":
        from .attest import main as attest_main

        return attest_main(update_manifest=args.update)

    if args.command == "run":
        case = registry.get(args.case)
        arm = args.arm or _default_arm(case)
        arm_config = _arm_config_from(args)
        factory = _endpoint_factory_from(args, arm_config)
        if args.seeds:
            rows = run_matrix(
                [args.case],
                [arm],
                list(range(args.seeds)),
                factory,
                arm_config,
                Path(args.out),
                native_tool_contract=native_tool_contract,
                endpoint_mode=args.endpoint_mode,
            )
            print(json.dumps(rows, indent=2, sort_keys=True))
            return 0
        endpoint = factory(args.case, arm, args.seed)
        result = run_case(
            args.case, arm, args.seed, endpoint, arm_config, Path(args.out), registry,
            native_tool_contract=native_tool_contract, endpoint_mode=args.endpoint_mode,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0 if result.status == "ok" else 2

    if args.command == "matrix":
        case = registry.get(args.case)
        arms = args.arms or (["done", "none"] if case["threat"] == "benign" else ["compliant", "compromised"])
        arm_config = _arm_config_from(args)
        factory = _endpoint_factory_from(args, arm_config)
        rows = run_matrix(
            [args.case], arms, list(range(args.seeds)), factory, arm_config, Path(args.out),
            native_tool_contract=native_tool_contract, endpoint_mode=args.endpoint_mode,
            comparison_pairs=_comparison_pairs_from(args.compare_arm),
        )
        print(json.dumps(rows, indent=2, sort_keys=True))
        return 0

    if args.command == "replay":
        from .outcomes import RunFailure

        try:
            report = replay_trace(Path(args.trace))
        except RunFailure as exc:
            print(json.dumps(exc.to_outcome(), indent=2, sort_keys=True))
            return 3
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["matches"] else 3

    return 1


if __name__ == "__main__":
    sys.exit(main())
