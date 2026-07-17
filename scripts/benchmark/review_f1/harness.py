#!/usr/bin/env python3
"""EV-13 review-finding-F1 harness: /v1/chat/completions review driver.

Drives a LOCAL model (served by llama-server) through a code-review task over
diff + surrounding-file context, per PR, for >=3 runs, and persists the parsed
review findings so :mod:`scorer` can compute micro-averaged P/R/F1.

BUILD-LEG CONTRACT (this file is inference-free unless a real transport is
wired at run time):
  * Real transport = stdlib :mod:`urllib` (guaranteed importable; no requests/
    httpx dependency). Tests inject a ``MockTransport``; ``--dry-run`` prints
    the request plan and sends nothing.
  * Per-PR INCREMENTAL persistence: one JSON file per PR, resume-safe. A PR
    whose file already holds all requested runs is skipped on ``--resume``.
  * Results indexed by MODEL/QUANT, NEVER by role (feedback_model_not_role_indexing).
  * Judge-swap plumbing (EV-6 <=2pp cross-family check) is recorded as config
    on every result; the semantic judge itself is a later inference entry.
  * enable_thinking=False in the payload (Qwen3.x rule); production-style
    sampling with a per-run seed so the >=3 runs vary for the StdDev protocol.

The response parser turns the model's JSON review output into findings of the
shape scorer expects: ``{criterion, location:{file,line_start,line_end}, comment}``.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

try:  # runnable both as a package module and as a bare script
    from . import scorer  # type: ignore
except ImportError:  # pragma: no cover - script-mode fallback
    import scorer  # type: ignore

DEFAULT_TEMPERATURE = 0.6
DEFAULT_BASE_SEED = 42
DEFAULT_MAX_TOKENS = 2048
DEFAULT_TIMEOUT = 600
DEFAULT_RUNS = 3

SYSTEM_PROMPT = (
    "You are a meticulous code reviewer. Given a pull-request diff and the "
    "surrounding file context, identify concrete bugs. Respond with ONLY a JSON "
    "object of the form {\"findings\": [{\"criterion\": one of "
    "runtime_error|logic_bug|performance|security, \"file\": path, "
    "\"line_start\": int, \"line_end\": int, \"comment\": short explanation}]}. "
    "Report a real defect for each finding; do not include style-only nits."
)


# --------------------------------------------------------------------------- #
# Transports
# --------------------------------------------------------------------------- #
class UrllibTransport:
    """Real transport (stdlib only). Only used at RUN time, never in tests."""

    def post(self, url: str, payload: dict, timeout: int) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (local server)
            return json.loads(resp.read().decode("utf-8"))


class MockTransport:
    """Deterministic transport for tests. Serves canned completions.

    ``responder`` maps (case_id, run_index) -> assistant message string.
    A single string or a dict may also be given.
    """

    def __init__(self, responder: Callable[[str, int], str] | dict | str):
        self.responder = responder
        self.calls: list[dict] = []

    def post(self, url: str, payload: dict, timeout: int) -> dict:
        meta = payload.get("_meta", {})
        case_id, run_index = meta.get("case_id", ""), meta.get("run_index", 0)
        if callable(self.responder):
            content = self.responder(case_id, run_index)
        elif isinstance(self.responder, dict):
            content = self.responder.get(case_id, '{"findings": []}')
        else:
            content = self.responder
        self.calls.append({"url": url, "case_id": case_id, "run_index": run_index})
        return {"choices": [{"message": {"role": "assistant", "content": content}}]}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class HarnessConfig:
    golden_path: str
    out_dir: str
    server_url: str = "http://127.0.0.1:8080"
    model: str = "unknown-model"
    quant: str = "unknown-quant"
    judge_model: str | None = None
    judge_quant: str | None = None
    runs: int = DEFAULT_RUNS
    temperature: float = DEFAULT_TEMPERATURE
    base_seed: int = DEFAULT_BASE_SEED
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout: int = DEFAULT_TIMEOUT
    resume: bool = True
    context_dir: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def model_quant_key(self) -> str:
        return f"{self.model}__{self.quant}"

    def result_root(self) -> Path:
        return Path(self.out_dir) / self.model_quant_key

    def judge_config(self) -> dict:
        """EV-6 judge-swap plumbing. Recorded on every result; the semantic
        judge run itself is a separate inference-gated manifest entry."""
        return {
            "judge_model": self.judge_model,
            "judge_quant": self.judge_quant,
            "cross_family_required": bool(self.judge_model)
            and self.judge_model != self.model,
            "swap_tolerance_pp": 2.0,
            "matcher": "deterministic-criterion-location (build leg); "
            "semantic-judge is a later inference entry",
        }


# --------------------------------------------------------------------------- #
# Request assembly + parsing
# --------------------------------------------------------------------------- #
def _load_diff(case: dict, config: HarnessConfig) -> str:
    inline = case.get("pr_ref", {}).get("diff")
    if inline:
        return inline
    diff_path = case.get("pr_ref", {}).get("diff_path")
    if diff_path and config.context_dir:
        p = Path(config.context_dir) / diff_path
        if p.exists():
            return p.read_text()
    return case.get("pr_ref", {}).get("diff_placeholder", "<diff not available in build leg>")


def build_payload(case: dict, run_index: int, config: HarnessConfig) -> dict:
    diff = _load_diff(case, config)
    pr = case.get("pr_ref", {})
    user = (
        f"Repository: {pr.get('repo', '?')}\nPR #{pr.get('number', '?')}: "
        f"{pr.get('title', '')}\n\n--- DIFF + CONTEXT ---\n{diff}\n"
    )
    return {
        "model": config.model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "temperature": config.temperature,
        "seed": config.base_seed + run_index,
        "max_tokens": config.max_tokens,
        "enable_thinking": False,
        "_meta": {"case_id": case["case_id"], "run_index": run_index},
    }


def parse_findings(content: str) -> list[dict]:
    """Deterministically parse a model completion into scorer findings.

    Tolerant: accepts a bare JSON object/array, or the first JSON object found
    in prose. Unparseable output yields an empty finding list (scored as all
    FN for that PR — a real, non-crashing outcome)."""
    obj = _extract_json(content)
    if obj is None:
        return []
    raw = obj.get("findings", obj) if isinstance(obj, dict) else obj
    if not isinstance(raw, list):
        return []
    findings = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        loc = None
        if item.get("file") is not None:
            loc = {
                "file": item.get("file"),
                "line_start": item.get("line_start"),
                "line_end": item.get("line_end"),
            }
        findings.append(
            {
                "criterion": item.get("criterion", "unspecified"),
                "location": loc,
                "comment": item.get("comment", ""),
            }
        )
    return findings


def _extract_json(content: str) -> Any:
    content = (content or "").strip()
    if not content:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


# --------------------------------------------------------------------------- #
# Persistence (one file per PR, resume-safe)
# --------------------------------------------------------------------------- #
def _load_golden(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _pr_result_path(config: HarnessConfig, case_id: str) -> Path:
    return config.result_root() / f"{case_id}.json"


def _completed_runs(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text())
        return len(data.get("runs", []))
    except (json.JSONDecodeError, OSError):
        return 0


def run(config: HarnessConfig, transport: Any) -> dict:
    """Drive every PR x run through ``transport``; persist per PR incrementally.

    Returns a summary dict incl. the aggregated Mean-F1/StdDev over the golden
    findings (deterministic scorer). Callers pass MockTransport in tests."""
    golden = _load_golden(config.golden_path)
    cases = golden["cases"]
    root = config.result_root()
    root.mkdir(parents=True, exist_ok=True)

    per_case_runs: dict[str, list[dict[str, list[dict]]]] = {}
    skipped = 0
    for case in cases:
        case_id = case["case_id"]
        path = _pr_result_path(config, case_id)
        existing = _load_pr_file(path)
        runs_findings: list[list[dict]] = list(existing.get("runs", []))
        if config.resume and len(runs_findings) >= config.runs:
            skipped += 1
            per_case_runs[case_id] = _as_run_maps(case_id, runs_findings)
            continue
        for run_index in range(len(runs_findings), config.runs):
            payload = build_payload(case, run_index, config)
            resp = transport.post(f"{config.server_url}/v1/chat/completions", payload, config.timeout)
            content = resp["choices"][0]["message"]["content"]
            runs_findings.append(parse_findings(content))
            _write_pr_file(path, config, case, runs_findings)  # persist AFTER each run
        per_case_runs[case_id] = _as_run_maps(case_id, runs_findings)

    runs_as_maps = _pivot_runs(cases, per_case_runs, config.runs)
    aggregate = scorer.aggregate_runs(cases, runs_as_maps)
    summary = {
        "model": config.model,
        "quant": config.quant,
        "model_quant_key": config.model_quant_key,
        "judge_config": config.judge_config(),
        "runs_requested": config.runs,
        "cases_scored": len(cases),
        "cases_skipped_resume": skipped,
        "golden_checksum": golden.get("checksum"),
        "aggregate": aggregate,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (root / "_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _load_pr_file(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _write_pr_file(path: Path, config: HarnessConfig, case: dict, runs_findings: list[list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case_id": case["case_id"],
        "pr_ref": case.get("pr_ref", {}),
        "model": config.model,
        "quant": config.quant,
        "judge_config": config.judge_config(),
        "runs": runs_findings,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)  # atomic swap -> resume-safe against interruption


def _as_run_maps(case_id: str, runs_findings: list[list[dict]]) -> list[dict[str, list[dict]]]:
    return [{case_id: rf} for rf in runs_findings]


def _pivot_runs(cases, per_case_runs, n_runs) -> list[dict[str, list[dict]]]:
    runs = [dict() for _ in range(n_runs)]
    for case in cases:
        case_runs = per_case_runs.get(case["case_id"], [])
        for i in range(n_runs):
            findings = case_runs[i][case["case_id"]] if i < len(case_runs) else []
            runs[i][case["case_id"]] = findings
    return runs


# --------------------------------------------------------------------------- #
# Dry-run + CLI
# --------------------------------------------------------------------------- #
def print_dry_run(config: HarnessConfig) -> dict:
    golden = _load_golden(config.golden_path)
    cases = golden["cases"]
    plan = {
        "server_url": config.server_url,
        "endpoint": "/v1/chat/completions",
        "model_quant_key": config.model_quant_key,
        "judge_config": config.judge_config(),
        "runs_per_pr": config.runs,
        "n_prs": len(cases),
        "total_requests": len(cases) * config.runs,
        "result_root": str(config.result_root()),
        "golden_checksum": golden.get("checksum"),
        "enable_thinking": False,
        "sampling": {"temperature": config.temperature, "base_seed": config.base_seed},
        "sample_output_paths": [str(_pr_result_path(config, c["case_id"])) for c in cases[:3]],
    }
    print("=== review_f1 harness DRY-RUN (no server contacted) ===")
    print(json.dumps(plan, indent=2))
    return plan


def _config_from_args(args) -> HarnessConfig:
    return HarnessConfig(
        golden_path=args.golden,
        out_dir=args.out,
        server_url=args.server_url,
        model=args.model,
        quant=args.quant,
        judge_model=args.judge_model,
        judge_quant=args.judge_quant,
        runs=args.runs,
        temperature=args.temperature,
        base_seed=args.seed,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        resume=not args.no_resume,
        context_dir=args.context_dir,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EV-13 review-finding-F1 harness")
    p.add_argument("--golden", required=True, help="assembled golden set JSON")
    p.add_argument("--out", required=True, help="results root (indexed by model__quant)")
    p.add_argument("--server-url", default="http://127.0.0.1:8080")
    p.add_argument("--model", required=True, help="model under review (index key, NOT role)")
    p.add_argument("--quant", required=True, help="quantization (index key)")
    p.add_argument("--judge-model", default=None, help="EV-6 cross-family judge (later entry)")
    p.add_argument("--judge-quant", default=None)
    p.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--seed", type=int, default=DEFAULT_BASE_SEED, help="base seed; run i uses seed+i")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    p.add_argument("--context-dir", default=None, help="dir holding PR diff files")
    p.add_argument("--resume", action="store_true", help="resume-safe (default on); explicit no-op")
    p.add_argument("--no-resume", action="store_true", help="ignore existing per-PR files")
    p.add_argument("--dry-run", action="store_true", help="print request plan; contact no server")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = _config_from_args(args)
    if args.dry_run:
        print_dry_run(config)
        return 0
    summary = run(config, UrllibTransport())  # RUN LEG ONLY (contacts local server)
    print(json.dumps(summary["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
