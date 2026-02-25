#!/usr/bin/env python3
"""Standalone adaptive memory viability pilot (independent of seeding)."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

import httpx
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from benchmark.debug_scorer import score_answer
from scripts.experiments.memory_variant_generator import AdaptiveVariantGenerator, GeneratorConfig
from scripts.experiments.memory_variant_registry import AdaptiveVariant, VariantRegistry
from scripts.experiments.memory_viability_report import build_decision_markdown


@dataclass
class Question:
    suite: str
    qid: str
    prompt: str
    context: str
    expected: str
    scoring_method: str
    scoring_config: dict[str, Any]


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9_]{3,}", text.lower())}


class GraphMemoryBuilder:
    """Retrieves and renders graph/motif memory snippets for a variant."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._graph_cache: dict[str, dict[str, Any]] = {}
        self._motif_cache: dict[str, list[dict[str, Any]]] = {}

    def _graph_path(self, graph_variant: str) -> Path:
        if graph_variant in {"none", ""}:
            return self.root / "raw_graph_latest.json"
        if graph_variant.endswith(".json"):
            return self.root / graph_variant
        if graph_variant.startswith("coarse_graph_"):
            return self.root / f"{graph_variant}.json"
        if graph_variant.startswith("raw_graph"):
            return self.root / f"{graph_variant}.json"
        return self.root / f"coarse_graph_{graph_variant}.json"

    def _load_graph(self, graph_variant: str) -> dict[str, Any]:
        key = graph_variant or "raw_graph_latest"
        if key in self._graph_cache:
            return self._graph_cache[key]
        path = self._graph_path(key)
        if not path.exists():
            path = self.root / "raw_graph_latest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self._graph_cache[key] = data
        return data

    def _load_motifs(self, motif_variant: str, kind: str) -> list[dict[str, Any]]:
        key = f"{kind}:{motif_variant}"
        if key in self._motif_cache:
            return self._motif_cache[key]
        path = self.root / f"motifs_{kind}_{motif_variant}.csv"
        rows: list[dict[str, Any]] = []
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                header = None
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    cols = [c.strip() for c in line.split(",")]
                    if header is None:
                        header = cols
                        continue
                    row = {header[i]: cols[i] if i < len(cols) else "" for i in range(len(header))}
                    rows.append(row)
        self._motif_cache[key] = rows
        return rows

    @staticmethod
    def _node_text(node: dict[str, Any]) -> str:
        return " ".join(
            [
                str(node.get("id", "")),
                str(node.get("category", "")),
                str(node.get("source_type", "")),
                str(node.get("sample_message", "")),
            ]
        )

    @staticmethod
    def _safe_int(val: Any, default: int = 0) -> int:
        try:
            return int(val)
        except Exception:
            return default

    @staticmethod
    def _safe_float(val: Any, default: float = 0.0) -> float:
        try:
            return float(val)
        except Exception:
            return default

    def build_memory(self, prompt: str, knobs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if knobs.get("memory_mode") == "baseline" or int(knobs.get("retrieval_k", 0) or 0) <= 0:
            return "", {"nodes": 0, "edges": 0, "failure_motifs": 0, "hypothesis_motifs": 0}

        graph = self._load_graph(str(knobs.get("graph_variant", "raw_graph_latest")))
        nodes = graph.get("nodes", []) or []
        edges = graph.get("edges", []) or []

        query_tokens = _tokenize(prompt)
        scored_nodes: list[tuple[float, dict[str, Any]]] = []
        for node in nodes:
            txt = self._node_text(node)
            node_tokens = _tokenize(txt)
            overlap = len(query_tokens & node_tokens)
            freq = math.log1p(float(node.get("count", 1) or 1))
            score = float(overlap) * 3.0 + freq
            if score > 0:
                scored_nodes.append((score, node))

        scored_nodes.sort(key=lambda x: x[0], reverse=True)
        retrieval_k = max(1, self._safe_int(knobs.get("retrieval_k", 8), 8))
        top_nodes = [n for _, n in scored_nodes[:retrieval_k]]
        if not top_nodes and nodes:
            top_nodes = sorted(nodes, key=lambda n: float(n.get("count", 0) or 0), reverse=True)[:retrieval_k]

        selected_ids = {str(n.get("id", "")) for n in top_nodes}
        edge_candidates = []
        for e in edges:
            src = str(e.get("source", ""))
            dst = str(e.get("target", ""))
            if src in selected_ids or dst in selected_ids:
                edge_candidates.append(e)
        edge_candidates.sort(key=lambda e: self._safe_int(e.get("count", 0)), reverse=True)
        edge_k = max(0, self._safe_int(knobs.get("edge_k", 8), 8))
        top_edges = edge_candidates[:edge_k]

        motif_variant = str(knobs.get("motif_variant", "level_category"))
        fail_k = max(0, self._safe_int(knobs.get("failure_motif_k", 4), 4))
        hypo_k = max(0, self._safe_int(knobs.get("hypothesis_motif_k", 4), 4))

        fail_rows = self._load_motifs(motif_variant, "failure")
        hypo_rows = self._load_motifs(motif_variant, "hypothesis")

        def motif_rank(row: dict[str, Any], kind: str) -> float:
            text = f"{row.get('from','')} {row.get('to','')}"
            overlap = len(_tokenize(text) & query_tokens)
            score = overlap * 2.0
            if kind == "failure":
                score += self._safe_float(row.get("risk_score", 0.0))
            else:
                score += self._safe_float(row.get("lift_vs_baseline", 0.0)) * 10.0
                score += self._safe_float(row.get("success_rate", 0.0))
            score += math.log1p(self._safe_int(row.get("count", 0)))
            return score

        fail_rows = sorted(fail_rows, key=lambda r: motif_rank(r, "failure"), reverse=True)[:fail_k]
        hypo_rows = sorted(hypo_rows, key=lambda r: motif_rank(r, "hypothesis"), reverse=True)[:hypo_k]

        lines: list[str] = []
        lines.append("[Strategic Memory]")
        lines.append("Use this only as guidance; solve the user task directly.")
        if top_nodes:
            lines.append("Relevant states:")
            for n in top_nodes:
                lines.append(
                    f"- {n.get('id','')} | count={self._safe_int(n.get('count',0))} | {str(n.get('sample_message','')).strip()}"
                )
        if top_edges:
            lines.append("Frequent transitions:")
            for e in top_edges:
                lines.append(
                    f"- {e.get('source','')} -> {e.get('target','')} (count={self._safe_int(e.get('count',0))})"
                )
        if fail_rows:
            lines.append("Failure motifs to avoid:")
            for r in fail_rows:
                lines.append(
                    f"- {r.get('from','')} -> {r.get('to','')} | risk={self._safe_float(r.get('risk_score',0.0)):.2f}"
                )
        if hypo_rows:
            lines.append("Success hypotheses to prefer:")
            for r in hypo_rows:
                lines.append(
                    f"- {r.get('from','')} -> {r.get('to','')} | lift={self._safe_float(r.get('lift_vs_baseline',0.0)):.3f}"
                )

        text = "\n".join(lines)
        budget = max(0, self._safe_int(knobs.get("memory_char_budget", 1800), 1800))
        if budget > 0 and len(text) > budget:
            text = text[:budget] + "\n[truncated]"

        meta = {
            "nodes": len(top_nodes),
            "edges": len(top_edges),
            "failure_motifs": len(fail_rows),
            "hypothesis_motifs": len(hypo_rows),
            "memory_chars": len(text),
        }
        return text, meta


def _load_questions(files: list[Path], suites: set[str]) -> list[Question]:
    out: list[Question] = []
    for path in files:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            continue
        suite = str(data.get("suite") or path.stem)
        if suite not in suites:
            continue
        default_method = str((data.get("scoring_default") or {}).get("method", "exact_match"))
        default_cfg = dict((data.get("scoring_default") or {}).get("config", {}) or {})
        for q in data.get("questions", []) or []:
            if not isinstance(q, dict):
                continue
            out.append(
                Question(
                    suite=suite,
                    qid=str(q.get("id", "")),
                    prompt=str(q.get("prompt", "")),
                    context=str(q.get("context", "")),
                    expected=str(q.get("expected", "")),
                    scoring_method=str(q.get("scoring_method", default_method)),
                    scoring_config=dict(q.get("scoring_config", default_cfg) or {}),
                )
            )
    return out


def _sample_questions(questions: list[Question], suites: set[str], n_per_suite: int, seed: int) -> list[Question]:
    by_suite: dict[str, list[Question]] = {s: [] for s in suites}
    for q in questions:
        by_suite.setdefault(q.suite, []).append(q)

    sampled: list[Question] = []
    for suite in sorted(suites):
        arr = by_suite.get(suite, [])
        rng = random.Random(seed + abs(hash(suite)) % 1_000_000)
        arr = sorted(arr, key=lambda x: x.qid)
        rng.shuffle(arr)
        sampled.extend(arr[: min(n_per_suite, len(arr))])
    return sampled


def _parse_search_space(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    params = (raw or {}).get("parameters", {})
    out: dict[str, Any] = {}
    if not isinstance(params, dict):
        return out
    for k, v in params.items():
        if not isinstance(v, dict):
            continue
        if isinstance(v.get("choices"), list):
            out[k] = v["choices"]
        elif "low" in v and "high" in v:
            low = int(v["low"])
            high = int(v["high"])
            step = int(v.get("step", 1))
            out[k] = list(range(low, high + 1, step))
    return out


def _load_arms_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    return raw


def _evaluate_single(
    client: httpx.Client,
    api_url: str,
    force_role: str,
    force_mode: str,
    timeout_s: float,
    max_turns: int,
    question: Question,
    memory_block: str,
) -> tuple[dict[str, Any], float, str]:
    prompt = question.prompt
    if memory_block:
        prompt = (
            f"{memory_block}\n\n"
            "Task:\n"
            f"{question.prompt}\n\n"
            "Answer succinctly and correctly."
        )

    payload = {
        "prompt": prompt,
        "context": question.context,
        "real_mode": True,
        "force_role": force_role,
        "force_mode": force_mode,
        "max_turns": max_turns,
        "cache_prompt": False,
    }

    t0 = time.perf_counter()
    try:
        resp = client.post(f"{api_url.rstrip('/')}/chat", json=payload, timeout=timeout_s)
        elapsed = time.perf_counter() - t0
        if resp.status_code != 200:
            return {}, elapsed, f"http_{resp.status_code}"
        data = resp.json()
        if not isinstance(data, dict):
            return {}, elapsed, "invalid_json"
        return data, elapsed, str(data.get("error") or "")
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {}, elapsed, f"request_error: {exc}"


def _accuracy(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    ok = sum(1 for r in rows if r.get("correct"))
    return ok / len(rows)


def _write_round_csv(path: Path, round_rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "round_index",
                "variant_id",
                "variant_name",
                "provenance",
                "is_control",
                "questions",
                "accuracy",
                "baseline_accuracy",
                "uplift_pp",
            ],
        )
        writer.writeheader()
        for row in round_rows:
            writer.writerow(row)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Adaptive memory viability pilot")
    p.add_argument("--stage", choices=["stage0", "stage1"], default="stage0")
    p.add_argument("--api-url", default="http://localhost:8000")
    p.add_argument("--force-role", default="worker_fast")
    p.add_argument("--force-mode", default="direct")
    p.add_argument("--timeout-s", type=float, default=240.0)
    p.add_argument("--max-turns", type=int, default=8)
    p.add_argument("--adaptive-per-round", type=int, default=4)
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--sample-per-suite", type=int, default=0)
    p.add_argument("--seeds", nargs="*", type=int, default=[])
    p.add_argument("--suites", nargs="+", default=["thinking", "coder"])
    p.add_argument("--question-files", nargs="+", default=[
        "benchmarks/prompts/debug/thinking.yaml",
        "benchmarks/prompts/debug/coder.yaml",
    ])
    p.add_argument("--search-space-config", default="configs/memory_viability/search_space.yaml")
    p.add_argument("--arms-config", default="configs/memory_viability/arms.yaml")
    p.add_argument("--use-llm-generator", action="store_true")
    p.add_argument("--stage0-uplift-threshold", type=float, default=2.0)
    p.add_argument("--viability-uplift-threshold", type=float, default=5.0)
    p.add_argument("--viability-positive-seeds", type=int, default=2)
    p.add_argument("--output-dir", default="")
    return p.parse_args()


def main() -> int:
    args = _args()

    suites = set(args.suites)
    if args.sample_per_suite <= 0:
        args.sample_per_suite = 20 if args.stage == "stage0" else 60
    if not args.seeds:
        args.seeds = [7] if args.stage == "stage0" else [7, 17, 29]
    if args.stage == "stage0":
        args.max_rounds = 1

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "logs" / "memory_viability" / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.jsonl"
    round_csv_path = run_dir / "round_summary.csv"
    decision_path = run_dir / "decision.md"
    arm_decisions_path = run_dir / "arm_decisions.json"
    config_path = run_dir / "run_config.json"

    qfiles = [Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p for p in args.question_files]
    questions = _load_questions(qfiles, suites)
    if not questions:
        print("No questions loaded", file=sys.stderr)
        return 1

    sample_map: dict[int, list[Question]] = {}
    for seed in args.seeds:
        sample_map[seed] = _sample_questions(questions, suites, args.sample_per_suite, seed)

    graph_builder = GraphMemoryBuilder(PROJECT_ROOT / "logs" / "audit_graph")

    registry = VariantRegistry(run_dir / "variants.jsonl")
    arms_cfg_path = Path(args.arms_config) if Path(args.arms_config).is_absolute() else PROJECT_ROOT / args.arms_config
    arms_cfg = _load_arms_config(arms_cfg_path)
    space = _parse_search_space(Path(args.search_space_config) if Path(args.search_space_config).is_absolute() else PROJECT_ROOT / args.search_space_config)
    generator = AdaptiveVariantGenerator(
        registry,
        config=GeneratorConfig(
            use_llm=args.use_llm_generator,
            orchestrator_url=args.api_url,
            search_space=space,
        ),
        seed=args.seeds[0],
    )

    controls = [
        registry.get_by_name("baseline"),
        registry.get_by_name("raw_subgraph"),
        registry.get_by_name("category_coarse"),
    ]
    controls = [c for c in controls if c is not None]
    adaptive_active: list[AdaptiveVariant] = []
    templates = (arms_cfg.get("adaptive_templates", []) if isinstance(arms_cfg, dict) else []) or []
    for item in templates:
        if not isinstance(item, dict):
            continue
        knobs = item.get("knobs")
        if not isinstance(knobs, dict):
            continue
        variant = AdaptiveVariant.make(
            name=str(item.get("name") or "adaptive_template"),
            knobs=knobs,
            provenance="template",
            generation=0,
        )
        adaptive_active.append(registry.register(variant))
    if len(adaptive_active) < args.adaptive_per_round:
        adaptive_active.extend(
            generator.propose_many(
                args.adaptive_per_round - len(adaptive_active),
                use_llm=args.use_llm_generator,
            )
        )

    config_path.write_text(
        json.dumps(
            {
                "stage": args.stage,
                "api_url": args.api_url,
                "force_role": args.force_role,
                "force_mode": args.force_mode,
                "timeout_s": args.timeout_s,
                "max_turns": args.max_turns,
                "adaptive_per_round": args.adaptive_per_round,
                "max_rounds": args.max_rounds,
                "sample_per_suite": args.sample_per_suite,
                "seeds": args.seeds,
                "suites": sorted(suites),
                "question_files": [str(p) for p in qfiles],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    all_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    round_seed_uplifts: list[dict[str, dict[int, float]]] = []

    with httpx.Client() as client, results_path.open("w", encoding="utf-8") as fout:
        for round_index in range(args.max_rounds):
            variants = controls + adaptive_active
            by_variant_rows: dict[str, list[dict[str, Any]]] = {v.variant_id: [] for v in variants}
            seed_acc: dict[str, dict[int, float]] = {v.variant_id: {} for v in variants}

            for seed in args.seeds:
                selected = sample_map[seed]
                for variant in variants:
                    vrows: list[dict[str, Any]] = []
                    for q in selected:
                        memory_block, mem_meta = graph_builder.build_memory(q.prompt, variant.knobs)
                        resp, elapsed, err = _evaluate_single(
                            client,
                            args.api_url,
                            args.force_role,
                            args.force_mode,
                            args.timeout_s,
                            args.max_turns,
                            q,
                            memory_block,
                        )
                        answer = str(resp.get("answer", "")) if resp else ""
                        try:
                            correct = bool(
                                score_answer(
                                    answer=answer,
                                    expected=q.expected,
                                    scoring_method=q.scoring_method,
                                    scoring_config=q.scoring_config,
                                )
                            ) and not err
                        except Exception:
                            correct = False

                        row = {
                            "timestamp": ts,
                            "round_index": round_index,
                            "seed": seed,
                            "variant_id": variant.variant_id,
                            "variant_name": variant.name,
                            "provenance": variant.provenance,
                            "suite": q.suite,
                            "question_id": q.qid,
                            "correct": correct,
                            "error": err,
                            "elapsed_s": elapsed,
                            "tokens_generated": int(resp.get("tokens_generated", 0) or 0),
                            "force_role": args.force_role,
                            "force_mode": args.force_mode,
                            "memory_meta": mem_meta,
                            "answer": answer,
                            "expected": q.expected,
                        }
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        vrows.append(row)
                        by_variant_rows[variant.variant_id].append(row)
                        all_rows.append(row)

                    seed_acc[variant.variant_id][seed] = _accuracy(vrows)

            baseline = registry.get_by_name("baseline")
            baseline_acc = 0.0
            if baseline is not None:
                baseline_acc = _accuracy(by_variant_rows.get(baseline.variant_id, []))

            seed_uplifts_for_round: dict[str, dict[int, float]] = {}
            for variant in variants:
                rows_v = by_variant_rows.get(variant.variant_id, [])
                acc = _accuracy(rows_v)
                uplift_pp = (acc - baseline_acc) * 100.0
                round_rows.append(
                    {
                        "round_index": round_index,
                        "variant_id": variant.variant_id,
                        "variant_name": variant.name,
                        "provenance": variant.provenance,
                        "is_control": variant.name in {"baseline", "raw_subgraph", "category_coarse"},
                        "questions": len(rows_v),
                        "accuracy": f"{acc:.6f}",
                        "baseline_accuracy": f"{baseline_acc:.6f}",
                        "uplift_pp": f"{uplift_pp:.3f}",
                    }
                )

                if variant.name not in {"baseline", "raw_subgraph", "category_coarse"}:
                    registry.record_result(variant.variant_id, uplift_pp)

                seed_uplifts: dict[int, float] = {}
                for seed in args.seeds:
                    v_seed = seed_acc.get(variant.variant_id, {}).get(seed, 0.0)
                    b_seed = seed_acc.get(baseline.variant_id, {}).get(seed, 0.0) if baseline else 0.0
                    seed_uplifts[seed] = (v_seed - b_seed) * 100.0
                seed_uplifts_for_round[variant.variant_id] = seed_uplifts

            round_seed_uplifts.append(seed_uplifts_for_round)

            if args.stage == "stage1" and round_index < args.max_rounds - 1:
                ranked = sorted(
                    adaptive_active,
                    key=lambda v: float(v.score or -9999.0),
                    reverse=True,
                )
                survivors = ranked[: min(2, len(ranked))]
                next_adaptive: list[AdaptiveVariant] = list(survivors)
                while len(next_adaptive) < args.adaptive_per_round:
                    cand = generator.propose_next(use_llm=args.use_llm_generator)
                    if cand.variant_id not in {v.variant_id for v in next_adaptive}:
                        next_adaptive.append(cand)
                adaptive_active = next_adaptive

    _write_round_csv(round_csv_path, round_rows)

    # Decision on the best adaptive variant in the last round.
    last_round = max((r["round_index"] for r in round_rows), default=0)
    candidates = [r for r in round_rows if r["round_index"] == last_round and not r["is_control"]]
    candidates.sort(key=lambda r: float(r["uplift_pp"]), reverse=True)

    best = candidates[0] if candidates else None
    decision = "no_go"
    rationale = "No adaptive variants evaluated."

    if best is not None:
        best_id = str(best["variant_id"])
        best_uplift = float(best["uplift_pp"])
        seed_uplifts = round_seed_uplifts[last_round].get(best_id, {})
        positive_seed_count = sum(1 for v in seed_uplifts.values() if v > 0.0)

        if args.stage == "stage0":
            if best_uplift >= args.stage0_uplift_threshold:
                decision = "continue"
                rationale = (
                    f"Best uplift {best_uplift:.2f}pp >= stage0 threshold "
                    f"{args.stage0_uplift_threshold:.2f}pp"
                )
            else:
                decision = "stop"
                rationale = (
                    f"Best uplift {best_uplift:.2f}pp < stage0 threshold "
                    f"{args.stage0_uplift_threshold:.2f}pp"
                )
        else:
            if (
                best_uplift >= args.viability_uplift_threshold
                and positive_seed_count >= args.viability_positive_seeds
            ):
                decision = "go"
                rationale = (
                    f"Best uplift {best_uplift:.2f}pp and {positive_seed_count}/{len(args.seeds)} seeds positive"
                )
            else:
                decision = "no_go"
                rationale = (
                    f"Best uplift {best_uplift:.2f}pp or seed consistency insufficient "
                    f"({positive_seed_count}/{len(args.seeds)} positive)"
                )

    arm_decisions_path.write_text(
        json.dumps(
            {
                "decision": decision,
                "rationale": rationale,
                "best_last_round": best,
                "thresholds": {
                    "stage0_uplift_pp": args.stage0_uplift_threshold,
                    "viability_uplift_pp": args.viability_uplift_threshold,
                    "viability_positive_seeds": args.viability_positive_seeds,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    decision_md = build_decision_markdown(
        stage=args.stage,
        run_dir=run_dir,
        decision=decision,
        rationale=rationale,
        round_rows=round_rows,
    )
    decision_path.write_text(decision_md, encoding="utf-8")

    print(str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
