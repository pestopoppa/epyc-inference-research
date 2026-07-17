#!/usr/bin/env python3
"""Retrieval recall@k / MRR harness for the granite-embedding-97m-r2 embedder.

Fills the gap that `bench_canonical` cannot cover: `bench_canonical` measures
tg/pp throughput only and has no notion of retrieval quality. This harness takes
a query set + a document corpus + gold relevant-doc labels, embeds them through a
*pluggable* embed function, retrieves top-k per query, and computes recall@k /
MRR / ndcg@k. Unblocks BULK-K-EMB-1 (granite-97m-r2 quality bench, Phase B).

Design contract (matches the FROZEN-serving-path constraints):

  * DEFAULT is a dry run: validate the corpus, resolve the model plan, print it,
    and exit 0. No model is loaded and no server is contacted in dry-run mode.
  * Real embedding execution is gated behind BOTH an explicit ``--execute`` flag
    AND the ``EMBEDDER_RECALL_EXECUTE=1`` environment flag. ``--execute`` without
    the env flag is refused (it never silently runs inference).
  * All real inference is embedding-only and is routed with
    ``workload_class=eval_batch`` at ``request_priority=background`` against the
    embedder ``/v1/embeddings`` endpoint. It never touches the ``/chat`` LLM path.
  * Results are indexed by ``model/quant`` (e.g. ``granite-embedding-97m-r2/Q8_0``),
    never by serving role or port. Ports live only in the serving section of the
    plan; they never key a result row.

The recall/MRR/ndcg computation is a set of pure functions with no I/O; it is
exercised by ``test_embedder_recall_bench.py`` with synthetic embeddings and gold
labels that assert exact metric values.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# The A-fast fallback corpus is authored in the epyc-root governance repo.
DEFAULT_CORPUS = Path(
    "/mnt/raid0/llm/epyc-root/data/benchmarks/eval-corpus-v0.jsonl"
)

# Explicit execution requires BOTH the flag and this environment variable.
EXECUTE_ENV_FLAG = "EMBEDDER_RECALL_EXECUTE"

# Inference routing invariants. Embedding jobs go through the eval-batch lane so
# they never contend with live /chat traffic.
WORKLOAD_CLASS = "eval_batch"
REQUEST_PRIORITY = "background"
EMBED_ENDPOINT_PATH = "/v1/embeddings"
CHAT_ENDPOINT_PATH = "/chat"

# recall@k targets + ndcg@k target from the granite bench plan.
DEFAULT_KS: tuple[int, ...] = (10, 50)
DEFAULT_NDCG_K = 10


@dataclass(frozen=True)
class ModelSpec:
    """A single benchmark subject, indexed by model+quant (never by role/port)."""

    model: str
    quant: str
    server_port: int
    dim: int | None = None

    @property
    def result_key(self) -> str:
        """Model/quant identity used to key every emitted result row."""
        return f"{self.model}/{self.quant}"

    def embed_endpoint(self, host: str = "localhost") -> str:
        return f"http://{host}:{self.server_port}{EMBED_ENDPOINT_PATH}"


# Candidate subjects for the granite bench. Ports match the warm/default-off
# embedder recipes (granite 8096, e5-base 8097, bge-m3 8098; bge-large-en 8090
# is the English-only reference). Ports are a serving detail only.
DEFAULT_MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("granite-embedding-97m-r2", "Q8_0", 8096, dim=768),
    ModelSpec("granite-embedding-97m-r2", "Q4_K_M", 8096, dim=768),
    ModelSpec("multilingual-e5-base", "Q8_0", 8097, dim=768),
    ModelSpec("bge-m3", "Q8_0", 8098, dim=1024),
    ModelSpec("bge-large-en-v1.5", "reference", 8090, dim=1024),
)


# --------------------------------------------------------------------------- #
# Corpus parsing (pure)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    relevant_doc_ids: tuple[str, ...]
    payload: dict[str, Any]


def load_corpus(path: Path) -> tuple[list[Document], list[Query]]:
    """Parse the JSONL corpus into typed documents and queries.

    Schema mirrors ``eval-corpus-v0.jsonl``: each line is a JSON object with a
    ``type`` of ``document`` or ``query``. Raises ValueError on malformed rows,
    duplicate ids, or an empty ``relevant_doc_ids`` list.
    """
    documents: list[Document] = []
    queries: list[Query] = []
    doc_ids: set[str] = set()
    query_ids: set[str] = set()
    text = Path(path).read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        record_type = row.get("type")
        if record_type == "document":
            for field_name in ("doc_id", "text"):
                if field_name not in row:
                    raise ValueError(f"{path}:{lineno}: missing document field {field_name!r}")
            doc_id = str(row["doc_id"])
            if doc_id in doc_ids:
                raise ValueError(f"{path}:{lineno}: duplicate doc_id {doc_id!r}")
            doc_ids.add(doc_id)
            documents.append(Document(doc_id=doc_id, text=str(row["text"]), payload=dict(row)))
        elif record_type == "query":
            for field_name in ("query_id", "query", "relevant_doc_ids"):
                if field_name not in row:
                    raise ValueError(f"{path}:{lineno}: missing query field {field_name!r}")
            query_id = str(row["query_id"])
            if query_id in query_ids:
                raise ValueError(f"{path}:{lineno}: duplicate query_id {query_id!r}")
            rel = row["relevant_doc_ids"]
            if not isinstance(rel, list) or not rel:
                raise ValueError(f"{path}:{lineno}: relevant_doc_ids must be a non-empty list")
            query_ids.add(query_id)
            queries.append(
                Query(
                    query_id=query_id,
                    text=str(row["query"]),
                    relevant_doc_ids=tuple(str(d) for d in rel),
                    payload=dict(row),
                )
            )
        else:
            raise ValueError(f"{path}:{lineno}: unsupported record type {record_type!r}")
    return documents, queries


def missing_relevance_refs(documents: Sequence[Document], queries: Sequence[Query]) -> list[str]:
    """Return ``query_id -> doc_id`` labels whose target doc is absent from the corpus."""
    doc_ids = {doc.doc_id for doc in documents}
    missing: list[str] = []
    for query in queries:
        for doc_id in query.relevant_doc_ids:
            if doc_id not in doc_ids:
                missing.append(f"{query.query_id} -> {doc_id}")
    return missing


# --------------------------------------------------------------------------- #
# Retrieval + metrics (pure)
# --------------------------------------------------------------------------- #

def l2_norm(vec: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in vec))


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity; returns 0.0 if either vector is degenerate (zero)."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} != {len(b)}")
    na = l2_norm(a)
    nb = l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    return dot / (na * nb)


def rank_documents(
    query_vec: Sequence[float],
    doc_vecs: dict[str, Sequence[float]],
) -> list[str]:
    """Rank doc_ids by descending cosine similarity to the query.

    Ties are broken by ascending doc_id so the ranking is fully deterministic
    (important for reproducible recall/MRR and for the fixture assertions).
    """
    scored = [(doc_id, cosine_similarity(query_vec, vec)) for doc_id, vec in doc_vecs.items()]
    scored.sort(key=lambda item: (-item[1], item[0]))
    return [doc_id for doc_id, _ in scored]


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Fraction of the gold relevant docs that appear in the top-k ranking."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    top_k = set(ranked_ids[:k])
    return len(top_k & relevant) / len(relevant)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Iterable[str]) -> float:
    """1 / (rank of the first relevant doc); 0.0 if none is retrieved."""
    relevant = set(relevant_ids)
    for index, doc_id in enumerate(ranked_ids, 1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Iterable[str], k: int) -> float:
    """Binary-gain nDCG@k (gain 1 for a relevant doc, 0 otherwise)."""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    dcg = 0.0
    for index, doc_id in enumerate(ranked_ids[:k], 1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def evaluate_query(
    ranked_ids: Sequence[str],
    relevant_ids: Iterable[str],
    ks: Sequence[int],
    ndcg_k: int,
) -> dict[str, float]:
    relevant = list(relevant_ids)
    row: dict[str, float] = {f"recall@{k}": recall_at_k(ranked_ids, relevant, k) for k in ks}
    row["rr"] = reciprocal_rank(ranked_ids, relevant)
    row[f"ndcg@{ndcg_k}"] = ndcg_at_k(ranked_ids, relevant, ndcg_k)
    return row


def aggregate_metrics(
    per_query: Sequence[dict[str, float]],
    ks: Sequence[int],
    ndcg_k: int,
) -> dict[str, float]:
    """Mean each recall@k + ndcg@k, and report MRR as the mean reciprocal rank."""
    if not per_query:
        raise ValueError("cannot aggregate zero queries")
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"recall@{k}"] = statistics.fmean(row[f"recall@{k}"] for row in per_query)
    metrics["mrr"] = statistics.fmean(row["rr"] for row in per_query)
    metrics[f"ndcg@{ndcg_k}"] = statistics.fmean(row[f"ndcg@{ndcg_k}"] for row in per_query)
    metrics["query_count"] = float(len(per_query))
    return metrics


# --------------------------------------------------------------------------- #
# Bench driver (pure given the injected embed function)
# --------------------------------------------------------------------------- #

# An embed function takes a list of texts + the ModelSpec being benched and
# returns one vector per text. Tests inject a deterministic stub; --execute
# builds one that routes through the eval-batch embedding lane.
EmbedFn = Callable[[list[str], ModelSpec], list[list[float]]]


def run_recall_bench(
    documents: Sequence[Document],
    queries: Sequence[Query],
    specs: Sequence[ModelSpec],
    embed_fn: EmbedFn,
    ks: Sequence[int] = DEFAULT_KS,
    ndcg_k: int = DEFAULT_NDCG_K,
) -> dict[str, dict[str, Any]]:
    """Compute recall@k / MRR / ndcg@k for each spec, keyed by ``model/quant``.

    Pure with respect to ``embed_fn``: given a deterministic embed function this
    returns deterministic metrics, so the whole path is fixture-testable without
    any model or server.
    """
    doc_ids = [doc.doc_id for doc in documents]
    doc_texts = [doc.text for doc in documents]
    query_texts = [query.text for query in queries]

    results: dict[str, dict[str, Any]] = {}
    for spec in specs:
        doc_matrix = embed_fn(list(doc_texts), spec)
        query_matrix = embed_fn(list(query_texts), spec)
        if len(doc_matrix) != len(doc_ids):
            raise ValueError(f"{spec.result_key}: embed_fn returned {len(doc_matrix)} doc vectors, expected {len(doc_ids)}")
        if len(query_matrix) != len(queries):
            raise ValueError(f"{spec.result_key}: embed_fn returned {len(query_matrix)} query vectors, expected {len(queries)}")
        doc_vecs = {doc_id: vec for doc_id, vec in zip(doc_ids, doc_matrix)}

        per_query: list[dict[str, float]] = []
        for query, query_vec in zip(queries, query_matrix):
            ranked = rank_documents(query_vec, doc_vecs)
            per_query.append(evaluate_query(ranked, query.relevant_doc_ids, ks, ndcg_k))

        row = aggregate_metrics(per_query, ks, ndcg_k)
        row["model"] = spec.model
        row["quant"] = spec.quant
        results[spec.result_key] = row  # model/quant-indexed, never role/port
    return results


# --------------------------------------------------------------------------- #
# Inference routing model (eval-batch lane; pure request builder)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EmbedRequest:
    url: str
    headers: dict[str, str]
    payload: dict[str, Any]


def _assert_not_chat_endpoint(url: str) -> None:
    if CHAT_ENDPOINT_PATH in url:
        raise ValueError(f"embedding inference must not target the /chat path: {url!r}")


def build_embed_request(spec: ModelSpec, inputs: Sequence[str], host: str = "localhost") -> EmbedRequest:
    """Build the eval-batch embedding request for a batch of inputs.

    The request carries ``workload_class=eval_batch`` at ``background`` priority
    and targets the embedder ``/v1/embeddings`` endpoint, never ``/chat``.
    """
    url = spec.embed_endpoint(host)
    _assert_not_chat_endpoint(url)
    payload = {
        "model": spec.model,
        "input": list(inputs),
        "encoding_format": "float",
        "workload_class": WORKLOAD_CLASS,
        "request_priority": REQUEST_PRIORITY,
    }
    headers = {
        "Content-Type": "application/json",
        "X-Workload-Class": WORKLOAD_CLASS,
        "X-Request-Priority": REQUEST_PRIORITY,
    }
    return EmbedRequest(url=url, headers=headers, payload=payload)


def make_eval_batch_embed_fn(host: str = "localhost", timeout_s: float = 120.0) -> EmbedFn:
    """Build a real embed function that routes through the eval-batch lane.

    Constructed ONLY on the ``--execute`` path. Importing urllib lazily keeps the
    default dry-run import surface tiny and makes it obvious that no network
    handle is created unless execution is explicitly requested.
    """
    from urllib import request as _urlrequest  # local import: execute-only

    def _embed(inputs: list[str], spec: ModelSpec) -> list[list[float]]:
        req_spec = build_embed_request(spec, inputs, host=host)
        _assert_not_chat_endpoint(req_spec.url)
        body = json.dumps(req_spec.payload).encode("utf-8")
        http_req = _urlrequest.Request(req_spec.url, data=body, headers=req_spec.headers, method="POST")
        with _urlrequest.urlopen(http_req, timeout=timeout_s) as resp:
            parsed = json.loads(resp.read().decode("utf-8"))
        data = parsed.get("data", [])
        return [list(item["embedding"]) for item in data]

    return _embed


# --------------------------------------------------------------------------- #
# Execution gating (pure)
# --------------------------------------------------------------------------- #

def resolve_execution_mode(execute_flag: bool, env_value: str | None) -> tuple[bool, str]:
    """Decide whether to actually run inference.

    Returns ``(will_execute, reason)``. Real execution requires the explicit
    ``--execute`` flag AND ``EMBEDDER_RECALL_EXECUTE=1``. Any other combination
    is a dry run; ``--execute`` without the env flag is an explicit blocker.
    """
    env_set = str(env_value).strip() == "1" if env_value is not None else False
    if not execute_flag:
        return False, "dry_run_default"
    if not env_set:
        return False, f"blocked_missing_env:{EXECUTE_ENV_FLAG}=1"
    return True, "execute_confirmed"


# --------------------------------------------------------------------------- #
# Planning (pure)
# --------------------------------------------------------------------------- #

def build_plan(
    corpus: Path,
    documents: Sequence[Document],
    queries: Sequence[Query],
    specs: Sequence[ModelSpec],
    ks: Sequence[int],
    ndcg_k: int,
    host: str,
    will_execute: bool,
    execute_reason: str,
) -> dict[str, Any]:
    doc_word_counts = [len(doc.text.split()) for doc in documents]
    query_word_counts = [len(query.text.split()) for query in queries]
    missing = missing_relevance_refs(documents, queries)
    metrics = [f"recall@{k}" for k in ks] + ["mrr", f"ndcg@{ndcg_k}"]

    model_rows = []
    for spec in specs:
        model_rows.append(
            {
                "result_key": spec.result_key,  # model/quant-indexed
                "model": spec.model,
                "quant": spec.quant,
                "dim": spec.dim,
                "server_port": spec.server_port,  # serving detail only
                "embed_endpoint": spec.embed_endpoint(host),
                "workload_class": WORKLOAD_CLASS,
                "request_priority": REQUEST_PRIORITY,
            }
        )

    return {
        "mode": "execute" if will_execute else "dry_run",
        "corpus": str(corpus),
        "documents": {
            "count": len(documents),
            "avg_word_count": round(statistics.fmean(doc_word_counts), 2) if doc_word_counts else 0.0,
            "median_word_count": round(statistics.median(doc_word_counts), 2) if doc_word_counts else 0.0,
        },
        "queries": {
            "count": len(queries),
            "avg_word_count": round(statistics.fmean(query_word_counts), 2) if query_word_counts else 0.0,
            "median_word_count": round(statistics.median(query_word_counts), 2) if query_word_counts else 0.0,
        },
        "missing_relevance_refs": missing,
        "ks": list(ks),
        "ndcg_k": ndcg_k,
        "metrics": metrics,
        "result_index": "model/quant",
        "routing": {
            "workload_class": WORKLOAD_CLASS,
            "request_priority": REQUEST_PRIORITY,
            "endpoint_path": EMBED_ENDPOINT_PATH,
            "never_chat": CHAT_ENDPOINT_PATH not in EMBED_ENDPOINT_PATH,
        },
        "execution_gate": {
            "execute_flag": will_execute or execute_reason.startswith("blocked"),
            "env_flag_name": EXECUTE_ENV_FLAG,
            "will_execute": will_execute,
            "reason": execute_reason,
        },
        "models": model_rows,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _select_specs(names: Sequence[str] | None) -> list[ModelSpec]:
    if not names:
        return list(DEFAULT_MODEL_SPECS)
    wanted = set(names)
    selected = [spec for spec in DEFAULT_MODEL_SPECS if spec.result_key in wanted or spec.model in wanted]
    if not selected:
        raise ValueError(f"no known model spec matched {sorted(wanted)!r}")
    return selected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS, help="Input JSONL corpus (documents + queries)")
    parser.add_argument("--models", nargs="+", default=None, help="Subset of model/quant result-keys or model names to bench")
    parser.add_argument("--k", dest="ks", type=int, nargs="+", default=list(DEFAULT_KS), help="recall@k cut-offs")
    parser.add_argument("--ndcg-k", type=int, default=DEFAULT_NDCG_K, help="ndcg@k cut-off")
    parser.add_argument("--host", default="localhost", help="Embedder server host (serving detail; execute-only)")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON path for model/quant-indexed results")
    parser.add_argument(
        "--execute",
        action="store_true",
        help=f"Run real embedding inference. Requires {EXECUTE_ENV_FLAG}=1; otherwise refused.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    will_execute, reason = resolve_execution_mode(args.execute, os.environ.get(EXECUTE_ENV_FLAG))

    specs = _select_specs(args.models)
    documents, queries = load_corpus(args.corpus)

    plan = build_plan(
        corpus=args.corpus,
        documents=documents,
        queries=queries,
        specs=specs,
        ks=args.ks,
        ndcg_k=args.ndcg_k,
        host=args.host,
        will_execute=will_execute,
        execute_reason=reason,
    )

    # Refuse an explicit --execute that is missing the env flag: print the plan
    # (so the operator sees exactly what would run) and exit non-zero.
    if args.execute and not will_execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        print(f"\nREFUSED: --execute requires {EXECUTE_ENV_FLAG}=1 in the environment.")
        return 2

    if not will_execute:
        # DEFAULT path: validate + resolve + print plan, exit 0. No model loaded.
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        if plan["missing_relevance_refs"]:
            print(f"\nWARNING: {len(plan['missing_relevance_refs'])} unresolved relevance reference(s).")
        return 0

    # Execute path (env + flag confirmed). Routes embeddings through the
    # eval-batch lane only. Not exercised by tests / this session.
    embed_fn = make_eval_batch_embed_fn(host=args.host)
    results = run_recall_bench(documents, queries, specs, embed_fn, ks=args.ks, ndcg_k=args.ndcg_k)
    report = {"plan": plan, "results": results}  # results keyed by model/quant
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
