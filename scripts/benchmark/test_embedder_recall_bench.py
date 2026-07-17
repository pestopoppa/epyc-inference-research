#!/usr/bin/env python3
"""Fixture tests for embedder_recall_bench.py (stdlib unittest; no inference).

Exercises the NON-inference logic only: corpus parsing, the pure recall@k / MRR /
ndcg@k computation on synthetic embeddings + gold labels, execution gating, the
eval-batch request builder, plan construction, and the dry-run CLI. No model is
loaded and no server is contacted. Run:

    .venv/bin/python scripts/benchmark/test_embedder_recall_bench.py
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import embedder_recall_bench as bench


# Synthetic 2-D embeddings with hand-computable rankings.
DOC_VECS = {
    "d1": [1.0, 0.0],
    "d2": [0.0, 1.0],
    "d3": [1.0, 1.0],
    "d4": [-1.0, 0.0],
}
# text -> vector lookup used by the stub embed function
EMBED_LOOKUP = {
    "doc-d1": DOC_VECS["d1"],
    "doc-d2": DOC_VECS["d2"],
    "doc-d3": DOC_VECS["d3"],
    "doc-d4": DOC_VECS["d4"],
    "query-1": [1.0, 0.0],
    "query-2": [0.0, 1.0],
}


def _stub_embed_fn(inputs, spec):
    # Deterministic, spec-independent lookup so the whole bench is fixture-pure.
    return [list(EMBED_LOOKUP[text]) for text in inputs]


class TestMetrics(unittest.TestCase):
    def test_cosine_similarity_known_value(self) -> None:
        self.assertAlmostEqual(bench.cosine_similarity([1.0, 0.0], [1.0, 1.0]), 1.0 / math.sqrt(2.0))
        self.assertAlmostEqual(bench.cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertAlmostEqual(bench.cosine_similarity([2.0, 0.0], [3.0, 0.0]), 1.0)

    def test_cosine_similarity_zero_vector_is_zero(self) -> None:
        self.assertEqual(bench.cosine_similarity([0.0, 0.0], [1.0, 1.0]), 0.0)

    def test_cosine_similarity_dimension_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            bench.cosine_similarity([1.0, 0.0], [1.0])

    def test_rank_documents_ties_break_by_doc_id(self) -> None:
        # d1 and d4 both score 0 against [0,1]; tie-break ascending doc_id.
        ranked = bench.rank_documents([0.0, 1.0], DOC_VECS)
        self.assertEqual(ranked, ["d2", "d3", "d1", "d4"])

    def test_recall_at_k_exact(self) -> None:
        ranked = ["d1", "d3", "d2", "d4"]
        self.assertEqual(bench.recall_at_k(ranked, {"d3"}, 1), 0.0)
        self.assertEqual(bench.recall_at_k(ranked, {"d3"}, 2), 1.0)
        self.assertEqual(bench.recall_at_k(ranked, {"d2", "d3"}, 2), 0.5)
        self.assertEqual(bench.recall_at_k(ranked, {"d2", "d3"}, 3), 1.0)

    def test_recall_at_k_empty_relevant_is_zero(self) -> None:
        self.assertEqual(bench.recall_at_k(["d1"], set(), 1), 0.0)

    def test_reciprocal_rank_exact(self) -> None:
        ranked = ["d1", "d3", "d2", "d4"]
        self.assertEqual(bench.reciprocal_rank(ranked, {"d3"}), 0.5)
        self.assertEqual(bench.reciprocal_rank(ranked, {"d1"}), 1.0)
        self.assertEqual(bench.reciprocal_rank(ranked, {"d4"}), 0.25)
        self.assertEqual(bench.reciprocal_rank(ranked, {"absent"}), 0.0)

    def test_ndcg_at_k_exact(self) -> None:
        # Relevant at rank 2 only: dcg = 1/log2(3); idcg (1 rel) = 1/log2(2) = 1.
        self.assertAlmostEqual(bench.ndcg_at_k(["d1", "d3"], {"d3"}, 2), 1.0 / math.log2(3.0))
        # Perfectly ranked two relevant docs -> 1.0
        self.assertAlmostEqual(bench.ndcg_at_k(["d2", "d3", "d1"], {"d2", "d3"}, 2), 1.0)

    def test_aggregate_metrics_matches_hand_computation(self) -> None:
        per_query = [
            {"recall@1": 0.0, "recall@2": 1.0, "rr": 0.5, "ndcg@2": 1.0 / math.log2(3.0)},
            {"recall@1": 0.5, "recall@2": 1.0, "rr": 1.0, "ndcg@2": 1.0},
        ]
        agg = bench.aggregate_metrics(per_query, ks=[1, 2], ndcg_k=2)
        self.assertEqual(agg["recall@1"], 0.25)
        self.assertEqual(agg["recall@2"], 1.0)
        self.assertEqual(agg["mrr"], 0.75)
        self.assertAlmostEqual(agg["ndcg@2"], (1.0 / math.log2(3.0) + 1.0) / 2.0)
        self.assertEqual(agg["query_count"], 2.0)


class TestRunRecallBench(unittest.TestCase):
    def _fixture(self):
        documents = [
            bench.Document("d1", "doc-d1", {}),
            bench.Document("d2", "doc-d2", {}),
            bench.Document("d3", "doc-d3", {}),
            bench.Document("d4", "doc-d4", {}),
        ]
        queries = [
            bench.Query("q1", "query-1", ("d3",), {}),
            bench.Query("q2", "query-2", ("d2", "d3"), {}),
        ]
        return documents, queries

    def test_end_to_end_metrics_exact(self) -> None:
        documents, queries = self._fixture()
        specs = [bench.ModelSpec("granite-embedding-97m-r2", "Q8_0", 8096)]
        results = bench.run_recall_bench(documents, queries, specs, _stub_embed_fn, ks=[1, 2], ndcg_k=2)
        row = results["granite-embedding-97m-r2/Q8_0"]
        self.assertEqual(row["recall@1"], 0.25)
        self.assertEqual(row["recall@2"], 1.0)
        self.assertEqual(row["mrr"], 0.75)
        self.assertAlmostEqual(row["ndcg@2"], (1.0 / math.log2(3.0) + 1.0) / 2.0)
        self.assertEqual(row["model"], "granite-embedding-97m-r2")
        self.assertEqual(row["quant"], "Q8_0")

    def test_results_are_model_quant_indexed_not_port(self) -> None:
        documents, queries = self._fixture()
        # Two quants of the SAME model share port 8096; keys must still be distinct
        # and encode model/quant, proving results are not port/role-indexed.
        specs = [
            bench.ModelSpec("granite-embedding-97m-r2", "Q8_0", 8096),
            bench.ModelSpec("granite-embedding-97m-r2", "Q4_K_M", 8096),
        ]
        results = bench.run_recall_bench(documents, queries, specs, _stub_embed_fn, ks=[1, 2], ndcg_k=2)
        self.assertEqual(
            set(results),
            {"granite-embedding-97m-r2/Q8_0", "granite-embedding-97m-r2/Q4_K_M"},
        )
        for key in results:
            self.assertNotIn("8096", key)
            self.assertNotIn("role", key)

    def test_embed_fn_vector_count_mismatch_raises(self) -> None:
        documents, queries = self._fixture()
        specs = [bench.ModelSpec("m", "Q8_0", 8096)]

        def _bad_embed(inputs, spec):
            return [[0.0, 1.0]]  # wrong count

        with self.assertRaises(ValueError):
            bench.run_recall_bench(documents, queries, specs, _bad_embed, ks=[1], ndcg_k=1)


class TestCorpusParsing(unittest.TestCase):
    def _write(self, rows) -> Path:
        tmp = Path(tempfile.mkdtemp()) / "corpus.jsonl"
        tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return tmp

    def test_load_corpus_parses_documents_and_queries(self) -> None:
        path = self._write(
            [
                {"type": "document", "doc_id": "a", "text": "alpha text"},
                {"type": "document", "doc_id": "b", "text": "beta text"},
                {"type": "query", "query_id": "q1", "query": "find alpha", "relevant_doc_ids": ["a"]},
            ]
        )
        documents, queries = bench.load_corpus(path)
        self.assertEqual([d.doc_id for d in documents], ["a", "b"])
        self.assertEqual(queries[0].relevant_doc_ids, ("a",))
        self.assertEqual(bench.missing_relevance_refs(documents, queries), [])

    def test_missing_relevance_ref_detected(self) -> None:
        path = self._write(
            [
                {"type": "document", "doc_id": "a", "text": "alpha"},
                {"type": "query", "query_id": "q1", "query": "x", "relevant_doc_ids": ["a", "zzz"]},
            ]
        )
        documents, queries = bench.load_corpus(path)
        self.assertEqual(bench.missing_relevance_refs(documents, queries), ["q1 -> zzz"])

    def test_duplicate_doc_id_raises(self) -> None:
        path = self._write(
            [
                {"type": "document", "doc_id": "a", "text": "alpha"},
                {"type": "document", "doc_id": "a", "text": "again"},
            ]
        )
        with self.assertRaises(ValueError):
            bench.load_corpus(path)

    def test_empty_relevant_list_raises(self) -> None:
        path = self._write(
            [
                {"type": "document", "doc_id": "a", "text": "alpha"},
                {"type": "query", "query_id": "q1", "query": "x", "relevant_doc_ids": []},
            ]
        )
        with self.assertRaises(ValueError):
            bench.load_corpus(path)

    def test_unsupported_record_type_raises(self) -> None:
        path = self._write([{"type": "widget", "doc_id": "a"}])
        with self.assertRaises(ValueError):
            bench.load_corpus(path)


class TestExecutionGating(unittest.TestCase):
    def test_dry_run_default_when_no_flag(self) -> None:
        self.assertEqual(bench.resolve_execution_mode(False, None), (False, "dry_run_default"))
        self.assertEqual(bench.resolve_execution_mode(False, "1"), (False, "dry_run_default"))

    def test_execute_flag_without_env_is_blocked(self) -> None:
        will, reason = bench.resolve_execution_mode(True, None)
        self.assertFalse(will)
        self.assertEqual(reason, "blocked_missing_env:EMBEDDER_RECALL_EXECUTE=1")
        will0, reason0 = bench.resolve_execution_mode(True, "0")
        self.assertFalse(will0)
        self.assertTrue(reason0.startswith("blocked_missing_env"))

    def test_execute_confirmed_requires_flag_and_env(self) -> None:
        self.assertEqual(bench.resolve_execution_mode(True, "1"), (True, "execute_confirmed"))


class TestEmbedRequestRouting(unittest.TestCase):
    def test_build_embed_request_uses_eval_batch_lane(self) -> None:
        spec = bench.ModelSpec("granite-embedding-97m-r2", "Q8_0", 8096)
        req = bench.build_embed_request(spec, ["hello", "world"], host="localhost")
        self.assertEqual(req.url, "http://localhost:8096/v1/embeddings")
        self.assertNotIn("/chat", req.url)
        self.assertEqual(req.payload["workload_class"], "eval_batch")
        self.assertEqual(req.payload["request_priority"], "background")
        self.assertEqual(req.payload["input"], ["hello", "world"])
        self.assertEqual(req.payload["model"], "granite-embedding-97m-r2")
        self.assertEqual(req.headers["X-Workload-Class"], "eval_batch")

    def test_assert_not_chat_endpoint_rejects_chat_path(self) -> None:
        with self.assertRaises(ValueError):
            bench._assert_not_chat_endpoint("http://localhost:8000/chat")


class TestPlan(unittest.TestCase):
    def _fixture(self):
        documents = [bench.Document("a", "alpha beta gamma", {})]
        queries = [bench.Query("q1", "find alpha", ("a",), {})]
        specs = [
            bench.ModelSpec("granite-embedding-97m-r2", "Q8_0", 8096),
            bench.ModelSpec("bge-m3", "Q8_0", 8098),
        ]
        return documents, queries, specs

    def test_plan_dry_run_shape(self) -> None:
        documents, queries, specs = self._fixture()
        plan = bench.build_plan(
            corpus=Path("/tmp/corpus.jsonl"),
            documents=documents,
            queries=queries,
            specs=specs,
            ks=[10, 50],
            ndcg_k=10,
            host="localhost",
            will_execute=False,
            execute_reason="dry_run_default",
        )
        self.assertEqual(plan["mode"], "dry_run")
        self.assertEqual(plan["result_index"], "model/quant")
        self.assertEqual(plan["metrics"], ["recall@10", "recall@50", "mrr", "ndcg@10"])
        self.assertEqual(plan["routing"]["workload_class"], "eval_batch")
        self.assertEqual(plan["routing"]["request_priority"], "background")
        self.assertTrue(plan["routing"]["never_chat"])
        self.assertFalse(plan["execution_gate"]["will_execute"])
        self.assertEqual(plan["execution_gate"]["reason"], "dry_run_default")
        self.assertEqual(plan["models"][0]["result_key"], "granite-embedding-97m-r2/Q8_0")
        self.assertEqual(plan["models"][0]["workload_class"], "eval_batch")
        self.assertEqual(plan["missing_relevance_refs"], [])


class TestCli(unittest.TestCase):
    def _write_corpus(self) -> Path:
        rows = [
            {"type": "document", "doc_id": "a", "text": "alpha text"},
            {"type": "document", "doc_id": "b", "text": "beta text"},
            {"type": "query", "query_id": "q1", "query": "find alpha", "relevant_doc_ids": ["a"]},
        ]
        tmp = Path(tempfile.mkdtemp()) / "corpus.jsonl"
        tmp.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return tmp

    def test_main_dry_run_default_exits_zero_and_prints_plan(self) -> None:
        corpus = self._write_corpus()
        buf = io.StringIO()
        env_backup = os.environ.pop(bench.EXECUTE_ENV_FLAG, None)
        try:
            with contextlib.redirect_stdout(buf):
                rc = bench.main(["--corpus", str(corpus)])
        finally:
            if env_backup is not None:
                os.environ[bench.EXECUTE_ENV_FLAG] = env_backup
        self.assertEqual(rc, 0)
        plan = json.loads(buf.getvalue())
        self.assertEqual(plan["mode"], "dry_run")
        self.assertEqual(plan["documents"]["count"], 2)
        self.assertEqual(plan["queries"]["count"], 1)

    def test_main_execute_without_env_is_refused(self) -> None:
        corpus = self._write_corpus()
        buf = io.StringIO()
        env_backup = os.environ.pop(bench.EXECUTE_ENV_FLAG, None)
        try:
            with contextlib.redirect_stdout(buf):
                rc = bench.main(["--corpus", str(corpus), "--execute"])
        finally:
            if env_backup is not None:
                os.environ[bench.EXECUTE_ENV_FLAG] = env_backup
        self.assertEqual(rc, 2)
        self.assertIn("REFUSED", buf.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
