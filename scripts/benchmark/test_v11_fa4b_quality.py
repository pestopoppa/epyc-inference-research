"""Model-free refusal and pairing checks for V11-FA-4b."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.benchmark import v11_fa4b_quality as m


def fixture_manifest():
    qs = []
    for suite in m.SUITES:
        for i in range(100):
            qs.append({"id": f"{suite}-{i:03d}", "suite": suite,
                       "prompt": f"Question {suite} {i}", "expected": "A",
                       "scoring_method": "multiple_choice", "scoring_config": {}})
    decl = {"images": {a: {"binary": f"/{a}/server", "hip_library": f"/{a}/hip.so",
                            "binary_sha256": "a" * 64, "hip_sha256": "b" * 64,
                            "source_commit": "c5"} for a in m.ARMS},
            "models": {"target": {"path": "/model", "sha256": "c" * 64},
                       "draft": {"path": "/draft", "sha256": "d" * 64}},
            "chat_template": {"path": "/template", "sha256": "e" * 64},
            "configs": {}}
    for kv in m.KVS:
        for draft in m.DRAFTS:
            argv = ["llama-server", "-m", "/model", "-np", "1", "--cache-ram", "0",
                    "-ctk", kv, "-ctv", kv, "--flash-attn", "on", "--device", "ROCm0",
                    "--chat-template-file", "/template"]
            if draft == "on":
                argv += ["-md", "/draft", "-ngld", "99", "--spec-type", "draft-dflash",
                         "--spec-draft-n-max", "7", "--device-draft", "ROCm0"]
            decl["configs"][f"{kv}:{draft}"] = {a: argv for a in m.ARMS}
    result = {"schema": m.SCHEMA, "declaration": decl, "seed": 42,
              "scored": qs, "greedy_ids": [q["id"] for q in qs[:100]],
              "prompt_sha256": {q["id"]: __import__("hashlib").sha256(q["prompt"].encode()).hexdigest()
                                for q in qs},
              "sources": {name: m.file_sha(m.HERE / name) for name in (
                  "v11_fa4b_quality.py", "v7_quality_gate_runner.py", "answer_scoring.py",
                  "dataset_adapters.py", "architect_bench_analyze.py", "code_exec_scorer.py")},
              "requests": m.protocol_requests(42)}
    result["seal_sha256"] = m.digest(result)
    return result


def proof_files(root, manifest, arm, kv, draft, port=19000):
    folder = root / "slices" / f"{arm}-{kv}-{draft}"
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / "server.log"
    proof = folder / "grant-proof.json"
    if not log.exists():
        log.write_text("")
    if not proof.exists():
        proof.write_text(json.dumps({"manifest_sha256": manifest["seal_sha256"],
            "arm": arm, "kv": kv, "draft": draft, "port": port, "actual_grant_ok": True,
            "hip_sha256": manifest["declaration"]["images"][arm]["hip_sha256"],
            "server_log": str(log), "server_identity": {"pid": 123, "uid": __import__('os').getuid(),
            "sid": 123, "pgid": 123, "ticks": 456}}))
    health = folder / "server-log-health.json"
    if not health.exists():
        health.write_text(json.dumps({"server_log": str(log), "sha256": m.file_sha(log),
                                      "bytes": log.stat().st_size, "bad_matches": []}))
    return proof, log


def record(root, manifest, q, phase, arm, kv, draft, index, correct=True, tokens=None):
    proof, log = proof_files(root, manifest, arm, kv, draft)
    raw = (json.dumps({"content": "A complete response", "tokens": tokens,
                       "truncated": False, "stop_type": "eos"}) if phase == "greedy" else
           json.dumps({"choices": [{"finish_reason": "stop",
                                     "message": {"content": "A complete response"}}]}))
    row = {"schema": m.SCHEMA, "manifest_sha256": manifest["seal_sha256"], "phase": phase,
           "arm": arm, "kv": kv, "draft": draft, "id": q["id"], "index": index,
           "prompt_sha256": manifest["prompt_sha256"][q["id"]],
           "request_sha256": m.digest(m.request_payload(manifest, phase, q)),
           "response_sha256": __import__("hashlib").sha256(raw.encode()).hexdigest(),
           "raw_response_text": raw, "server_log_window_sha256": __import__('hashlib').sha256(b'').hexdigest(),
           "server_log_start": 0, "server_log_end": 0,
           "grant_proof_path": str(proof), "grant_proof_sha256": m.file_sha(proof),
           "server_log_path": str(log), "server_pid": 123,
           "server_startticks": 456, "server_port": 19000,
           "content": "A complete response", "correct": correct if phase == "score" else None,
           "tokens": tokens if phase == "greedy" else None,
           "truncated": False, "termination_unknown": False,
           "finish_reason": "stop" if phase == "score" else None,
           "stop_type": "eos" if phase == "greedy" else None}
    row["seal_sha256"] = m.digest(row)
    return row


class QualitySourceTests(unittest.TestCase):
    def test_manifest_requires_all_configs_and_unique_paired_ids(self):
        x = fixture_manifest()
        m.validate_manifest(x)
        x["declaration"]["configs"].pop("f16:on")
        x["seal_sha256"] = m.digest({k: v for k, v in x.items() if k != "seal_sha256"})
        with self.assertRaisesRegex(ValueError, "four"):
            m.validate_manifest(x)
        y = fixture_manifest()
        y["scored"][1]["id"] = y["scored"][0]["id"]
        y["seal_sha256"] = m.digest({k: v for k, v in y.items() if k != "seal_sha256"})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            m.validate_manifest(y)
        z = fixture_manifest()
        z["declaration"]["configs"]["q8_0:on"]["OFF"] = z["declaration"]["configs"]["q8_0:on"]["OFF"] + ["--threads", "16"]
        z["seal_sha256"] = m.digest({k: v for k, v in z.items() if k != "seal_sha256"})
        with self.assertRaisesRegex(ValueError, "launch flags differ"):
            m.validate_manifest(z)
        w = fixture_manifest()
        w["requests"]["greedy"]["n_predict"] = 256
        w["seal_sha256"] = m.digest({k: v for k, v in w.items() if k != "seal_sha256"})
        with self.assertRaisesRegex(ValueError, "request protocol drift"):
            m.validate_manifest(w)

    def test_actual_returned_tokens_only(self):
        q = fixture_manifest()["scored"][0]
        good = m.parse_response("greedy", {"content": "A", "tokens": [10, 42],
                                           "stop_type": "eos", "truncated": False}, q)
        self.assertEqual(good["tokens"], [10, 42])
        for bad in (None, [], ["10"], [True], [-1]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                m.parse_response("greedy", {"content": "A", "tokens": bad,
                                            "stop_type": "eos", "truncated": False}, q)

    def test_bounded_length_is_scored_and_reportable(self):
        q = fixture_manifest()["scored"][0]
        with patch.object(m, "score_response_or_error", return_value=(False, None)):
            scored = m.parse_response("score", {"choices": [{"finish_reason": "length",
                                    "message": {"content": "A"}}]}, q)
        self.assertFalse(scored["correct"])
        self.assertTrue(scored["truncated"])
        greedy = m.parse_response("greedy", {"content": "A", "tokens": [1, 2],
                                             "stop_type": "limit", "truncated": True}, q)
        self.assertEqual(greedy["tokens"], [1, 2])
        self.assertTrue(greedy["truncated"])
        unknown = m.parse_response("greedy", {"content": "A", "tokens": [1],
                                              "stop_type": None}, q)
        self.assertTrue(unknown["termination_unknown"])

    def test_garbage_nonfinite_and_undecidable_refuse(self):
        q = fixture_manifest()["scored"][0]
        with self.assertRaisesRegex(ValueError, "garbage"):
            m.parse_response("greedy", {"content": "!!!!!!!!", "tokens": [1],
                                        "stop_type": "eos", "truncated": False}, q)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            m.parse_response("greedy", {"content": "A", "tokens": [1],
                                        "stop_type": "eos", "truncated": False,
                                        "timings": {"predicted_ms": float("nan")}}, q)
        with patch.object(m, "score_response_or_error", return_value=(None, "checker_error")):
            with self.assertRaisesRegex(ValueError, "undecidable"):
                m.parse_response("score", {"choices": [{"finish_reason": "stop",
                                    "message": {"content": "A"}}]}, q)

    def test_capture_persists_first_request_and_resumes_without_requery(self):
        x = fixture_manifest()
        q = x["scored"][0]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proof, log = proof_files(root, x, "ON", "q8_0", "off")
            response = json.dumps({"content": "A", "tokens": [101, 202],
                                   "stop_type": "eos", "truncated": False}).encode()
            with patch.object(m.urllib.request, "urlopen", return_value=io.BytesIO(response)) as post:
                m.capture(x, "greedy", "ON", "q8_0", "off", 0, 1, 19000, root, 2, proof, log, lambda: None)
                m.capture(x, "greedy", "ON", "q8_0", "off", 0, 1, 19000, root, 2, proof, log, lambda: None)
                self.assertEqual(post.call_count, 1)
            path = next((root / "ON/q8_0-off/greedy").glob("000-*.json"))
            row = m.read_json(path)
            self.assertEqual(row["tokens"], [101, 202])
            self.assertEqual(row["id"], q["id"])
            row["tokens"] = [999]
            path.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, "record seal"):
                m.capture(x, "greedy", "ON", "q8_0", "off", 0, 1, 19000, root, 2, proof, log, lambda: None)
            row["tokens"] = [101, 202]
            row["raw_response_text"] = "{}"
            row["seal_sha256"] = m.digest({k: v for k, v in row.items() if k != "seal_sha256"})
            with self.assertRaisesRegex(ValueError, "raw response hash"):
                m.validate_record(row, "greedy", q, x, "ON", "q8_0", "off")

    def test_server_log_failure_refuses_before_durable_capture(self):
        x = fixture_manifest()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            proof, log = proof_files(root, x, "ON", "q8_0", "off")
            response = json.dumps({"content": "A", "tokens": [101],
                                   "stop_type": "eos", "truncated": False}).encode()
            class Response(io.BytesIO):
                def __enter__(self):
                    with log.open("a") as stream:
                        stream.write("non-finite target features\n")
                    return self
            with patch.object(m.urllib.request, "urlopen", return_value=Response(response)):
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    m.capture(x, "greedy", "ON", "q8_0", "off", 0, 1, 19000, root, 2, proof, log, lambda: None)
            self.assertEqual(list((root / "ON/q8_0-off/greedy").glob("000-*.json")), [])
            failures = list((root / "ON/q8_0-off/greedy/failures").glob("000-*.json"))
            self.assertEqual(len(failures), 1)
            self.assertIn("nonfinite", m.read_json(failures[0])["error"])

    def test_analysis_requires_every_cell_and_known_paired_stats(self):
        x = fixture_manifest()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for kv in m.KVS:
                for draft in m.DRAFTS:
                    for arm in m.ARMS:
                        for phase in m.PHASES:
                            ids = [q["id"] for q in x["scored"]] if phase == "score" else x["greedy_ids"]
                            folder = root / arm / f"{kv}-{draft}" / phase
                            folder.mkdir(parents=True)
                            for i, qid in enumerate(ids):
                                q = next(q for q in x["scored"] if q["id"] == qid)
                                r = record(root, x, q, phase, arm, kv, draft, i,
                                           correct=(i != 0 or arm == "ON"), tokens=[i, 5])
                                if phase == "score" and arm == "ON" and i == 1:
                                    r["truncated"] = True
                                    r["finish_reason"] = "length"
                                    r["raw_response_text"] = json.dumps({"choices": [{"finish_reason": "length",
                                                           "message": {"content": "A complete response"}}]})
                                    r["response_sha256"] = __import__("hashlib").sha256(r["raw_response_text"].encode()).hexdigest()
                                    r["seal_sha256"] = m.digest({k: v for k, v in r.items() if k != "seal_sha256"})
                                m.write_new(folder / f"{i:03d}-{__import__('hashlib').sha256(qid.encode()).hexdigest()[:16]}.json", r)
            report = m.analyze(x, root)
            self.assertEqual(len(report["cells"]), 4)
            case = report["cells"]["q8_0:off"]
            self.assertEqual(case["n_scored"], 200)
            self.assertEqual(case["n_greedy"], 100)
            self.assertEqual(case["mcnemar"]["on_only"], 1)
            self.assertEqual(case["mcnemar"]["p_two_sided"], 1.0)
            self.assertEqual(case["token_identical"], 100)
            self.assertIsNone(case["first_divergence"][x["greedy_ids"][0]])
            self.assertEqual(case["termination"]["ON"]["score"]["truncated"], 1)
            missing = next((root / "OFF/q8_0-off/greedy").glob("000-*.json"))
            missing.unlink()
            with self.assertRaisesRegex(ValueError, "99/100"):
                m.analyze(x, root)

    def test_first_divergence_exact_ids(self):
        self.assertIsNone(m.first_divergence([1, 2], [1, 2]))
        self.assertEqual(m.first_divergence([1, 2], [1, 3]), 1)
        self.assertEqual(m.first_divergence([1, 2], [1, 2, 3]), 2)


if __name__ == "__main__":
    unittest.main()
