"""Prepared native provenance/truncation controls; no models or network."""
import hashlib
import json
from pathlib import Path
import pytest
import rus3_native_reader as reader


def capture(tmp_path):
    scorer = b'class ScoringUnavailableError(Exception): pass\ndef score_answer(answer, expected, **kwargs): return answer == expected\n'
    (tmp_path / "scorer-source.py").write_bytes(scorer)
    (tmp_path / "owner-guard-source").write_bytes(b"fixture guard")
    (tmp_path / "launch-identity.json").write_bytes(b"fixture owner receipt")
    config = {"run_id": "run", "role": "r", "model": "model", "max_tokens": 50,
              "temperature": 0.7, "seed": 42, "k": 4, "protocol_id": "",
              "scorer_sha256": hashlib.sha256(scorer).hexdigest(),
              "owner_guard_sha256": hashlib.sha256(b"fixture guard").hexdigest()}
    config_bytes = json.dumps(config).encode()
    (tmp_path / "config.json").write_bytes(config_bytes)
    question = {"id": "q", "suite": "s", "prompt": "what", "expected": "42", "scoring_method": "exact_match"}
    (tmp_path / "questions.jsonl").write_text(json.dumps(question) + "\n")
    manifest = {"config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "questions_sha256": hashlib.sha256((tmp_path / "questions.jsonl").read_bytes()).hexdigest(),
                "launch_identity_sha256": hashlib.sha256(b"fixture owner receipt").hexdigest(),
                "scorer_sha256": config["scorer_sha256"], "owner_guard_sha256": config["owner_guard_sha256"]}
    for name, key in (("grant-receipt", "grant_receipt_sha256"),
                      ("preflight-receipt", "preflight_sha256"),
                      ("owner-start-receipt", "owner_start_sha256")):
        data = ("synthetic " + name).encode()
        (tmp_path / name).write_bytes(data)
        manifest[key] = hashlib.sha256(data).hexdigest()
    manifest["producer_sha256"] = hashlib.sha256(b"synthetic producer").hexdigest()
    manifest["source_sha256"] = manifest["producer_sha256"]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    rows = []
    # A low-token truncated row must be excluded; a valid maximum stop remains.
    for index, (tokens, passed, finish) in enumerate([(1, False, "length"), (10, True, "stop"),
                                                     (40, False, "stop"), (40, False, "stop")]):
        answer = "42" if passed else "wrong"
        response = {"model": "model", "usage": {"completion_tokens": tokens}, "choices": [{
            "finish_reason": finish, "message": {"content": answer, "reasoning_content": None}}]}
        raw = json.dumps(response).encode()
        name = f"{index}.json"
        (tmp_path / name).write_bytes(raw)
        request = json.dumps(reader.request_body(config, question, 42 + index)).encode()
        request_name = f"{index}.request.json"
        (tmp_path / request_name).write_bytes(request)
        rows.append({"schema": "epyc.rus3.native-sample.v1", **config, "suite": "s", "question_id": "q",
                     "sample_index": index, "status": "native", "finish_reason": finish,
                     **manifest, "tokens_generated": tokens, "passed": passed, "scoring_status": "scored", "answer": answer, "served_model": "model",
                     "trace_source": "unavailable",
                     "reasoning_content": None, "raw_response_path": name,
                     "seed": 42 + index, "raw_request_path": request_name,
                     "request_sha256": hashlib.sha256(request).hexdigest(),
                     "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                     "raw_response_sha256": hashlib.sha256(raw).hexdigest()})
    (tmp_path / "samples.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    return rows


def test_explicit_native_truncation_keeps_legitimate_maximum(tmp_path):
    capture(tmp_path)
    row = reader.analyze(tmp_path, tmp_path / "scorer-source.py")["rows"][0]
    assert row["counts"]["truncated"] == 1
    assert row["counts"]["retained"] == 3
    assert row["mixed_questions"] == 1 and row["within_pairs"] == 2
    assert row["within_pooled_auroc"] == 0.0
    assert row["counts"]["trace_unavailable"] == 3
    assert row["review_ratio_within_auroc"] is None


def test_direct_smoke_without_grant_receipt_is_not_authenticated(tmp_path):
    capture(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(line) for line in (tmp_path / "samples.jsonl").read_text().splitlines()]
    for name, key in (("grant-receipt", "grant_receipt_sha256"),
                      ("preflight-receipt", "preflight_sha256"),
                      ("owner-start-receipt", "owner_start_sha256")):
        (tmp_path / name).unlink()
        manifest.pop(key)
        for row in rows:
            row.pop(key)
    manifest_path.write_text(json.dumps(manifest))
    (tmp_path / "samples.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    report = reader.analyze(tmp_path, tmp_path / "scorer-source.py")
    assert report["owner_evidence_authenticated"] is False
    assert report["capture_complete"] is False


def test_complete_structural_capture_is_not_authenticated(tmp_path):
    rows = capture(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    argv = ["/fake/server"]
    identity = {"pid": 42, "pgid": 42, "start_ticks": 123,
                "boot_id": "01234567-89ab-cdef-0123-456789abcdef", "argv": argv,
                "argv_sha256": hashlib.sha256(json.dumps(argv, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()}
    maps = [{"path": "/fake/libggml.so", "sha256": "a" * 64}]
    for row in rows:
        row.update(server_pid=42, server_pgid=42, server_start_ticks=123,
                   server_boot_id=identity["boot_id"], started_monotonic=2.0,
                   ended_monotonic=3.0, ggml_map_identity=maps)
    sample_file = tmp_path / "samples.jsonl"
    sample_file.write_text("\n".join(map(json.dumps, rows)) + "\n")
    complete = {"status": "captured", "run_id": "run", "samples_sha256": hashlib.sha256(sample_file.read_bytes()).hexdigest(),
                "grant_receipt_sha256": manifest["grant_receipt_sha256"], "preflight_sha256": manifest["preflight_sha256"],
                "owner_start_sha256": manifest["owner_start_sha256"], "source_sha256": manifest["producer_sha256"],
                "config_sha256": manifest["config_sha256"], "questions_sha256": manifest["questions_sha256"],
                "scorer_sha256": manifest["scorer_sha256"], "server_identity": identity,
                "ggml_maps_before": maps, "ggml_maps_after": maps, "started_monotonic": 1.0, "ended_monotonic": 4.0}
    (tmp_path / "COMPLETE.json").write_text(json.dumps(complete))
    report = reader.analyze(tmp_path, tmp_path / "scorer-source.py")
    assert report["capture_complete"] is True
    assert report["owner_evidence_authenticated"] is False
    assert report["capture_readiness"] == "OBSERVED_UNATTESTED"


def test_raw_native_digest_mismatch_is_not_a_result(tmp_path):
    capture(tmp_path)
    (tmp_path / "1.json").write_text("{}")
    with pytest.raises(ValueError, match="digest mismatch"):
        reader.analyze(tmp_path, tmp_path / "scorer-source.py")


def test_duplicate_sample_refuses_instead_of_increasing_pair_count(tmp_path):
    rows = capture(tmp_path)
    with (tmp_path / "samples.jsonl").open("a") as output:
        output.write(json.dumps(rows[1]) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        reader.analyze(tmp_path, tmp_path / "scorer-source.py")


def test_question_artifact_mutation_refuses(tmp_path):
    capture(tmp_path)
    (tmp_path / "questions.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="manifest artifact digest"):
        reader.analyze(tmp_path, tmp_path / "scorer-source.py")


def test_passed_tampering_refuses(tmp_path):
    rows = capture(tmp_path)
    rows[1]["passed"] = False
    (tmp_path / "samples.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    with pytest.raises(ValueError, match="scorer recomputation"):
        reader.analyze(tmp_path, tmp_path / "scorer-source.py")


@pytest.mark.parametrize("content", [None, ""])
def test_completed_empty_answer_is_scored_in_retained_denominator(tmp_path, content):
    rows = capture(tmp_path)
    raw_path = tmp_path / rows[1]["raw_response_path"]
    response = json.loads(raw_path.read_bytes())
    response["choices"][0]["message"]["content"] = content
    raw = json.dumps(response).encode()
    raw_path.write_bytes(raw)
    rows[1].update(answer=content, passed=False, raw_response_sha256=hashlib.sha256(raw).hexdigest())
    (tmp_path / "samples.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    row = reader.analyze(tmp_path, tmp_path / "scorer-source.py")["rows"][0]
    assert row["counts"]["retained"] == 3
    assert row["counts"]["unscoreable"] == 0
    assert row["all_wrong_retained_only_questions"] == 1


def test_full_k_does_not_hide_deadline_incomplete_marker(tmp_path):
    capture(tmp_path)
    (tmp_path / "INCOMPLETE.json").write_text(json.dumps({"reason": "deadline", "safe_cancel_or_drain": "NOT_PROVEN"}))
    report = reader.analyze(tmp_path, tmp_path / "scorer-source.py")
    assert report["missing_samples"] == 0
    assert report["capture_complete"] is False
    assert report["capture_markers"]["INCOMPLETE.json"]["record"]["reason"] == "deadline"
    assert report["safe_cancel_or_drain"] == "NOT_PROVEN"


def test_refusal_retains_raw_response_integrity_gate(tmp_path):
    rows = capture(tmp_path)
    rows[1]["status"] = "refused"
    (tmp_path / "samples.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
    (tmp_path / rows[1]["raw_response_path"]).write_text("{}")
    with pytest.raises(ValueError, match="digest mismatch"):
        reader.analyze(tmp_path, tmp_path / "scorer-source.py")
