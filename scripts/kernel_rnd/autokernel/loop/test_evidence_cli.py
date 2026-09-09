import json

from autokernel.loop import evidence_cli as C
from autokernel.loop import scoped_evidence as E


H = "a" * 64
H2 = "b" * 64


def fixture():
    scope = {"target": "cpu", "backend": "llama_cpu", "model": "m",
             "quant": "q8", "workload": "decode", "allocation": "cores0-47"}
    claim = {"schema": E.CLAIM_KEY_SCHEMA, "target_scope": scope,
             "control_identity": {"digest": H},
             "intervention_identity": {"digest": H2},
             "mechanism_identity": {"name": "thp", "digest": H},
             "estimand": "level", "metric": "tokens_per_second",
             "metric_direction": "higher",
             "effect_question": {"kind": "absolute_effect_bound", "bound": 2.0,
                                 "unit": "percent"},
             "dependency_identities": {"recipe:cpu": H}}
    finding = {"schema": E.FINDING_SCHEMA, "finding_id": "finding-1",
               "source": {"schema": E.SOURCE_REF_SCHEMA, "event_id": "event-1",
                          "artifact_digest": H, "locator": "journal:1"},
               "claim_key": claim, "conclusion": "positive", "value": 3.0,
               "tested_scope": scope,
               "tested_question": claim["effect_question"],
               "raw_grade": {"grade": "PASS"}, "epoch": "epoch-1",
               "record_class": "strict_search",
               "dependency_generations": {"recipe:cpu": 0},
               "intended_use_disposition": {
                   "intended_use": "screen_out", "disposition": "certificate_candidate"},
               "authority_reference": "actor-labelled-pass", "frontier": 1}
    query = {"schema": C.QUERY_SCHEMA, "scope": scope, "claim_key": claim,
             "intended_use": "explore", "current_epoch": "epoch-1", "limit": 40,
             "projection_available": True}
    return finding, query


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def paths(tmp_path, invalidations=()):
    finding, query = fixture()
    findings = tmp_path / "findings.json"
    invalids = tmp_path / "invalidations.json"
    query_path = tmp_path / "query.json"
    write(findings, [finding])
    write(invalids, list(invalidations))
    write(query_path, query)
    return findings, invalids, query_path


def test_offline_cli_explains_projection_and_never_authorizes(tmp_path, capsys):
    arguments = paths(tmp_path)
    assert C.main(["--findings", str(arguments[0]), "--invalidations", str(arguments[1]),
                   "--query", str(arguments[2])]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["execution_authorized"] is False
    assert output["trusted_certificate_adapters_connected"] is False
    assert output["retrieval"]["retrieval_complete"] is True
    assert output["retrieval"]["supported_for_intended_use"] is True
    assert len(output["retrieval"]["findings"]) == 1


def test_json_pass_label_cannot_become_certificate(tmp_path, capsys):
    arguments = paths(tmp_path)
    query = json.loads(arguments[2].read_text(encoding="utf-8"))
    query["intended_use"] = "screen_out"
    write(arguments[2], query)
    assert C.main(["--findings", str(arguments[0]), "--invalidations", str(arguments[1]),
                   "--query", str(arguments[2])]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["retrieval"]["retrieval_complete"] is True
    assert output["retrieval"]["supported_for_intended_use"] is False
    assert "trusted registered" in " ".join(output["retrieval"]["reasons"])


def test_malformed_invalidation_is_quarantined_and_durable_out_matches(tmp_path, capsys):
    bad = {"schema": E.INVALIDATION_SCHEMA, "event_id": "bad-1",
           "dependency_id": "recipe:cpu", "generation": True,
           "kind": "recipe", "frontier": 2}
    arguments = paths(tmp_path, [bad])
    out = tmp_path / "out" / "inspection.json"
    assert C.main(["--findings", str(arguments[0]), "--invalidations", str(arguments[1]),
                   "--query", str(arguments[2]), "--out", str(out)]) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert json.loads(out.read_text(encoding="utf-8")) == stdout
    assert stdout["projection"]["quarantines"][0]["affected_dependencies"] == ["recipe:cpu"]
    assert not list(out.parent.glob(".scoped-evidence-*"))


def test_bad_query_schema_exits_two_on_stderr(tmp_path, capsys):
    arguments = paths(tmp_path)
    query = json.loads(arguments[2].read_text(encoding="utf-8"))
    query["schema"] = "unknown.v9"
    write(arguments[2], query)
    assert C.main(["--findings", str(arguments[0]), "--invalidations", str(arguments[1]),
                   "--query", str(arguments[2])]) == 2
    streams = capsys.readouterr()
    assert not streams.out and "unsupported" in streams.err


def test_bool_limit_is_schema_error(tmp_path, capsys):
    arguments = paths(tmp_path)
    query = json.loads(arguments[2].read_text(encoding="utf-8"))
    query["limit"] = True
    write(arguments[2], query)
    assert C.main(["--findings", str(arguments[0]), "--invalidations", str(arguments[1]),
                   "--query", str(arguments[2])]) == 2
    assert "integer" in capsys.readouterr().err


def test_unknown_intended_use_is_schema_error(tmp_path, capsys):
    arguments = paths(tmp_path)
    query = json.loads(arguments[2].read_text(encoding="utf-8"))
    query["intended_use"] = "screen-otu"
    write(arguments[2], query)
    assert C.main(["--findings", str(arguments[0]), "--invalidations", str(arguments[1]),
                   "--query", str(arguments[2])]) == 2
    assert "unsupported" in capsys.readouterr().err


def test_duplicate_json_identifier_is_not_silently_collapsed(tmp_path, capsys):
    arguments = paths(tmp_path)
    arguments[2].write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
    assert C.main(["--findings", str(arguments[0]), "--invalidations", str(arguments[1]),
                   "--query", str(arguments[2])]) == 2
    assert "duplicate JSON object key" in capsys.readouterr().err
