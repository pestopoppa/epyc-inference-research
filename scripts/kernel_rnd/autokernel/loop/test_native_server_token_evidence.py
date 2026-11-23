"""Frozen-prompt v1 byte pin, v2 old-reader refusal and same-server token evidence join.

All responses below are LABELLED SYNTHETIC fixtures: they prove the join/derivation logic,
not any server behaviour, determinism of a real kernel, or hardware.
"""
import copy
import hashlib
import json

import pytest

from . import native_server_token_evidence as te
from . import planned_serving as ps
from . import serving
from .test_glm_frozen_requests import _manifest, _request
from .test_planned_serving import _prompt, _prompts

V1_BODY = (b'{"cache_prompt":false,"n_predict":256,"prompt":"first exact prompt",'
           b'"temperature":0.6,"top_k":20,"top_p":0.95}')
V1_BODY_SHA = "bcdaaf09d9fa092227b1eb66d4507552aff7891f0ba79cfa52b7433ad98a9108"
V1_MANIFEST_DIGEST = "e7bb1bd5831b2e15cf3abf63f27bb652063a575d6b46bc2b92e192cddf2b6162"


def test_v1_serialization_and_hash_are_pinned_byte_for_byte():
    recipe = serving.Recipe("old-gpu", "/m", temperature=0.6)
    prompt = ps.FrozenPrompt.from_dict(_prompt("p1", "first exact prompt", recipe))
    assert prompt.body == V1_BODY and hashlib.sha256(prompt.body).hexdigest() == V1_BODY_SHA
    manifest = _prompts(recipe)
    assert manifest.schema == ps.PROMPT_SCHEMA and manifest.digest == V1_MANIFEST_DIGEST
    assert ps.PROMPT_SCHEMA == "epyc.autokernel.frozen_prompt_manifest.v1"


def test_old_reader_refuses_v2_rows_and_v1_reader_refuses_option_smuggling():
    v2 = _manifest(_request()).to_dict()
    with pytest.raises(ps.PlannedServingError):  # v1 row grammar applied to a v2 row
        ps.FrozenPrompt.from_dict(v2["prompts"][0], schema=ps.PROMPT_SCHEMA)
    relabelled = {**v2, "schema": ps.PROMPT_SCHEMA}
    with pytest.raises(ps.PlannedServingError):
        ps.FrozenPromptManifest.from_dict(relabelled)
    with pytest.raises(ps.PlannedServingError, match="unsupported"):
        ps.FrozenPrompt.from_dict(v2["prompts"][0], schema="epyc.autokernel.frozen_prompt_manifest.v3")


def _row(prompt_id, phase, slot, request, tokens=None, *, error=None, predicted="auto", response="auto"):
    rbytes = json.dumps(request, sort_keys=True).encode()
    if response == "auto":
        response = {"content": "x", "stop": True}
        if tokens is not None:
            response["tokens"] = list(tokens)
        response["timings"] = {"predicted_n": len(tokens) if predicted == "auto" and tokens is not None
                               else predicted if predicted != "auto" else request["n_predict"]}
    sha = None if error or response is None else hashlib.sha256(json.dumps(response).encode()).hexdigest()
    return {"request": request, "response": response,
            "raw": {"prompt_id": prompt_id, "phase": phase, "slot_index": slot, "error": error,
                    "request_sha256": hashlib.sha256(rbytes).hexdigest(), "response_sha256": sha}}


def _req(seed=42, **over):
    base = {"prompt": [1, 2, 3], "n_predict": 4, "temperature": 0.0, "top_k": 1,
            "cache_prompt": False, "seed": seed, "ignore_eos": True, "return_tokens": True,
            "stream": False}
    return {**base, **over}


def test_v1_rows_yield_unknown_never_invented_seed_or_tokens():
    request = {"prompt": "p", "n_predict": 4, "temperature": 0.0, "top_k": 1, "cache_prompt": False}
    rows = [_row("p1", "warmup", 0, request), _row("p1", "measurement", 0, request)]
    joined = te.join_rows(rows)
    assert all(item["basis"] == te.BASIS_V1 and item["seed"] is None and item["tokens"] is None
               for item in joined)
    witness = te.repeat_witness(rows)
    assert witness["overall"] == "unknown" and "p1" in witness["unknown_prompts"]
    assert witness["is_t0_pass"] is False


def test_v2_identical_tokens_across_requests_is_identical_witness():
    rows = [_row("p1", "warmup", 0, _req(), [5, 6, 7, 8]),
            _row("p1", "measurement", 0, _req(), [5, 6, 7, 8]),
            _row("p2", "warmup", 1, _req(), [9, 9, 9, 9]),
            _row("p2", "measurement", 1, _req(), [9, 9, 9, 9])]
    witness = te.repeat_witness(rows)
    assert witness["overall"] == "identical" and not witness["unknown_prompts"]
    assert [g["verdict"] for g in witness["groups"]] == ["identical", "identical"]
    assert witness["rows"][0]["seed"] == 42 and witness["rows"][0]["tokens"] == [5, 6, 7, 8]
    assert witness["is_t0_pass"] is False


def test_v2_divergence_reports_first_index():
    rows = [_row("p1", "warmup", 0, _req(), [5, 6, 7, 8]),
            _row("p1", "measurement", 0, _req(), [5, 6, 0, 8])]
    witness = te.repeat_witness(rows)
    assert witness["overall"] == "diverged"
    assert witness["groups"][0]["first_divergence_index"] == 2
    assert witness["groups"][0]["agreement_prefix_ratio"] == 0.5


def test_one_uncovered_prompt_keeps_overall_unknown_no_reduced_list():
    rows = [_row("p1", "warmup", 0, _req(), [1, 1, 1, 1]), _row("p1", "measurement", 0, _req(), [1, 1, 1, 1]),
            _row("p2", "warmup", 1, _req(), [2, 2, 2, 2])]  # p2 has a single request only
    witness = te.repeat_witness(rows)
    assert witness["overall"] == "unknown" and "p2" in witness["unknown_prompts"]


def test_different_seeds_do_not_pool_and_non_greedy_or_failed_are_unknown():
    a = [_row("p1", "warmup", 0, _req(1), [1, 1, 1, 1]), _row("p1", "measurement", 0, _req(2), [1, 1, 1, 1])]
    assert te.repeat_witness(a)["overall"] == "unknown"
    hot = _req(temperature=0.8, top_k=40)
    b = [_row("p1", "warmup", 0, hot, [1, 1, 1, 1]), _row("p1", "measurement", 0, hot, [1, 1, 1, 1])]
    assert te.repeat_witness(b)["overall"] == "unknown"
    c = [_row("p1", "warmup", 0, _req(), [1, 1, 1, 1]),
         _row("p1", "measurement", 0, _req(), None, error="timeout", response=None)]
    assert te.repeat_witness(c)["overall"] == "unknown"
    d = [_row("p1", "warmup", 0, _req(return_tokens=False)), _row("p1", "measurement", 0, _req(return_tokens=False))]
    joined = te.join_rows(d)
    assert joined[0]["token_output_requested"] is False and joined[0]["tokens"] is None
    assert te.repeat_witness(d)["overall"] == "unknown"


@pytest.mark.parametrize("mutate", [
    lambda r: r["response"].pop("tokens"),
    lambda r: r["response"].update(tokens=[1, 2, 3]),                  # ignore_eos short count
    lambda r: r["response"].update(tokens=[1, 2, 3, True]),
    lambda r: r["response"].update(tokens=[1, 2, 3, -1]),
    lambda r: r["response"].update(tokens=[1, 2, 3, 4, 5]),            # more than n_predict
    lambda r: r["response"].update(timings={"predicted_n": 3}),        # contradicts list
])
def test_malformed_token_evidence_refuses(mutate):
    row = _row("p1", "warmup", 0, _req(), [1, 2, 3, 4])
    row = copy.deepcopy(row)
    mutate(row)
    with pytest.raises(te.TokenEvidenceRefused):
        te.join_row(row)


def test_partial_or_malformed_v2_request_fields_refuse():
    partial = {"prompt": "p", "n_predict": 4, "temperature": 0.0, "top_k": 1, "seed": 1}
    with pytest.raises(te.TokenEvidenceRefused):
        te.join_row(_row("p1", "warmup", 0, partial, response={"content": "x"}))
    with pytest.raises(te.TokenEvidenceRefused):
        te.join_row(_row("p1", "warmup", 0, _req(seed=-1), [1, 2, 3, 4]))
    with pytest.raises(te.TokenEvidenceRefused):
        te.join_row(_row("p1", "warmup", 0, _req(stream=True), [1, 2, 3, 4]))
