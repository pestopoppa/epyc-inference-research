"""`model_identity.check`'s divergence persistence (part B, ak-longctx-identity-oracle-
20261007): a full per-repeat token/content record, with the first divergence index,
written under `record_dir` on any non-`pass` verdict, and referenced first in `detail` --
fakes only, no servers."""
from __future__ import annotations

import json
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from . import model_identity


def _recipe(build, port=8080):
    return SimpleNamespace(
        backend="cpu", build_dir=build,
        command_argv=(f"{build}/bin/llama-server", "-m", "/m.gguf", "--port", str(port)),
        topology_prefix=("numactl", "--interleave=all"), port=port,
        launch_env=(("LD_LIBRARY_PATH", f"{build}/bin"),))


REQS = tuple((f"p{i}", json.dumps({"prompt": "x", "temperature": 0}).encode()) for i in range(3))


def _row(prompt_id, digest, content, tokens=None):
    return (prompt_id, digest, content[:80], {"content": content, "tokens": tokens})


def _serve(script):
    """`script` is a list of row-lists, one per `serve_fn` call, in call order. An
    `Exception` instance in the script is raised instead of returned, for the
    harness-fault (serve_fn exception) partial-persistence paths."""
    calls = []

    def serve_fn(recipe, selected):
        calls.append((recipe.build_dir, len(selected)))
        out = script.pop(0)
        if isinstance(out, Exception):
            raise out
        return out
    return serve_fn, calls


def test_no_record_dir_leaves_detail_exactly_as_before():
    """Default behaviour (no `record_dir`): nothing written, no `record` key."""
    anchor_rows = [_row("p0", "h1", "same"), _row("p1", "h2", "same"), _row("p2", "h3", "same")]
    candidate_rows = [_row("p0", "h1", "same"), _row("p1", "h2", "same"), _row("p2", "XX", "diff")]
    serve_fn, _calls = _serve([anchor_rows, candidate_rows, anchor_rows])  # re-serve agrees
    result = model_identity.check(anchor_recipe=_recipe("/a"), candidate_recipe=_recipe("/c"),
                                  requests=REQS, n_requests=3, serve_fn=serve_fn)
    assert result.status == "wrong"
    detail = json.loads(result.detail)
    assert "record" not in detail


def test_wrong_verdict_persists_full_rows_and_the_first_divergence_token_index(tmp_path):
    record_dir = tmp_path / "identity-divergence"
    anchor_rows = [_row("p0", "h1", "hello world", [1, 2, 3]),
                  _row("p1", "h2", "same", [9]),
                  _row("p2", "h3", "same", [9])]
    candidate_rows = [_row("p0", "hX", "hello there", [1, 2, 4]),
                      _row("p1", "h2", "same", [9]),
                      _row("p2", "h3", "same", [9])]
    serve_fn, _calls = _serve([anchor_rows, candidate_rows, anchor_rows])  # re-serve agrees
    result = model_identity.check(anchor_recipe=_recipe("/a"), candidate_recipe=_recipe("/c"),
                                  requests=REQS, serve_fn=serve_fn, record_dir=record_dir)
    assert result.status == "wrong"
    assert "p0" in result.reason and "token divergence at index 2" in result.reason
    detail = json.loads(result.detail)
    path = Path(detail["record"])
    assert path.is_file() and path.parent == record_dir
    payload = json.loads(path.read_text())
    assert payload["schema"] == "epyc.autokernel.model_identity_divergence.v1"
    assert payload["kind"] == "wrong"
    # "which comparison" (request 0 / p0) and "where in it" (token index 2).
    assert payload["first_divergent"] == {
        "kind": "candidate_disagrees_with_the_reproducible_anchor",
        "request_index": 0, "prompt_id": "p0", "level": "token", "divergence_index": 2}
    assert payload["anchor_first"][0]["tokens"] == [1, 2, 3]
    assert payload["candidate"][0]["tokens"] == [1, 2, 4]
    assert payload["anchor_first"][0]["content"] == "hello world"
    # ALL observations, not just the differing pair: the confirming anchor re-serve too.
    assert payload["anchor_reserve"][0]["content"] == "hello world"
    # The fake serve_fn ignores `selected` and returns its whole 3-row scripted list for
    # `anchor` (never sliced); `candidate` IS sliced to `n` (n_requests defaults to 2).
    assert len(payload["anchor_first"]) == 3 and len(payload["candidate"]) == 2
    # detail itself also carries the locator for a quick look without opening the file.
    assert detail["first_divergent"]["divergence_index"] == 2


def test_race_verdict_persists_the_candidate_repeats(tmp_path):
    record_dir = tmp_path / "identity-divergence"
    anchor_rows = [_row("p0", "h1", "a"), _row("p1", "h2", "b")]
    racy_candidate = [_row("p0", "h1", "a"), _row("p1", "h2", "b"),  # rep 1
                      _row("p0", "h1", "a"), _row("p1", "XX", "bad"),  # rep 2 (p1 differs)
                      _row("p0", "h1", "a"), _row("p1", "h2", "b")]  # rep 3
    anchor_reproduced = anchor_rows * 3
    serve_fn, calls = _serve([anchor_rows, racy_candidate, anchor_reproduced])
    reqs = tuple((f"p{i}", json.dumps({"prompt": "x", "temperature": 0}).encode())
                 for i in range(2))
    result = model_identity.check(anchor_recipe=_recipe("/a"), candidate_recipe=_recipe("/c"),
                                  requests=reqs, repeats=3, serve_fn=serve_fn,
                                  record_dir=record_dir)
    assert result.status == "wrong" and "race" in result.reason
    detail = json.loads(result.detail)
    payload = json.loads(Path(detail["record"]).read_text())
    assert payload["kind"] == "race"
    # ALL observations: every candidate repeat, every anchor repeat, and the single
    # initial anchor serve -- not just the racy request.
    assert len(payload["candidate_repeats"]) == 6
    assert len(payload["anchor_repeats"]) == 6
    assert payload["anchor_first"][1]["content"] == "b"
    # "which comparison" (request 1 / p1, repeats 0 vs 1) and "where" (char index 1:
    # "b" vs "bad" share their first character, then "b" runs out).
    assert payload["first_divergent"] == {
        "kind": "candidate_repeats_disagree_with_each_other", "request_index": 1,
        "prompt_id": "p1", "repeat_a": 0, "repeat_b": 1, "level": "char",
        "divergence_index": 1}


def test_anchor_self_inconsistent_persists_every_served_row_the_incident_this_fixes(tmp_path):
    """The diagnosed defect: an unmodified anchor whose own repeated greedy completions
    disagree (here because its own drafter races) makes the gate `oracle_unavailable` for
    EVERY candidate. Part B's job is only to make that refusal localizable -- the actual
    fix (not running speculative decoding inside the oracle at all) is `longctx._oracle_arm`,
    covered in test_longctx.py."""
    record_dir = tmp_path / "identity-divergence"
    anchor_rows = [_row("p0", "h1", "a"), _row("p1", "h2", "b")]
    racy_candidate = [_row("p0", "h1", "a"), _row("p1", "h2", "b"),
                      _row("p0", "h1", "a"), _row("p1", "XX", "bad"),
                      _row("p0", "h1", "a"), _row("p1", "h2", "b")]
    racy_anchor = [_row("p0", "h1", "a"), _row("p1", "h2", "b"),
                  _row("p0", "h1", "a"), _row("p1", "ZZ", "bad"),
                  _row("p0", "h1", "a"), _row("p1", "h2", "b")]
    serve_fn, _calls = _serve([anchor_rows, racy_candidate, racy_anchor])
    reqs = tuple((f"p{i}", json.dumps({"prompt": "x", "temperature": 0}).encode())
                 for i in range(2))
    result = model_identity.check(anchor_recipe=_recipe("/a"), candidate_recipe=_recipe("/c"),
                                  requests=reqs, repeats=3, serve_fn=serve_fn,
                                  record_dir=record_dir)
    assert result.status == "unavailable" and "cannot judge a race" in result.reason
    detail = json.loads(result.detail)
    payload = json.loads(Path(detail["record"]).read_text())
    assert payload["kind"] == "anchor_self_inconsistent"
    # ALL observations: both repeat batches and the single initial anchor serve.
    assert len(payload["anchor_repeats"]) == 6 and len(payload["candidate_repeats"]) == 6
    assert len(payload["anchor_first"]) == 2
    assert payload["first_divergent"] == {
        "kind": "anchor_repeats_disagree_with_each_other", "request_index": 1,
        "prompt_id": "p1", "repeat_a": 0, "repeat_b": 1, "level": "char",
        "divergence_index": 1}


def test_anchor_unstable_between_launches_persists_both_launches(tmp_path):
    record_dir = tmp_path / "identity-divergence"
    anchor_rows = [_row("p0", "h1", "a")]
    candidate_rows = [_row("p0", "hX", "b")]        # differs -> triggers an anchor re-serve
    second_anchor_rows = [_row("p0", "hZ", "c")]    # the anchor disagrees with itself
    serve_fn, _calls = _serve([anchor_rows, candidate_rows, second_anchor_rows])
    reqs = tuple((f"p{i}", json.dumps({"prompt": "x", "temperature": 0}).encode())
                 for i in range(1))
    result = model_identity.check(anchor_recipe=_recipe("/a"), candidate_recipe=_recipe("/c"),
                                  requests=reqs, serve_fn=serve_fn, record_dir=record_dir)
    assert result.status == "unavailable" and "differ between two launches" in result.reason
    detail = json.loads(result.detail)
    payload = json.loads(Path(detail["record"]).read_text())
    assert payload["kind"] == "anchor_unstable"
    # ALL observations: the first anchor serve, the confirming re-serve, AND the
    # candidate (which is what triggered the re-serve by matching the first anchor serve).
    assert payload["anchor_first"][0]["content"] == "a"
    assert payload["anchor_reserve"][0]["content"] == "c"
    assert payload["candidate"][0]["content"] == "b"
    assert payload["first_divergent"] == {
        "kind": "anchor_disagrees_between_two_launches", "request_index": 0,
        "prompt_id": "p0", "level": "char", "divergence_index": 0}


# ------------------------------------------------------------ harness faults: partial evidence

def test_candidate_serve_failure_persists_the_anchor_it_already_served(tmp_path):
    """A harness fault (serve_fn exception) is never a correctness verdict, but every
    observation already made before it fired must still be persisted -- here the anchor
    served fine and the candidate serve then raised."""
    record_dir = tmp_path / "identity-divergence"
    anchor_rows = [_row("p0", "h1", "a"), _row("p1", "h2", "b")]
    serve_fn, _calls = _serve([anchor_rows, RuntimeError("candidate boom")])
    result = model_identity.check(anchor_recipe=_recipe("/a"), candidate_recipe=_recipe("/c"),
                                  requests=REQS, serve_fn=serve_fn, record_dir=record_dir)
    assert result.status == "unavailable" and "serving failed" in result.reason
    detail = json.loads(result.detail)
    payload = json.loads(Path(detail["record"]).read_text())
    assert payload["kind"] == "serving_failed"
    assert payload["anchor_first"][0]["content"] == "a"
    assert "candidate" not in payload and "candidate_repeats" not in payload


def test_anchor_serve_failure_with_nothing_yet_observed_persists_nothing(tmp_path):
    """When the very first serve_fn call raises, there is nothing to persist -- `detail`
    stays empty exactly as it did before record_dir existed."""
    record_dir = tmp_path / "identity-divergence"
    serve_fn, _calls = _serve([RuntimeError("anchor boom")])
    result = model_identity.check(anchor_recipe=_recipe("/a"), candidate_recipe=_recipe("/c"),
                                  requests=REQS, serve_fn=serve_fn, record_dir=record_dir)
    assert result.status == "unavailable" and "serving failed" in result.reason
    assert result.detail == ""
    assert not record_dir.is_dir()


def test_anchor_repetition_serve_failure_persists_the_anchor_and_candidate_repeats(tmp_path):
    """The racy branch's own anchor re-serve (to check whether the anchor itself races)
    can fail too; the candidate's repeats and the first anchor serve were already made
    and must still be persisted."""
    record_dir = tmp_path / "identity-divergence"
    anchor_rows = [_row("p0", "h1", "a"), _row("p1", "h2", "b")]
    racy_candidate = [_row("p0", "h1", "a"), _row("p1", "h2", "b"),
                      _row("p0", "h1", "a"), _row("p1", "XX", "bad"),
                      _row("p0", "h1", "a"), _row("p1", "h2", "b")]
    serve_fn, _calls = _serve([anchor_rows, racy_candidate, RuntimeError("anchor repeat boom")])
    reqs = tuple((f"p{i}", json.dumps({"prompt": "x", "temperature": 0}).encode())
                 for i in range(2))
    result = model_identity.check(anchor_recipe=_recipe("/a"), candidate_recipe=_recipe("/c"),
                                  requests=reqs, repeats=3, serve_fn=serve_fn,
                                  record_dir=record_dir)
    assert result.status == "unavailable" and "anchor repetition serve failed" in result.reason
    detail = json.loads(result.detail)
    payload = json.loads(Path(detail["record"]).read_text())
    assert payload["kind"] == "anchor_repeat_serve_failed"
    assert len(payload["candidate_repeats"]) == 6
    assert payload["anchor_first"][1]["content"] == "b"
    assert "anchor_repeats" not in payload


def test_anchor_reserve_failure_persists_the_anchor_and_candidate_already_served(tmp_path):
    """The final branch's confirming anchor re-serve can also fail; the anchor and
    candidate already served (the ones that triggered the re-serve) must persist."""
    record_dir = tmp_path / "identity-divergence"
    anchor_rows = [_row("p0", "h1", "a")]
    candidate_rows = [_row("p0", "hX", "b")]
    serve_fn, _calls = _serve([anchor_rows, candidate_rows, RuntimeError("reserve boom")])
    reqs = tuple((f"p{i}", json.dumps({"prompt": "x", "temperature": 0}).encode())
                 for i in range(1))
    result = model_identity.check(anchor_recipe=_recipe("/a"), candidate_recipe=_recipe("/c"),
                                  requests=reqs, serve_fn=serve_fn, record_dir=record_dir)
    assert result.status == "unavailable" and "anchor re-serve failed" in result.reason
    detail = json.loads(result.detail)
    payload = json.loads(Path(detail["record"]).read_text())
    assert payload["kind"] == "anchor_reserve_failed"
    assert payload["anchor_first"][0]["content"] == "a"
    assert payload["candidate"][0]["content"] == "b"
    assert "anchor_reserve" not in payload


def test_gates_passes_record_dir_through_the_sixth_target_element(monkeypatch, tmp_path):
    """`check_model_identity_targets` threads an optional sixth per-target element
    (`record_dir`) into `model_identity.check`, alongside the existing fifth (`prepare`)."""
    from . import gates, model_identity as mi

    record_dir = tmp_path / "identity-divergence"
    seen = {}

    def fake_check(**kwargs):
        seen.update(kwargs)
        return mi.IdentityResult("wrong", "differs", json.dumps({"record": "r.json"}))

    monkeypatch.setattr(mi, "check", fake_check)
    targets = [("own", _recipe("/a"), _recipe("/c"), REQS, None, record_dir)]
    verdict = gates.check_model_identity_targets(targets)
    assert not verdict.passed
    assert seen["record_dir"] == record_dir
    assert seen.get("prepare") is None


def test_record_file_path_survives_the_600_char_gate_truncation(tmp_path):
    """`gates.check_model_identity_targets` truncates each target's `detail` to 600
    chars; the record path must therefore be the FIRST key so it is never the part that
    gets cut. Exercises the real `model_identity.check` (via its `serve_fn` seam) so the
    truncation is proven against the real detail shape, not a stub."""
    record_dir = tmp_path / "identity-divergence"
    anchor_rows = [_row("p0", "h1", "x" * 2000)]
    candidate_rows = [_row("p0", "hX", "y" * 2000)]
    serve_fn, _calls = _serve([anchor_rows, candidate_rows, anchor_rows])
    reqs = tuple((f"p{i}", json.dumps({"prompt": "x", "temperature": 0}).encode())
                 for i in range(1))
    result = model_identity.check(anchor_recipe=_recipe("/a"), candidate_recipe=_recipe("/c"),
                                  requests=reqs, serve_fn=serve_fn, record_dir=record_dir)
    assert result.status == "wrong"
    # The gate truncates to 600 chars; a 2000-char content pair pushes well past that, far
    # enough that the truncated JSON is no longer even well-formed -- which is exactly why
    # the record path must be first: a regex extraction still finds it.
    assert len(result.detail) > 600
    truncated = result.detail[:600]
    with pytest.raises(json.JSONDecodeError):
        json.loads(truncated)
    match = re.search(r'"record":\s*"([^"]+)"', truncated)
    assert match is not None
    path = Path(match.group(1))
    assert path.is_file()
    payload = json.loads(path.read_text())
    assert payload["anchor_first"][0]["content"] == "x" * 2000  # the full text, not the 80-char preview

