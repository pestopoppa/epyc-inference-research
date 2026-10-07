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
    """`script` is a list of row-lists, one per `serve_fn` call, in call order."""
    calls = []

    def serve_fn(recipe, selected):
        calls.append((recipe.build_dir, len(selected)))
        return script.pop(0)
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
    assert payload["divergence"] == {"level": "token", "index": 2}
    assert payload["anchor"][0]["tokens"] == [1, 2, 3]
    assert payload["candidate"][0]["tokens"] == [1, 2, 4]
    assert payload["anchor"][0]["content"] == "hello world"
    # detail itself also carries the single differing pair for a quick look without
    # opening the file.
    assert detail["divergence"] == {"level": "token", "index": 2}


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
    assert len(payload["candidate_repeats"]) == 6
    assert payload["anchor"][1]["content"] == "b"


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
    assert len(payload["anchor_repeats"]) == 6 and len(payload["candidate_repeats"]) == 6


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
    assert payload["first"][0]["content"] == "a"
    assert payload["second"][0]["content"] == "c"


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
    assert payload["anchor"][0]["content"] == "x" * 2000  # the full text, not the 80-char preview

