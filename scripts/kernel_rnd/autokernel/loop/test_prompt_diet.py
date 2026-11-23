"""OAB-6 prompt diet in the actor context renderer: golden renders, the before/after
measurement on the recorded DS41 run-8 planner prompt, and the information-kept proofs.

Zero inference: every number here is a byte count of text the renderer produces. Tokens are
ESTIMATES at the measured DS41 run-8 ratio (75,978 chars = 26,293 tokens, `actor_context`
docstring), not a tokenizer run."""
import json
import re
from pathlib import Path

from autokernel.loop import actor_context, actors

FIXTURES = Path(__file__).with_name("fixtures")
PROMPT = (FIXTURES / "ds41-run8-planner-prompt.txt").read_text()
CHARS_PER_TOKEN = 75978 / 26293


def _block(header: str) -> str:
    """The text of the first fenced json block after `header` in the recorded prompt."""
    start = PROMPT.index(header)
    match = re.compile(r"```json\n(.*?)\n```", re.S).search(PROMPT, start)
    return match.group(1)


def _expand(root, value):
    """Inverse of `_dedupe_subtrees`: replace every pointer with the subtree it names."""
    if isinstance(value, dict):
        return {k: _expand(root, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(root, v) for v in value]
    resolved = actor_context._resolve(root, value)
    return _expand(root, resolved) if resolved is not value else value


def _dump(value):
    return json.dumps(value, indent=2, sort_keys=True)


def test_ds41_target_block_before_after_and_information_kept():
    rendered = _block("## Selected target")
    after_target = json.loads(rendered)
    before_target = _expand(after_target, after_target)
    assert "<same as" in rendered and "<same as" not in _dump(before_target)
    # The recorded block is exactly what the renderer makes from the pre-diet target.
    assert actors._dedupe_subtrees(before_target) == after_target
    assert _dump(actors._dedupe_subtrees(before_target)) == rendered
    # Information kept: expanding every pointer restores the pre-diet target exactly.
    assert _expand(after_target, after_target) == before_target
    before, after = len(_dump(before_target)), len(rendered)
    assert after < before
    saved_tokens = (before - after) / CHARS_PER_TOKEN
    assert saved_tokens > 1000          # ESTIMATE at the measured chars/token ratio
    print(f"DS41 target block: {before} -> {after} chars "
          f"({before - after} saved, ~{saved_tokens:.0f} tokens ESTIMATE)")


def test_shared_rows_golden_drop_ids_clip_prose_keep_what_planner_needs():
    shared = json.loads(_block("## Shared historical mechanisms"))
    row = dict(shared["rows"][0])
    kept = set(row)
    fat = {**row, "attempt_id": "a" * 64, "result_sha256": "b" * 64, "original_epoch": "c" * 64,
           "source_store": "/s", "hypothesis_id": "h", "recorded_at": "t", "campaign_id": "c",
           "refusal_reason": "r" * 1200}
    slim = actors._slim_shared_history({**shared, "rows": [fat]})
    out = slim["rows"][0]
    assert set(out) == kept                                   # only id fields went
    assert out["refusal_reason"] == "r" * 500 + " …[clipped 700 chars]"
    for key in kept - {"refusal_reason"}:                     # model/quant/recipe/surface/caveats
        value = row[key]       # recorded prose is already clipped: its first 500 chars survive
        assert out[key] == value or (isinstance(value, str) and out[key].startswith(value[:500])), key
    assert {k: v for k, v in slim.items() if k != "rows"} == {k: v for k, v in shared.items()
                                                              if k != "rows"}
    assert all(k not in r for r in actors._slim_shared_history(shared)["rows"]
               for k in actors.SHARED_ROW_ID_FIELDS)


def test_nothing_to_diet_renders_byte_identical():
    feedback = {"as_of": "t", "rows": [{"a": 1}], "scope": {"model": "m"}}
    ctx = {"serving_observations": feedback}
    assert _dump(feedback) in actors.render_context(ctx)
    small = {"recipe": {"threads": 48}, "scope": {"threads": 48}}      # repeats, but < 200 chars
    assert actors._dedupe_subtrees(small) == small
    assert actors._slim_shared_history({"rows": [{"k": "short", "n": 1}]}) == {
        "rows": [{"k": "short", "n": 1}]}


def test_serving_observations_repeats_point_at_first_copy_and_resolve_back():
    launch = {"dsos": [{"path": f"/lib/lib{i}.so", "sha256": "a" * 64} for i in range(6)]}
    feedback = {"anchor": {"launch": launch}, "scope": {"launch": launch, "epoch": "e1"}}
    text = actors.render_context({"serving_observations": feedback})
    block = json.loads(re.search(r"```json\n(.*?)\n```", text, re.S).group(1))
    assert block["scope"]["launch"] == "<same as $.anchor.launch>"
    assert block["scope"]["epoch"] == "e1"                    # ids are NOT dropped here
    assert _expand(block, block) == feedback
    assert len(text) < len(actors.render_context({"serving_observations": {
        "anchor": {"launch": launch}, "scope": {"launch": {**launch, "x": 1}, "epoch": "e1"}}}))
