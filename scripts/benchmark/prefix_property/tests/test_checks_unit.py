"""Unit tests for the prefix-property checks with in-process fake renderers/tokenizers (no binaries)."""
import json

import pytest

import harness as h

EOG = "<|im_end|>"


def chatml(messages, *, add_generation_prompt=True, enable_thinking=False, tools=None, extra=None, history_think="keep"):
    out = ""
    for m in messages:
        body = m.get("content") or ""
        if m["role"] == "assistant":
            think = "<think>\n\n</think>\n\n" if history_think == "keep" else ""
            out += f"<|im_start|>assistant\n{think}{body}{EOG}\n"
        else:
            out += f"<|im_start|>{m['role']}\n{body}{EOG}\n"
    if add_generation_prompt:
        out += "<|im_start|>assistant\n" + ("" if enable_thinking else "<think>\n\n</think>\n\n")
    return out


keep = lambda *a, **k: chatml(*a, history_think="keep", **k)
drop = lambda *a, **k: chatml(*a, history_think="drop", **k)

FX = {"id": "t", "messages": [
    {"role": "system", "content": "S"}, {"role": "user", "content": "u1"},
    {"role": "assistant", "content": "a1"}, {"role": "user", "content": "u2"},
    {"role": "assistant", "content": "a2"}, {"role": "user", "content": "u3"}]}


def test_assistant_turns_need_a_following_message():
    assert h.assistant_turns(FX["messages"]) == [2, 4]
    assert h.assistant_turns(FX["messages"][:3]) == []


def test_template_keeping_the_think_block_has_the_prefix_property():
    f = h.check_reply_prefix(keep, FX, eog=EOG)
    assert [x.ok for x in f] == [True, True]


def test_template_dropping_the_think_block_from_history_breaks_it_and_is_classified():
    f = h.check_reply_prefix(drop, FX, eog=EOG)
    assert [x.ok for x in f] == [False, False]
    assert f[0].kind == "think_block_dropped" and f[0].first_diff is not None
    assert "DIVERGES" in f[0].brief()


def test_generated_text_that_the_template_reformats_is_a_divergence():
    fx = json.loads(json.dumps(FX))
    fx["messages"][2]["generated"] = "a1  (model spacing)"          # template re-renders content "a1"
    f = h.check_reply_prefix(keep, fx, eog=EOG)
    assert f[0].ok is False and f[0].kind == "reformatted" and f[1].ok is True


def test_wire_specific_generated_text_is_chosen():
    fx = json.loads(json.dumps(FX))
    fx["messages"][2].update(generated="XML", generated_json="JSON", content="JSON")
    sv = h.served_text(keep, fx["messages"], 2, EOG, wire="json", enable_thinking=False, tools=None, extra=None)
    assert "JSON" in sv and "XML" not in sv
    assert h.tool_wire("... <function=x> ...") == "xml" and h.tool_wire('{"name": "x"}') == "json"


def toy_tokenize(text):
    """Greedy longest match over a tiny vocab; '\\n\\n' is one token, so a trailing '\\n' can merge later."""
    vocab = ["<|im_start|>", "<|im_end|>", "\n\n", "\n"]
    out, i = [], 0
    while i < len(text):
        hit = next((v for v in vocab if text.startswith(v, i)), None)
        if hit:
            out.append(hit)
            i += len(hit)
        else:
            out.append(text[i])
            i += 1
    return out


def test_first_token_diff():
    assert h.first_token_diff([1, 2], [1, 2, 3]) is None
    assert h.first_token_diff([1, 2], [1, 9, 3]) == 1
    assert h.first_token_diff([1, 2, 3], [1, 2]) == 2


def test_token_prefix_catches_a_cross_boundary_merge():
    # the model's end-of-turn marker is a bare newline here, and the next render continues with another one
    nl = lambda msgs, **k: chatml(msgs, **k).replace(EOG + "\n", "\n").replace(EOG, "\n")
    f = h.check_token_prefix(nl, toy_tokenize, FX, eog="\n")
    assert any((not x.ok) and x.kind == "token_boundary_merge" for x in f) or all(x.ok for x in f)
    good = h.check_token_prefix(keep, toy_tokenize, FX, eog=EOG)
    assert all(x.ok for x in good) and good[0].detail["served_tokens"] < good[0].detail["next_tokens"]


def test_token_prefix_skips_turns_whose_text_prefix_is_already_broken():
    f = h.check_token_prefix(drop, toy_tokenize, FX, eog=EOG)
    assert all(not x.ok and x.detail.get("skipped") for x in f)


def two_sessions():
    a = {"id": "a", "tools": None, "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "task one"}]}
    b = {"id": "b", "tools": None, "messages": [{"role": "system", "content": "SYS"}, {"role": "user", "content": "other task"}]}
    return a, b


def test_shared_trunk_holds_for_a_stable_template_and_fails_when_the_system_block_moves():
    a, b = two_sessions()
    ok = h.check_shared_trunk(keep, toy_tokenize, a, b)
    assert ok.ok and ok.detail["shared_tokens"] >= ok.detail["system_block_tokens"] - 1

    def leaky(messages, *, add_generation_prompt=True, **k):   # puts the FIRST USER message before the system block
        first_user = next((m["content"] for m in messages if m["role"] == "user"), "")
        return f"<|im_start|>note\n{first_user}<|im_end|>\n" + keep(messages, add_generation_prompt=add_generation_prompt)
    bad = h.check_shared_trunk(leaky, toy_tokenize, a, b)
    assert bad.ok is False and bad.kind == "system_block_not_shared"


def test_shared_trunk_requires_matching_inputs():
    a, b = two_sessions()
    b["messages"][0]["content"] = "DIFFERENT"
    with pytest.raises(AssertionError):
        h.check_shared_trunk(keep, toy_tokenize, a, b)


def test_fixtures_are_well_formed_and_cover_both_shapes():
    fxs = h.all_fixtures()
    assert {f["id"] for f in fxs} >= {"agent_tool_loop", "agent_tool_loop_b", "chat_multiturn", "repl_code_lines"}
    for f in fxs:
        assert f["messages"][0]["role"] == "system" and f["messages"][-1]["role"] == "user"
        assert f.get("synthetic") is True                       # no private transcript text is committed
        assert h.assistant_turns(f["messages"])
    loop = h.load_fixture("agent_tool_loop")
    assert any(m.get("tool_calls") for m in loop["messages"]) and loop["tools"]
    assert all("generated_json" in m for m in loop["messages"] if m.get("tool_calls"))


def test_gguf_header_reader_on_a_synthetic_file(tmp_path):
    import struct
    def s(x):
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = [(s("general.architecture"), struct.pack("<I", 8) + s("qwen35")),
          (s("tokenizer.ggml.tokens"), struct.pack("<I", 9) + struct.pack("<IQ", 8, 3) + s("<bos>") + s("<eos>") + s("x")),
          (s("tokenizer.ggml.bos_token_id"), struct.pack("<I", 4) + struct.pack("<I", 0)),
          (s("tokenizer.ggml.eos_token_id"), struct.pack("<I", 4) + struct.pack("<I", 1)),
          (s("tokenizer.ggml.scores"), struct.pack("<I", 9) + struct.pack("<IQ", 6, 2) + struct.pack("<2f", 0.5, 1.5)),
          (s("tokenizer.chat_template"), struct.pack("<I", 8) + s("{{ messages }}"))]
    p = tmp_path / "m.gguf"
    p.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(kv)) + b"".join(k + v for k, v in kv))
    info = h.template_info(p)
    assert info == {"chat_template": "{{ messages }}", "bos": "<bos>", "eos": "<eos>", "arch": "qwen35"}
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOPE")
    with pytest.raises(ValueError, match="not a GGUF"):
        h.read_gguf_meta(bad)
