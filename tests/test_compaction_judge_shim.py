import json
import pathlib
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts/toc"))
import compaction_judge_shim as S  # noqa: E402

CASES = json.load(open(HERE / "fixtures/compaction_judge/cmp3_live_r3_cases.json"))["cases"]
SLIP = pathlib.Path("/mnt/raid0/llm/tmp/dive-ak-intake-20261121/extra/slipstream/repo/src/minisweagent/agents/judge.py")


def stub_for(case_by_prompt, fail_first=0, extra_msg=None):
    calls = {"n": 0}

    def post(url, payload, timeout):
        calls["n"] += 1
        assert url.endswith("/v1/chat/completions")
        if calls["n"] <= fail_first:
            raise OSError("connection refused")
        case = case_by_prompt[payload["messages"][0]["content"]]
        msg = {"role": "assistant", "content": "9" if case["label"] == "accept" else "1"}
        msg.update(extra_msg or {})
        return {"choices": [{"message": msg}], "usage": {"prompt_tokens": 500, "completion_tokens": 1}}

    return post, calls


def prompts():
    return {S.build_prompt(c["summary"], c["actions"]): c for c in CASES}


def test_grammar_is_single_integer_0_to_10():
    assert S.SCORE_GRAMMAR.strip() == 'root ::= "10" | [0-9]'
    for ok in ("0", "7", "10", " 8\n"):
        assert S.parse_score(ok) is not None
    for bad in ("11", "7.5", "score: 7", "", "-1", "1 0", None):
        assert S.parse_score(bad) is None, bad


def test_payload_contract_thinking_off_grammar_greedy_no_cache():
    p = S.build_payload("x")
    assert p["chat_template_kwargs"] == {"enable_thinking": False}
    assert p["grammar"] == S.SCORE_GRAMMAR and p["temperature"] == 0.0 and p["max_tokens"] == 8
    assert p["cache_prompt"] is False


def test_prompt_is_verbatim_slipstream_v2_fast_when_source_present():
    if not SLIP.exists():
        return
    m = re.search(r'_JUDGE_PROMPT_V2_FAST = """(.*?)"""', SLIP.read_text(), re.DOTALL)
    assert m and m.group(1) == S.PROMPT_V2_FAST


def test_fixtures_come_from_our_transcripts_and_cover_both_labels():
    labels = {c["label"] for c in CASES}
    assert labels == {"accept", "reject"} and len(CASES) == 12
    assert sum(c["strength"] == "firm" and c["label"] == "reject" for c in CASES) == 4
    assert all(c["summary"].startswith("## Goal") for c in CASES)


def test_accept_and_reject_fixtures_decided_through_shim():
    post, _ = stub_for(prompts())
    for c in CASES:
        r = S.judge_one(c["summary"], c["actions"], threshold=7, post=post)
        assert r["decision"] == c["label"], c["id"]
        assert r["usage"]["prompt_tokens"] == 500


def test_thinking_present_is_not_accepted():
    post, _ = stub_for(prompts(), extra_msg={"reasoning_content": "hmm"})
    r = S.judge_one(CASES[0]["summary"], CASES[0]["actions"], post=post, retries=2)
    assert r["decision"] == "unknown" and "thinking was not disabled" in r["error"]


def test_transport_retry_then_fail_closed():
    post, calls = stub_for(prompts(), fail_first=2)
    assert S.judge_one(CASES[0]["summary"], CASES[0]["actions"], post=post, retries=3)["decision"] == "accept"
    post, calls = stub_for(prompts(), fail_first=9)
    r = S.judge_one(CASES[0]["summary"], CASES[0]["actions"], post=post, retries=3)
    assert r["decision"] == "unknown" and r["score"] is None and calls["n"] == 3


def test_unparseable_is_unknown_never_accept():
    def post(u, p, t):
        return {"choices": [{"message": {"content": "I think 8"}}]}
    assert S.judge_one("s", [], post=post, retries=1)["decision"] == "unknown"


def test_thinking_blocks_stripped_from_summary():
    assert "SECRET" not in S.build_prompt("<think>SECRET</think>## Goal\nx", [])


def test_real_http_path_against_local_stub_server():
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.update(body)
            out = json.dumps({"choices": [{"message": {"content": "10"}}]}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        r = S.judge_one("## Goal\nx", [{"command": "ls"}], base_url=f"http://127.0.0.1:{srv.server_port}", timeout=5)
    finally:
        srv.shutdown()
    assert r["decision"] == "accept" and seen["grammar"] == S.SCORE_GRAMMAR
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}
