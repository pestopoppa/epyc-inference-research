"""KPF-64.3 amendment 3: the replay prompt cap is token-derived so no prompt exceeds one slot (262144/np4 = 65536),
which is what excluded codex-00 from every A-D arm (request 9 = 66,982 tokens)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import adapter
import pytest


def turns_of(sizes):
    """Each turn adds `size` chars on top of the history (ref output 'r')."""
    return [{"pre": "x" * n, "ref": "r", "reset": False} for n in sizes]


def test_char_cap_is_token_derived_and_below_legacy():
    cap = adapter.prompt_char_cap()
    assert cap == int((65536 - 128 - 1024) * 2.5) == 160960
    assert cap < 200_000                       # the legacy cap that let codex-00 through
    assert adapter.prompt_char_cap(100_000) == 100_000   # an explicit lower cap still wins
    # at the worst MEASURED density (2.62 chars/token, codex-04) a capped prompt fits with n_predict to spare
    assert cap / 2.62 + 128 < 65536
    # the codex-00 case: 66,982 tokens at 2.9 chars/token is ~194k chars, which the new cap refuses
    assert 66982 * 2.9 > cap


def test_truncate_stops_at_char_cap_and_keeps_prefix():
    t = turns_of([100_000, 30_000, 30_000, 30_000])   # prompts: 100k, 130k+, 160k+, 190k+ (history grows)
    kept = adapter.truncate(t, adapter.prompt_char_cap(), 200)
    assert 1 <= len(kept) < len(t)
    assert all(len(p) <= adapter.prompt_char_cap() for _, p, _ in adapter.iter_prompts(kept))
    assert kept == t[:len(kept)]


class _Tok(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        out = json.dumps({"tokens": list(range(len(body["content"]) // 2))}).encode()   # 2 chars/token: denser than 2.5
        self.send_response(200)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_truncate_with_exact_token_counter():
    srv = HTTPServer(("127.0.0.1", 0), _Tok)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        count = adapter.tokenize_counter(f"http://127.0.0.1:{srv.server_address[1]}")
        t = turns_of([1000, 1000, 1000, 1000])        # prompts 1000, 2001, 3002, ... chars -> /2 tokens
        kept = adapter.truncate(t, 10**9, 200, max_tokens=1200, count_tokens=count)
        assert len(kept) == 2                          # 1000 and 1000+1+1000 chars = 500 and 1000 tok; third is 1500
    finally:
        srv.shutdown()


def test_recap_dir_applies_new_cap_to_frozen_files(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    big = turns_of([100_000, 30_000, 30_000, 30_000, 30_000])     # only the first 3 requests fit a 160,960-char cap
    small = turns_of([5_000] * 5)
    rows = [{"id": "claude-00", "store": "claude", "parent": None, "children": 2, "replayable": True, "provider": None,
             "requests": 5, "prompt_chars": 1, "ideal_chars": 1, "ideal_union_chars": 1, "max_prompt_chars": 1},
            {"id": "claude-00.c00", "store": "claude", "parent": "claude-00", "provider": None,
             "requests": 5, "prompt_chars": 1, "ideal_chars": 1, "ideal_union_chars": 1, "max_prompt_chars": 1},
            {"id": "claude-00.c01", "store": "claude", "parent": "claude-00", "provider": None,
             "requests": 5, "prompt_chars": 1, "ideal_chars": 1, "ideal_union_chars": 1, "max_prompt_chars": 1}]
    (src / "manifest.json").write_text(json.dumps({"schema": "kpf64.sessions.v2", "params": {}, "sessions": rows,
                                                   "totals": {"requests": 15}}))
    (src / "claude-00.json").write_text(json.dumps({"id": "claude-00", "store": "claude", "turns": small, "replayable": True,
                                                    "children": [{"id": "claude-00.c00", "parent_turn": 0},
                                                                 {"id": "claude-00.c01", "parent_turn": 1}]}))
    (src / "claude-00.c00.json").write_text(json.dumps({"id": "claude-00.c00", "turns": big, "parent": "claude-00"}))
    (src / "claude-00.c01.json").write_text(json.dumps({"id": "claude-00.c01", "turns": small, "parent": "claude-00"}))
    before = (src / "claude-00.c00.json").read_text()
    out = adapter.recap_dir(src, dst, adapter.prompt_char_cap(), min_requests=3)
    assert (src / "claude-00.c00.json").read_text() == before            # source untouched
    kept = json.loads((dst / "claude-00.c00.json").read_text())["turns"]
    assert 3 <= len(kept) < 5
    assert out["recap"]["requests_after"] < 15 and out["recap"]["dropped"] == []
    # a child cut below min_child_requests drops it, and a parent left with < min_children children goes with it
    with pytest.raises(adapter.ZeroSessionsError):      # children cut to 1 request -> dropped -> parent dropped -> nothing left
        adapter.recap_dir(src, tmp_path / "dst2", 6_000, min_requests=1)
