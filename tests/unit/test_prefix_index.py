"""RTG-58 P2 — the revived radix tree (``src/inference/prefix_index.py``).

Pure, offline tests of the data structure and of the server-truth
reconciliation. Admission and wiring tests are in
``test_prefix_index_admission.py``; the flag-off proof is in
``test_prefix_index_flag_off.py``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.backends.context_limits import PoolOccupancy, SlotState, parse_slots
from src.inference import prefix_index as pi
from src.inference.prefix_index import PrefixIndex, RadixTree, block_hashes

BLOCK = 64


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _occ(*slots) -> PoolOccupancy:
    return PoolOccupancy(url="http://x:1", slots=tuple(
        SlotState(slot_id=sid, n_ctx=8192, is_processing=busy, n_prompt_tokens=n,
                  n_remain=None, n_decoded=1, id_task=task)
        for sid, busy, n, task in slots))


def _text(tag: str, blocks: int) -> str:
    return "".join(f"{tag}{i:05d}".ljust(BLOCK, ".") for i in range(blocks))


@pytest.fixture
def idx(tmp_path):
    clock = Clock()
    index = PrefixIndex("http://localhost:8083", clock=clock, wall=clock,
                        launch_id=lambda url: "L1", directory=lambda: tmp_path, block=BLOCK)
    index.clock = clock  # type: ignore[attr-defined]
    return index


# ── block hashes ──────────────────────────────────────────────────────────────


def test_block_hashes_are_chained_prefix_identities():
    a = _text("a", 4)
    assert len(block_hashes(a, BLOCK)) == 4
    assert block_hashes(a + "tail", BLOCK) == block_hashes(a, BLOCK)  # partial block not hashed
    b = a[:BLOCK] + "X" + a[BLOCK + 1:]
    ha, hb = block_hashes(a, BLOCK), block_hashes(b, BLOCK)
    assert ha[0] == hb[0] and all(x != y for x, y in zip(ha[1:], hb[1:]))  # chained
    assert block_hashes("", BLOCK) == () and block_hashes(None, BLOCK) == ()


# ── the tree ──────────────────────────────────────────────────────────────────


def test_split_match_and_prune():
    t = RadixTree()
    t.insert("A", ("1", "2", "3", "4"))
    t.insert("B", ("1", "2", "9"))
    assert t.match(("1", "2", "3", "4", "5")) == (4, {"A"})
    assert t.match(("1", "2", "9")) == (3, {"B"})
    assert t.match(("1", "2", "7")) == (2, {"A", "B"})
    assert t.match(("8",)) == (0, set())
    assert t.node_count() == 3  # [1,2] -> [3,4], [9]
    assert t.remove("A")
    assert t.match(("1", "2", "3")) == (2, {"B"})
    assert t.node_count() == 2  # [3,4] pruned
    t.remove("B")
    assert t.node_count() == 0 and len(t) == 0


def test_accept_filter_returns_deepest_accepted_holder():
    t = RadixTree()
    t.insert("busy", ("1", "2", "3"))
    t.insert("idle", ("1", "2"))
    assert t.match(("1", "2", "3"), accept=lambda e: e == "idle") == (2, {"idle"})


def test_union_counts_shared_prefixes_once():
    t = RadixTree()
    t.insert("A", tuple("abcdef"))
    t.insert("B", tuple("abcxyz"))
    assert t.union_blocks({"A", "B"}) == 9  # abc + def + xyz
    assert t.union_blocks({"A"}) == 6


def test_stale_slot_regression_from_the_january_radix_cache():
    """The deleted radix_cache.py stamped slot_id on every node of a path and
    cleared only the leaf on eviction, so an interior node kept naming a slot that
    had been reassigned. Here a reinsert REPLACES the entry's path everywhere."""
    t = RadixTree()
    t.insert("slot:0", ("t", "r", "u", "n", "k"))
    t.insert("slot:1", ("t", "r", "u", "x"))
    t.insert("slot:0", ("o", "t", "h", "e", "r"))  # slot 0 now holds something else
    depth, held = t.match(("t", "r", "u", "n", "k"))
    assert held == {"slot:1"} and depth == 3      # slot 0 no longer claims the trunk
    assert t.match(("o", "t", "h"))[1] == {"slot:0"}


# ── observation, binding, staleness ────────────────────────────────────────────


def test_exact_slot_binds_to_id_task_and_matches_idle(idx):
    text = _text("p", 20)
    idx.observe_served(text, slot_id=1, prompt_tokens=400, generated_tokens=10)
    assert idx.lookup(text).source == "served"          # pending: unverified, no credit
    idx.reconcile(_occ((0, False, 0, None), (1, False, 410, 7)))
    m = idx.lookup(text + "more")
    assert (m.source, m.slot_id, m.matched_chars) == ("slot_idle", 1, 20 * BLOCK)
    assert m.tokens_est == int(20 * BLOCK * 400 / len(text))
    assert idx.status()["slots"]["1"]["id_task"] == 7 and idx.stats["bound_exact"] == 1


def test_chat_lane_binding_is_inferred_from_slots_and_never_guessed(idx):
    text = _text("c", 10)
    idx.observe_served(text, slot_id=None, prompt_tokens=200, generated_tokens=5)
    # two idle slots with the expected token count -> ambiguous -> no bind
    idx.reconcile(_occ((0, False, 205, 3), (1, False, 205, 4)))
    assert idx.lookup(text).source == "served"
    idx.reconcile(_occ((0, False, 205, 3), (1, True, 900, 5)))
    m = idx.lookup(text)
    assert (m.source, m.slot_id) == ("slot_idle", 0) and idx.stats["bound_inferred"] == 1


@pytest.mark.parametrize("after,reason", [
    ((1, False, 410, 8), "id_task_changed"),
    ((1, False, 0, 7), "cleared"),
    ((1, False, 300, 7), "tokens_shrunk"),
])
def test_content_changed_under_us_is_dropped(idx, after, reason):
    text = _text("s", 12)
    idx.observe_served(text, slot_id=1, prompt_tokens=400, generated_tokens=10)
    idx.reconcile(_occ((1, False, 410, 7)))
    assert idx.lookup(text).source == "slot_idle"
    idx.reconcile(_occ(after))
    assert idx.lookup(text).source is None
    assert idx.stats["stale_drops"][reason] == 1


def test_missing_slot_and_server_relaunch_drop(idx, tmp_path):
    text = _text("r", 8)
    idx.observe_served(text, slot_id=0, prompt_tokens=100, generated_tokens=0)
    idx.reconcile(_occ((0, False, 100, 1)))
    idx.reconcile(_occ((1, False, 0, None)))
    assert idx.stats["stale_drops"]["slot_missing"] == 1
    idx.observe_served(text, slot_id=0, prompt_tokens=100, generated_tokens=0)
    idx.reconcile(_occ((0, False, 100, 2)))
    idx._launch_id_fn = lambda url: "L2"
    idx.reconcile(_occ((0, False, 100, 2)))
    assert idx.lookup(text).source is None and idx.stats["stale_drops"]["launch_changed"] >= 1


def test_busy_slot_counts_only_with_fork(idx):
    text = _text("b", 16)
    idx.observe_served(text, slot_id=2, prompt_tokens=300, generated_tokens=0)
    idx.reconcile(_occ((2, False, 300, 5)))
    idx.reconcile(_occ((2, True, 300, 5)))
    assert idx.lookup(text).source is None          # a busy slot cannot take the request
    assert idx.lookup(text, fork=True).source == "slot_busy"


def test_unverified_entry_times_out_to_served_or_away(idx):
    text = _text("v", 6)
    idx.observe_served(text, slot_id=None, prompt_tokens=50, generated_tokens=0)
    idx.observe_served(_text("w", 6), slot_id=3, prompt_tokens=50, generated_tokens=0)
    idx.clock.t += pi.VERIFY_TIMEOUT_S + 1
    idx.reconcile(None)
    kinds = idx.status()["entries"]
    assert kinds == {"served": 1} and idx.stats["verify_timeouts"] == 2


def test_index_stays_bounded_without_any_reconcile(idx):
    """Pin policy off (the shadow-window default) means nothing calls reconcile;
    observation alone must still expire pending entries and honour the caps."""
    for i in range(3 * pi.MAX_PENDING_ENTRIES):
        idx.observe_served(_text(f"c{i}", 6), slot_id=None, prompt_tokens=50, generated_tokens=0)
        idx.clock.t += 0.01
    kinds = idx.status()["entries"]
    assert kinds["pending"] == pi.MAX_PENDING_ENTRIES
    assert kinds["served"] == pi.MAX_SERVED_ENTRIES
    assert kinds["pending"] + kinds["served"] == len(idx._entries)
    # time, not count: everything pending is past the verify timeout
    idx.clock.t += pi.VERIFY_TIMEOUT_S + 1
    idx.observe_served(_text("last", 6), slot_id=None, prompt_tokens=50, generated_tokens=0)
    kinds = idx.status()["entries"]
    assert kinds == {"pending": 1, "served": pi.MAX_SERVED_ENTRIES}
    assert idx.status()["nodes"] <= (pi.MAX_SERVED_ENTRIES + 1) * 6


# ── in-flight: fork lookups, trunk owner, pinning, unique cells ────────────────


def test_inflight_trunk_owner_and_exact_junction(idx):
    trunk = _text("T", 40)
    idx.begin(1, trunk + "child one", prompt_tokens=2600, prefill_s=60)
    owner = idx.trunk_owner(2, trunk + "child two", min_tokens=100)
    assert owner is not None and owner.entry_id == "inflight:1"
    assert owner.junction_chars == len(trunk) + len("child ")
    assert idx.trunk_owner(2, trunk + "x", min_tokens=10**6) is None   # below threshold
    assert idx.trunk_owner(0, trunk + "x", min_tokens=1) is None       # only EARLIER siblings
    idx.prefilled(1)
    assert idx.trunk_owner(2, trunk + "child two", min_tokens=100) is None
    m = idx.lookup(trunk + "child two", fork=True)
    assert m.source == "inflight" and m.junction_chars == len(trunk) + len("child ")
    assert idx.lookup(trunk, fork=False).source is None    # no fork: in-flight is not reusable
    idx.end(1)
    assert idx.lookup(trunk, fork=True).source is None


def test_trunk_owner_released_when_server_shows_no_prefill(idx):
    """Review D1: ``/slots`` showing no prefill ends the hold only once the owner
    was SEEN prefilling in a read that postdates its admission (one /slots TTL
    = half the grace), or after the dispatch grace (two TTLs)."""
    grace = pi.PrefixIndex.trunk_grace_s()
    trunk = _text("Q", 40)
    idx.begin(1, trunk, prefill_s=600)
    # A cached read from before the owner reached the server: zero prefilling.
    assert idx.trunk_owner(2, trunk, min_tokens=1, server_prefilling=0) is not None
    # A prefilling slot in a read that may predate the dispatch does not count.
    assert idx.trunk_owner(2, trunk, min_tokens=1, server_prefilling=1) is not None
    assert idx.trunk_owner(2, trunk, min_tokens=1, server_prefilling=0) is not None
    idx.clock.t += grace / 2.0
    assert idx.trunk_owner(2, trunk, min_tokens=1, server_prefilling=1) is not None  # seen
    assert idx.trunk_owner(2, trunk, min_tokens=1, server_prefilling=0) is None  # now over


def test_trunk_owner_never_seen_prefilling_ends_after_the_grace(idx):
    grace = pi.PrefixIndex.trunk_grace_s()
    trunk = _text("R", 40)
    idx.begin(1, trunk, prefill_s=600)
    idx.clock.t += grace - 0.01
    assert idx.trunk_owner(2, trunk, min_tokens=1, server_prefilling=0) is not None
    idx.clock.t += 0.02
    assert idx.trunk_owner(2, trunk, min_tokens=1, server_prefilling=0) is None
    assert idx.trunk_owner(2, trunk, min_tokens=1, server_prefilling=None) is not None


def test_pin_candidate_only_fresh_idle_long_enough(idx):
    text = _text("P", 30)
    idx.observe_served(text, slot_id=3, prompt_tokens=3000, generated_tokens=0)
    idx.reconcile(_occ((3, False, 3000, 1)))
    assert idx.pin_candidate(text, min_tokens=100, fresh_s=2.0) == 3
    assert idx.pin_candidate(text, min_tokens=10**6, fresh_s=2.0) is None
    idx.clock.t += 5
    assert idx.pin_candidate(text, min_tokens=100, fresh_s=2.0) is None   # stale view
    idx.reconcile(_occ((3, True, 3000, 1)))
    assert idx.pin_candidate(text, min_tokens=100, fresh_s=2.0) is None   # busy: never


def test_unique_cells_union_vs_sum(idx):
    trunk = _text("U", 30)
    for t in (1, 2, 3):
        idx.begin(t, trunk + _text(f"c{t}", 2))
    cells = idx.unique_cells()
    assert cells["entries"] == 3
    assert cells["sum_tokens_est"] > cells["union_tokens_est"]
    assert cells["shareable_tokens_est"] == cells["sum_tokens_est"] - cells["union_tokens_est"]


# ── host-wide ledger ────────────────────────────────────────────────────────────


def test_ledger_shares_verified_slots_across_workers(tmp_path):
    clock = Clock()
    mk = lambda: PrefixIndex("http://127.0.0.1:8083", clock=clock, wall=clock,  # noqa: E731
                             launch_id=lambda url: "L1", directory=lambda: tmp_path, block=BLOCK)
    a, b = mk(), mk()
    text = _text("H", 12)
    a.observe_served(text, slot_id=1, prompt_tokens=100, generated_tokens=0)
    clock.t += 1
    a.reconcile(_occ((1, False, 100, 4)))
    state = json.loads(a.ledger_path().read_text())
    assert state["slots"]["1"]["id_task"] == 4 and "text" not in json.dumps(state)
    b.reconcile(_occ((1, False, 100, 4)))          # worker B never served it
    assert b.lookup(text).source == "slot_idle"
    clock.t += 1
    b.reconcile(_occ((1, False, 100, 5)))          # B sees the slot change; A learns from B
    clock.t += 1
    a.reconcile(None)
    assert a.lookup(text).source is None


def test_ledger_off_writes_nothing(tmp_path):
    index = PrefixIndex("http://x:9", launch_id=lambda u: None, directory=lambda: tmp_path,
                        host_wide_ledger=False, block=BLOCK)
    index.observe_served(_text("n", 4), slot_id=0, prompt_tokens=10, generated_tokens=0)
    index.reconcile(_occ((0, False, 10, 1)))
    assert list(tmp_path.iterdir()) == []


# ── registry and the serving-record hook ────────────────────────────────────────


def test_one_index_per_physical_server(monkeypatch):
    pi.reset_indexes()
    try:
        assert pi.get_index("http://localhost:8083") is pi.get_index("http://127.0.0.1:8083/")
        assert pi.get_index("http://localhost:8083") is not pi.get_index("http://localhost:8084")
    finally:
        pi.reset_indexes()


def test_observe_record_is_inert_with_the_flag_off(monkeypatch):
    pi.reset_indexes()
    monkeypatch.delenv(pi.FLAG_ENV, raising=False)
    pi.observe_record({"dispatched": True, "outcome": "ok",
                       "server": {"base_url": "http://localhost:8083"}}, "x" * 4096)
    assert pi.peek_index("http://localhost:8083") is None


def test_observe_record_feeds_prediction_error(monkeypatch, tmp_path):
    pi.reset_indexes()
    monkeypatch.setenv(pi.FLAG_ENV, "1")
    monkeypatch.setenv(pi.HOST_WIDE_ENV, "0")
    try:
        text = "z" * 4096
        pi.observe_record({
            "dispatched": True, "outcome": "ok",
            "server": {"base_url": "http://localhost:8099"},
            "timings": {"prompt_n": 900, "cache_n": 100, "predicted_n": 5},
            "notes": {"server_slot": 2},
            "kv_admission": {"prefix_index": {"predicted_cache_tokens": 300,
                                              "predicted_slot": 2}},
        }, text)
        idx = pi.peek_index("http://localhost:8099")
        assert idx is not None
        assert idx.stats["predictions"] == 1 and idx.stats["prediction_abs_err_tokens"] == 200
        assert idx.stats["prediction_over"] == 1 and idx.stats["slot_prediction_hits"] == 1
        assert idx.status()["entries"] == {"pending": 1}
    finally:
        pi.reset_indexes()


def test_parse_slots_carries_id_task():
    occ = parse_slots("u", [{"id": 0, "id_task": 41, "is_processing": False,
                             "n_prompt_tokens": 9}, {"id": 1, "id_task": True}])
    assert occ.slots[0].id_task == 41 and occ.slots[1].id_task is None
    # id_task never changes slot equality (compare=False), like prompt_text
    assert occ.slots[0] == SlotState(slot_id=0, n_ctx=None, is_processing=False,
                                     n_prompt_tokens=9, n_remain=None)


def test_key_text_matches_the_fingerprinted_text():
    from src.backends import serving_calls

    req = SimpleNamespace(prompt="hello world", chat_payload=None)
    assert pi.key_text_for_request(req) == serving_calls._prompt_text_for_fingerprint(req)
    tool_req = SimpleNamespace(prompt="", chat_payload={"messages": [{"role": "user"}]})
    assert pi.key_kind_for_request(tool_req) == "approx"
    assert pi.key_kind_for_request(req) == "exact"
