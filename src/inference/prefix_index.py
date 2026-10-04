"""Orchestrator-side prefix index: a revived radix tree (RTG-58 P2, KPF-2x).

Feature flag ``ORCHESTRATOR_PREFIX_INDEX`` (default OFF). With it off nothing in
this module runs: every call site checks :func:`enabled` first, and no index,
file or thread is ever created.

What it is
==========
One index per PHYSICAL llama-server (keyed ``host_port`` exactly as the
long-prefill lease keys it, so roles sharing a server share an index — KV-0b).
It records which prompt prefixes are where, as far as the orchestrator can know:

* ``slot``     — the prompt a server slot holds, server-VERIFIED: bound to the
                 slot's ``id_task`` in ``GET /slots`` (see Staleness below).
* ``pending``  — a served prompt not yet bound to a slot (the chat lane gets no
                 ``id_slot`` back; binding is inferred from ``/slots``).
* ``served``   — a served prompt whose slot could not be determined. It may live
                 in ``--cache-ram``; it is used for ordering only, never credit.
* ``inflight`` — an admitted request (SGLang's ``lock_ref``): the trunk it is
                 prefilling right now. In-process only.

What it drives (each behind the flag; see ``kv_pool_admission``)
================================================================
(a) trunk-first: a request whose trunk an in-flight sibling is still
    prefilling waits for that prefill (``ORCHESTRATOR_PREFIX_INDEX_FORK=auto``
    and a server whose ``/props.slot_fork`` reports the fork on: without the
    server-side fork, RTG-58 P1, a busy slot's cells cannot be shared, so
    holding would only add latency and a fork credit would over-admit);
(b) LPM ordering: a bounded longest-prefix-match bypass in the admission queue;
(c) slot pinning: ``ORCHESTRATOR_PREFIX_INDEX_PIN=idle`` pins ``id_slot`` only to
    a VERIFIED IDLE slot holding the longest prefix (default ``off``; UFH14-B4);
(d) unique cells: the union of resident/in-flight prefixes vs their sum, and
    (fork on) a reservation credit for cells the server will share;
(e) the fork plan: source kind/slot and junction, recorded per admission.

Keys
====
The key is the text the serving record fingerprints
(``serving_calls._prompt_text_for_fingerprint``): the ``/completion`` prompt
verbatim, the single user message on the chat lane, and ``tools`` + ``messages``
JSON for a chat payload. On the single-message chat lane the rendered prompt is
``HEAD + content + TAIL``, so content prefixes ARE rendered prefixes. For a JSON
payload the prefix relation holds at message granularity (``key_kind=approx``).
Nothing is canonicalized: the server tokenizes raw bytes, and a canonical key
would invent matches the server never sees (prefix_cache KV-6).

The tree is path-compressed over chained block hashes (``BLOCK_CHARS`` chars per
block, vLLM-style ``h_i = H(h_{i-1} || block_i)``), so a 60k-token trunk is ~500
hashes, not 60k nodes, and slot records fit a small host-wide JSON ledger. A
match is exact to the block: it errs LOW by < one block, never high.

Staleness — the server is the source of truth
=============================================
A slot entry is bound to the ``id_task`` ``/slots`` reports for that slot when
the slot is idle and holds ~the tokens our call left there (prompt + generated,
± ``SLOT_TOKEN_TOLERANCE``). It is dropped the moment ``/slots`` shows a different
``id_task`` (someone else's task ran there), fewer tokens than we left (cleared /
truncated / purged idle slot), the slot missing, or a different server launch.
On v10 (no content hash) "same id_task, same token count" is the strongest
content-identity test available.

KPF-27e — the RTG-58 P1 server interface (``kpf-p1-20261004/INTERFACE.md``).
Each field is used when present and falls back to the v10 inference above when
absent (the server emits them only with ``--slot-fork-min-tokens > 0``):

* ``/slots`` ``content_epoch`` -> staleness: an entry bound at (slot, epoch E,
  N tokens) stays valid while the epoch is E and ``n_prompt_tokens >= N``; an
  appending foreign task (new ``id_task``, same epoch) no longer drops it.
* ``/slots`` ``prefix_hash`` (slot and per checkpoint) -> verification without a
  tokenizer: a hash change at a known length drops the entry
  (``prefix_hash_changed``); ``fnv1a64_tokens`` reproduces the hash for a
  ``/tokenize``-based check.
* ``/slots`` ``checkpoints`` -> a ``checkpoint``-mode fork from a BUSY slot is
  predicted at its largest checkpoint <= the match, not the full LCP.
* ``/slots`` ``kv_cells`` / ``kv_pool`` -> the gate's occupancy counts forked
  cells once (``PoolOccupancy.projected_unique_tokens``); the index status
  shows the server's unique-cell pool.
* ``timings.id_slot`` / ``timings.id_task`` -> exact binding on every endpoint,
  the chat lane included (bound when ``/slots[slot].id_task`` equals the task).
* ``/props.slot_fork`` -> ``fork_enabled(caps)``: ``FORK=auto`` (documented)
  follows the server; ``FORK=1`` is the same but warns when the server reports
  no fork (and stays off: never fork credit without real sharing);
  ``min_tokens`` raises the trunk-hold and fork-credit floors.
* ``checkpoint_at: [{"message": k, "at": ...}]`` -> sent by the scout fan-out
  (the one fan-out site that builds its own payload) at the end of the shared
  system message; the generic lanes do not rewrite client payloads.

Scouts write no ``serving_call`` record (they stream straight to the server),
so their gate feeds this index directly with the same projection as
``observe_record`` (``scout_stage._scout_observe``).

Bounds: ``slot`` entries are one per server slot; ``pending`` is capped at
``MAX_PENDING_ENTRIES`` (LRU) and expires after ``VERIFY_TIMEOUT_S``; ``served``
is capped at ``MAX_SERVED_ENTRIES``; ``inflight`` ends at release. Expiry runs on
every observation, so the index is bounded even when nothing polls ``/slots``.

Concurrency: one ``threading.Lock`` per index, a leaf lock (nothing is called
out while it is held). Host-wide (6 uvicorn workers): verified slot entries are
shared through ``{tmp_dir}/kv_prefix_index.{host}_{port}.json``, rewritten
atomically under an flock and merged newest-wins per slot; pending and in-flight
entries stay per process.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from src.backends.context_limits import DEFAULT_OCCUPANCY_TTL_S

logger = logging.getLogger(__name__)

FLAG_ENV = "ORCHESTRATOR_PREFIX_INDEX"
BLOCK_CHARS_ENV = "ORCHESTRATOR_PREFIX_INDEX_BLOCK_CHARS"
DEFAULT_BLOCK_CHARS = 512
HOST_WIDE_ENV = "ORCHESTRATOR_PREFIX_INDEX_HOST_WIDE"
FORK_ENV = "ORCHESTRATOR_PREFIX_INDEX_FORK"
PIN_ENV = "ORCHESTRATOR_PREFIX_INDEX_PIN"
PIN_MIN_TOKENS_ENV = "ORCHESTRATOR_PREFIX_INDEX_PIN_MIN_TOKENS"
DEFAULT_PIN_MIN_TOKENS = 2048
PIN_FRESH_S_ENV = "ORCHESTRATOR_PREFIX_INDEX_PIN_FRESH_S"
DEFAULT_PIN_FRESH_S = 2.0
LPM_ENV = "ORCHESTRATOR_PREFIX_INDEX_LPM"
LPM_MIN_TOKENS_ENV = "ORCHESTRATOR_PREFIX_INDEX_LPM_MIN_TOKENS"
DEFAULT_LPM_MIN_TOKENS = 2048
LPM_MAX_SKIPS_ENV = "ORCHESTRATOR_PREFIX_INDEX_LPM_MAX_SKIPS"
DEFAULT_LPM_MAX_SKIPS = 1
TRUNK_MIN_TOKENS_ENV = "ORCHESTRATOR_PREFIX_INDEX_TRUNK_MIN_TOKENS"
DEFAULT_TRUNK_MIN_TOKENS = 4096
TRUNK_HOLD_S_ENV = "ORCHESTRATOR_PREFIX_INDEX_TRUNK_HOLD_S"
DEFAULT_TRUNK_HOLD_S = 120.0
TRUNK_GRACE_S_ENV = "ORCHESTRATOR_PREFIX_INDEX_TRUNK_GRACE_S"
#: An in-flight trunk owner counts as "prefill over" by ``/slots`` (zero slots
#: prefilling) only once it was SEEN prefilling, or after this grace since its
#: admission. ``/slots`` is a cached read (``DEFAULT_OCCUPANCY_TTL_S``, 1.5 s), so
#: a read taken before the owner reached the server shows zero prefilling slots;
#: two TTLs guarantee the read postdates the dispatch (review D1).
DEFAULT_TRUNK_GRACE_S = 2.0 * DEFAULT_OCCUPANCY_TTL_S
CREDIT_FRESH_S_ENV = "ORCHESTRATOR_PREFIX_INDEX_CREDIT_FRESH_S"
#: A slot entry is a credit / LPM / fork source only while ``/slots`` verified it
#: this recently (review D6: a ledger-adopted or long-unverified entry must not be
#: credited when ``/slots`` is unavailable). Every gate poll reconciles first, so
#: with ``/slots`` up this never bites.
DEFAULT_CREDIT_FRESH_S = 5.0
#: Chars per token for a prefix whose server-measured ratio is unknown (in-flight
#: siblings, unmeasured slot entries). Measured prose runs ~3.6 (the old 4 sized
#: prose ~10% low, review D3); still above the gate's conservative 3.0, so a fork
#: credit subtracted from a 3.0-sized reservation never over-credits.
PREFIX_CHARS_PER_TOKEN = 3.6
VERIFY_TIMEOUT_S = 30.0
SLOT_TOKEN_TOLERANCE = 4
MAX_SERVED_ENTRIES = 32
#: Cap on unbound ``pending`` entries (LRU). Every served call adds one; without
#: this the index grows without bound when nothing reconciles (pin policy off).
MAX_PENDING_ENTRIES = 64
FILE_PREFIX = "kv_prefix_index."
SCHEMA = "epyc.orchestrator.kv_prefix_index.v1"
_HASH_HEX = 16
_OFF = {"0", "false", "no", "off", ""}
_ON = {"1", "true", "yes", "on"}
#: Outcomes whose prompt the server finished prefilling (as prefix_history).
SERVED_OUTCOMES = ("ok", "early_stop")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


# -- configuration --------------------------------------------------------------


def enabled() -> bool:
    """The feature flag. Default OFF; every call site checks this first."""
    return os.environ.get(FLAG_ENV, "0").strip().lower() in _ON


def fork_caps_from_props(props: Any) -> dict[str, Any] | None:
    """KPF-27e: the server's ``/props.slot_fork`` = ``{min_tokens, mode,
    checkpoint_at}``, normalized; None when absent (a v10 server, or a P1 server
    run with ``--slot-fork-min-tokens 0``, which emits nothing)."""
    raw = props.get("slot_fork") if isinstance(props, dict) else None
    if not isinstance(raw, dict):
        return None
    mode = raw.get("mode")
    return {
        "min_tokens": max(0, _int(raw.get("min_tokens")) or 0),
        "mode": mode if mode in ("none", "kv", "checkpoint") else "none",
        "checkpoint_at": raw.get("checkpoint_at") is True,
    }


_fork_without_caps_warned = False


def fork_enabled(caps: dict[str, Any] | None = None) -> bool:
    """The server shares a busy slot's cells with a new task (RTG-58 P1 live).

    ``ORCHESTRATOR_PREFIX_INDEX_FORK``:

    * ``auto`` (the documented value for the shadow and P1 windows) = on exactly
      when the server's ``/props.slot_fork`` (``caps``, KPF-27e) reports its fork
      enabled (``mode`` ``kv`` or ``checkpoint``). A v10 server emits no
      ``slot_fork``, so ``auto`` is off there: no trunk hold, no fork credit.
    * ``1`` = the same, but asserting the fork: with ``caps`` None (a v10 server,
      a P1 server run with ``--slot-fork-min-tokens 0``, or an unreadable
      ``/props``) it logs a warning and stays OFF. Fork credit and trunk holds
      without real sharing would over-admit the pool by the trunk size, so the
      gate refuses rather than trusting the env (Fable re-review, MEDIUM).
    * anything else = off."""
    global _fork_without_caps_warned
    if not enabled():
        return False
    raw = os.environ.get(FORK_ENV, "0").strip().lower()
    if raw != "auto" and raw not in _ON:
        return False
    if caps is None:
        if raw != "auto" and not _fork_without_caps_warned:
            _fork_without_caps_warned = True
            logger.warning(
                "prefix index: %s=%s but the server reports no /props.slot_fork "
                "(v10, fork configured off, or /props unreadable): fork features "
                "stay OFF (use %s=auto to follow the server)", FORK_ENV, raw, FORK_ENV)
        return False
    return caps.get("mode") in ("kv", "checkpoint")


def fork_requested() -> bool:
    """``ORCHESTRATOR_PREFIX_INDEX_FORK`` asks for the fork at all (``1`` or
    ``auto``): only then is the server's ``/props.slot_fork`` worth reading."""
    return enabled() and os.environ.get(FORK_ENV, "0").strip().lower() in (_ON | {"auto"})


def fnv1a64_tokens(token_ids: Iterable[int]) -> str:
    """The server's ``prefix_hash`` (KPF-27e): FNV-1a 64 over the token ids as
    int32 little-endian, lower-case hex, 16 chars. To VERIFY a text prefix,
    tokenize it through the same server (``/tokenize``) and compare."""
    h = 0xCBF29CE484222325
    for tid in token_ids:
        for b in int(tid).to_bytes(4, "little", signed=True):
            h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def lpm_enabled() -> bool:
    return enabled() and os.environ.get(LPM_ENV, "1").strip().lower() not in _OFF


def pin_policy() -> str:
    """``off`` (default) or ``idle``."""
    if not enabled():
        return "off"
    value = os.environ.get(PIN_ENV, "off").strip().lower()
    return value if value in ("off", "idle") else "off"


def block_chars() -> int:
    return max(64, _env_int(BLOCK_CHARS_ENV, DEFAULT_BLOCK_CHARS))


def host_wide() -> bool:
    return os.environ.get(HOST_WIDE_ENV, "1").strip().lower() not in _OFF


# -- keys -----------------------------------------------------------------------


def block_hashes(text: str | None, block: int | None = None) -> tuple[str, ...]:
    """Chained truncated sha256 of each FULL ``block``-char block of ``text``.

    ``h_i`` covers ``text[:(i + 1) * block]``, so equal ``h_i`` means equal
    prefixes of that length. A trailing partial block is not hashed. Pure."""
    if not text:
        return ()
    block = block or block_chars()
    digest = hashlib.sha256()
    out: list[str] = []
    for end in range(block, len(text) + 1, block):
        digest.update(text[end - block:end].encode("utf-8", "surrogatepass"))
        out.append(digest.copy().hexdigest()[:_HASH_HEX])
    return tuple(out)


def key_text_for_request(request: Any) -> str | None:
    """The index key for a backend request: the text the serving record
    fingerprints (UFH14-B4), so observation and lookup agree byte for byte."""
    from src.backends import serving_calls

    return serving_calls._prompt_text_for_fingerprint(request)


def key_kind_for_request(request: Any) -> str:
    return "approx" if isinstance(getattr(request, "chat_payload", None), dict) else "exact"


def common_prefix_len(a: str, b: str) -> int:
    """Exact common-prefix length (binary search over C-speed slice compares)."""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


# -- the radix tree ---------------------------------------------------------------


class _Node:
    __slots__ = ("edge", "children", "parent", "entries")

    def __init__(self, edge: tuple[str, ...] = (), parent: "_Node | None" = None) -> None:
        self.edge = edge
        self.children: dict[str, _Node] = {}
        self.parent = parent
        self.entries: set[str] = set()


class RadixTree:
    """Path-compressed trie over hash sequences; every entry is ONE path.

    Each node keeps the set of entries whose path passes through it, so removing
    an entry walks its own path and prunes what nobody else holds. This is the
    fix for the January 2026 ``radix_cache.py`` stale-slot bug, where ``insert``
    stamped one ``slot_id`` on every node of the path (last writer wins on
    shared nodes) and eviction cleared only the leaf, leaving interior nodes
    pointing at a slot that no longer held them.
    """

    def __init__(self) -> None:
        self.root = _Node()
        self._paths: dict[str, tuple[str, ...]] = {}

    def __contains__(self, entry_id: str) -> bool:
        return entry_id in self._paths

    def __len__(self) -> int:
        return len(self._paths)

    def path(self, entry_id: str) -> tuple[str, ...] | None:
        return self._paths.get(entry_id)

    def insert(self, entry_id: str, hashes: Iterable[str]) -> None:
        """Insert (or REPLACE) ``entry_id``'s path."""
        hashes = tuple(hashes)
        if entry_id in self._paths:
            self.remove(entry_id)
        self._paths[entry_id] = hashes
        node = self.root
        i = 0
        while i < len(hashes):
            child = node.children.get(hashes[i])
            if child is None:
                leaf = _Node(hashes[i:], node)
                node.children[hashes[i]] = leaf
                leaf.entries.add(entry_id)
                return
            edge = child.edge
            k = 0
            while k < len(edge) and i + k < len(hashes) and edge[k] == hashes[i + k]:
                k += 1
            if k < len(edge):  # split the edge at k
                mid = _Node(edge[:k], node)
                mid.entries = set(child.entries)
                node.children[hashes[i]] = mid
                child.edge = edge[k:]
                child.parent = mid
                mid.children[child.edge[0]] = child
                child = mid
            child.entries.add(entry_id)
            node = child
            i += k

    def remove(self, entry_id: str) -> bool:
        hashes = self._paths.pop(entry_id, None)
        if hashes is None:
            return False
        node = self.root
        i = 0
        visited: list[_Node] = []
        while i < len(hashes):
            child = node.children.get(hashes[i])
            if child is None:
                break
            child.entries.discard(entry_id)
            visited.append(child)
            i += len(child.edge)
            node = child
        for n in reversed(visited):  # prune nodes nobody holds any more
            if not n.entries and not n.children and n.parent is not None:
                n.parent.children.pop(n.edge[0], None)
        return True

    def match(self, hashes: Iterable[str],
              accept: Callable[[str], bool] | None = None) -> tuple[int, set[str]]:
        """``(blocks, entries)``: the deepest prefix of ``hashes`` held by at least
        one ACCEPTED entry, and every accepted entry holding it."""
        hashes = tuple(hashes)
        node = self.root
        i = 0
        best_depth: int = 0
        best: set[str] = set()
        while i < len(hashes):
            child = node.children.get(hashes[i])
            if child is None:
                break
            k = 0
            edge = child.edge
            while k < len(edge) and i + k < len(hashes) and edge[k] == hashes[i + k]:
                k += 1
            held = {e for e in child.entries if accept is None or accept(e)}
            if not held:
                break  # entries only shrink going down
            best_depth, best = i + k, held
            if k < len(edge):
                break
            node = child
            i += k
        return best_depth, best

    def union_blocks(self, entry_ids: set[str]) -> int:
        """Distinct blocks covered by ``entry_ids`` (shared prefixes counted once)."""
        total = 0
        stack = list(self.root.children.values())
        while stack:
            n = stack.pop()
            if n.entries & entry_ids:
                total += len(n.edge)
                stack.extend(n.children.values())
        return total

    def node_count(self) -> int:
        count, stack = 0, list(self.root.children.values())
        while stack:
            n = stack.pop()
            count += 1
            stack.extend(n.children.values())
        return count


# -- entries ----------------------------------------------------------------------


@dataclass
class Entry:
    id: str
    kind: str                      # slot | pending | served | inflight
    hashes: tuple[str, ...]
    chars: int
    key_kind: str = "exact"
    prompt_tokens: int | None = None    # server-measured (prompt_n + cache_n)
    expected_tokens: int | None = None  # what the slot should hold after the call
    slot_id: int | None = None
    id_task: int | None = None
    busy: bool = False
    origin: str | None = None       # exact (server id_slot) | inferred (/slots)
    ts: float = 0.0                 # monotonic: LRU / verify timeout
    wall_ts: float = 0.0            # wall clock: host-wide merge
    verified_at: float | None = None
    text: str | None = field(default=None, repr=False)  # in-flight only
    prefill_deadline: float | None = None
    prefilled: bool = False
    # In-flight only: ``/slots`` showed a prefilling slot in a read that
    # postdates this entry's admission (review D1). Sticky. Once the entry is
    # bound to its own slot (``slot_id``, see ``_bind_inflight``) only THAT slot
    # prefilling counts (Fable re-review, LOW).
    seen_prefilling: bool = False
    # In-flight only: ``slot -> id_task`` from the last ``/slots`` read before
    # admission; a slot whose task changed since is a candidate for this entry.
    pre_tasks: dict[int, int | None] | None = field(default=None, repr=False)
    # KPF-27e (server fork on; None/() otherwise): the slot's ``content_epoch``,
    # ``n_prompt_tokens`` and ``prefix_hash`` when bound, and its checkpoints
    # ``(n_tokens, prefix_hash)`` as last verified.
    content_epoch: int | None = None
    bound_tokens: int | None = None
    slot_hash: str | None = None
    checkpoints: tuple[tuple[int, str | None], ...] = ()

    def tokens_for_chars(self, chars: int) -> int:
        """Token estimate for a prefix of ``chars`` of this entry's key: the
        measured ratio, else ``PREFIX_CHARS_PER_TOKEN``."""
        if self.prompt_tokens and self.chars > 0:
            return int(chars * self.prompt_tokens / self.chars)
        return int(int(chars) / PREFIX_CHARS_PER_TOKEN)

    def tokens_per_char(self) -> float | None:
        if self.prompt_tokens and self.chars > 0:
            return self.prompt_tokens / self.chars
        return None


@dataclass(frozen=True)
class Match:
    """The best prefix the index knows for a key."""

    matched_chars: int = 0
    tokens_est: int = 0
    source: str | None = None        # slot_idle | slot_busy | inflight | served
    entry_id: str | None = None
    slot_id: int | None = None
    tokens_per_char: float | None = None
    # exact junction with an in-flight sibling (in-process text compare)
    junction_chars: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "matched_chars": self.matched_chars,
            "tokens_est": self.tokens_est,
            "source": self.source,
            "slot_id": self.slot_id,
            "junction_chars": self.junction_chars,
        }


def _new_stats() -> dict[str, Any]:
    return {
        "lookups": 0, "matches": {}, "observed": 0, "bound_exact": 0, "bound_inferred": 0,
        "verify_timeouts": 0, "stale_drops": {}, "reconciles": 0,
        "predictions": 0, "prediction_abs_err_tokens": 0, "prediction_over": 0,
        "prediction_under": 0, "slot_predictions": 0, "slot_prediction_hits": 0,
        "trunk_holds": 0, "trunk_hold_s_total": 0.0, "trunk_hold_timeouts": 0,
        "lpm_passes": 0, "pins": 0, "fork_credit_tokens": 0, "fork_credit_reattributed": 0,
        "bound_by_task": 0, "inflight_bound": 0, "fork_credit_reinflated": 0,
        "ledger_reads": 0, "ledger_writes": 0, "ledger_errors": 0,
    }


def _bump(d: dict[str, Any], key: str, n: int = 1) -> None:
    d[key] = d.get(key, 0) + n


# -- the per-server index -----------------------------------------------------------


def _default_launch_id(url: str) -> str | None:
    from src.backends import serving_calls

    return serving_calls.server_identity(serving_calls._port_of(url)).get("launch_id")


def _default_dir() -> Path:
    from src.runtime import long_prefill_lease

    return long_prefill_lease.lease_dir()


def server_key(url: str) -> str:
    from src.runtime.long_prefill_lease import server_key as _key

    return _key(url)


class PrefixIndex:
    """The prefix index of ONE physical server."""

    def __init__(
        self,
        url: str,
        *,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        launch_id: Callable[[str], str | None] | None = None,
        directory: Callable[[], Path] | None = None,
        host_wide_ledger: bool | None = None,
        block: int | None = None,
    ) -> None:
        self.url = url
        self.key = server_key(url)
        self.block = block or block_chars()
        self._clock = clock
        self._wall = wall
        self._launch_id_fn = launch_id or _default_launch_id
        self._dir_fn = directory or _default_dir
        self._host_wide = host_wide_ledger
        self._lock = threading.Lock()
        self._tree = RadixTree()
        self._entries: dict[str, Entry] = {}
        self._launch: str | None = None
        self._ledger_sig: tuple[int, int] | None = None
        self._last_reconcile: float | None = None
        # KPF-27e: the server's last ``kv_pool`` {size, used, shared} (None on v10).
        self._server_pool: dict[str, int] | None = None
        # slot -> id_task in the last ``/slots`` read (in-flight slot binding).
        self._slot_tasks: dict[int, int | None] | None = None
        self.stats = _new_stats()

    # -- helpers (caller holds the lock) -----------------------------------------
    def _put(self, entry: Entry) -> None:
        self._entries[entry.id] = entry
        self._tree.insert(entry.id, entry.hashes)

    def _drop(self, entry_id: str, reason: str | None = None) -> Entry | None:
        entry = self._entries.pop(entry_id, None)
        self._tree.remove(entry_id)
        if entry is not None and reason:
            _bump(self.stats["stale_drops"], reason)
        return entry

    def _of_kind(self, *kinds: str) -> list[Entry]:
        return [e for e in self._entries.values() if e.kind in kinds]

    def _trim_served(self) -> None:
        served = sorted(self._of_kind("served"), key=lambda e: e.ts)
        for e in served[:max(0, len(served) - MAX_SERVED_ENTRIES)]:
            self._drop(e.id)

    def _expire_pending(self, now: float) -> None:
        """Bound the unbound: a pending entry past ``VERIFY_TIMEOUT_S`` (or beyond
        the ``MAX_PENDING_ENTRIES`` LRU cap) leaves; a chat-lane one (no slot)
        becomes ``served`` (ordering only). Runs on every observation, not only
        at reconcile, so the index stays bounded when nothing polls ``/slots``."""
        pending = sorted(self._of_kind("pending"), key=lambda e: e.ts)
        excess = max(0, len(pending) - MAX_PENDING_ENTRIES)
        for i, e in enumerate(pending):
            timed_out = now - e.ts > VERIFY_TIMEOUT_S
            if not timed_out and i >= excess:
                break  # sorted oldest first: the rest are younger and within the cap
            self._drop(e.id)
            if timed_out:
                self.stats["verify_timeouts"] += 1
            if e.slot_id is None:
                self._put(Entry(id=f"served:{e.id.split(':', 1)[1]}", kind="served",
                                hashes=e.hashes, chars=e.chars, key_kind=e.key_kind,
                                prompt_tokens=e.prompt_tokens, ts=e.ts, wall_ts=e.wall_ts))
        self._trim_served()

    def _read_launch(self) -> str | None:
        """The server's launch id (a sidecar ``stat``: call it OUTSIDE the lock)."""
        try:
            launch = self._launch_id_fn(self.url)
        except Exception:
            launch = None
        return str(launch) if launch else None

    def _check_launch(self, launch: str | None) -> list[int]:
        """Drop everything after a server relaunch. Returns slot ids dropped."""
        dropped: list[int] = []
        if launch != self._launch:
            if self._launch is not None or launch is None:
                for e in list(self._entries.values()):
                    if e.kind != "inflight":
                        if e.slot_id is not None and e.kind == "slot":
                            dropped.append(e.slot_id)
                        self._drop(e.id, "launch_changed" if self._launch is not None else None)
            self._launch = launch
        return dropped

    # -- observation ---------------------------------------------------------------
    def observe_served(self, text: str | None, *, slot_id: int | None,
                       prompt_tokens: int | None, generated_tokens: int | None,
                       key_kind: str = "exact", id_task: int | None = None) -> None:
        """A call finished on this server: remember where its prompt went.

        ``id_task`` (KPF-27e ``timings.id_task``, server fork on) binds the entry
        to the exact task: ``/slots[slot].id_task`` equals it once the task is
        done, so binding needs no token-count inference."""
        hashes = block_hashes(text, self.block)
        if not hashes:
            return
        now, wall = self._clock(), self._wall()
        expected = (prompt_tokens + max(0, int(generated_tokens or 0))
                    if prompt_tokens else None)
        launch = self._read_launch()
        with self._lock:
            self._check_launch(launch)
            self.stats["observed"] += 1
            if slot_id is not None:
                # The server named the slot: its old content is gone.
                for e in self._of_kind("slot", "pending"):
                    if e.slot_id == slot_id:
                        self._drop(e.id)
                eid = f"pending:slot{slot_id}"
            else:
                eid = f"pending:{uuid.uuid4().hex[:12]}"
            self._put(Entry(
                id=eid, kind="pending", hashes=hashes, chars=len(text or ""),
                key_kind=key_kind, prompt_tokens=prompt_tokens, expected_tokens=expected,
                slot_id=slot_id, origin="exact" if slot_id is not None else None,
                id_task=id_task if slot_id is not None else None,
                ts=now, wall_ts=wall,
            ))
            self._expire_pending(now)  # after the put: the index is bounded on exit

    def record_prediction(self, predicted_cache_tokens: int | None, actual_cache_n: int | None,
                          predicted_slot: int | None, actual_slot: int | None) -> None:
        """Online prediction error: what admission predicted vs what the server did."""
        with self._lock:
            if predicted_cache_tokens is not None and actual_cache_n is not None:
                self.stats["predictions"] += 1
                err = int(predicted_cache_tokens) - int(actual_cache_n)
                self.stats["prediction_abs_err_tokens"] += abs(err)
                if err > 0:
                    self.stats["prediction_over"] += 1
                elif err < 0:
                    self.stats["prediction_under"] += 1
            if predicted_slot is not None and actual_slot is not None:
                self.stats["slot_predictions"] += 1
                if predicted_slot == actual_slot:
                    self.stats["slot_prediction_hits"] += 1

    # -- reconciliation ----------------------------------------------------------
    def reconcile(self, occupancy: Any) -> None:
        """Make the index agree with the server's ``/slots`` (a ``PoolOccupancy``).

        Never raises. A None occupancy (no /slots) changes nothing except the
        verify timeout of pending entries."""
        try:
            launch = self._read_launch()
            with self._lock:
                relaunched = self._check_launch(launch)
            self._merge_ledger()
            slots = list(getattr(occupancy, "slots", ()) or ()) if occupancy is not None else None
            now = self._clock()
            changed: dict[int, Entry | None] = {}
            with self._lock:
                for sid in relaunched:
                    changed[sid] = None
                self.stats["reconciles"] += 1
                if slots is not None:
                    self._last_reconcile = now
                    pool = getattr(occupancy, "kv_pool", None)
                    self._server_pool = dict(pool) if isinstance(pool, dict) else None
                    by_id = {getattr(s, "slot_id", None): s for s in slots}
                    for e in self._of_kind("slot"):
                        s = by_id.get(e.slot_id)
                        reason = self._stale_reason(e, s)
                        if reason:
                            self._drop(e.id, reason)
                            changed[e.slot_id] = None  # type: ignore[index]
                        else:
                            e.busy = bool(getattr(s, "is_processing", False))
                            e.verified_at = now
                            _refresh_server_facts(e, s)
                    claimed = {e.slot_id for e in self._of_kind("slot")}
                    for e in sorted(self._of_kind("pending"), key=lambda x: -x.ts):
                        s = self._bind_target(e, by_id, slots, claimed)
                        if s is None:
                            continue
                        self._drop(e.id)
                        sid = int(getattr(s, "slot_id"))
                        for old in self._of_kind("slot"):
                            if old.slot_id == sid:
                                self._drop(old.id)
                        bound = Entry(
                            id=f"slot:{sid}", kind="slot", hashes=e.hashes, chars=e.chars,
                            key_kind=e.key_kind, prompt_tokens=e.prompt_tokens,
                            expected_tokens=e.expected_tokens, slot_id=sid,
                            id_task=_int(getattr(s, "id_task", None)), busy=False,
                            origin=e.origin or "inferred", ts=e.ts, wall_ts=self._wall(),
                            verified_at=now,
                            content_epoch=_int(getattr(s, "content_epoch", None)),
                            bound_tokens=_int(getattr(s, "n_prompt_tokens", None)),
                            slot_hash=getattr(s, "prefix_hash", None) or None,
                        )
                        _refresh_server_facts(bound, s)
                        self._put(bound)
                        claimed.add(sid)
                        changed[sid] = bound
                        self.stats["bound_exact" if bound.origin == "exact"
                                   else "bound_inferred"] += 1
                        if e.id_task is not None:
                            self.stats["bound_by_task"] += 1
                    self._bind_inflight(now, slots)
                    self._slot_tasks = {
                        int(getattr(s, "slot_id")): _int(getattr(s, "id_task", None))
                        for s in slots if _int(getattr(s, "slot_id", None)) is not None}
                self._expire_pending(now)
            if changed:
                self._write_ledger(changed)
        except Exception:
            logger.debug("prefix index: reconcile failed for %s", self.url, exc_info=True)

    def _bind_inflight(self, now: float, slots: list) -> None:
        """Bind an in-flight entry to the slot serving it, so trunk-first reads
        the OWNER's slot instead of "some slot is prefilling" (Fable re-review,
        LOW). ``timings.id_slot`` arrives only with the final chunk, after the
        prefill, so the binding comes from ``/slots``: a processing slot whose
        ``id_task`` changed since the read before the entry's admission is a
        candidate. Bound only when that is unambiguous both ways (one candidate
        slot for the entry, one unbound entry for the slot) and the read
        postdates the dispatch (entry age >= one ``/slots`` TTL), so the owner
        is already among the candidates; otherwise the entry keeps the
        server-wide rule. Caller holds the lock."""
        unbound = [e for e in self._entries.values()
                   if e.kind == "inflight" and e.slot_id is None and e.pre_tasks is not None
                   and not e.prefilled and now - e.ts >= self.trunk_grace_s() / 2.0]
        if not unbound:
            return
        claimed = {e.slot_id for e in self._entries.values()
                   if e.kind == "inflight" and e.slot_id is not None}
        cands: dict[str, list[int]] = {}
        for e in unbound:
            pre = e.pre_tasks or {}
            cands[e.id] = [
                int(getattr(s, "slot_id")) for s in slots
                if getattr(s, "is_processing", False)
                and _int(getattr(s, "slot_id", None)) is not None
                and int(getattr(s, "slot_id")) not in claimed
                and _int(getattr(s, "id_task", None)) is not None
                and _int(getattr(s, "id_task", None)) != pre.get(int(getattr(s, "slot_id")))]
        for e in unbound:
            mine = cands[e.id]
            if len(mine) != 1:
                continue
            sid = mine[0]
            if sum(1 for c in cands.values() if sid in c) != 1:
                continue  # another unbound entry could be on that slot too
            e.slot_id = sid
            e.origin = "inferred"
            self.stats["inflight_bound"] += 1

    @staticmethod
    def _stale_reason(e: Entry, s: Any) -> str | None:
        if s is None:
            return "slot_missing"
        n = _int(getattr(s, "n_prompt_tokens", None)) or 0
        if n <= 0:
            return "cleared"
        epoch = _int(getattr(s, "content_epoch", None))
        if e.content_epoch is not None and epoch is not None:
            # KPF-27e binding rule (INTERFACE §6 q1): valid while content_epoch is
            # unchanged and n_prompt_tokens has not dropped below what we bound.
            # A different id_task alone is NOT staleness here: a task that only
            # extends our prefix leaves the epoch alone.
            if epoch != e.content_epoch:
                return "epoch_changed"
            if e.expected_tokens is not None and n < e.expected_tokens - SLOT_TOKEN_TOLERANCE:
                return "tokens_shrunk"
            if PrefixIndex._hash_mismatch(e, s, n):
                return "prefix_hash_changed"
            return None
        task = _int(getattr(s, "id_task", None))
        if e.id_task is not None and task is not None and task != e.id_task:
            return "id_task_changed"
        if e.expected_tokens is not None and n < e.expected_tokens - SLOT_TOKEN_TOLERANCE:
            return "tokens_shrunk"
        return None

    @staticmethod
    def _hash_mismatch(e: Entry, s: Any, n: int) -> bool:
        """KPF-27e verification without a tokenizer: the slot's content hash at
        the bound length, and every checkpoint hash we saw at a position, must
        not change while the epoch says the prefix is intact."""
        h = getattr(s, "prefix_hash", None)
        if (e.slot_hash and h and e.bound_tokens is not None and n == e.bound_tokens
                and not getattr(s, "is_processing", False) and h != e.slot_hash):
            return True
        known = dict(e.checkpoints)
        for cp in getattr(s, "checkpoints", ()) or ():
            old = known.get(getattr(cp, "n_tokens", None))
            new = getattr(cp, "prefix_hash", None)
            if old and new and old != new:
                return True
        return False

    @staticmethod
    def _holds(s: Any, e: Entry) -> bool:
        if getattr(s, "is_processing", False):
            return False
        n = _int(getattr(s, "n_prompt_tokens", None)) or 0
        task = _int(getattr(s, "id_task", None))
        if e.id_task is not None and task is not None:
            # KPF-27e: the server named the task (``timings.id_task``) — the slot
            # holds our call exactly when it still reports that task.
            return task == e.id_task and n > 0 and (
                e.expected_tokens is None or n >= e.expected_tokens - SLOT_TOKEN_TOLERANCE)
        if e.expected_tokens is None:
            return n > 0
        return abs(n - e.expected_tokens) <= SLOT_TOKEN_TOLERANCE

    def _bind_target(self, e: Entry, by_id: dict, slots: list, claimed: set) -> Any:
        if e.slot_id is not None:
            s = by_id.get(e.slot_id)
            return s if s is not None and self._holds(s, e) else None
        if e.expected_tokens is None:
            return None
        cands = [s for s in slots
                 if getattr(s, "slot_id", None) not in claimed and self._holds(s, e)]
        return cands[0] if len(cands) == 1 else None  # ambiguous -> never guess

    # -- in-flight locks -----------------------------------------------------------
    def begin(self, ticket: int, text: str | None, *, prompt_tokens: int | None = None,
              prefill_s: float | None = None, hashes: tuple[str, ...] | None = None) -> None:
        """Register an admitted request's trunk. Pass ``hashes`` (this index's
        ``block_hashes(text)``) to skip re-hashing: the gate hashes once per
        ticket, outside its lock (review D4)."""
        hashes = block_hashes(text, self.block) if hashes is None else hashes
        if not hashes:
            return
        now = self._clock()
        with self._lock:
            self._put(Entry(
                id=f"inflight:{ticket}", kind="inflight", hashes=hashes,
                chars=len(text or ""), expected_tokens=prompt_tokens, ts=now,
                wall_ts=self._wall(), text=text,
                prefill_deadline=None if prefill_s is None else now + max(0.0, prefill_s),
                pre_tasks=dict(self._slot_tasks) if self._slot_tasks is not None else None,
            ))

    def prefilled(self, ticket: int) -> None:
        with self._lock:
            e = self._entries.get(f"inflight:{ticket}")
            if e is not None:
                e.prefilled = True

    def end(self, ticket: int) -> None:
        with self._lock:
            self._drop(f"inflight:{ticket}")

    @staticmethod
    def trunk_grace_s() -> float:
        return max(0.0, _env_float(TRUNK_GRACE_S_ENV, DEFAULT_TRUNK_GRACE_S))

    def _unbound_prefilling(self, server_prefilling: int | None,
                            prefilling_slots: set[int] | None) -> int | None:
        """Prefilling slots an UNBOUND in-flight entry may be on: the server-wide
        count, less slots bound to other in-flight entries. Caller holds the lock."""
        if prefilling_slots is None:
            return server_prefilling
        bound = {e.slot_id for e in self._entries.values()
                 if e.kind == "inflight" and e.slot_id is not None}
        return len(prefilling_slots - bound)

    def _note_prefilling(self, now: float, server_prefilling: int | None,
                         prefilling_slots: set[int] | None = None) -> None:
        """Mark in-flight entries SEEN prefilling. Only a read that postdates the
        entry's admission counts: the ``/slots`` read is up to one TTL (half the
        grace) old, so the entry must be at least that old. An entry bound to
        its slot (``_bind_inflight``) is seen only when THAT slot is prefilling
        in ``prefilling_slots``. Caller holds the lock."""
        min_age = self.trunk_grace_s() / 2.0
        loose = self._unbound_prefilling(server_prefilling, prefilling_slots)
        for e in self._entries.values():
            if e.kind != "inflight" or e.seen_prefilling or now - e.ts < min_age:
                continue
            if e.slot_id is not None and prefilling_slots is not None:
                e.seen_prefilling = e.slot_id in prefilling_slots
            elif loose:
                e.seen_prefilling = True

    def _prefill_over(self, e: Entry, now: float, server_prefilling: int | None,
                      prefilling_slots: set[int] | None = None) -> bool:
        """Is ``e``'s trunk prefill over? ``prefill_done`` (first output), the
        floor-rate deadline, or ``/slots`` showing the owner no longer
        prefilling — the last only once the entry was seen prefilling or is
        past the dispatch grace, so a cached read from before the owner reached
        the server cannot end the hold (review D1). A bound entry reads its own
        slot; an unbound one needs no prefilling slot that is not bound to
        another entry."""
        if e.prefilled:
            return True
        if e.prefill_deadline is not None and now >= e.prefill_deadline:
            return True
        if e.slot_id is not None and prefilling_slots is not None:
            idle = e.slot_id not in prefilling_slots
        else:
            idle = self._unbound_prefilling(server_prefilling, prefilling_slots) == 0
        return idle and (e.seen_prefilling or now - e.ts >= self.trunk_grace_s())

    # -- lookups -------------------------------------------------------------------
    def lookup(self, text: str | None, *, exclude_ticket: int | None = None,
               fork: bool = False, count: bool = True,
               hashes: tuple[str, ...] | None = None, fork_mode: str | None = None,
               fresh_s: float | None = None) -> Match:
        """The best prefix of ``text`` the server can reuse.

        Without fork: an IDLE verified slot (the server's LCP pick, or the
        ``--cache-ram`` restore of it). With fork (RTG-58 P1): also a busy slot
        or an in-flight sibling whose prefill is over. A slot entry counts only
        while ``/slots`` verified it within ``fresh_s`` (default
        ``ORCHESTRATOR_PREFIX_INDEX_CREDIT_FRESH_S``; review D6). With
        ``fork_mode == "checkpoint"`` (``/props.slot_fork``, KPF-27e) a busy
        slot's match is cut to its largest reported checkpoint. ``hashes``
        skips re-hashing ``text`` (review D4)."""
        hashes = block_hashes(text, self.block) if hashes is None else hashes
        if not hashes:
            return Match()
        now = self._clock()
        if fresh_s is None:
            fresh_s = _env_float(CREDIT_FRESH_S_ENV, DEFAULT_CREDIT_FRESH_S)
        excl = None if exclude_ticket is None else f"inflight:{exclude_ticket}"
        with self._lock:
            if count:
                self.stats["lookups"] += 1
            entries = self._entries

            def usable(eid: str) -> bool:
                e = entries.get(eid)
                if e is None or eid == excl:
                    return False
                if e.kind == "slot":
                    if e.verified_at is None or now - e.verified_at > fresh_s:
                        return False  # unverified (ledger-adopted / stale): no credit
                    return fork or not e.busy
                if e.kind == "inflight":
                    return fork and self._prefill_over(e, now, None)
                return False

            depth, held = self._tree.match(hashes, usable)
            if depth <= 0:
                sdepth, sheld = self._tree.match(
                    hashes, lambda eid: eid in entries and entries[eid].kind in ("served", "pending"))
                if sdepth <= 0:
                    return Match()
                e = max((entries[i] for i in sheld), key=lambda x: x.ts)
                chars = sdepth * self.block
                m = Match(matched_chars=chars, tokens_est=e.tokens_for_chars(chars),
                          source="served", entry_id=e.id, tokens_per_char=e.tokens_per_char())
            else:
                ranked = sorted((entries[i] for i in held), key=lambda x: (
                    0 if x.kind == "slot" and not x.busy else 1, -x.ts))
                e = ranked[0]
                chars = depth * self.block
                source = ("slot_busy" if e.busy else "slot_idle") if e.kind == "slot" else "inflight"
                junction = (common_prefix_len(text, e.text)
                            if e.kind == "inflight" and e.text and text else None)
                tokens = e.tokens_for_chars(chars)
                if source == "slot_busy" and fork_mode == "checkpoint":
                    tokens = _fork_position(e, tokens)
                m = Match(matched_chars=chars, tokens_est=tokens,
                          source=source, entry_id=e.id, slot_id=e.slot_id,
                          tokens_per_char=e.tokens_per_char(), junction_chars=junction)
            if count:
                _bump(self.stats["matches"], m.source or "none")
            return m

    def trunk_owner(self, ticket: int, text: str | None, *, min_tokens: int,
                    server_prefilling: int | None = None,
                    hashes: tuple[str, ...] | None = None,
                    prefilling_slots: set[int] | None = None) -> Match | None:
        """An EARLIER in-flight request still prefilling a trunk of at least
        ``min_tokens`` that ``text`` shares (trunk-first, KPF-21).
        ``server_prefilling`` = prefilling slots in the gate's ``/slots`` read;
        ``prefilling_slots`` = their ids (per-owner reads for bound entries)."""
        hashes = block_hashes(text, self.block) if hashes is None else hashes
        if not hashes:
            return None
        now = self._clock()
        me = f"inflight:{ticket}"
        with self._lock:
            entries = self._entries
            self._note_prefilling(now, server_prefilling, prefilling_slots)

            def owner(eid: str) -> bool:
                e = entries.get(eid)
                if e is None or e.kind != "inflight" or eid == me:
                    return False
                try:
                    if int(eid.split(":", 1)[1]) > ticket:
                        return False  # only an earlier sibling can be the trunk
                except ValueError:
                    return False
                return not self._prefill_over(e, now, server_prefilling, prefilling_slots)

            depth, held = self._tree.match(hashes, owner)
            if depth <= 0:
                return None
            e = min((entries[i] for i in held), key=lambda x: x.ts)
            chars = depth * self.block
            tokens = e.tokens_for_chars(chars)
            if tokens < min_tokens:
                return None
            return Match(matched_chars=chars, tokens_est=tokens, source="inflight",
                         entry_id=e.id, junction_chars=(common_prefix_len(text, e.text)
                                                        if e.text and text else None))

    def pin_candidate(self, text: str | None, *, min_tokens: int, fresh_s: float) -> int | None:
        """The slot to pin (policy ``idle``): a verified IDLE slot holding the
        longest prefix, verified within ``fresh_s``. Never a busy slot."""
        m = self.lookup(text, fork=False, count=False)
        if m.source != "slot_idle" or m.slot_id is None or m.tokens_est < min_tokens:
            return None
        with self._lock:
            e = self._entries.get(m.entry_id or "")
            if (e is None or e.busy or e.verified_at is None
                    or self._clock() - e.verified_at > fresh_s):
                return None
            self.stats["pins"] += 1
        return m.slot_id

    def unique_cells(self) -> dict[str, Any]:
        """Union vs sum of the prefixes resident in slots and in flight (tokens).
        ``sum - union`` is what cross-slot sharing (P1) would save."""
        with self._lock:
            live = {e.id for e in self._entries.values() if e.kind in ("slot", "inflight")}
            union_blocks = self._tree.union_blocks(live)
            sum_blocks = sum(len(self._entries[i].hashes) for i in live)
            ratios = [self._entries[i].tokens_per_char() for i in live]
            ratios = [r for r in ratios if r]
            tpc = (sum(ratios) / len(ratios)) if ratios else 1.0 / PREFIX_CHARS_PER_TOKEN
            server_pool = dict(self._server_pool) if self._server_pool else None
        out: dict[str, Any] = {
            "entries": len(live),
            "union_tokens_est": int(union_blocks * self.block * tpc),
            "sum_tokens_est": int(sum_blocks * self.block * tpc),
            "shareable_tokens_est": int((sum_blocks - union_blocks) * self.block * tpc),
        }
        if server_pool:
            # KPF-27e: the server's own unique-cell count (``/slots`` kv_pool).
            out["server_kv_pool"] = server_pool
        return out

    def status(self) -> dict[str, Any]:
        with self._lock:
            kinds: dict[str, int] = {}
            for e in self._entries.values():
                kinds[e.kind] = kinds.get(e.kind, 0) + 1
            slots = {
                str(e.slot_id): {"chars": e.chars, "prompt_tokens": e.prompt_tokens,
                                 "busy": e.busy, "origin": e.origin, "id_task": e.id_task}
                for e in self._entries.values() if e.kind == "slot"
            }
            out = {
                "server": self.key, "block_chars": self.block, "entries": kinds,
                "nodes": self._tree.node_count(), "slots": slots,
                "stats": json.loads(json.dumps(self.stats)),
            }
        out["unique_cells"] = self.unique_cells()
        return out

    # -- host-wide ledger ------------------------------------------------------------
    def _ledger_on(self) -> bool:
        return host_wide() if self._host_wide is None else bool(self._host_wide)

    def ledger_path(self) -> Path:
        return self._dir_fn() / f"{FILE_PREFIX}{self.key}.json"

    def _write_ledger(self, changed: dict[int, Entry | None]) -> None:
        if not self._ledger_on() or not changed:
            return
        path = self.ledger_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path.with_name(path.name + ".lock"), "a") as lock_fh:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
                try:
                    try:
                        state = json.loads(path.read_text())
                        if not isinstance(state, dict) or state.get("schema") != SCHEMA:
                            state = {}
                    except (OSError, ValueError):
                        state = {}
                    if state.get("launch_id") != self._launch:
                        state = {}
                    slots = state.setdefault("slots", {})
                    wall = self._wall()
                    for sid, e in changed.items():
                        if e is None:
                            slots[str(sid)] = {"dropped": True, "wall_ts": wall}
                        else:
                            slots[str(sid)] = {
                                "hashes": list(e.hashes), "chars": e.chars,
                                "key_kind": e.key_kind, "prompt_tokens": e.prompt_tokens,
                                "expected_tokens": e.expected_tokens, "id_task": e.id_task,
                                "origin": e.origin, "wall_ts": e.wall_ts or wall,
                                "content_epoch": e.content_epoch,
                            }
                    state.update(schema=SCHEMA, launch_id=self._launch, block_chars=self.block)
                    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
                    tmp.write_text(json.dumps(state, separators=(",", ":")))
                    os.replace(tmp, path)
                finally:
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            with self._lock:
                self.stats["ledger_writes"] += 1
        except Exception:
            with self._lock:
                self.stats["ledger_errors"] += 1
            logger.debug("prefix index: cannot write %s", path, exc_info=True)

    def _merge_ledger(self) -> None:
        """Adopt slot entries other workers verified (newest wins per slot)."""
        if not self._ledger_on():
            return
        path = self.ledger_path()
        try:
            st = path.stat()
        except OSError:
            return
        sig = (st.st_mtime_ns, st.st_size)
        if sig == self._ledger_sig:
            return
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError):
            with self._lock:
                self.stats["ledger_errors"] += 1
            return
        with self._lock:
            self._ledger_sig = sig
            self.stats["ledger_reads"] += 1
            if (not isinstance(state, dict) or state.get("schema") != SCHEMA
                    or state.get("block_chars") != self.block
                    or state.get("launch_id") != self._launch):
                return
            for sid_s, rec in (state.get("slots") or {}).items():
                try:
                    sid = int(sid_s)
                except ValueError:
                    continue
                if not isinstance(rec, dict):
                    continue
                local = self._entries.get(f"slot:{sid}")
                wall = float(rec.get("wall_ts") or 0.0)
                if local is not None and local.wall_ts >= wall:
                    continue
                if rec.get("dropped"):
                    if local is not None:
                        self._drop(local.id, "peer_dropped")
                    continue
                hashes = tuple(str(h) for h in rec.get("hashes") or ())
                if not hashes:
                    continue
                self._put(Entry(
                    id=f"slot:{sid}", kind="slot", hashes=hashes,
                    chars=int(rec.get("chars") or len(hashes) * self.block),
                    key_kind=str(rec.get("key_kind") or "exact"),
                    prompt_tokens=_int(rec.get("prompt_tokens")),
                    expected_tokens=_int(rec.get("expected_tokens")), slot_id=sid,
                    id_task=_int(rec.get("id_task")), origin=rec.get("origin"),
                    ts=self._clock(), wall_ts=wall,
                    content_epoch=_int(rec.get("content_epoch")),
                    # verified_at stays None: adopted, not verified HERE, so it is
                    # no credit source until this worker's /slots read confirms it.
                ))


def _refresh_server_facts(e: Entry, s: Any) -> None:
    """Copy a verified slot's KPF-27e facts onto its entry (no-op on v10)."""
    cps = getattr(s, "checkpoints", ()) or ()
    e.checkpoints = tuple((int(cp.n_tokens), getattr(cp, "prefix_hash", None)) for cp in cps)
    if e.content_epoch is None:
        # Verified by the legacy rule just now, so the current epoch is ours.
        e.content_epoch = _int(getattr(s, "content_epoch", None))


def _fork_position(e: Entry, tokens: int) -> int:
    """KPF-27e: where a ``checkpoint``-mode fork from a BUSY slot can land — the
    largest checkpoint at or below ``tokens`` (0 = none, so no fork). Without
    reported checkpoints, ``tokens`` (the v10-era assumption)."""
    if not e.checkpoints:
        return tokens
    return max((n for n, _ in e.checkpoints if n <= tokens), default=0)


# -- registry: one index per physical server ------------------------------------------

_indexes: dict[str, PrefixIndex] = {}
_registry_lock = threading.Lock()


def get_index(url: str) -> PrefixIndex:
    key = server_key(url)
    with _registry_lock:
        idx = _indexes.get(key)
        if idx is None:
            idx = PrefixIndex(url)
            _indexes[key] = idx
        return idx


def peek_index(url: str) -> PrefixIndex | None:
    """The index for ``url`` if one exists (never creates)."""
    with _registry_lock:
        return _indexes.get(server_key(url))


def reset_indexes() -> None:
    """Drop every index (tests)."""
    global _fork_without_caps_warned
    with _registry_lock:
        _indexes.clear()
    _fork_without_caps_warned = False


def all_status() -> dict[str, Any]:
    with _registry_lock:
        items = list(_indexes.items())
    return {k: v.status() for k, v in items}


# -- the serving-record hook ------------------------------------------------------------


def observe_record(record: dict[str, Any], key_text: str | None, *,
                   key_kind: str = "exact") -> None:
    """Feed one ``serving_call.v1`` record into its server's index. Never raises;
    a no-op with the flag off."""
    try:
        if not enabled() or not key_text:
            return
        if not record.get("dispatched") or record.get("outcome") not in SERVED_OUTCOMES:
            return
        url = (record.get("server") or {}).get("base_url")
        if not url:
            return
        notes = record.get("notes") or {}
        timings = notes.get("timings") or record.get("timings") or {}
        prompt_n, cache_n = _int(timings.get("prompt_n")), _int(timings.get("cache_n"))
        prompt_tokens = prompt_n + (cache_n or 0) if prompt_n is not None else None
        generated = _int(timings.get("predicted_n")) or 0
        slot = _int(notes.get("server_slot"))
        if slot is None:
            # KPF-27e: ``timings.id_slot`` on every endpoint (server fork on), so
            # the chat lane binds exactly too.
            slot = _int(timings.get("id_slot"))
            slot = slot if slot is not None and slot >= 0 else None
        task = _int(timings.get("id_task"))
        idx = get_index(url)
        pred = ((record.get("kv_admission") or {}).get("prefix_index") or {})
        idx.record_prediction(_int(pred.get("predicted_cache_tokens")), cache_n,
                              _int(pred.get("predicted_slot")), slot)
        idx.observe_served(key_text, slot_id=slot, prompt_tokens=prompt_tokens,
                           generated_tokens=generated, key_kind=key_kind, id_task=task)
    except Exception:
        logger.debug("prefix index: observe_record failed", exc_info=True)


__all__ = [
    "Entry",
    "Match",
    "PrefixIndex",
    "RadixTree",
    "all_status",
    "block_hashes",
    "enabled",
    "fnv1a64_tokens",
    "fork_caps_from_props",
    "fork_enabled",
    "get_index",
    "key_text_for_request",
    "lpm_enabled",
    "observe_record",
    "peek_index",
    "pin_policy",
    "reset_indexes",
]
