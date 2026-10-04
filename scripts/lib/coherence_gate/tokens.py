"""Token-id PROVENANCE for the degeneracy detector.

The INF-70 v1 clients called `classify(dict(tokens=list(range(npred)), ...))`. Those synthetic,
all-distinct ids make uniq == 1, top == 1/n and run == 1, so three of the five triggers could never
fire and the classifier was a near no-op on every chat run (q38t7-rescore/AUDIT.md section 0). This
module makes that structurally impossible: a verdict is computed from exactly one of

  token_ids   real ids the caller captured (e.g. llama-server /completion `tokens`, /tokenize);
              VALIDATED here and refused when they are synthetic or cannot belong to the text;
  tokenizer   ids produced by a tokenizer the caller injected (`HFTokenizer`, `ServerTokenizer`, or
              any `Tokenizer`), whose id is recorded on the verdict;
  text_only   a DECLARED surrogate (`surrogate:regex-word.v1`): letter runs, digit groups of 1-3,
              punctuation CLUSTERS (BPE merges `****`, `):`, fences), newlines. The thresholds were
              calibrated on BPE ids, so the verdict says `calibrated: false`. Never chosen silently.
              Measured 2026-10-04 against degeneracy.v2 on real Qwen3.8 ids: 3/153 false DEGENERATE on
              the unique INF-70 chat outputs (code/LaTeX-heavy), 14/15 genuine gdn-rowexact loops caught.
              A per-character punctuation split was 13/153 false; splitting every digit was 15/153 false;
              unsplit digit runs missed 4/15 `2222...` loops.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

MODE_IDS = "token_ids"
MODE_TOKENIZER = "tokenizer"
MODE_TEXT_ONLY = "text_only"
SURROGATE_ID = "surrogate:regex-word.v1"
FAKE_MIN_N = 4          # list(range(n)) itself (starts at 0) is refused from this length
FAKE_RUN_N = 16         # any step-1 run this long, at any offset, is refused. Shorter offset runs are
                        # legitimate: byte-level BPE digits/ASCII are consecutive ids ("1234").
SPECIAL_SLACK = 8       # ids allowed beyond the UTF-8 byte count (specials that decode to "")
_SURROGATE_RE = re.compile(r"[^\W\d]+|\d{1,3}|[^\w\s]+|\n", re.UNICODE)


class TokenProvenanceError(ValueError):
    """Token ids are synthetic, malformed, inconsistent with the text, or absent without a
    declared fallback. The gate refuses to compute a verdict from them."""


class Tokenizer(Protocol):
    tokenizer_id: str

    def __call__(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class FnTokenizer:
    """Wrap any `text -> ids` callable with the id that will be recorded on every verdict."""
    fn: Callable[[str], list[int]]
    tokenizer_id: str

    def __call__(self, text: str) -> list[int]:
        return list(self.fn(text))


class HFTokenizer:
    """A HuggingFace `tokenizers` tokenizer.json (no special tokens added). The recorded id pins the
    file by content hash, so two verdicts name the same vocabulary only if they used the same file."""

    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer as _T  # lazy: optional dependency
        p = Path(path)
        self._tok = _T.from_file(str(p))
        digest = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        self.tokenizer_id = f"hf:{p.parent.name}/{p.name}@sha256:{digest}"

    def __call__(self, text: str) -> list[int]:
        return self._tok.encode(text, add_special_tokens=False).ids


class ServerTokenizer:
    """llama-server `POST /tokenize` (a lookup, not inference). Use the server that produced the
    outputs, or one serving the same GGUF, so the ids are the vocabulary the model decoded in."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.tokenizer_id = f"llama-server:{self.base}/tokenize"

    def __call__(self, text: str) -> list[int]:
        req = urllib.request.Request(f"{self.base}/tokenize", data=json.dumps({"content": text}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 - caller-chosen local URL
            return list(json.loads(r.read())["tokens"])


def surrogate_ids(text: str) -> list[int]:
    """Deterministic text-only surrogate: regex words/punctuation/newlines, interned in first-seen
    order. Repetition structure (top/run/loop) survives; absolute ids mean nothing."""
    vocab: dict[str, int] = {}
    return [vocab.setdefault(t, len(vocab)) for t in _SURROGATE_RE.findall(text)]


def fake_reason(ids: list, text: str | None = None) -> str | None:
    """Why `ids` cannot be a real tokenization (None when they can).

    * not a list of non-negative ints;
    * the INF-70 v1 placeholder: a step-1 run starting at 0 of >= FAKE_MIN_N ids (`list(range(n))`),
      or a step-1 run of >= FAKE_RUN_N ids at any offset;
    * more ids than the text's UTF-8 bytes + SPECIAL_SLACK (byte-level BPE emits >= 1 byte per id).
    """
    if not isinstance(ids, (list, tuple)) or not all(isinstance(t, int) and not isinstance(t, bool) and t >= 0
                                                      for t in ids):
        return "token ids must be a list of non-negative ints"
    n = len(ids)
    step1 = n >= FAKE_MIN_N and all(b - a == 1 for a, b in zip(ids, ids[1:]))
    if step1 and (ids[0] == 0 or n >= FAKE_RUN_N):
        return (f"token ids are a step-1 range ({ids[0]}..{ids[-1]}, n={n}): the INF-70 v1 "
                f"`list(range(n))` placeholder, not a tokenization")
    if text is not None and n > len(text.encode("utf-8")) + SPECIAL_SLACK:
        return f"{n} token ids for a {len(text.encode('utf-8'))}-byte text: ids do not belong to this text"
    return None


def validate_ids(ids: list, text: str) -> list[int]:
    why = fake_reason(ids, text)
    if why:
        raise TokenProvenanceError(why)
    return list(ids)


def resolve(text: str, ids: list | None, tokenizer: Tokenizer | None, allow_text_only: bool,
            force_mode: str | None = None) -> tuple[list[int], dict]:
    """Return (ids, provenance). Real ids win; then the tokenizer; then the declared surrogate.
    Supplied ids are validated even when a fallback exists: a fake id list is a harness bug, and
    silently replacing it would hide the bug that made INF-70's checks vacuous."""
    if ids is not None:
        validate_ids(ids, text)
    mode = force_mode or (MODE_IDS if ids is not None else MODE_TOKENIZER if tokenizer is not None
                          else MODE_TEXT_ONLY if allow_text_only else None)
    if mode == MODE_IDS:
        if ids is None:
            raise TokenProvenanceError("mode token_ids requested but the row carries no token ids")
        return list(ids), {"mode": MODE_IDS, "source": "row.token_ids", "calibrated": True}
    if mode == MODE_TOKENIZER:
        if tokenizer is None:
            raise TokenProvenanceError("mode tokenizer requested but no tokenizer was injected")
        got = validate_ids(tokenizer(text), text)
        return got, {"mode": MODE_TOKENIZER, "source": tokenizer.tokenizer_id, "calibrated": True}
    if mode == MODE_TEXT_ONLY:
        if not allow_text_only:
            raise TokenProvenanceError("text-only mode must be declared (allow_text_only=True / --text-only)")
        return surrogate_ids(text), {"mode": MODE_TEXT_ONLY, "source": SURROGATE_ID, "calibrated": False}
    raise TokenProvenanceError(
        "no token ids, no tokenizer, and text-only mode not declared: pass real ids (row.token_ids), "
        "inject a tokenizer (--tokenizer tokenizer.json | --tokenize-url), or declare --text-only")
