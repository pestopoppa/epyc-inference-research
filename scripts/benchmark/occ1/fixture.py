"""History fixture for OCC-1: SQuAD v1.1 dev passages as one continuous "history" stream.

Design (paired): the history is cut into fixed CHUNK_CHARS chunks that are IDENTICAL across
arms; every arm answers the SAME questions about the SAME chunk and only the carrier differs
(raw text vs N rendered frames). That makes every per-question outcome a matched pair.

Scoring is official SQuAD v1.1 EM/F1 (normalisation ported from snapcompact research/squad.py,
itself the official evaluate-v1.1 logic).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
SQUAD_SHA256 = "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9"
DEFAULT_CACHE = Path("/mnt/raid0/llm/cache/occ1")

# = 6x10 font capacity of one 1568x1568 frame (261 cols x 156 rows); same as upstream TEXT_CHUNK.
CHUNK_CHARS = 40716


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ensure_squad(cache: Path = DEFAULT_CACHE, download: bool = False) -> Path:
    """Return the hash-verified SQuAD dev path. Never silently accepts a drifted file."""
    path = cache / "squad-dev-v1.1.json"
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} missing; re-run with --download (fetches {SQUAD_URL})")
        import urllib.request

        cache.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SQUAD_URL, path)
    got = sha256_file(path)
    if got != SQUAD_SHA256:
        raise ValueError(f"SQuAD dev hash drift: {got} != pinned {SQUAD_SHA256}")
    return path


def load_paragraphs(path: Path) -> list[dict]:
    """Flattened [{ctx, qas, title}] in deterministic dataset order, whitespace collapsed."""
    data = json.loads(path.read_text())["data"]
    out = []
    for art in data:
        for p in art["paragraphs"]:
            out.append({"ctx": " ".join(p["context"].split()), "qas": p["qas"], "title": art["title"]})
    return out


def build_flow(paras: list[dict], max_chars: int | None = None) -> tuple[str, list[int]]:
    """Space-joined passage stream + start offset of each passage."""
    parts: list[str] = []
    offsets: list[int] = []
    n = 0
    for p in paras:
        offsets.append(n)
        parts.append(p["ctx"] + " ")
        n += len(p["ctx"]) + 1
        if max_chars is not None and n >= max_chars:
            break
    return "".join(parts), offsets


@dataclass(frozen=True)
class Chunk:
    index: int
    start: int
    end: int
    text: str


def chunk_flow(flow: str, chunk_chars: int = CHUNK_CHARS, full_only: bool = True) -> list[Chunk]:
    """Fixed-size chunks. full_only drops a short tail so every chunk has the same size."""
    chunks = []
    for i, start in enumerate(range(0, len(flow), chunk_chars)):
        end = min(start + chunk_chars, len(flow))
        if full_only and end - start < chunk_chars:
            break
        chunks.append(Chunk(i, start, end, flow[start:end]))
    return chunks


def sample_chunk_questions(
    paras: list[dict], offsets: list[int], chunk: Chunk, n: int, seed: int
) -> list[dict]:
    """Up to n questions from passages fully inside the chunk, evenly spread across it.

    Deterministic in (seed, chunk.start) and independent of the arm, so all arms get the same
    list. Each record carries a stable qid and pos_rel (passage start within the chunk, 0..1).
    """
    rng = random.Random(seed * 1_000_003 + chunk.start)
    eligible = [
        i
        for i in range(len(offsets))
        if offsets[i] >= chunk.start and offsets[i] + len(paras[i]["ctx"]) <= chunk.end
    ]
    if not eligible:
        return []
    n = min(n, len(eligible))
    step = len(eligible) / n
    picked = []
    for k in range(n):
        pi = eligible[int(k * step)]
        qa = rng.choice(paras[pi]["qas"])
        picked.append(
            {
                "qid": qa["id"],
                "q": " ".join(qa["question"].split()),
                "golds": sorted({a["text"] for a in qa["answers"]}),
                "pos_rel": (offsets[pi] - chunk.start) / (chunk.end - chunk.start),
            }
        )
    return picked


# --- official SQuAD v1.1 normalisation / metrics ---


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, golds: list[str]) -> float:
    return float(any(normalize_answer(pred) == normalize_answer(g) for g in golds))


def f1(pred: str, golds: list[str]) -> float:
    best = 0.0
    p_tok = normalize_answer(pred).split()
    for g in golds:
        g_tok = normalize_answer(g).split()
        common = Counter(p_tok) & Counter(g_tok)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        prec, rec = overlap / len(p_tok), overlap / len(g_tok)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def parse_numbered(text: str, n: int) -> list[str]:
    """Answers from a numbered list; missing entries become '' (scored 0, counted as parse miss)."""
    answers = [""] * n
    for line in text.splitlines():
        m = re.match(r"\s*(\d+)[.):]\s*(.*\S)?\s*$", line)
        if m and m.group(2):
            idx = int(m.group(1)) - 1
            if 0 <= idx < n and not answers[idx]:
                answers[idx] = m.group(2).strip()
    return answers
