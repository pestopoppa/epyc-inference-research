"""RULER NIAH cells for the YaRN gate (``niah_multikey_2``, ``niah_multikey_3``, ``niah_multivalue``).

Port of NVIDIA RULER ``scripts/data/synthetic/niah.py`` (checkout ``/mnt/raid0/llm/data/eval/ruler/repo``
@ ab17b78) with the task parameters of ``scripts/synthetic.yaml``, the ``niah`` template + answer_prefix of
``constants.py`` and RULER's ``generate_input_output`` logic (needle insertion at one of 40 depths for the
essay haystack; needle-sentence haystack with random insertion indexes; the single-answer template rewrite
Some->A / are all->is / are->is / numbers->number). Stdlib + ``tokenizers`` only (numpy is not needed).

DEVIATIONS (each is also listed in every item's ``pins["deviations"]``):
 1. word_keys: wonderwords is absent and RULER's ``json/english_words.json`` in the checkout is an unfetched
    git-LFS pointer (132 bytes), so no word list exists on disk. Word needles are ``w1-w2`` pairs of distinct
    words drawn from the vocabulary of the local Paul Graham essays (lowercase alphabetic tokens of >= 4
    letters, sorted unique), mimicking RULER's ``adjective-noun`` shape. Recorded as ``words_source``.
 2. essay_source: RULER downloads PaulGrahamEssays.json (absent). We use the needle-haystack repo's
    ``PaulGrahamEssays/*.txt`` sorted by filename, joined with a newline, whitespace-collapsed and split
    on spaces exactly as RULER does; ``essay_sha256`` is the sha256 of the newline-joined concatenation.
 3. sentence_split: nltk is absent; ``_sent_tokenize`` is a regex splitter (split after . ! ? followed by
    whitespace and an upper-case letter / quote / bracket).
 4. sizing: RULER binary-searches the haystack size with its own tokenizer plus tokens_to_generate. Here the
    search uses the SERVED model's tokenizer.json (``add_special_tokens=False``) over the FULL user message
    (template + context + query + answer prefix), target = ``target_real`` with no generation reserve, so
    ``real_n_offline`` lands at or just under ``target_real``. ``n_source = "model_tokenizer_json"``.
 5. chat: the chat template is applied server-side. The whole RULER input is one user message, context
    first, question last; the RULER ``answer_prefix`` is appended at the END of that user message (it is
    NOT an assistant prefill), and the 'base' model template is used (no [INST] wrappers).
 6. rng: a private ``random.Random`` per item (seed derived from ``seed`` and the item id) instead of
    RULER's global seed; same seed -> byte-identical messages, but not RULER's sample stream.
"""
from __future__ import annotations

import hashlib
import random
import re
import subprocess
import uuid
from pathlib import Path
from typing import Callable

from scripts.benchmark.yarn_gate import cells

RULER_REPO = "/mnt/raid0/llm/data/eval/ruler/repo"
RULER_SHA_SHORT = "ab17b78"
ESSAY_DIR = "/mnt/raid0/llm/data/eval/needle/repo/needlehaystack/PaulGrahamEssays"

TASKS = ("niah_multikey_2", "niah_multikey_3", "niah_multivalue")
_SHORT = {"niah_multikey_2": "mk_2", "niah_multikey_3": "mk_3", "niah_multivalue": "mv"}
# synthetic.yaml parameters
_CFG = {
    "niah_multikey_2": dict(type_haystack="needle", type_needle_k="words", type_needle_v="numbers", num_needle_v=1),
    "niah_multikey_3": dict(type_haystack="needle", type_needle_k="uuids", type_needle_v="uuids", num_needle_v=1),
    "niah_multivalue": dict(type_haystack="essay", type_needle_k="words", type_needle_v="numbers", num_needle_v=4),
}
# constants.py TASKS['niah']
TEMPLATE = ("Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. "
            "I will quiz you about the {type_needle_v} afterwards.\n{context}\nWhat are all the special magic "
            "{type_needle_v} for {query} mentioned in the provided text?")
ANSWER_PREFIX = " The special magic {type_needle_v} for {query} mentioned in the provided text are"
NEEDLE = "One of the special magic {type_needle_v} for {key} is: {value}."
DEPTHS = [int(round(i * 100 / 39)) for i in range(40)]  # np.round(np.linspace(0, 100, 40))

DEVIATIONS = [
    "word_keys: wonderwords absent and english_words.json is an LFS pointer; keys are w1-w2 pairs from the "
    "Paul Graham essay vocabulary (>=4 lowercase letters)",
    "essay_source: needle-haystack repo PaulGrahamEssays/*.txt sorted, newline-joined (RULER's downloaded "
    "PaulGrahamEssays.json absent)",
    "sentence_split: regex splitter instead of nltk sent_tokenize",
    "sizing: binary search with served model tokenizer.json over the full user message, no generation reserve",
    "chat: one user message, answer_prefix appended at the end of the user message, base template",
    "rng: per-item private random.Random, not RULER's global seed stream",
]


def available() -> tuple[bool, str]:
    for label, p in (("RULER checkout", Path(RULER_REPO) / "scripts/data/synthetic/niah.py"),
                     ("essay dir", Path(ESSAY_DIR)),
                     ("tokenizer.json", Path(cells.DEFAULT_TOKENIZER_JSON))):
        if not p.exists():
            return False, f"{label} missing: {p}"
    if not sorted(Path(ESSAY_DIR).glob("*.txt")):
        return False, f"essay dir has no *.txt: {ESSAY_DIR}"
    return True, "ok"


# ── resources ────────────────────────────────────────────────────────────────────────────────────────
def _load_essays() -> tuple[list[str], str, list[str]]:
    files = sorted(Path(ESSAY_DIR).glob("*.txt"), key=lambda p: p.name)
    text = "\n".join(f.read_text(encoding="utf-8", errors="replace") for f in files)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    words = re.sub(r"\s+", " ", text).split(" ")
    vocab = sorted({w for w in re.findall(r"[a-z]{4,}", text.lower())})
    return words, sha, vocab


def _ruler_sha() -> str:
    try:
        out = subprocess.run(["git", "-C", RULER_REPO, "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return RULER_SHA_SHORT


_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


def _sent_tokenize(text: str) -> list[str]:
    return [s for s in _SENT_RE.split(text) if s]


def _make_count(tokenizer_json: str | None) -> tuple[Callable[[str], int], str]:
    from tokenizers import Tokenizer
    path = tokenizer_json or cells.DEFAULT_TOKENIZER_JSON
    tok = Tokenizer.from_file(path)
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return (lambda t: len(tok.encode(t, add_special_tokens=False).ids)), sha


# ── generator (niah.generate_input_output) ───────────────────────────────────────────────────────────
class _Gen:
    def __init__(self, task: str, haystack_words: list[str], vocab: list[str]):
        self.cfg = _CFG[task]
        self.hay = haystack_words
        self.vocab = vocab

    def rand(self, rng: random.Random, kind: str) -> str:
        if kind == "numbers":
            return str(rng.randint(10 ** 6, 10 ** 7 - 1))
        if kind == "words":
            a, b = rng.sample(self.vocab, 2)
            return f"{a}-{b}"
        if kind == "uuids":
            return str(uuid.UUID(int=rng.getrandbits(128), version=4))
        raise NotImplementedError(kind)

    def build(self, num_haystack: int, seed: int) -> tuple[str, list[str], str]:
        """-> (user_text, answers, query_key). Deterministic in (num_haystack, seed)."""
        c = self.cfg
        rng = random.Random(seed)
        tv = c["type_needle_v"]
        key = self.rand(rng, c["type_needle_k"])
        values = [self.rand(rng, tv) for _ in range(c["num_needle_v"])]
        needles = [NEEDLE.format(type_needle_v=tv, key=key, value=v) for v in values]
        random.Random(seed).shuffle(needles)

        if c["type_haystack"] == "essay":
            hay = self.hay
            if num_haystack <= len(hay):
                text = " ".join(hay[:num_haystack])
            else:
                reps = (num_haystack + len(hay) - 1) // len(hay)
                text = " ".join((hay * reps)[:num_haystack])
            sents = _sent_tokenize(text.strip())
            pos = [0] + sorted(int(len(sents) * (d / 100)) for d in rng.sample(DEPTHS, len(needles))) + [len(sents)]
            parts = []
            for i in range(1, len(pos)):
                parts.append(" ".join(sents[pos[i - 1]:pos[i]]))
                if i - 1 < len(needles):
                    parts.append(needles[i - 1])
            context = " ".join(parts)
        else:
            sentences = [NEEDLE.format(type_needle_v=tv, key=self.rand(rng, c["type_needle_k"]),
                                       value=self.rand(rng, tv)) for _ in range(num_haystack)]
            idx = sorted(rng.sample(range(num_haystack), len(needles)), reverse=True)
            for i, n in zip(idx, needles):
                sentences.insert(i, n)
            context = "\n".join(sentences)

        template = TEMPLATE + ANSWER_PREFIX
        tvf = tv
        if len(values) == 1:  # num_needle_q * num_needle_v == 1
            template = template.replace("Some", "A").replace("are all", "is").replace("are", "is") \
                               .replace("answers", "answer")
            tvf = tv[:-1]
        text = template.format(type_needle_v=tvf, context=context, query=key)
        return text, values, key


def _search(gen: _Gen, count: Callable[[str], int], target: int, seed: int, incremental: int) -> int:
    per = max(count(gen.build(incremental, seed)[0]) / incremental, 1e-6)
    lo, hi = incremental, max(int(target / per * 3), incremental * 2)
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if count(gen.build(mid, seed)[0]) <= target:
            best, lo = mid, mid + 1
        else:
            hi = mid - 1
    return best if best is not None else 1


def build_cell(task: str, target_real: int, n: int, *, seed: int = 20261022, count=None, tokenizer_json=None,
               max_tokens: int = 128) -> list[dict]:
    if task not in TASKS:
        raise ValueError(f"unknown RULER task {task!r}; have {TASKS}")
    if count is None:
        count, tok_sha = _make_count(tokenizer_json)
    else:
        tok_sha = "injected_count_callable"
    words, essay_sha, vocab = _load_essays()
    if len(vocab) < 2:
        raise RuntimeError(f"essay vocabulary too small in {ESSAY_DIR}")
    gen = _Gen(task, words, vocab)
    essay = _CFG[task]["type_haystack"] == "essay"
    incremental = 500 if essay else 25
    short = _SHORT[task]
    label = f"{target_real // 1024}k"
    cell = f"ruler_{short}@{label}"
    base_seed = int(hashlib.sha256(f"{seed}:{task}:{target_real}".encode()).hexdigest()[:12], 16)
    size = _search(gen, count, target_real, base_seed, incremental)
    ruler_sha = _ruler_sha()
    items = []
    for i in range(n):
        item_id = f"{cell}#{i:04d}"
        iseed = int(hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()[:12], 16)
        used = size
        step = max(1, size // 200)
        while True:
            text, gold, key = gen.build(used, iseed)
            real = count(text)
            if real <= target_real or used <= 1:
                break
            used -= step
        pins = {"ruler_repo": RULER_REPO, "ruler_sha": ruler_sha, "seed": seed, "item_seed": iseed,
                "tokenizer_sha256": tok_sha, "num_haystack": used, "query_key": key,
                "words_source": "essay_vocab_hyphen_pairs" if _CFG[task]["type_needle_k"] == "words" else None,
                "deviations": list(DEVIATIONS)}
        if essay:
            pins["essay_sha256"] = essay_sha
        items.append({
            "item_id": item_id, "cell": cell, "suite": "ruler", "task": task, "context_id": item_id,
            "messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "thinking": False,
            "scorer": "string_match_all", "gold": list(gold), "nominal_len": target_real,
            "real_n_offline": real, "n_source": "model_tokenizer_json", "pins": pins,
        })
    return items
