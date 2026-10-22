"""YARN-AGG-1 LongBioBench cell builder: local, offline, zero inference.

Source: the pinned clone ``/mnt/raid0/llm/data/eval/longbiobench/repo`` @ dc6f6ed1aab5dd072f0837f3160a45beb4df8dbe.

What is IMPORTED from the clone (loaded by file path under a private module name, never at import of this module,
never via a permanent ``sys.path`` edit): ``data_utils/bios.py`` -> ``Bio`` (bio rendering, standard + paraphrase
templates, ``to_question``), plus the ``data_utils/attributes/*.txt`` pools and ``data_utils/templates/*.json``.
It needs ``pandas`` (bios.py imports it, unused) and the ``from __init__ import DATA_UTIL_PATH`` shim, both handled.

What is PORTED (``data_utils/prepare_data.py::BioQABenchmarkGenerator`` cannot be imported: it hard-imports
``tiktoken``, ``loguru``, ``tqdm``, bare ``from bios import`` and draws from the *global* ``random``, i.e. is not
seedable per item): ``sample_attributes``, ``single_retrieval`` (+ ``_paraphrase``), ``multiple_retrieval``
(+ ``get_context_questions_for_multiple_retrieval``), ``remove_suffix``; prompts from
``scripts/data_loader_prompts.py`` (``SingleQA`` / ``MultiQA`` system+user text); the grader from
``scripts/evaluator.py::GeneralEvaluator.cal_hit``.

Variants -> repo: standard = ``--skill standard -nr 1``; multi_standard_5 / _10 = ``--skill standard -nr 5|10``
(``multiple_retrieval``, all questions about ONE attribute, MultiQA prompt); paraphrase = ``--skill paraphrase -nr 1``
(``single_retrieval_paraphrase``: distractors AND the target bio use the paraphrase templates).

DEVIATIONS (also stamped on every item as ``pins["deviations"]``): see ``DEVIATIONS`` below.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import math
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

from scripts.benchmark.yarn_gate import cells

VARIANTS = ("standard", "multi_standard_5", "multi_standard_10", "paraphrase")
SUITE = "longbio"
REPO_DIR = "/mnt/raid0/llm/data/eval/longbiobench/repo"
REPO_SHA = "dc6f6ed1aab5dd072f0837f3160a45beb4df8dbe"
POOL_SIZE = 10000          # repo default num_bios
BIRTHDATE_POOL = ("1950-01-01", "2001-12-31")
DENSITY = 1.0              # repo default --distractor_density (every distractor keeps the queried attribute)
_ATTR_TXT = ("birthplace", "first_name", "gender", "hobby", "last_name", "major", "middle_name", "university",
             "work_city")
_TEMPLATE_JSON = ("birthdate", "birthplace", "hobby", "major", "university", "work_city")
_NON_QUESTION = ("first_name", "middle_name", "last_name", "gender")
LOWER_FRAC = 0.995         # accept a sized message in [LOWER_FRAC*T, T]
MIN_FRAC = 0.95            # hard floor; below this we raise

SYSTEM_SINGLE = ("Your task is to answer the user’s question based on a long context, which consists of many "
                 "bios. Output the answer only. Don't explain or output other things.")
SYSTEM_MULTI = ("Your task is to answer all the user’s questions based on a long context, which consists of "
                "many bios. Output only the answers for each question sequentially. Don't explain or output other "
                "things.")

DEVIATIONS = [
    "per-item random.Random (sha256 of seed|variant|target|index) replaces the repo's global `random` and its fixed "
    "Random(2024) template rng; same distributions, but seedable and distinct per item",
    "sizing is exact against the SERVED model tokenizer over the whole user message (calibrated fill), not the "
    "repo's `length*1024-150-200*n` reserve with its own tokenizer",
    "no assistant prefill ('Based on the provided context, <question_prefix>'): chat message ends at the question "
    "(thinking off, system message keeps 'Output the answer only')",
    "multi-retrieval target people are sampled WITHOUT replacement (repo `random.choices` can pick one person twice)",
    "target-position fraction is drawn once per target and mapped onto the final distractor count (repo draws from "
    "the already-built list; same uniform distribution over [0,1))",
    "grader: repo `GeneralEvaluator.cal_hit` (case-insensitive containment, multi = ordered sequential containment) "
    "instead of arms.py's literal 'exact match'; multi scores the fraction of answers found in order rather than "
    "the repo's all-or-nothing",
    "Bio objects share one cached TemplateProcessor (repo re-reads the 6 template JSONs for every Bio; output is identical)",
]


# ── repo access ─────────────────────────────────────────────────────────────────────────────────────
def _required_paths(repo_dir: str | None = None) -> list[Path]:
    r = Path(repo_dir or REPO_DIR)
    out = [r / "data_utils" / "bios.py", r / "data_utils" / "__init__.py", r / "scripts" / "data_loader_prompts.py"]
    out += [r / "data_utils" / "attributes" / f"{a}.txt" for a in _ATTR_TXT]
    out += [r / "data_utils" / "templates" / f"{a}.json" for a in _TEMPLATE_JSON]
    return out


def _git_head(repo: Path) -> str:
    r = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "git rev-parse failed")
    return r.stdout.strip()


def available(tokenizer_json: str | None = None, *, check_tokenizer: bool = True) -> tuple[bool, str]:
    """(ok, reason). On failure the reason names the exact missing path / mismatching sha / missing module."""
    repo = Path(REPO_DIR)
    if not repo.is_dir():
        return False, f"missing LongBioBench clone: {repo}"
    try:
        head = _git_head(repo)
    except Exception as e:  # noqa: BLE001
        return False, f"cannot read HEAD of {repo}: {e}"
    if head != REPO_SHA:
        return False, f"LongBioBench clone {repo} HEAD {head} != pinned {REPO_SHA}"
    for p in _required_paths():
        if not p.is_file():
            return False, f"missing file: {p}"
    if importlib.util.find_spec("pandas") is None:
        return False, "missing python module 'pandas' (imported by the repo's data_utils/bios.py)"
    if check_tokenizer:
        tj = Path(tokenizer_json or cells.DEFAULT_TOKENIZER_JSON)
        if not tj.is_file():
            return False, f"missing tokenizer file: {tj}"
    return True, "ok"


@functools.lru_cache(maxsize=1)
def _repo_bios_module():
    """Load the repo's bios.py by path under a private name; temporarily provide the bare ``__init__`` module it
    imports. No permanent sys.path / sys.modules pollution."""
    repo = Path(REPO_DIR)
    du = repo / "data_utils"
    init_spec = importlib.util.spec_from_file_location("_longbio_repo_init", du / "__init__.py")
    init_mod = importlib.util.module_from_spec(init_spec)
    init_spec.loader.exec_module(init_mod)
    spec = importlib.util.spec_from_file_location("_longbio_repo_bios", du / "bios.py")
    mod = importlib.util.module_from_spec(spec)
    saved = sys.modules.get("__init__")
    sys.modules["__init__"] = init_mod
    try:
        spec.loader.exec_module(mod)
    finally:
        if saved is None:
            sys.modules.pop("__init__", None)
        else:
            sys.modules["__init__"] = saved
    # one shared TemplateProcessor (the repo builds one per Bio: 6 JSON reads each)
    tp = mod.TemplateProcessor(str(du / "templates"))
    mod.TemplateProcessor = lambda template_path: tp
    return mod


@functools.lru_cache(maxsize=1)
def _attribute_pool() -> dict[str, list[str]]:
    d = Path(REPO_DIR) / "data_utils" / "attributes"
    mod = _repo_bios_module()
    return {a: mod.read_list_from_txt(str(d / f"{a}.txt")) for a in sorted(_ATTR_TXT)}


def _int_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16], 16)


def _random_date(rng: random.Random):
    from datetime import datetime, timedelta
    a = datetime.strptime(BIRTHDATE_POOL[0], "%Y-%m-%d")
    b = datetime.strptime(BIRTHDATE_POOL[1], "%Y-%m-%d")
    return a + timedelta(days=rng.randint(0, (b - a).days))


@functools.lru_cache(maxsize=4)
def _bio_pool(seed: int):
    """Port of BioQABenchmarkGenerator.__init__ bio pool: POOL_SIZE bios with unique full names."""
    mod = _repo_bios_module()
    pool_attrs = _attribute_pool()
    rng = random.Random(_int_seed(seed, "pool"))
    bios, seen = [], set()
    while len(bios) < POOL_SIZE:
        attrs = {k: rng.choice(v) for k, v in pool_attrs.items()}        # sample_attributes()
        bd = _random_date(rng)
        attrs["birthdate"] = bd.strftime("%Y-%m-%d")
        attrs["age"] = 2024 - bd.year
        key = (attrs["first_name"], attrs["middle_name"], attrs["last_name"])
        if key in seen:
            continue
        seen.add(key)
        bios.append(mod.Bio(**attrs))
    return tuple(bios)


def _q_attributes() -> list[str]:
    return [a for a in sorted(_attribute_pool()) if a not in _NON_QUESTION] + ["birthdate"]


def _tokenizer_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_counter(tokenizer_json: str) -> Callable[[str], int]:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tokenizer_json)
    return lambda text: len(tok.encode(text, add_special_tokens=False).ids)


# ── item construction ────────────────────────────────────────────────────────────────────────────────
def _variant_spec(variant: str) -> tuple[int, bool]:
    if variant == "standard":
        return 1, False
    if variant == "paraphrase":
        return 1, True
    if variant in ("multi_standard_5", "multi_standard_10"):
        return int(variant.rsplit("_", 1)[1]), False
    raise ValueError(f"unknown longbio variant {variant!r}; expected one of {VARIANTS}")


def _user_content(ctx: str, questions: list[str]) -> str:
    if len(questions) == 1:
        return f"Context: {ctx}\n\nQuestion: {questions[0]}"
    qs = "\n".join(f"Question{i + 1}: {q}" for i, q in enumerate(questions))
    return f"Context: {ctx}\n\nThe Questions are as follows:\n{qs}\nAnswer each question in sequence."


def _build_item(variant: str, target_real: int, i: int, seed: int, count: Callable[[str], int],
                max_tokens: int) -> dict:
    m, paraphrase = _variant_spec(variant)
    label = f"{target_real // 1024}k"
    iseed = _int_seed(seed, variant, target_real, i)
    pick = random.Random(iseed)            # who / which attribute / where
    d_rng = random.Random(iseed ^ 0x5DEECE66D)   # distractor template choices
    t_rng = random.Random(iseed ^ 0xB5AD4ECE)    # target template choices
    pool = _bio_pool(seed)

    # 1. targets + question(s)  (single_retrieval / multiple_retrieval step 1)
    bio_qs = pick.sample(pool, m)
    attr = pick.choice(_q_attributes())
    qs = [b.to_question(attr)[1] for b in bio_qs]
    answers = [str(getattr(b, attr)) for b in bio_qs]
    fracs = sorted(pick.random() for _ in range(m))
    target_names = {b.full_names for b in bio_qs}
    descs = [b.to_description(t_rng, use_paraphrase=paraphrase)["description"] for b in bio_qs]
    for d, a in zip(descs, answers):
        if a not in d:
            raise RuntimeError(f"target bio does not state its {attr} {a!r}: {d!r}")

    system = SYSTEM_MULTI if m > 1 else SYSTEM_SINGLE

    # 2. lazy distractor stream (port of the `while True` fill loop, budget replaced by exact calibration)
    stream: list[str] = []
    cost: list[int] = []

    def ensure(k: int) -> None:
        while len(stream) < k:
            while True:
                b = pool[pick.randrange(len(pool))]
                if b.full_names not in target_names:
                    break
            exc = [attr] if not (d_rng.random() < DENSITY) else []
            desc = b.to_description(d_rng, exclude_attributions=exc, use_paraphrase=paraphrase)["description"]
            stream.append(desc)
            cost.append(count(" " + desc.strip()))

    suffix = "This is the end of this bio."

    def assemble(k: int) -> str:
        ensure(k)
        ctx = list(stream[:k])
        off = 0
        for fr, d in zip(fracs, descs):
            p = min(k - 1, int(fr * k)) if k > 0 else 0
            ctx.insert(p + off, d)
            off += 1
        if ctx[-1].endswith(suffix + " "):   # repo remove_suffix on the final element
            ctx[-1] = ctx[-1][: -len(suffix) - 1].rstrip()
        return "".join(ctx)

    def measure(k: int) -> int:
        return count(_user_content(assemble(k), qs))

    # initial k from additive per-bio costs
    base = count(_user_content("", qs)) + sum(count(" " + d.strip()) for d in descs)
    budget = target_real - base
    k, acc = 0, 0
    while True:
        ensure(k + 1)
        if acc + cost[k] > budget:
            break
        acc += cost[k]
        k += 1
    best: tuple[int, int] | None = None   # (N, k) with N <= T
    tried: dict[int, int] = {}
    for _ in range(10):
        n = tried[k] = measure(k)
        if n <= target_real and (best is None or n > best[0]):
            best = (n, k)
        if LOWER_FRAC * target_real <= n <= target_real:
            break
        avg = sum(cost) / len(cost)
        aim = target_real - 0.0025 * target_real
        dk = int((aim - n) / avg)
        if dk == 0:
            dk = -1 if n > target_real else (1 if n < LOWER_FRAC * target_real else 0)
        if dk == 0:
            break
        k = max(1, k + dk)
        if k in tried:
            break
    if best is None:  # still over: step down one bio at a time
        while k > 1:
            k -= 1
            n = measure(k)
            if n <= target_real:
                best = (n, k)
                break
    if best is None or best[0] < MIN_FRAC * target_real:
        raise RuntimeError(f"could not size {variant}@{label} item {i} into [{MIN_FRAC}T, T] (T={target_real}, "
                           f"best={best})")
    n_real, k = best
    ctx_text = assemble(k)
    user = _user_content(ctx_text, qs)
    item_id = f"longbio_{variant}@{label}#{i:04d}"
    return {
        "item_id": item_id, "cell": f"longbio_{variant}@{label}", "suite": SUITE, "task": variant,
        "context_id": item_id,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "max_tokens": max_tokens, "thinking": False, "scorer": "longbio_exact_all", "gold": answers,
        "datapoint": {"attribute": attr, "names": [b.full_names for b in bio_qs], "questions": qs,
                      "n_distractors": k, "retrievals": m, "paraphrase": paraphrase},
        "nominal_len": target_real, "real_n_offline": n_real, "n_source": "model_tokenizer_json",
    }


def build_cell(variant: str, target_real: int, n: int, *, seed: int = 20261022, count=None,
               tokenizer_json=None, max_tokens: int = 256) -> list[dict]:
    """n LongBioBench items of ``variant`` whose user message is in [0.95*T, T] real tokens (normally >= 0.995*T).

    ``count`` is an injectable ``count(text) -> int``; when None the served model's tokenizer.json is used
    (``tokenizer_json`` or ``cells.DEFAULT_TOKENIZER_JSON``, ``add_special_tokens=False``).
    """
    _variant_spec(variant)
    ok, why = available(tokenizer_json, check_tokenizer=count is None)
    if not ok:
        raise RuntimeError(f"longbio unavailable: {why}")
    tj = str(tokenizer_json or cells.DEFAULT_TOKENIZER_JSON)
    if count is None:
        counter, tsha, src = _make_counter(tj), _tokenizer_sha(tj), "model_tokenizer_json"
    else:
        counter, tsha, src = count, None, "injected_counter"
    pins = {"longbio_repo": REPO_DIR, "longbio_sha": REPO_SHA, "seed": seed, "tokenizer_sha256": tsha,
            "grader": "longbio_exact_all", "deviations": list(DEVIATIONS)}
    out = []
    for i in range(n):
        it = _build_item(variant, target_real, i, seed, counter, max_tokens)
        it["n_source"] = src
        it["pins"] = dict(pins)
        cells.validate_item(it)
        out.append(it)
    return out


# ── scorer ───────────────────────────────────────────────────────────────────────────────────────────
_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", str(s)).strip().lower()


@cells.register("longbio_exact_all")
def longbio_exact_all(item, output: str, *, model: str) -> dict:
    """Repo ``GeneralEvaluator.cal_hit``: each gold answer must occur (case-insensitive, whitespace-collapsed) in
    the output, in order for multi-retrieval. Score = fraction of gold answers found in sequence (1.0 for single
    retrieval == the repo's hit; the repo's multi hit is the all-or-nothing ``score == 1.0``)."""
    gold = item["gold"] if isinstance(item["gold"], list) else [item["gold"]]
    gold = [_norm(g) for g in gold]
    pred = _norm(output or "")
    pos, hits = 0, 0
    for g in gold:
        j = pred.find(g, pos) if g else -1
        if j >= 0:
            hits += 1
            pos = j + len(g)
    return {"score": hits / len(gold) if gold else 0.0, "hits": hits, "n_refs": len(gold),
            "all_hit": hits == len(gold) and bool(gold)}
