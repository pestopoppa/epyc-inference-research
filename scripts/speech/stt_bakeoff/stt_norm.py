"""stt_norm.py — the pinned STT output normalisers (CS-8.STT1 (a)/(b), DP-B).

Three named, versioned, content-hashed normalisers, all pure functions of
``(text, lang, side)``:

``stt_norm/v1`` — Annex S P-STT-1 §1.3 exactly, in the normative order
    (``whisper_stt.NORMALIZATION_STEPS``):

    1. Unicode NFKC, then ``casefold()`` (never ``lower()``).
    2. Non-lexical marker removal from the CLOSED list ``whisper_stt.NONLEXICAL_MARKERS``
       (compared in NFKC+casefold form), counted per utterance. A marker-shaped span
       NOT on the list (``[applause]``, ``(inaudible)``) is NOT removed: it is reported
       in ``unknown_markers`` and the utterance routes to the ``unknown_marker`` class.
       ``<unk>`` is exempt from marker shape: it is the Parakeet-family unknown token,
       counted in ``unk_count`` and scored as an ordinary (wrong) token, never excluded.
    3. Every character outside ``[\\w\\s']`` becomes a SPACE (never deleted).
    4. Apostrophes kept (``don't`` stays one token).
    5. Hyphen split (already done by step 3; declared so the choice is visible).
    6. Numerals, HYPOTHESIS side only, through a CLOSED enumerated table into the
       reference's spelled form. A digit-bearing token not in the table, or a complex
       numeral span (``3.5``, ``1,920``, ``$5``, ``5%``), is ``numeral_uncovered`` —
       COULD_NOT_CHECK, never guessed. On the REFERENCE side digits are never converted
       (the reference is the fixed side); a digit-bearing reference token is flagged
       ``numeral_uncovered`` too, because the hyp->ref direction cannot be checked
       against a reference that is not in spelled form.
    7. Whitespace collapse and strip — always last.

    v1 has NO filler step: fillers are ordinary words.

``stt_norm/v2`` — CANDIDATE (never pooled or differenced with v1 until CS-2 ratifies):
    v1 plus (i) typographic apostrophes ``’`` (U+2019) and ``ʼ`` (U+02BC) folded to
    ``'`` immediately after casefold — NFKC does not fold them, so under v1 ``l’uomo``
    is two tokens while ``l'uomo`` is one; (ii) fr/it elision rule: an elided form
    (``l'uomo``, ``dell'anno``, ``j'ai``) stays ONE token after the fold — no
    splitting; (iii) closed it/de/fr numeral tables beside the English one.
    German compounds are NOT merged or split by the normaliser; the scorer counts
    them as their own error class (``de_compound_split_join``) and forgives nothing.
    DP-B: fillers (``uh``, ``um``, ``euh``, ``ähm`` …) are ordinary words — never deleted
    from either side. ``FILLERS`` exists ONLY to flag ``fillers_in_ref``.

``stt_norm/v2-secondary-fillers`` — SECONDARY scorer for external anchors only:
    v2 plus a closed, counted filler removal (``FILLERS``) on both sides. Labelled
    ``secondary``; never pooled or compared with v1 or v2-primary.

The API takes the text and its language and NOTHING ELSE about the partner string:
a normaliser that can read the reference can rewrite the hypothesis toward it, which
manufactures matches. There is deliberately no parameter through which the partner
string could be passed (``inspect.signature`` is asserted in the tests).

Pure function module: no I/O except reading this file's own bytes for the content
hash, no locale, no host state, no inference.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from scripts.kernel_rnd.autokernel.adapters import whisper_stt

NORMALISER_ID = "stt_norm/v1"
V2_CANDIDATE_ID = "stt_norm/v2"
SECONDARY_ID = "stt_norm/v2-secondary-fillers"

REFERENCE = "reference"
HYPOTHESIS = "hypothesis"
SIDES = (REFERENCE, HYPOTHESIS)

STATUS_OK = "ok"
STATUS_UNKNOWN_MARKER = "unknown_marker"
STATUS_NUMERAL_UNCOVERED = "numeral_uncovered"

UNK_TOKEN = "<unk>"

V1_STEPS: Tuple[str, ...] = tuple(whisper_stt.NORMALIZATION_STEPS)
V2_STEPS: Tuple[str, ...] = (
    V1_STEPS[:2] + ("fold_typographic_apostrophes",) + V1_STEPS[2:]
)
SECONDARY_STEPS: Tuple[str, ...] = (
    V2_STEPS[:-1] + ("remove_enumerated_fillers",) + V2_STEPS[-1:]
)

_TYPOGRAPHIC_APOSTROPHES = ("’", "ʼ")


def _nfkc_casefold(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _surface_form(text: str) -> str:
    """Table values pass through steps 1, 3 and 7 so they are already in normalised form."""
    return " ".join(re.sub(r"[^\w\s']", " ", _nfkc_casefold(text)).split())


# ---------------------------------------------------------------------------
# Closed marker list (step 2)
# ---------------------------------------------------------------------------

#: The closed list, compared AFTER NFKC+casefold. Longest first so an alternation
#: never removes a prefix of a longer marker.
MARKERS_CASEFOLDED: Tuple[str, ...] = tuple(sorted(
    {_nfkc_casefold(m) for m in whisper_stt.NONLEXICAL_MARKERS},
    key=lambda m: (-len(m), m)))
_MARKER_RE = re.compile("|".join(re.escape(m) for m in MARKERS_CASEFOLDED))

#: Marker-SHAPED spans. Anything matching here after closed-list removal is an
#: ``unknown_marker`` — reported, never removed (a general bracket-deletion rule would
#: let a candidate wrap its errors in brackets and score a perfect match).
_MARKER_SHAPE_RE = re.compile(r"\[[^\[\]]*\]|\([^()]*\)|\{[^{}]*\}|<[^<>]*>")


# ---------------------------------------------------------------------------
# Closed numeral tables (step 6) — the calibration tables
# ---------------------------------------------------------------------------

_EN_ONES = ("zero one two three four five six seven eight nine ten eleven twelve "
            "thirteen fourteen fifteen sixteen seventeen eighteen nineteen").split()
_EN_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty", 60: "sixty",
            70: "seventy", 80: "eighty", 90: "ninety"}
_EN_ORD_ONES = ("_ first second third fourth fifth sixth seventh eighth ninth tenth "
                "eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth "
                "eighteenth nineteenth").split()
_EN_ORD_TENS = {20: "twentieth", 30: "thirtieth"}


def _en_cardinal(n: int) -> str:
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        t, u = divmod(n, 10)
        return _EN_TENS[t * 10] + ("" if u == 0 else " " + _EN_ONES[u])
    if n == 100:
        return "one hundred"
    raise ValueError(n)


def _en_ordinal(n: int) -> str:
    if n < 20:
        return _EN_ORD_ONES[n]
    t, u = divmod(n, 10)
    if u == 0:
        return _EN_ORD_TENS[t * 10]
    return _EN_TENS[t * 10] + " " + _EN_ORD_ONES[u]


def _en_suffix(n: int) -> str:
    if 10 <= n % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def _build_en_table() -> Dict[str, str]:
    """English calibration table (v1 and v2): EXACTLY these forms, nothing else.

    * cardinals 0..100 (``21`` -> ``twenty one``; ``100`` -> ``one hundred``);
    * ordinals 1st..31st (``21st`` -> ``twenty first``);
    * round hundreds 200..900 (``two hundred``) and ``1000`` -> ``one thousand``;
    * years 1900..1999 in the LibriSpeech reading (``1920`` -> ``nineteen twenty``,
      ``1900`` -> ``nineteen hundred``), EXCEPT 1901..1909 (``nineteen oh five`` vs
      ``nineteen o five`` vs ``nineteen hundred and five`` is ambiguous — uncovered);
    * ``2000`` -> ``two thousand``.

    Everything else digit-bearing is ``numeral_uncovered``. Extending the table is a
    new normaliser version, never an in-loop edit.
    """
    table: Dict[str, str] = {}
    for n in range(0, 101):
        table[str(n)] = _en_cardinal(n)
    for n in range(1, 32):
        table[f"{n}{_en_suffix(n)}"] = _en_ordinal(n)
    for h in range(2, 10):
        table[str(h * 100)] = _EN_ONES[h] + " hundred"
    table["1000"] = "one thousand"
    table["1900"] = "nineteen hundred"
    for yy in range(10, 100):
        table[str(1900 + yy)] = "nineteen " + _en_cardinal(yy)
    table["2000"] = "two thousand"
    return {k: _surface_form(v) for k, v in table.items()}


_IT_ONES = ("zero uno due tre quattro cinque sei sette otto nove dieci undici dodici "
            "tredici quattordici quindici sedici diciassette diciotto diciannove").split()
_IT_TENS = {20: "venti", 30: "trenta", 40: "quaranta", 50: "cinquanta", 60: "sessanta",
            70: "settanta", 80: "ottanta", 90: "novanta"}


def _build_it_table() -> Dict[str, str]:
    """Italian cardinals 0..100 (v2 candidate). Tens drop their final vowel before
    ``uno``/``otto`` (``ventuno``, ``ventotto``); ``3`` in a compound is ``tré``
    (``ventitré``); ``100`` -> ``cento``."""
    table: Dict[str, str] = {}
    for n in range(0, 20):
        table[str(n)] = _IT_ONES[n]
    for n in range(20, 100):
        t, u = divmod(n, 10)
        tens = _IT_TENS[t * 10]
        if u == 0:
            word = tens
        elif u in (1, 8):
            word = tens[:-1] + _IT_ONES[u]
        elif u == 3:
            word = tens + "tré"
        else:
            word = tens + _IT_ONES[u]
        table[str(n)] = word
    table["100"] = "cento"
    return {k: _surface_form(v) for k, v in table.items()}


_DE_ONES = ("null eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf "
            "dreizehn vierzehn fünfzehn sechzehn siebzehn achtzehn neunzehn").split()
_DE_TENS = {20: "zwanzig", 30: "dreißig", 40: "vierzig", 50: "fünfzig", 60: "sechzig",
            70: "siebzig", 80: "achtzig", 90: "neunzig"}


def _build_de_table() -> Dict[str, str]:
    """German cardinals 0..100 (v2 candidate), one closed compound token each
    (``21`` -> ``einundzwanzig``); ``100`` -> ``hundert``. Values pass through casefold,
    so ``dreißig`` is stored as ``dreissig`` — the same fold the reference receives."""
    table: Dict[str, str] = {}
    for n in range(0, 20):
        table[str(n)] = _DE_ONES[n]
    for n in range(20, 100):
        t, u = divmod(n, 10)
        tens = _DE_TENS[t * 10]
        if u == 0:
            word = tens
        else:
            unit = "ein" if u == 1 else _DE_ONES[u]
            word = unit + "und" + tens
        table[str(n)] = word
    table["100"] = "hundert"
    return {k: _surface_form(v) for k, v in table.items()}


_FR_ONES = ("zéro un deux trois quatre cinq six sept huit neuf dix onze douze treize "
            "quatorze quinze seize dix-sept dix-huit dix-neuf").split()
_FR_TENS = {20: "vingt", 30: "trente", 40: "quarante", 50: "cinquante", 60: "soixante"}


def _build_fr_table() -> Dict[str, str]:
    """French cardinals 0..100 (v2 candidate), traditional spelling: ``21`` ->
    ``vingt et un``, ``22`` -> ``vingt-deux``, ``71`` -> ``soixante et onze``, ``80`` ->
    ``quatre-vingts``, ``81`` -> ``quatre-vingt-un``, ``91`` -> ``quatre-vingt-onze``.
    Hyphens become separators (step 3), so the stored values are space-split. ``1`` is
    ``un`` (never ``une``); a reference reading ``une`` is a calibration finding."""
    table: Dict[str, str] = {}
    for n in range(0, 20):
        table[str(n)] = _FR_ONES[n]
    for n in range(20, 70):
        t, u = divmod(n, 10)
        tens = _FR_TENS[t * 10]
        if u == 0:
            word = tens
        elif u == 1:
            word = tens + " et un"
        else:
            word = tens + "-" + _FR_ONES[u]
        table[str(n)] = word
    for n in range(70, 80):
        u = n - 60
        table[str(n)] = "soixante et onze" if u == 11 else "soixante-" + _FR_ONES[u]
    table["80"] = "quatre-vingts"
    for n in range(81, 100):
        table[str(n)] = "quatre-vingt-" + _FR_ONES[n - 80]
    table["100"] = "cent"
    return {k: _surface_form(v) for k, v in table.items()}


NUMERAL_TABLE_EN: Mapping[str, str] = _build_en_table()
NUMERAL_TABLE_IT: Mapping[str, str] = _build_it_table()
NUMERAL_TABLE_DE: Mapping[str, str] = _build_de_table()
NUMERAL_TABLE_FR: Mapping[str, str] = _build_fr_table()

V1_NUMERAL_TABLES: Mapping[str, Mapping[str, str]] = {"en": NUMERAL_TABLE_EN}
V2_NUMERAL_TABLES: Mapping[str, Mapping[str, str]] = {
    "en": NUMERAL_TABLE_EN, "it": NUMERAL_TABLE_IT,
    "de": NUMERAL_TABLE_DE, "fr": NUMERAL_TABLE_FR,
}

#: Complex numeral spans, detected on the post-marker text BEFORE punctuation turns
#: them into separate digit tokens: digit-punct-digit (``3.5``, ``1,920``, ``10:30``,
#: ``1/2`` — NFKC turns ``½`` into ``1⁄2``), currency/percent/degree adjacent to a
#: digit (``$5``, ``5%``, ``20°``), and a leading-decimal (``.5``). Converting any of
#: them token by token would silently produce a WRONG spelled form, so they are
#: ``numeral_uncovered`` instead.
_NUM_SYMBOLS = "$€£¥¢%‰°#№+"
_COMPLEX_NUMERAL_RE = re.compile(
    r"\w*\d\w*(?:[^\w\s]+\w*\d\w*)+"
    rf"|[{re.escape(_NUM_SYMBOLS)}]\s?\d\w*"
    rf"|\d\w*\s?[{re.escape(_NUM_SYMBOLS)}]"
    r"|(?:^|\s)[.,]\d\w*")


# ---------------------------------------------------------------------------
# Closed filler list — FLAGGING ONLY (DP-B), removal only in the secondary scorer
# ---------------------------------------------------------------------------

#: Closed, enumerated per-language filler tokens (normalised form). Used to compute
#: ``fillers_in_ref`` on every record and, ONLY in ``stt_norm/v2-secondary-fillers``,
#: for counted removal. v1 and v2-primary never delete a filler.
FILLERS: Mapping[str, Tuple[str, ...]] = {
    lang: tuple(sorted({_surface_form(w) for w in words}))
    for lang, words in {
        "en": ("uh", "um", "uhm", "umm", "er", "erm", "hmm", "mm", "mhm"),
        "fr": ("euh", "heu", "hum", "bah", "ben"),
        "de": ("äh", "ähm", "öh", "öhm", "hm", "hmm"),
        "it": ("ehm", "eh", "ehmm", "mmh", "uhm", "ehh"),
    }.items()
}


def _lang_key(lang: str) -> str:
    if not isinstance(lang, str) or not lang.strip():
        raise TypeError(f"lang must be a non-empty string, got {lang!r}")
    return re.split(r"[-_]", lang.strip().casefold(), maxsplit=1)[0]


def count_fillers(tokens: Iterable[str], lang: str) -> int:
    """Number of tokens on the closed filler list for ``lang``. Flagging only."""
    fl = set(FILLERS.get(_lang_key(lang), ()))
    return sum(1 for t in tokens if t in fl)


# ---------------------------------------------------------------------------
# The normaliser
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NormResult:
    """Output of one normalisation. ``status`` is ``ok``, ``unknown_marker`` or
    ``numeral_uncovered`` (the latter two are COULD_NOT_CHECK for the utterance)."""
    normaliser_id: str
    side: str
    lang: str
    text: str
    tokens: Tuple[str, ...]
    markers_removed_count: int
    unknown_markers: Tuple[str, ...]
    numeral_uncovered: Tuple[str, ...]
    numerals_converted: int
    unk_count: int
    fillers_removed_count: int

    @property
    def status(self) -> str:
        if self.unknown_markers:
            return STATUS_UNKNOWN_MARKER
        if self.numeral_uncovered:
            return STATUS_NUMERAL_UNCOVERED
        return STATUS_OK


@dataclass(frozen=True)
class NormaliserSpec:
    """A named normaliser. ``normalise(text, lang, *, side)`` is its only entry point."""
    normaliser_id: str
    column: str
    candidate: bool
    secondary: bool
    steps: Tuple[str, ...]
    numeral_tables: Mapping[str, Mapping[str, str]] = field(repr=False)
    fold_typographic_apostrophes: bool
    remove_fillers: bool
    transforms_used: Tuple[str, ...] = ()

    def normalise(self, text: str, lang: str, *, side: str) -> NormResult:
        return _normalise(self, text, lang, side)

    def sha256(self) -> str:
        return normaliser_sha256(self.normaliser_id)


def _normalise(spec: NormaliserSpec, text: str, lang: str, side: str) -> NormResult:
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    lkey = _lang_key(lang)

    # 1. NFKC, then casefold.
    s = _nfkc_casefold(text)
    if spec.fold_typographic_apostrophes:
        for ch in _TYPOGRAPHIC_APOSTROPHES:
            s = s.replace(ch, "'")
    unk_count = s.count(UNK_TOKEN)

    # 2. Closed marker removal, counted; unknown marker-shaped spans reported, kept.
    s, markers_removed = _MARKER_RE.subn(" ", s)
    unknown = tuple(m.group(0) for m in _MARKER_SHAPE_RE.finditer(s)
                    if m.group(0) != UNK_TOKEN)

    # Pre-scan for complex numeral spans before punctuation splits them.
    complex_spans = tuple(m.group(0).strip() for m in _COMPLEX_NUMERAL_RE.finditer(s))

    # 3-5. Punctuation -> separator; apostrophe kept; hyphen therefore split.
    s = re.sub(r"[^\w\s']", " ", s)
    tokens: List[str] = s.split()

    # 6. Numerals: hypothesis side only, closed table; uncovered is COULD_NOT_CHECK.
    uncovered: List[str] = list(complex_spans)
    converted = 0
    if side == HYPOTHESIS:
        table = spec.numeral_tables.get(lkey, {})
        out: List[str] = []
        for tok in tokens:
            if any(ch.isdigit() for ch in tok):
                spelled = table.get(tok)
                if spelled is None:
                    uncovered.append(tok)
                    out.append(tok)
                else:
                    out.extend(spelled.split())
                    converted += 1
            else:
                out.append(tok)
        tokens = out
    else:
        uncovered.extend(tok for tok in tokens if any(ch.isdigit() for ch in tok))

    fillers_removed = 0
    if spec.remove_fillers:
        fl = set(FILLERS.get(lkey, ()))
        kept = [t for t in tokens if t not in fl]
        fillers_removed = len(tokens) - len(kept)
        tokens = kept

    # 7. Whitespace collapse and strip — last.
    norm_text = " ".join(" ".join(tokens).split())
    return NormResult(
        normaliser_id=spec.normaliser_id, side=side, lang=lkey, text=norm_text,
        tokens=tuple(norm_text.split()), markers_removed_count=markers_removed,
        unknown_markers=unknown, numeral_uncovered=tuple(uncovered),
        numerals_converted=converted, unk_count=unk_count,
        fillers_removed_count=fillers_removed)


V1 = NormaliserSpec(
    normaliser_id=NORMALISER_ID, column="v1", candidate=False, secondary=False,
    steps=V1_STEPS, numeral_tables=V1_NUMERAL_TABLES,
    fold_typographic_apostrophes=False, remove_fillers=False)
V2_CANDIDATE = NormaliserSpec(
    normaliser_id=V2_CANDIDATE_ID, column="v2_candidate", candidate=True, secondary=False,
    steps=V2_STEPS, numeral_tables=V2_NUMERAL_TABLES,
    fold_typographic_apostrophes=True, remove_fillers=False)
SECONDARY = NormaliserSpec(
    normaliser_id=SECONDARY_ID, column="secondary", candidate=True, secondary=True,
    steps=SECONDARY_STEPS, numeral_tables=V2_NUMERAL_TABLES,
    fold_typographic_apostrophes=True, remove_fillers=True,
    transforms_used=("filler_removal",))

NORMALISERS: Mapping[str, NormaliserSpec] = {
    s.normaliser_id: s for s in (V1, V2_CANDIDATE, SECONDARY)}
COLUMNS: Mapping[str, NormaliserSpec] = {s.column: s for s in NORMALISERS.values()}


def get_normaliser(normaliser_id: str) -> NormaliserSpec:
    try:
        return NORMALISERS[normaliser_id]
    except KeyError:
        raise KeyError(f"unknown normaliser {normaliser_id!r}; known: {sorted(NORMALISERS)}")


def normalise_v1(text: str, lang: str, *, side: str) -> NormResult:
    """``stt_norm/v1`` — P-STT-1 §1.3. Takes no partner string, by construction."""
    return _normalise(V1, text, lang, side)


def normalise_v2_candidate(text: str, lang: str, *, side: str) -> NormResult:
    """``stt_norm/v2`` CANDIDATE. Never pooled or differenced with v1."""
    return _normalise(V2_CANDIDATE, text, lang, side)


def normalise_secondary(text: str, lang: str, *, side: str) -> NormResult:
    """Secondary (external-anchor) scorer: v2 + closed counted filler removal."""
    return _normalise(SECONDARY, text, lang, side)


# ---------------------------------------------------------------------------
# Content hash and contract checks
# ---------------------------------------------------------------------------

def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


@lru_cache(maxsize=None)
def normaliser_sha256(normaliser_id: str = NORMALISER_ID) -> str:
    """SHA-256 over the normaliser id, THIS FILE's source bytes and the id's tables.

    The whole source file is hashed (not just the functions on the v1 path), so any
    edit to the module bumps every normaliser hash: conservative, because a helper
    shared by the pipeline cannot change without the instrument identity changing.
    """
    spec = get_normaliser(normaliser_id)
    h = hashlib.sha256()
    h.update(b"stt_bakeoff.stt_norm\0")
    h.update(spec.normaliser_id.encode("utf-8") + b"\0")
    h.update(Path(__file__).read_bytes())
    h.update(b"\0")
    h.update(_canonical({
        "steps": list(spec.steps),
        "markers": list(MARKERS_CASEFOLDED),
        "numeral_tables": {k: dict(v) for k, v in spec.numeral_tables.items()},
        "fillers": {k: list(v) for k, v in FILLERS.items()},
        "fold_typographic_apostrophes": spec.fold_typographic_apostrophes,
        "remove_fillers": spec.remove_fillers,
        "transforms_used": list(spec.transforms_used),
    }))
    return h.hexdigest()


def contract_check(normaliser_id: str = NORMALISER_ID):
    """``whisper_stt.check_normalizer_contract`` on the declared steps/transforms.

    v1 PASSES. The v2 candidate and the secondary scorer FAIL it by design: they are
    different instruments, which is exactly why their records are labelled candidate /
    secondary and refused for pooling with v1.
    """
    spec = get_normaliser(normaliser_id)
    return whisper_stt.check_normalizer_contract(
        steps=spec.steps, transforms_used=spec.transforms_used)


@dataclass(frozen=True)
class PropertyReport:
    check: object
    symmetric: Optional[bool]
    idempotent: Optional[bool]
    deterministic: Optional[bool]
    violations: Tuple[str, ...]


def check_properties(texts: Iterable[Tuple[str, str]],
                     normaliser_id: str = NORMALISER_ID) -> PropertyReport:
    """Assert §1.3 symmetry, idempotence and determinism over ``(text, lang)`` pairs.

    * determinism — two runs give identical results;
    * idempotence — ``norm(norm(x).text).text == norm(x).text`` on BOTH sides;
    * symmetry — the side-independent steps agree: whenever no numeral was converted
      or flagged on either side, the reference-side and hypothesis-side outputs are the
      same string (the side argument may only gate step 6).

    The verdict is ``whisper_stt.check_normalizer_properties``; an empty input is
    COULD_NOT_CHECK (a vacuous pass over nothing is not a pass).
    """
    spec = get_normaliser(normaliser_id)
    pairs = list(texts)
    if not pairs:
        chk = whisper_stt.check_normalizer_properties(
            symmetric=None, idempotent=None, deterministic=None)
        return PropertyReport(chk, None, None, None, ("no texts supplied",))
    sym = idem = det = True
    violations: List[str] = []
    for text, lang in pairs:
        for side in SIDES:
            a = spec.normalise(text, lang, side=side)
            b = spec.normalise(text, lang, side=side)
            if a != b:
                det = False
                violations.append(f"non-deterministic on {side}: {text!r}")
            again = spec.normalise(a.text, lang, side=side)
            if again.text != a.text:
                idem = False
                violations.append(f"non-idempotent on {side}: {text!r}")
        r = spec.normalise(text, lang, side=REFERENCE)
        h = spec.normalise(text, lang, side=HYPOTHESIS)
        numeral_free = (not r.numeral_uncovered and not h.numeral_uncovered
                        and h.numerals_converted == 0)
        if numeral_free and r.text != h.text:
            sym = False
            violations.append(f"asymmetric outside step 6: {text!r}")
    chk = whisper_stt.check_normalizer_properties(
        symmetric=sym, idempotent=idem, deterministic=det)
    return PropertyReport(chk, sym, idem, det, tuple(violations))
