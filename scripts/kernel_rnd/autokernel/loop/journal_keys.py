"""journal_keys.py: key hypothesis-journal entries and measured negatives by op_family + shape_class, not by model (RSL-R7).

A KEY is {"op_family": <census_families family>, "shape_class": {dim: value | [lo, hi]}, "device": "gpu"|"cpu" (optional)}.
Dims are free names; the conventional ones are n (verify/batch rows), kv (context depth), m (matrix rows). A dim missing from
either side is a wildcard, so an empty shape_class means "the whole family". Two keys OVERLAP when op_family is equal, the
devices agree (when both are stated) and every dim present in BOTH has intersecting ranges. The model is deliberately NOT part
of the key: a measured negative on model A's range blocks the same idea on model B's overlapping range. A candidate that has
new evidence says so with "reopens": [mechanism_id, ...] and is then compared against everything else.

Numeric-dim conventions: `quant` is the GGML type id of the weights (q8_0=8, iq4_nl=20, iq3_s=21); `kv_type` is the KV-cache
type id; `mech` separates mechanisms that share a family+range (1 = small-N barrier change, 2 = 4-row tile dispatch). A
candidate that omits a dim is a wildcard against it, so a quant- or mechanism-specific negative narrows only candidates that
declare the dim.

A negative reaches the sidecar `<store>/journal_keys.json` ({mechanism_id: key + status}); only a status=measured_null entry
blocks. Entries with no key are NOT blocked or blocking: they keep working exactly as before and are reported as `unkeyed`.

A hand-written hypothesis (an idea the census cannot see: kernel fusion, a drafting scheme, a removed pass, a device placement
such as a hybrid CPU+GPU Flash-Next split) carries its key in one HTML comment, `<!-- journal-key: {...} -->`, plus
"touches": [{"path", "op" | "family", "fraction"}]. price() turns that into a saving from CENSUS ROWS: fraction x the surface's
share_ms (step_ms x share_of_step) on every touched device, summed; a negative fraction is an added cost on that device. A touch
with no matching priced surface adds nothing and is counted, never guessed.
"""
import json
import os
import re

KEY_RE = re.compile(r"<!--\s*journal-key:\s*(\{.*?\})\s*-->", re.S)
SIDECAR = "journal_keys.json"
BLOCKING = "measured_null"


def parse_key(text: str):
    """The key in a hypothesis file's journal-key comment, or None (unkeyed; malformed JSON counts as unkeyed)."""
    m = KEY_RE.search(text or "")
    try:
        return json.loads(m.group(1)) if m else None
    except ValueError:
        return None


def key_of(entry: dict):
    """Key of a negatives.json-style record (op_family present), else None."""
    if not entry.get("op_family"):
        return None
    return {k: entry[k] for k in ("op_family", "shape_class", "device", "status", "mechanism_id") if k in entry}


def problems(key: dict) -> list:
    """Why a key is malformed (empty list = fine)."""
    out = [] if key.get("op_family") else ["op_family missing"]
    for dim, v in (key.get("shape_class") or {}).items():
        if not _range(v):
            out.append(f"shape_class.{dim}: want a number or [lo, hi] with lo <= hi")
    return out


def _range(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return (v, v)
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v) and v[0] <= v[1]:
        return (v[0], v[1])
    return None


def overlaps(a: dict, b: dict) -> bool:
    if a.get("op_family") != b.get("op_family"):
        return False
    if a.get("device") and b.get("device") and a["device"] != b["device"]:
        return False
    sa, sb = a.get("shape_class") or {}, b.get("shape_class") or {}
    for dim in sa.keys() & sb.keys():
        ra, rb = _range(sa[dim]), _range(sb[dim])
        if ra is None or rb is None or ra[1] < rb[0] or rb[1] < ra[0]:
            return False
    return True


def suppressed_by(candidate, negatives: dict):
    """mechanism_id of the first measured negative whose key overlaps `candidate`, else None. An unkeyed candidate is never
    suppressed; entries listed in candidate["reopens"] are skipped."""
    if not candidate or problems(candidate):
        return None
    skip = set(candidate.get("reopens") or [])
    for mid, k in sorted(negatives.items()):
        if mid not in skip and k.get("status", BLOCKING) == BLOCKING and not problems(k) and overlaps(candidate, k):
            return mid
    return None


#: census family -> the ggml op op_perf_screen.grid matches on; the ONLY place this map lives (defect 20).
OP_PERF_FAMILIES = {"dense-q8-gemv": "MUL_MAT", "dense-iq4xs-gemv": "MUL_MAT", "lm-head-gemv": "MUL_MAT", "mmid-moe": "MUL_MAT_ID"}
OP_PERF_DIMS = ("k", "m")        # the journal-key dims op_perf_screen.grid accepts (n comes from the path's registry n-set)
OP_PERF_RANGE_MAX = 1 << 16      # a [lo, hi] k/m range is expanded to its ints (grid filters by set membership) up to this width


def op_perf_class(obj):
    """THE mapping from a candidate's shape description to the `shape_class` op_perf_screen.grid takes:
    {"op_family": <ggml op>, "dims": {dim: [int, ...]}}, or None when the family has no served-shape grid (flash-attn, ...).

    `obj` is a candidate record (its `journal_key`), a bare journal key, or the LEGACY op-perf class itself (a `shape_class`
    that already has `dims`): the legacy form is accepted unchanged, so every existing caller keeps working. Of a journal key
    only k/m carry over; a dim the key leaves out is a wildcard, and a k/m range expands to its ints (never silently widened)."""
    if not isinstance(obj, dict):
        return None
    sc = obj.get("shape_class")
    if isinstance(sc, dict) and isinstance(sc.get("dims"), dict) and sc.get("op_family"):
        return sc
    key = obj.get("journal_key") if isinstance(obj.get("journal_key"), dict) else obj
    op = OP_PERF_FAMILIES.get(key.get("op_family"))
    if not op:
        return None
    dims = {}
    for d, v in (key.get("shape_class") or {}).items():
        r = _range(v)
        if d in OP_PERF_DIMS and r and r[1] - r[0] <= OP_PERF_RANGE_MAX:
            dims[d] = list(range(int(r[0]), int(r[1]) + 1))
    return {"op_family": op, "dims": dims}


def load(store_root) -> dict:
    try:
        with open(os.path.join(store_root, SIDECAR), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def key_entry(store_root, mechanism_id: str, key: dict, status: str = BLOCKING) -> dict:
    """Key any journal entry by mechanism_id (idempotent: a re-key overwrites). Returns the sidecar."""
    bad = problems(key)
    if bad:
        raise ValueError(f"{mechanism_id}: {'; '.join(bad)}")
    side = load(store_root)
    side[mechanism_id] = {**{k: key[k] for k in ("op_family", "shape_class", "device") if k in key}, "status": status}
    os.makedirs(store_root, exist_ok=True)
    with open(os.path.join(store_root, SIDECAR), "w", encoding="utf-8") as f:
        json.dump(side, f, indent=1, sort_keys=True)
    return side


def unkeyed(store_root, mechanism_ids) -> list:
    """The ids with no key in the sidecar (reported, never blocked)."""
    side = load(store_root)
    return sorted(m for m in mechanism_ids if m not in side)


def price(entry: dict, surfaces: list, family_of, weight_of=lambda path: 1.0) -> dict:
    """Saving of a hand-written entry from census surfaces: {saving_ms_per_step, score, by_device, n_unpriced, touches}."""
    by_dev, touches, unpriced, total, score = {}, [], 0, 0.0, 0.0
    for t in entry.get("touches") or []:
        hit = [s for s in surfaces if s["path"] == t.get("path") and (s["op"] == t["op"] if t.get("op") else
               family_of(s["path"], s["op"]) == t.get("family"))]
        ms = [s.get("share_ms") if s.get("share_ms") is not None else
              (s["step_ms"] * s["share_of_step"] if s.get("step_ms") and s.get("share_of_step") is not None else None)
              for s in hit]
        ms = [x for x in ms if x is not None]
        if not ms or not isinstance(t.get("fraction"), (int, float)):
            unpriced += 1
            touches.append({**t, "saving_ms_per_step": None})
            continue
        sv = round(t["fraction"] * sum(ms), 3)
        dev = hit[0].get("device") or t["path"]
        by_dev[dev] = round(by_dev.get(dev, 0.0) + sv, 3)
        total, score = total + sv, score + weight_of(t["path"]) * sv
        touches.append({**t, "saving_ms_per_step": sv})
    priced = len(touches) > unpriced
    return {"saving_ms_per_step": round(total, 3) if priced else None, "score": round(score, 3) if priced else None,
            "by_device": by_dev, "n_unpriced": unpriced, "touches": touches}


def hypothesis_entries(hypotheses_dir) -> list:
    """[{id, key}] for every hypothesis file with a journal-key that carries touches (the ones rank() can price)."""
    out = []
    for name in sorted(os.listdir(hypotheses_dir)):
        k = name.endswith(".md") and parse_key(open(os.path.join(hypotheses_dir, name), encoding="utf-8").read())
        if k and k.get("touches"):
            out.append({"id": name[:-3], **k})
    return out
