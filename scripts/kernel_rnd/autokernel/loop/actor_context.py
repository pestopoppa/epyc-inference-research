"""Variable-mode actor context: the bundle is a DIRECTORY, the prompt is its INDEX.

The planner/author prompt used to inline the whole rendered context bundle
(`actors.render_context`): DS41 run 7 sent 95.7k chars, run 8 75.9k after the seat's
diet. opencode's conversation is append-only, so every byte of it sat in every step:
the plain seat opened at 39,451 tokens and peaked at 93,490 on a 98,304-token slot
(run 8), and every arm of the DS41-C20c seat A/B compacted. This is the RLM pattern
adapted to opencode: the prompt carries a compact index (task, reply schema, a table
of contents with sizes, a resolved target card and the few sections the planner
always needs), and the rest of the bundle lives in a per-call directory OUTSIDE the
conversation, where the model pulls only what it needs with its native tools.

MEASURED (static, Qwen3.8-27B vocab via `llama-tokenize --no-escape`): the real run-8
planner prompt is 75,978 chars / 26,293 tokens inline and 15,864 / 5,510 as an index;
run 7's pre-diet prompt (95.7k was its argv-quoted size) is 92,757 / 32,917 -> 16,138 /
5,629. The rest of run 8's 39,451 first-step tokens (~13.2k) is opencode's own system
prompt, tool schemas and the lane's AGENTS.md, which no prompt change can remove.

WHY FILES AND NOT MCP TOOLS. The plain seat (the campaign default) runs bare
`opencode run --auto` with no `OPENCODE_CONFIG`: the global config has no `mcp` block,
so no MCP server exists in that seat. Its native tools do: the DS41 plain transcripts
show `bash`, `read` (1-indexed `offset`/`limit`), `grep` and `glob`, and `read`/`grep`
of paths OUTSIDE `--dir` work, because opencode's `external_directory` default is
`ask` and `--auto` approves every non-denied ask (run 8 read
`/mnt/raid0/llm/llama.cpp-experimental-deepseek41-20260923/...` and `/tmp/...` from a
lane at `.../workers/lane0`). The bounded seat has the same native tools (its
`tool_output` cap applies to them), so one file tree serves both seats.

LOSSLESS BY CONSTRUCTION. Nothing is re-rendered: the bundle is the exact
`render_context` text, cut at its own section headers. The section files concatenate
back to that text byte for byte, and each JSON block the inline prompt carries is also
exploded into a small tree of pretty-printed files (one per key / list item, split
only when large) that implodes back to the identical object. Variable mode therefore
changes WHERE the planner reads the bundle, never WHAT it may read -- the property the
A/B against inline mode needs.

WHAT STAYS INLINE (`INLINE_SECTIONS`), from the DS41 run-7/8 prompts and replies:

* `program` -- only the target-specific directives run.py PREPENDS to program.md
  (scope "half", author source only, the CPU override of the GPU text below). They
  change what a legal proposal is; program.md itself (`program_strategy`, ~19k chars,
  mostly GPU strategy the CPU directive overrides) goes to a file.
* `profile` -- the hotspot table and ranked families. The one complete run-7
  hypothesis took its `target_symbol` verbatim from that table
  (`mul_mat_qX_K_q8_2_X4_T<...DequantizerQ4K_AVX2, 1>`), the critic rejects a
  mechanism "unsupported by the profile", and the ranked-family text is the
  instruction to start with the highest-share mechanism. ~6k chars.
* the directive blocks -- superseded ("consider these FIRST"), the two
  diminishing-returns escapes ("mandatory for this turn"), characterised ("do NOT
  re-measure"), already tried, and the rejection feedback ("answer these"). Each is an
  instruction for THIS turn and each is small (run 8: 32 chars in total).
* a TARGET CARD (new, variable mode only): model, threads, speculation, env, topology
  and build dir resolved from the target JSON. Run 8's first tool call went straight to
  the build tree the target names; the other ~22k chars of that JSON (digests,
  provenance, a repeated full-transfer target) are reference material.

To files, with a SUMMARY inline (`SUMMARY_SECTIONS`): `node_profile` -- the per-op
wall shares from the instrumented sibling build. `render_context` did not print it at
all before 2026-09-24 although the CPU directive says "Read node_profile"; it now renders
in both modes (~3.9k chars on the run-8 observation), and here its head (instrument
caveat + mechanism-family share table) stays in the index while the per-op, weight-path,
host-phase and engram tables are a file named as required reading.

To files: the target JSON, program.md, shared history (12-21k chars, "suggestions
only"), serving observations ("recall") and the operator inbox. The inbox is named as
required reading in the index: run 7's hypothesis cited two of its figures, so the
planner does use it -- the A/B shows whether it reads it when not force-fed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re
import time
import uuid
from typing import Any, Iterable

#: Sibling of the lane worktree (never inside it: a file there would ride into the
#: authored diff and into the author's `git status` ground-truth check).
BUNDLE_DIR = "actor-context"
#: INF-78 OAB-7: the bundle travels to the ORCHESTRATOR as `ChatRequest.context_bundle`
#: and its REPL holds it as the variable `context` (`orchestrator_bundle`). Only the
#: `orchestrator` backend kind honours it; every other kind stays inline.
ORCH_MODE = "orchestrator-variable"
MODES = ("inline", "variable", ORCH_MODE)
#: `seat.arm` suffix on the VB-AK-SEAT call record for a variable-mode call.
ARM_SUFFIX = "+ctx-variable"
#: ... and for an orchestrator-variable call.
ORCH_ARM_SUFFIX = "+ctx-orch-variable"
#: The orchestrator's payload schema (epyc-orchestrator src/repl_environment/context_bundle.py).
ORCH_BUNDLE_SCHEMA = "epyc.orchestrator.context_bundle.v1"

#: (key, header prefixes) in the order `actors.render_context` emits them. The split
#: only honours a header whose order is after the last one matched, and stops matching
#: inside the inbox (always last), so operator prose that happens to contain a header
#: cannot re-open an earlier section. A mis-split is a mislabel, never a loss: the cut
#: points only partition the text.
SECTION_HEADERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("target", ("## Selected target (original launch, model, requests and build)",)),
    ("program", ("## Standing constraints and settled questions (read this first)",)),
    ("superseded", ("## Formed but never measured — consider these FIRST",)),
    ("profile", ("## CPU profile for the selected experimental target",
                 "## Where the device time actually goes (rocprofv3, current champion)")),
    ("node_profile", ("## Node profile — per-op wall SHARES on an instrumented sibling "
                      "of the same anchor",)),
    ("exhausted_families", ("## DIMINISHING-RETURNS ESCAPE — mandatory for this turn",)),
    ("stagnant_families", ("## Family-level diminishing returns — abstraction escape required",)),
    ("characterised", ("## Characterised — do NOT re-measure these",)),
    ("already_tried", ("## Already tried",)),
    ("shared_history", ("## Shared historical mechanisms — transfer NOT established",)),
    ("serving_observations", ("## Original serving observations — recall, not qualified gains",)),
    ("hypothesis_rejections", ("## Your hypothesis was rejected — answer these, do not re-derive",)),
    ("patch_rejections", ("## Your patch was rejected — answer these, do not re-derive",)),
    ("inbox", ("## Operator suggestions (async; use if relevant)",)),
)
#: Sections whose body carries one ```json block the inline prompt shows.
JSON_SECTIONS = frozenset({"target", "shared_history", "serving_observations"})
#: Kept verbatim in the prompt (see the module docstring for the evidence).
INLINE_SECTIONS = frozenset({"preamble", "program", "superseded", "profile",
                             "exhausted_families", "stagnant_families", "characterised",
                             "already_tried", "hypothesis_rejections", "patch_rejections"})
#: File sections whose HEAD (everything before its second `### ` sub-heading) is also
#: shown in the index, followed by a pointer to the full file. `node_profile` renders at
#: ~3.9k chars on the run-8 observation (a quarter of the ~16k index): its head -- the
#: instrument caveat and the mechanism-family wall-share table, ~1.2k -- is what orients
#: a proposal (dense vs expert matmul), while the per-op / weight-path / host-phase /
#: engram tables and the limitations are reference the planner pulls when it needs them.
#: A section with fewer than two sub-headings (an absent profile: status + reason) is
#: its own summary, shown whole.
SUMMARY_SECTIONS = frozenset({"node_profile"})
#: Required reading named in the index for a file section, beyond the heading heuristic.
REQUIRED_SECTIONS = {
    "node_profile": "node_profile -- the CPU directive says to read it: per-op, weight-path, "
                    "host-phase and engram SHARES for this anchor",
}
#: A JSON container is split into a directory only past this size and above this depth;
#: smaller ones stay one pretty-printed file. On the run-8 bundle that is one file per
#: shared-history row (~2k chars) and per target key (<= 11.5k), 30 JSON files: a
#: first cut at 1.5k / depth 3 made 196 files of a few bytes each, one call per field.
SPLIT_MIN_CHARS = 6000
MAX_SPLIT_DEPTH = 2
TOC_MAX_CHILDREN = 14
#: Sub-headings listed per file-only markdown section (file:line jump table).
TOC_MAX_HEADINGS = 40

_JSON_BLOCK = re.compile(r"```json\n(.*)\n```", re.S)
_POINTER = re.compile(r"^<same as (\$[^>]*)>$")
_POINTER_STEP = re.compile(r"\.([^.\[\]]+)|\[(\d+)\]")
_HEADING = re.compile(r"^(#{1,4}) (.+)$")
_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class Section:
    key: str
    text: str

    @property
    def inline(self) -> bool:
        return self.key in INLINE_SECTIONS

    @property
    def summarized(self) -> bool:
        return self.key in SUMMARY_SECTIONS


def section_summary(section: Section) -> str:
    """The head of a `SUMMARY_SECTIONS` section: its text up to the second `### `
    line (outside code fences), or the whole section when it has fewer than two."""
    starts = []
    offset = 0
    fence = False
    for line in section.text.splitlines(keepends=True):
        if line.startswith("```"):
            fence = not fence
        elif not fence and line.startswith("### "):
            starts.append(offset)
        offset += len(line)
    return section.text if len(starts) < 2 else section.text[:starts[1]]


def split_sections(text: str) -> list[Section]:
    """Cut the rendered context at its own section headers. `''.join` of the texts
    is the input, always; `program` is further cut at program.md's first H1."""
    cuts: list[tuple[int, str]] = []
    order = {key: i for i, (key, _) in enumerate(SECTION_HEADERS)}
    last = -1
    offset = 0
    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\n")
        if last < order["inbox"]:
            for key, prefixes in SECTION_HEADERS:
                if order[key] > last and bare in prefixes:
                    cuts.append((offset, key))
                    last = order[key]
                    break
        offset += len(line)
    sections: list[Section] = []
    if not cuts or cuts[0][0] > 0:
        sections.append(Section("preamble", text[:cuts[0][0] if cuts else len(text)]))
    for i, (start, key) in enumerate(cuts):
        end = cuts[i + 1][0] if i + 1 < len(cuts) else len(text)
        body = text[start:end]
        if key == "program":
            sections.extend(_split_program(body))
        else:
            sections.append(Section(key, body))
    return [s for s in sections if s.text]


def _split_program(body: str) -> list[Section]:
    """The run.py directives before program.md's first `# ` line stay `program`; the
    strategy document itself becomes `program_strategy`."""
    match = re.search(r"(?m)^# ", body)
    if match is None:
        return [Section("program", body)]
    return [Section("program", body[:match.start()]),
            Section("program_strategy", body[match.start():])]


def json_payload(section: Section) -> Any:
    """The JSON object a JSON section shows inline (None when it carries none)."""
    match = _JSON_BLOCK.search(section.text)
    return json.loads(match.group(1)) if match else None


def _dump(value: Any) -> str:
    """Exactly how `render_context` prints a JSON block."""
    return json.dumps(value, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# JSON tree: explode / implode
# ---------------------------------------------------------------------------
def _names(keys: Iterable[str]) -> list[str]:
    used: set[str] = set()
    out = []
    for key in keys:
        base = (_SAFE.sub("_", key).strip("._") or "_")[:60]
        name, n = base, 1
        while name.lower() in used or name.lower() == "_index":
            n += 1
            name = f"{base}~{n}"
        used.add(name.lower())
        out.append(name)
    return out


def explode(value: Any, path: Path, *, depth: int = 0) -> None:
    """Write `value` at `path` (a `.json` file, or a directory with `_index.json`
    naming its children in order). Lossless: `implode(path) == value`."""
    if not _is_dir(value, depth):
        path.with_name(path.name + ".json").write_text(_dump(value) + "\n", encoding="utf-8")
        return
    path.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        keys = sorted(value)
        names = _names(keys)
        index = {"type": "object", "entries": [[k, n] for k, n in zip(keys, names)]}
        children = [(n, value[k]) for k, n in zip(keys, names)]
    else:
        width = max(4, len(str(len(value) - 1)))
        names = [f"{i:0{width}d}" for i in range(len(value))]
        index = {"type": "array", "entries": names}
        children = list(zip(names, value))
    (path / "_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    for name, child in children:
        explode(child, path / name, depth=depth + 1)


def implode(path: Path) -> Any:
    """Inverse of `explode` (`path` without the `.json` suffix)."""
    if path.is_dir():
        index = json.loads((path / "_index.json").read_text(encoding="utf-8"))
        if index["type"] == "object":
            return {key: implode(path / name) for key, name in index["entries"]}
        return [implode(path / name) for name in index["entries"]]
    return json.loads(path.with_name(path.name + ".json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------
def _k(chars: int) -> str:
    return f"{chars / 1000:.1f}k" if chars >= 1000 else str(chars)


def _is_dir(value: Any, depth: int) -> bool:
    """Whether `explode` writes `value` (at `depth`) as a directory."""
    return (isinstance(value, (dict, list)) and bool(value) and depth < MAX_SPLIT_DEPTH
            and len(_dump(value)) > SPLIT_MIN_CHARS)


def _json_toc(value: Any, label: str, *, depth: int = 0) -> list[str]:
    """Key -> size lines for every directory level `explode` writes; a key ending in
    `/` is a directory, anything else one `.json` file. Largest-first beyond the cap."""
    if not isinstance(value, (dict, list)) or depth >= MAX_SPLIT_DEPTH:
        return []
    items = (sorted(value.items()) if isinstance(value, dict)
             else [(f"[{i}]", v) for i, v in enumerate(value)])
    ranked = items if len(items) <= TOC_MAX_CHILDREN else sorted(
        items, key=lambda kv: -len(_dump(kv[1])))[:TOC_MAX_CHILDREN]
    parts = [f"{key}{'/' if _is_dir(child, depth + 1) else ''} {_k(len(_dump(child)))}"
             for key, child in ranked]
    more = len(items) - len(ranked)
    lines = [f"{'  ' * depth}- {label}: " + " · ".join(parts)
             + (f" · (+{more} more)" if more else "")]
    for key, child in ranked:
        if _is_dir(child, depth + 1):
            lines.extend(_json_toc(child, f"{label}.{key}" if not key.startswith("[")
                                   else f"{label}{key}", depth=depth + 1))
    return lines


def _resolve(root: Any, value: Any, _hops: int = 0) -> Any:
    """Follow a `<same as $.path>` pointer left by `_dedupe_subtrees`."""
    match = _POINTER.match(value) if isinstance(value, str) else None
    if match is None or _hops > 8:
        return value
    node = root
    for key, index in _POINTER_STEP.findall(match.group(1)[1:]):
        try:
            node = node[int(index)] if index else node[key]
        except (KeyError, IndexError, TypeError, ValueError):
            return value
    return _resolve(root, node, _hops + 1)


def _get(root: Any, *path: str) -> Any:
    node = root
    for key in path:
        node = _resolve(root, node)
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return _resolve(root, node)


#: (label, path) of the target facts every proposal needs, resolved through pointers.
CARD_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("scope", ("scope",)),
    ("common CPU scope", ("common_cpu_scope", "scope")),
    ("backend", ("recipe", "backend")),
    ("model", ("recipe", "template", "model")),
    ("threads", ("recipe", "template", "threads")),
    ("ctx / batch / ubatch", ()),
    ("speculation", ("recipe", "template", "spec_decode")),
    ("env", ("recipe", "template", "env")),
    ("topology", ("recipe", "topology_prefix")),
    ("build dir (anchor binary)", ("recipe", "build_dir")),
    ("build recipe", ("build_recipe", "name")),
    ("frozen requests", ("requests",)),
    ("hotspot status", ("hotspot_status",)),
)


#: OAB-11: the card's build-dir label when the seat runs the lane guard. The plain
#: planner read the anchor's SOURCE because the only tree the prompt named was the
#: anchor build's; under the guard the card names the lane first as THE source tree.
GUARDED_BUILD_LABEL = "build dir (anchor BINARY: read-only, binaries only; never read source there)"


def target_card(target: Any, *, lane: Path | str | None = None) -> list[str]:
    """The resolved facts every proposal needs. With `lane` (the lane guard), the first
    row names the lane as THE source tree and the build dir is labelled a binary."""
    if not isinstance(target, dict):
        return []
    lines = [f"- source tree (THE tree to read and cite; your working directory): {lane}"
             ] if lane is not None else []
    for label, path in CARD_FIELDS:
        if lane is not None and path == ("recipe", "build_dir"):
            label = GUARDED_BUILD_LABEL
        if not path:
            dims = [_get(target, "recipe", "template", key) for key in ("ctx", "batch", "ubatch")]
            value = None if all(d is None for d in dims) else " / ".join(str(d) for d in dims)
        else:
            value = _get(target, *path)
        if value is None:
            continue
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            shown = " ".join(value)
        elif isinstance(value, (dict, list)):
            shown = json.dumps(value, sort_keys=True)
        else:
            shown = str(value)
        lines.append(f"- {label}: {shown}")
    return lines


def _headings(text: str) -> list[tuple[int, str]]:
    out = []
    fence = False
    for n, line in enumerate(text.splitlines(), 1):
        if line.startswith("```"):
            fence = not fence
        match = None if fence else _HEADING.match(line.lstrip("- ") if line.startswith("- #")
                                                   else line)
        if match:
            out.append((n, line.lstrip("- ").strip()))
    return out


_MUST_READ = re.compile(r"settled|not this loop|do not|standing rules|dead", re.I)


@dataclass
class Bundle:
    """One materialized call: where it lives, the index prompt text, its files."""
    directory: Path
    sections: list[Section]
    index: str
    files: dict[str, dict[str, Any]] = field(default_factory=dict)
    inline_chars: int = 0

    def seal(self, prompt: str) -> None:
        """Bind the bundle to the exact prompt sent (`manifest.json`): the call
        record's `prompt.sha256` joins to it, and `inline_equivalent_prompt_chars`
        is what inline mode would have sent for the same call."""
        manifest = {
            "schema": "epyc.autokernel.actor_context_bundle.v1",
            "mode": "variable",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prompt": {"sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                       "chars": len(prompt)},
            "inline_equivalent_prompt_chars": len(prompt) - len(self.index) + self.inline_chars,
            "context_chars": self.inline_chars,
            "index_chars": len(self.index),
            "sections": [{"key": s.key, "chars": len(s.text), "inline": s.inline}
                         for s in self.sections],
            "files": self.files,
        }
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def materialize(context_text: str, parent: Path, *, role: str, scope,
                lane: Path | None = None) -> Bundle:
    """Write one call's bundle under `parent/` and return its index prompt text.
    `lane` (lane guard on) makes the target card name it as THE source tree.

    The bundle directory is SCRATCH, allocated from `scope` (a `scratch.Scope`, the
    actor call's): it is marked, journalled, and released when the call ends. Its
    evidence -- the manifest's prompt digest and per-file digests -- rides on the call
    record, so nothing reads the directory after the call."""
    parent = Path(parent)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    name = f"{stamp}-{role}-{uuid.uuid4().hex[:8]}"
    directory = scope.dir(BUNDLE_DIR, name, at=parent / name)
    sections = split_sections(context_text)
    (directory / "sections").mkdir()
    files: dict[str, dict[str, Any]] = {}
    rows = []
    payloads: dict[str, Any] = {}
    for n, section in enumerate(sections, 1):
        rel = f"sections/{n:02d}-{section.key}.md"
        (directory / rel).write_text(section.text, encoding="utf-8")
        rows.append((section, rel))
        if section.key in JSON_SECTIONS:
            payload = json_payload(section)
            if payload is not None:
                payloads[section.key] = payload
                explode(payload, directory / "json" / section.key)
    from .scratch import MARKER
    for path in sorted(p for p in directory.rglob("*") if p.is_file() and p.name != MARKER):
        data = path.read_bytes()
        files[str(path.relative_to(directory))] = {
            "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    index = _index_text(directory, rows, payloads, lane=lane)
    (directory / "INDEX.md").write_text(index, encoding="utf-8")
    return Bundle(directory=directory, sections=sections, index=index, files=files,
                  inline_chars=len(context_text))


def _index_text(directory: Path, rows: list[tuple[Section, str]],
                payloads: dict[str, Any], *, lane: Path | None = None) -> str:
    total = sum(len(s.text) for s, _ in rows)
    on_disk = sum(len(s.text) for s, _ in rows if not s.inline)
    out = [
        "## Context bundle — VARIABLE mode: most of it is on disk, not in this prompt",
        f"Bundle directory: `{directory}`",
        f"The full planning context is {total:,} chars. The sections marked INLINE "
        f"below are reproduced in full at the end of this index; the other {on_disk:,} "
        "chars are files in that directory. Read only what you need with your file "
        "tools: `read` (1-indexed `offset`/`limit`), `grep` with a `path`, `glob`. The "
        "files are exactly what an inline prompt would have shown -- nothing added, "
        "nothing dropped -- and they stay on disk for this whole call: after a context "
        f"compaction, re-read `{directory}/INDEX.md` instead of working from memory. "
        "`sections/` holds each section verbatim; `json/<section>/` holds the same JSON "
        "split per key (a directory's `_index.json` lists its children in order).",
        "",
    ]
    must: list[str] = []
    for section, rel in rows:
        if section.inline:
            continue
        if section.key == "inbox":
            must.append(f"- `{rel}` — operator suggestions for this campaign (read it all)")
        if section.key in REQUIRED_SECTIONS:
            must.append(f"- `{rel}` — {REQUIRED_SECTIONS[section.key]}")
        for line, heading in _headings(section.text):
            if section.key != "inbox" and _MUST_READ.search(heading):
                must.append(f"- `{rel}` L{line}: {heading}")
    if must:
        out.append("Read these before you propose. The critic reviews your proposal "
                   "against the FULL bundle and rejects one that contradicts it:")
        out.extend(must)
        out.append("")
    out.append("| # | section | file | chars | lines | in prompt |")
    out.append("|---|---|---|---|---|---|")
    for n, (section, rel) in enumerate(rows, 1):
        extra = ""
        if section.key in payloads:
            extra = (f" (+ `json/{section.key}/`)" if _is_dir(payloads[section.key], 0)
                     else f" (+ `json/{section.key}.json`)")
        placement = ("INLINE" if section.inline else
                     "file + summary" if section.summarized else "file")
        out.append(f"| {n} | {section.key} | `{rel}`{extra} | {len(section.text):,} | "
                   f"{section.text.count(chr(10))} | {placement} |")
    out.append("")
    headed = [(section, rel) for section, rel in rows
              if not section.inline and section.key not in payloads]
    if headed:
        out.append("Headings in the file-only markdown sections (file → line):")
        for section, rel in headed:
            heads = _headings(section.text)
            shown = heads[:TOC_MAX_HEADINGS]
            out.append(f"- `{rel}`: " + "; ".join(f"L{n} {h}" for n, h in shown)
                       + (f"; (+{len(heads) - len(shown)} more)" if len(heads) > len(shown)
                          else ""))
        out.append("")
    if payloads:
        out.append("JSON keys with sizes (chars, as pretty-printed):")
        for key, payload in payloads.items():
            out.extend(_json_toc(payload, key))
        out.append("")
    card = target_card(payloads.get("target"), lane=lane)
    if card:
        out.append("Target card (resolved from the target section; full JSON in `json/target/`):")
        out.extend(card)
        out.append("")
    out.append("=== INLINE sections (verbatim) ===")
    out.append("")
    index = "\n".join(out)
    parts = []
    for section, rel in rows:
        if section.inline:
            parts.append(section.text)
        elif section.summarized:
            head = section_summary(section)
            if head == section.text:
                parts.append(section.text)
            else:
                parts.append(head.rstrip("\n") + "\n"
                             f"(summary -- the rest of this section, "
                             f"{len(section.text) - len(head):,} chars, is "
                             f"`{directory}/{rel}`)\n\n")
    return index + "".join(parts).rstrip("\n")


# ---------------------------------------------------------------------------
# INF-78 OAB-7: the bundle as the orchestrator REPL's `context` variable
# ---------------------------------------------------------------------------
#: One line per section for the orchestrator's section table ("about" column).
SECTION_ABOUT = {
    "preamble": "the task line",
    "target": "target JSON: launch, model, requests, build (card below)",
    "program": "run.py scope directives for this target",
    "program_strategy": "program.md strategy (GPU text; the CPU directive overrides it)",
    "superseded": "formed but never measured: consider FIRST",
    "profile": "CPU/GPU profile: hotspot table and ranked families",
    "node_profile": "per-op wall SHARES (summary in prompt; required reading)",
    "exhausted_families": "diminishing-returns escape: mandatory this turn",
    "stagnant_families": "family-level diminishing returns: escape required",
    "characterised": "characterised: do NOT re-measure",
    "already_tried": "already tried",
    "shared_history": "shared historical mechanisms (suggestions only)",
    "serving_observations": "original serving observations (recall)",
    "hypothesis_rejections": "your hypothesis was rejected: answer these",
    "patch_rejections": "your patch was rejected: answer these",
    "inbox": "operator suggestions (required reading)",
}


@dataclass
class OrchestratorBundle:
    """One orchestrator-variable call: the payload the CLI ships as
    `ChatRequest.context_bundle`, and the index text that replaces the context block
    in the prompt. Nothing is written to disk here -- the backend writes the payload
    beside the lane (`actor_orchestrator`), because the orchestrator, not the model,
    reads it."""
    sections: list[Section]
    payload: dict[str, Any]
    index: str
    inline_chars: int = 0

    def seal(self, prompt: str, *, role: str | None = None) -> None:
        """Bind the payload to the exact prompt sent (its `manifest`, echoed by digest
        in the orchestrator's `context_pulls`)."""
        self.payload["manifest"] = {
            "schema": "epyc.autokernel.actor_context_bundle.v1",
            "mode": ORCH_MODE,
            "role": role,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prompt": {"sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                       "chars": len(prompt)},
            "inline_equivalent_prompt_chars": len(prompt) - len(self.index) + self.inline_chars,
            "context_chars": self.inline_chars,
            "index_chars": len(self.index),
            "sections": [{"key": s.key, "chars": len(s.text), "inline": s.inline}
                         for s in self.sections],
        }


def orchestrator_bundle(context_text: str, *, role: str,
                        lane: Path | str | None = None) -> OrchestratorBundle:
    """Split `render_context`'s text into the orchestrator payload (every section,
    verbatim, in order: the section texts concatenate back to `context_text`) and the
    index the prompt carries instead (required reading, JSON keys, target card, and the
    same evidence-chosen INLINE set as file-variable mode)."""
    sections = split_sections(context_text)
    payloads: dict[str, Any] = {}
    entries = []
    for section in sections:
        payload = json_payload(section) if section.key in JSON_SECTIONS else None
        if payload is not None:
            payloads[section.key] = payload
        about = SECTION_ABOUT.get(section.key, "")
        entries.append({"name": section.key, "text": section.text,
                        "kind": "json" if payload is not None else "text",
                        "inline": section.inline, "description": about})
    index = _orch_index_text(sections, payloads, lane=lane)
    return OrchestratorBundle(sections=sections,
                              payload={"schema": ORCH_BUNDLE_SCHEMA, "sections": entries},
                              index=index, inline_chars=len(context_text))


def _orch_index_text(sections: list[Section], payloads: dict[str, Any], *,
                     lane: Path | str | None = None) -> str:
    total = sum(len(s.text) for s in sections)
    held = sum(len(s.text) for s in sections if not s.inline)
    out = [
        "## Context bundle -- ORCHESTRATOR mode: it is the REPL variable `context`",
        f"The full planning context is {total:,} chars in {len(sections)} sections. The "
        "sections marked INLINE are reproduced in full at the end of this index; the other "
        f"{held:,} chars are in `context` and NOT in this prompt. Pull what you need -- "
        "context.get('<section>'), context.grep('<regex>', section='<section>'), "
        "context.json('target.recipe') -- into variables, and print only the slice you "
        "use: pulls cost nothing, printed text is capped per turn. The sections are exactly "
        "what an inline prompt would have shown, nothing added, nothing dropped. The "
        "orchestrator's section table (sizes) follows this prompt.",
        "",
    ]
    must: list[str] = []
    for section in sections:
        if section.inline:
            continue
        if section.key == "inbox":
            must.append("- `inbox` -- operator suggestions for this campaign (read it all)")
        if section.key in REQUIRED_SECTIONS:
            must.append(f"- `{section.key}` -- {REQUIRED_SECTIONS[section.key]}")
        for line, heading in _headings(section.text):
            if section.key != "inbox" and _MUST_READ.search(heading):
                must.append(f"- `{section.key}` line {line}: {heading}")
    if must:
        out.append("Read these before you propose (line numbers are within "
                   "context.get('<section>')). The critic reviews your proposal against the "
                   "FULL bundle and rejects one that contradicts it:")
        out.extend(must)
        out.append("")
    if payloads:
        out.append("JSON keys with sizes (chars, as pretty-printed); reach one with "
                   "context.json('<section>.<key>'):")
        for key, payload in payloads.items():
            out.extend(_json_toc(payload, key))
        out.append("")
    card = target_card(payloads.get("target"), lane=lane)
    if card:
        out.append("Target card (resolved from the target section; the full JSON is "
                   "context['target']):")
        out.extend(card)
        out.append("")
    out.append("=== INLINE sections (verbatim) ===")
    out.append("")
    index = "\n".join(out)
    parts = []
    for section in sections:
        if section.inline:
            parts.append(section.text)
        elif section.summarized:
            head = section_summary(section)
            if head == section.text:
                parts.append(section.text)
            else:
                parts.append(head.rstrip("\n") + "\n"
                             f"(summary -- the rest of this section, "
                             f"{len(section.text) - len(head):,} chars, is "
                             f"context.get(\"{section.key}\"))\n\n")
    return index + "".join(parts).rstrip("\n")


__all__ = ["ARM_SUFFIX", "BUNDLE_DIR", "Bundle", "GUARDED_BUILD_LABEL", "INLINE_SECTIONS", "JSON_SECTIONS", "MODES",
           "REQUIRED_SECTIONS", "SECTION_HEADERS", "SUMMARY_SECTIONS", "Section", "explode",
           "implode", "json_payload", "materialize", "section_summary", "split_sections",
           "target_card", "ORCH_ARM_SUFFIX", "ORCH_BUNDLE_SCHEMA", "ORCH_MODE",
           "OrchestratorBundle", "orchestrator_bundle"]
