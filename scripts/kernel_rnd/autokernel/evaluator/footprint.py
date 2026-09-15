#!/usr/bin/env python3
"""Static patch-footprint producer for the cross-workload keep screen.

The producer is deliberately compiler-only: unified diff, ``compile_commands.json``
and compiler depfiles are its authorities.  It neither builds nor executes a model.
The result is a prediction used to select proof work; it is never an inertness proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import surface

SCHEMA = "epyc.autokernel.patch_footprint.v1"
HUNK_KINDS = ("body", "predicate", "table", "decl", "build", "comment")
CHANGE_CLASSES = ("kernel_body", "dispatch_predicate", "shared_machinery",
                  "recipe", "build", "mixed", "opaque")


class FootprintError(ValueError):
    """The static footprint could not be produced without under-approximation."""


def _uniq(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


@dataclass(frozen=True)
class Hunk:
    file: str
    old_range: str
    new_range: str
    kind: str
    text: str

    @property
    def hunk_id(self) -> str:
        digest = hashlib.sha256(self.text.encode()).hexdigest()[:16]
        return f"{self.file}:{self.old_range}:{self.new_range}:{digest}"

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file, "old_range": self.old_range,
                "new_range": self.new_range, "kind": self.kind,
                "hunk_id": self.hunk_id}


@dataclass(frozen=True)
class DispatchSite:
    file: str
    line: int
    predicate_text: str
    discriminators: Mapping[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file, "line": self.line,
                "predicate_text": self.predicate_text,
                "discriminators": {k: list(v) for k, v in self.discriminators.items()}}


@dataclass(frozen=True)
class EnvKnob:
    name: str
    read_site: str
    default_before: str
    default_after: str

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "read_site": self.read_site,
                "default_before": self.default_before,
                "default_after": self.default_after}


@dataclass(frozen=True)
class PatchFootprint:
    source_tree: str
    production_base_commit: str
    candidate_source_commit: str
    patch_bundle_sha256: str
    actual_files: tuple[str, ...]
    actual_hunk_ids: tuple[str, ...]
    actual_symbols: tuple[str, ...]
    feature_flag_assignments: tuple[tuple[str, str], ...]
    dispatch_predicates: tuple[str, ...]
    mechanism_id: str
    change_class: str
    hunks: tuple[Hunk, ...]
    dispatch_sites: tuple[DispatchSite, ...]
    env_knobs: tuple[EnvKnob, ...]
    build_flags: tuple[Mapping[str, Any], ...]
    objects_expected_changed: Mapping[str, tuple[str, ...]]
    opaque_reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        # The composition keys intentionally retain their exact names so this record
        # can feed CompositionEvidence without an actor-authored translation layer.
        return {
            "schema": SCHEMA,
            "range": {"parent": self.production_base_commit,
                      "child": self.candidate_source_commit,
                      "tree": self.source_tree},
            "source_tree": self.source_tree,
            "production_base_commit": self.production_base_commit,
            "candidate_source_commit": self.candidate_source_commit,
            "patch_bundle_sha256": self.patch_bundle_sha256,
            "actual_files": list(self.actual_files),
            "actual_hunk_ids": list(self.actual_hunk_ids),
            "actual_symbols": list(self.actual_symbols),
            "feature_flag_assignments": dict(self.feature_flag_assignments),
            "dispatch_predicates": list(self.dispatch_predicates),
            "mechanism_id": self.mechanism_id,
            "change_class": self.change_class,
            "hunks": [h.to_dict() for h in self.hunks],
            "dispatch_sites": [s.to_dict() for s in self.dispatch_sites],
            "env_knobs": [k.to_dict() for k in self.env_knobs],
            "build_flags": list(self.build_flags),
            "objects_expected_changed": {k: list(v) for k, v in sorted(self.objects_expected_changed.items())},
            "opaque_reasons": list(self.opaque_reasons),
        }

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                                         separators=(",", ":")).encode()).hexdigest()


_HUNK = re.compile(r"^@@ -(\d+(?:,\d+)?) \+(\d+(?:,\d+)?) @@(.*)$")
_FUNC = re.compile(r"(?:^|\s)([A-Za-z_][\w:<>~]*)\s*\([^;{}]*\)\s*(?:const\s*)?\{?\s*$")
_ENV = re.compile(r'(?:getenv|secure_getenv)\s*\(\s*"([A-Z][A-Z0-9_]*)"\s*\)')
_TYPE = re.compile(r"\bGGML_TYPE_[A-Z0-9_]+\b")
_OP = re.compile(r"\bGGML_OP_[A-Z0-9_]+\b")
_BACKEND = re.compile(r"\b(?:GGML_BACKEND_[A-Z0-9_]+|GGML_USE_(?:HIP|CUDA|METAL|VULKAN))\b")
_ARCH = re.compile(r"\b(?:GGML_CUDA_CC_[A-Z0-9_]+|gfx\d+[a-z0-9]*)\b", re.I)
_SHAPE = re.compile(r"\b(?:ne\d*\s*(?:\[[0-3]\])?|ncols_dst|nrows|cols_per_block)\s*(?:[<>=!]+)\s*[^&|;)]+")
_THREAD = re.compile(r"\b(?:nth|nthreads?|thread_count)\s*(?:[<>=!]+)\s*[^&|;)]+")


def _kind(path: str, body: str) -> str:
    changed = "\n".join(x[1:] for x in body.splitlines() if x[:1] in "+-")
    if path.endswith(("CMakeLists.txt", ".cmake", ".mk")) or "target_compile" in changed:
        return "build"
    code = [x.strip() for x in changed.splitlines() if x.strip()]
    if code and all(x.startswith(("//", "/*", "*", "# ")) for x in code):
        return "comment"
    if re.search(r"\b(if|else if|switch|case)\b|#\s*(?:if|ifdef|ifndef)", changed):
        return "predicate"
    if re.search(r"\b(?:static|constexpr|const)\b[^;=]*[=\[{]", changed):
        return "table"
    if code and all(re.match(r"(?:template\s*<|(?:static\s+)?(?:bool|int|void|struct|class|enum)\b)", x)
                    for x in code):
        return "decl"
    return "body"


def parse_unified_diff(text: str) -> tuple[Hunk, ...]:
    if not isinstance(text, str):
        raise FootprintError("diff must be text")
    path = ""
    pending: list[str] = []
    ranges: tuple[str, str] | None = None
    out: list[Hunk] = []
    for line in text.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
        match = _HUNK.match(line)
        if match:
            if ranges is not None:
                body = "\n".join(pending)
                out.append(Hunk(path, ranges[0], ranges[1], _kind(path, body), body))
            ranges = (match.group(1), match.group(2))
            pending = [line]
        elif ranges is not None:
            if line.startswith("diff --git "):
                body = "\n".join(pending)
                out.append(Hunk(path, ranges[0], ranges[1], _kind(path, body), body))
                ranges = None
                pending = []
            else:
                pending.append(line)
    if ranges is not None:
        body = "\n".join(pending)
        out.append(Hunk(path, ranges[0], ranges[1], _kind(path, body), body))
    if text.strip() and not out:
        raise FootprintError("non-empty diff contains no parseable hunks")
    return tuple(out)


def _disc(text: str) -> dict[str, tuple[str, ...]]:
    return {
        "types": _uniq(_TYPE.findall(text)),
        "ops": _uniq(_OP.findall(text)),
        "shape_terms": _uniq(m.group(0).strip() for m in _SHAPE.finditer(text)),
        "backend": _uniq(_BACKEND.findall(text)),
        "arch": _uniq(m.group(0) for m in _ARCH.finditer(text)),
        "threads": _uniq(m.group(0).strip() for m in _THREAD.finditer(text)),
    }


def _dispatch_sites(hunks: Sequence[Hunk]) -> tuple[DispatchSite, ...]:
    out: list[DispatchSite] = []
    for h in hunks:
        # A switch label and its return predicate commonly occupy adjacent lines
        # (the historical MMVQ crossover is exactly that shape).  Preserve the
        # site line, but union discriminators over the hunk so the case's type and
        # consequent's shape term are not artificially separated.
        hunk_disc = _disc(h.text)
        line_no = int(h.new_range.split(",", 1)[0])
        for raw in h.text.splitlines()[1:]:
            if raw.startswith("-"):
                continue
            code = raw[1:] if raw[:1] in "+ " else raw
            stripped = code.strip()
            local_disc = _disc(stripped)
            discriminators = {
                key: _uniq((*local_disc[key], *hunk_disc[key])) for key in local_disc
            }
            if (re.search(r"\b(if|else if|case|switch)\b|#\s*(if|ifdef|ifndef)", stripped)
                    and any(discriminators.values())):
                out.append(DispatchSite(h.file, line_no, stripped, discriminators))
            if not raw.startswith("+"):
                line_no += 1
            elif raw.startswith("+"):
                line_no += 1
    return tuple(out)


def _default_for(name: str, lines: Sequence[str]) -> str:
    window = "\n".join(lines)
    # Prefer the explicit NULL/empty ternary at the getenv read site.
    m = re.search(r"(?:NULL|\\0)[^?]*\?\s*(true|false|[01])\s*:", window)
    if m:
        return "ON" if m.group(1) in ("true", "1") else "OFF"
    # Marker strings are auditable build metadata and settle otherwise indirect reads.
    if re.search(rf"DEFAULT_ON=[^\n\"]*\b{re.escape(name)}\b", window):
        return "ON"
    if re.search(rf"DEFAULT_(?:OFF|INERT)=[^\n\"]*\b{re.escape(name)}\b", window):
        return "OFF"
    return "UNKNOWN"


def _env_knobs(hunks: Sequence[Hunk]) -> tuple[EnvKnob, ...]:
    found: dict[tuple[str, str], EnvKnob] = {}
    for h in hunks:
        lines = h.text.splitlines()
        old = [x[1:] for x in lines if not x.startswith("+")]
        new = [x[1:] for x in lines if not x.startswith("-")]
        names = set(_ENV.findall("\n".join(old + new)))
        # Build-marker-only changes name the same effective defaults and are useful
        # when initialization and getenv reads land in separate hunks.
        names.update(re.findall(r"\bGGML_[A-Z0-9_]+\b", "\n".join(lines)))
        for name in names:
            if not (_ENV.search("\n".join(lines)) or "DEFAULT_" in "\n".join(lines)):
                continue
            before, after = _default_for(name, old), _default_for(name, new)
            if before != after or _ENV.search("\n".join(lines)):
                found[(h.file, name)] = EnvKnob(name, f"{h.file}:{h.new_range.split(',')[0]}",
                                                before, after)
    return tuple(found[k] for k in sorted(found))


def _compile_objects(compile_commands: Sequence[Mapping[str, Any]], files: Sequence[str]) -> tuple[str, ...]:
    wanted = set(files)
    out: set[str] = set()
    for entry in compile_commands:
        src = str(entry.get("file", ""))
        if not any(src == f or src.endswith("/" + f) for f in wanted):
            continue
        obj = entry.get("output")
        if not obj:
            argv = entry.get("arguments")
            if not isinstance(argv, list):
                argv = str(entry.get("command", "")).split()
            if "-o" in argv and argv.index("-o") + 1 < len(argv):
                obj = argv[argv.index("-o") + 1]
        if obj:
            out.add(str(obj))
    return _uniq(out)


def produce_patch_footprint(*, source_tree: str, parent: str, child: str,
                            diff_text: str, compile_commands: Sequence[Mapping[str, Any]],
                            depfiles: Mapping[str, str], variant: str = "default") -> PatchFootprint:
    hunks = parse_unified_diff(diff_text)
    files = _uniq(h.file for h in hunks)
    compile_objects = set(_compile_objects(compile_commands, files))
    dep_objects: set[str] = set()
    opaque: list[str] = []
    for name, text in depfiles.items():
        for edge in surface.parse_make_depfile(text, origin_ref=name):
            if any(p in files or any(p.endswith("/" + f) for f in files) for p in edge.prerequisites):
                dep_objects.add(edge.target)
    objects = _uniq(compile_objects | dep_objects)
    for f in files:
        mapped = any(str(e.get("file", "")).endswith(f) for e in compile_commands)
        mapped |= any(f in edge.prerequisites or any(p.endswith("/" + f) for p in edge.prerequisites)
                      for name, text in depfiles.items()
                      for edge in surface.parse_make_depfile(text, origin_ref=name))
        if not mapped:
            opaque.append(f"unmapped_touched_file:{f}")

    symbols: set[str] = set()
    for h in hunks:
        header = h.text.splitlines()[0] if h.text else ""
        context = header.split("@@", 2)[-1].strip()
        match = _FUNC.search(context)
        if match:
            symbols.add(f"{h.file}:{match.group(1)}")
    sites = _dispatch_sites(hunks)
    predicates = _uniq(s.predicate_text for s in sites)
    knobs = _env_knobs(hunks)
    kinds = {h.kind for h in hunks}
    if "build" in kinds:
        change_class = "build" if len(kinds) == 1 else "mixed"
    elif sites:
        change_class = "dispatch_predicate" if kinds <= {"predicate", "comment"} else "mixed"
    elif any("ggml-cpu" in f and not f.endswith((".cu", ".cuh")) for f in files):
        change_class = "shared_machinery"
    elif any(f.endswith((".cu", ".cuh")) for f in files):
        change_class = "kernel_body"
    else:
        change_class = "opaque" if opaque else "kernel_body"
    flags = tuple((k.name, k.default_after) for k in knobs if k.default_after != "UNKNOWN")
    digest = hashlib.sha256(diff_text.encode()).hexdigest()
    return PatchFootprint(
        source_tree, parent, child, digest, files, _uniq(h.hunk_id for h in hunks),
        _uniq(symbols), tuple(sorted(flags)), predicates, "static_patch_footprint.v1",
        change_class, hunks, sites, knobs, (), {variant: objects}, _uniq(opaque))


def derive_affected_surface_from_footprint(
        footprint: PatchFootprint, *, indexes: Sequence[surface.BuildDependencyIndex],
        registrations: surface.SymbolRegistrationIndex | None = None,
        candidate_id: str | None = None) -> surface.AffectedSurface:
    """Feed producer-owned range/files into ``surface.derive_affected_surface``.

    This is the missing producer seam: the actor cannot substitute its declared file
    list.  The footprint came from git, while each supplied index is independently
    provenance-checked by ``derive_affected_surface`` as build-system output.
    """
    if not isinstance(footprint, PatchFootprint):
        raise FootprintError("footprint must be a PatchFootprint")
    entries = tuple(surface.DiffEntry(path=path, change_kind="modified")
                    for path in footprint.actual_files)
    source_diff = surface.SourceDiff(
        base_commit=footprint.production_base_commit,
        candidate_commit=footprint.candidate_source_commit,
        entries=entries,
        origin_ref=f"patch_footprint:{footprint.patch_bundle_sha256}")
    surface_change_class = (
        "core_header" if footprint.change_class == "shared_machinery"
        and any(path.endswith((".h", ".hpp", ".cuh")) for path in footprint.actual_files)
        else None)
    return surface.derive_affected_surface(
        candidate_id=candidate_id or footprint.candidate_source_commit,
        diff=source_diff, indexes=indexes, registrations=registrations,
        change_class=surface_change_class)


def produce_from_git(*, tree: Path, parent: str, child: str,
                     compile_commands_path: Path, depfile_paths: Sequence[Path],
                     variant: str = "default") -> PatchFootprint:
    tree = Path(tree)
    diff = subprocess.run(["git", "diff", "--no-ext-diff", "--unified=12",
                           parent, child, "--"], cwd=tree, check=True,
                          text=True, stdout=subprocess.PIPE).stdout
    commands = json.loads(Path(compile_commands_path).read_text())
    if not isinstance(commands, list):
        raise FootprintError("compile_commands.json must contain a list")
    depfiles = {str(p): Path(p).read_text() for p in depfile_paths}
    return produce_patch_footprint(source_tree=str(tree), parent=parent, child=child,
                                   diff_text=diff, compile_commands=commands,
                                   depfiles=depfiles, variant=variant)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="emit a static patch_footprint.v1 JSON record")
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--child", required=True)
    parser.add_argument("--compile-commands", type=Path, required=True)
    parser.add_argument("--depfile", type=Path, action="append", default=[], required=True)
    parser.add_argument("--variant", default="default")
    args = parser.parse_args(argv)
    result = produce_from_git(tree=args.tree, parent=args.parent, child=args.child,
                              compile_commands_path=args.compile_commands,
                              depfile_paths=args.depfile, variant=args.variant)
    print(json.dumps(result.to_dict(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
