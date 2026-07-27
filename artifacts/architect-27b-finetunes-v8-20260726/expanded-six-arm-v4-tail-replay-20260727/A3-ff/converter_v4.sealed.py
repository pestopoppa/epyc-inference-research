"""Convert SEARCH/REPLACE model outputs into unified-diff predictions for the
official SWE-bench harness.

Usage: python3 convert_sr_to_patch.py <arm_pq.jsonl> <arm_name> <out_predictions.json>
Apply order per instance: exact substring match; trailing-whitespace-normalized
line-sequence match; unique common-indentation-normalized line-window match.
Unmatched SR block => that block skipped (counted); instance with zero applied
blocks => empty patch (harness scores unresolved, correctly).
"""
import argparse
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ART = Path(__file__).parent
REPOS = ART / "swebench_repos"
with open(ART / "swebench_verified.json") as f:
    rows = {r["instance_id"]: r for r in json.load(f)}
_PINNED_PATHS: dict[tuple[str, str], tuple[str, ...]] = {}

SR = re.compile(r"<<<<<<<+\s*SEARCH\s*\n(.*?)\n?=======\s*\n(.*?)\n?>>>>>>>+\s*REPLACE\s*(\S*)",
                re.DOTALL)
CURRENT_CAPTURE_SCHEMA = "v7_quality_gate_capture.v4"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

def show(repo: str, commit: str, path: str) -> str | None:
    d = REPOS / repo.replace("/", "__")
    p = subprocess.run(["git", "--git-dir", str(d), "show", f"{commit}:{path}"],
                       capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


def pinned_repo_paths(repo: str, commit: str) -> tuple[str, ...]:
    """Return the immutable file list at an instance's pinned base commit."""
    key = (repo, commit)
    if key not in _PINNED_PATHS:
        directory = REPOS / repo.replace("/", "__")
        result = subprocess.run(
            ["git", "--git-dir", str(directory), "ls-tree", "-r", "--name-only", commit],
            capture_output=True,
            text=True,
        )
        _PINNED_PATHS[key] = tuple(result.stdout.splitlines()) if result.returncode == 0 else ()
    return _PINNED_PATHS[key]

def ws_norm_find(hay: str, needle: str) -> tuple[int, int] | None:
    """Find needle in hay comparing lines stripped of trailing ws; return char span."""
    h_lines = hay.split("\n")
    n_lines = [line.rstrip() for line in needle.split("\n")]
    if not n_lines:
        return None
    stripped = [line.rstrip() for line in h_lines]
    for i in range(len(stripped) - len(n_lines) + 1):
        if stripped[i:i + len(n_lines)] == n_lines:
            start = sum(len(line) + 1 for line in h_lines[:i])
            length = sum(len(line) + 1 for line in h_lines[i:i + len(n_lines)]) - 1
            return start, start + length
    return None


def common_indent_prefix(lines: list[str]) -> str:
    """Return the literal whitespace prefix common to every nonblank line."""
    indents = [re.match(r"[ \t]*", line).group(0) for line in lines if line.strip()]
    if not indents:
        return ""
    prefix = indents[0]
    for indent in indents[1:]:
        while not indent.startswith(prefix):
            prefix = prefix[:-1]
    return prefix


def indent_norm_find(hay: str, needle: str) -> tuple[str, tuple[int, int] | None, str]:
    """Find one indentation-normalized line window, failing closed on ambiguity.

    The caller receives the source window's common indentation so replacement text
    can be translated back to that base indentation.  This intentionally compares
    only after exact and trailing-whitespace matching have failed.
    """
    h_lines = hay.split("\n")
    n_lines = needle.split("\n")
    if not n_lines:
        return "not_found", None, ""
    needle_indent = common_indent_prefix(n_lines)
    normalized_needle = [line[len(needle_indent):].rstrip() for line in n_lines]
    matches: list[tuple[int, int, str]] = []
    for i in range(len(h_lines) - len(n_lines) + 1):
        window = h_lines[i:i + len(n_lines)]
        window_indent = common_indent_prefix(window)
        normalized_window = [line[len(window_indent):].rstrip() for line in window]
        if normalized_window != normalized_needle:
            continue
        start = sum(len(line) + 1 for line in h_lines[:i])
        length = sum(len(line) + 1 for line in window) - 1
        matches.append((start, start + length, window_indent))
    if len(matches) == 1:
        start, end, window_indent = matches[0]
        return "unique", (start, end), window_indent
    if len(matches) > 1:
        return "ambiguous", None, ""
    return "not_found", None, ""


def applicable_match_status(content: str, search: str) -> str:
    """Require one existing match before recovering an explicit path wrapper."""
    exact_matches = []
    start = 0
    while True:
        found = content.find(search, start)
        if found < 0:
            break
        exact_matches.append(found)
        start = found + 1
    if len(exact_matches) == 1:
        return "unique_exact"
    if len(exact_matches) > 1:
        return "ambiguous_exact"

    content_lines = content.split("\n")
    search_lines = [line.rstrip() for line in search.split("\n")]
    ws_matches = [
        index for index in range(len(content_lines) - len(search_lines) + 1)
        if [line.rstrip() for line in content_lines[index:index + len(search_lines)]] == search_lines
    ]
    if len(ws_matches) == 1:
        return "unique_whitespace_normalized"
    if len(ws_matches) > 1:
        return "ambiguous_whitespace_normalized"
    indent_status, _span, _indent = indent_norm_find(content, search)
    return "unique_indent_normalized" if indent_status == "unique" else indent_status


def normalize_explicit_path_wrapper(inst, path: str, search: str,
                                    files: dict[str, str]) -> tuple[str | None, dict]:
    """Recover only unambiguous ``path:`` and ``path/to/`` prompt wrappers.

    This is intentionally not suffix inference: the stripped spelling must name
    exactly one file in the pinned tree, and that file must contain one match
    under the converter's existing exact/whitespace/indent sequence.
    """
    wrapper = None
    candidate = path
    if path.startswith("path:"):
        wrapper, candidate = "path:", path.removeprefix("path:")
    elif path.startswith("path/to/"):
        wrapper, candidate = "path/to/", path.removeprefix("path/to/")
    if wrapper is None:
        return path, {"outcome": "not_requested", "candidate": None}
    candidate = candidate.strip()
    detail = {"wrapper": wrapper, "candidate": candidate}
    if candidate == "file.py":
        detail["outcome"] = "rejected_generic_placeholder"
        return None, detail
    matches = [name for name in pinned_repo_paths(inst["repo"], inst["base_commit"])
               if name == candidate]
    if len(matches) != 1:
        detail["outcome"] = (
            "rejected_ambiguous_pinned_file" if len(matches) > 1
            else "rejected_no_pinned_file"
        )
        return None, detail
    resolved = matches[0]
    content = files.get(resolved)
    if content is None:
        content = show(inst["repo"], inst["base_commit"], resolved)
    if content is None:
        detail["outcome"] = "rejected_unreadable_pinned_file"
        return None, detail
    match_status = applicable_match_status(content, search)
    detail["candidate"] = resolved
    detail["match_status"] = match_status
    if not match_status.startswith("unique_"):
        detail["outcome"] = "rejected_ambiguous_applicable_match" if (
            match_status.startswith("ambiguous")
        ) else "rejected_no_applicable_match"
        return None, detail
    detail["outcome"] = "normalized"
    return resolved, detail


def reindent_replacement(replace: str, search: str, matched_indent: str) -> str:
    """Translate replacement text from SEARCH's base indentation to the source's."""
    search_indent = common_indent_prefix(search.split("\n"))
    replacement_lines = replace.split("\n")
    reindented = []
    for line in replacement_lines:
        if not line.strip():
            reindented.append(line)
            continue
        if line.startswith(search_indent):
            reindented.append(matched_indent + line[len(search_indent):])
        elif matched_indent.startswith(search_indent):
            reindented.append(matched_indent[len(search_indent):] + line)
        elif search_indent.startswith(matched_indent):
            remove = len(search_indent) - len(matched_indent)
            leading = len(line) - len(line.lstrip(" \t"))
            reindented.append(line[min(remove, leading):])
        else:
            reindented.append(matched_indent + line)
    return "\n".join(reindented)

def fingerprint(value: str) -> str:
    """Return a stable content fingerprint without retaining duplicate payloads."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def record_fingerprint(record) -> str:
    """Fingerprint a source row canonically, independent of JSONL whitespace."""
    return fingerprint(json.dumps(record, sort_keys=True, separators=(",", ":")))


def text_fingerprint(text: str) -> dict[str, int | str]:
    """Return runner-compatible identity evidence for an unmodified response."""
    encoded = text.encode("utf-8")
    return {
        "chars": len(text),
        "utf8_bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def capture_integrity(x, response: str, expected_runner_sha256: str | None = None) -> dict:
    """Verify capture provenance without altering the response passed to conversion."""
    computed = text_fingerprint(response)
    runner_fingerprint = x.get("response_fingerprint")
    capture_schema_version = x.get("capture_schema_version")
    current_capture = capture_schema_version == CURRENT_CAPTURE_SCHEMA
    if runner_fingerprint is None:
        fingerprint_status = "legacy_missing"
    elif runner_fingerprint == computed:
        fingerprint_status = "verified"
    else:
        fingerprint_status = "mismatch"
    prompt = x.get("prompt")
    reasoning = x.get("reasoning")
    prompt_status = "verified" if isinstance(prompt, str) and x.get("prompt_fingerprint") == text_fingerprint(prompt) else "missing_or_mismatch"
    reasoning_status = "verified" if isinstance(reasoning, str) and x.get("reasoning_fingerprint") == text_fingerprint(reasoning) else "missing_or_mismatch"
    source_status = "verified" if (
        expected_runner_sha256
        and x.get("runner_source_sha256") == expected_runner_sha256
    ) else "missing_or_mismatch"
    request_error = bool(x.get("request_error"))
    scoring_eligible = (
        current_capture
        and fingerprint_status == "verified"
        and prompt_status == "verified"
        and reasoning_status == "verified"
        and source_status == "verified"
        and not request_error
    )
    return {
        "capture_schema_version": capture_schema_version,
        "runner_source_sha256": x.get("runner_source_sha256"),
        "response_fingerprint": computed,
        "runner_response_fingerprint": runner_fingerprint,
        "response_fingerprint_status": fingerprint_status,
        "current_capture_required": current_capture,
        "prompt_fingerprint_status": prompt_status,
        "reasoning_fingerprint_status": reasoning_status,
        "runner_source_status": source_status,
        "request_error": x.get("request_error", ""),
        "scoring_eligible": scoring_eligible,
    }


def apply_blocks(inst, response, diagnostics=None):
    """Apply existing SEARCH/REPLACE semantics and optionally record block outcomes.

    ``diagnostics`` is deliberately write-only: it does not participate in matching,
    patch construction, or the legacy three-value return contract.
    """
    files: dict[str, str] = {}
    source_file_found: dict[str, bool] = {}
    applied = skipped = 0
    for block_index, (search, replace, path) in enumerate(SR.findall(response or "")):
        raw_path = path
        path = path.strip()
        # strip a literal diff-style a/ or b/ prefix ONLY — lstrip("ab/") is
        # char-set stripping and mangles real paths (astropy/... -> stropy/...)
        if path.startswith(("a/", "b/")):
            path = path[2:]
        path = path.strip()
        path, normalization = normalize_explicit_path_wrapper(inst, path, search, files)
        detail = {
            "block_index": block_index,
            "raw_path": raw_path,
            "path": path,
            "path_normalization": normalization,
            "search_chars": len(search),
            "search_sha256": fingerprint(search),
            "replace_chars": len(replace),
            "replace_sha256": fingerprint(replace),
        }
        if not path:
            skipped += 1
            detail["source_file_found"] = None
            detail["outcome"] = (
                "skipped_missing_path" if normalization["outcome"] == "not_requested"
                else f"skipped_path_normalization_{normalization['outcome']}"
            )
            if diagnostics is not None:
                diagnostics.append(detail)
            continue
        if path not in files:
            base = show(inst["repo"], inst["base_commit"], path)
            files[path] = base if base is not None else ""
            source_file_found[path] = base is not None
        content = files[path]
        detail["source_file_found"] = source_file_found[path]
        detail["input_sha256"] = fingerprint(content)
        if search.strip() == "":                       # new-file creation
            files[path] = replace
            applied += 1
            detail["outcome"] = "applied_new_file"
            detail["output_sha256"] = fingerprint(files[path])
            if diagnostics is not None:
                diagnostics.append(detail)
            continue
        idx = content.find(search)
        if idx >= 0:
            files[path] = content[:idx] + replace + content[idx + len(search):]
            applied += 1
            detail["outcome"] = "applied_exact"
            detail["output_sha256"] = fingerprint(files[path])
            if diagnostics is not None:
                diagnostics.append(detail)
            continue
        span = ws_norm_find(content, search)
        if span:
            files[path] = content[:span[0]] + replace + content[span[1]:]
            applied += 1
            detail["outcome"] = "applied_whitespace_normalized"
            detail["output_sha256"] = fingerprint(files[path])
        else:
            indent_status, indent_span, matched_indent = indent_norm_find(content, search)
            if indent_status == "unique":
                reindented_replace = reindent_replacement(replace, search, matched_indent)
                files[path] = (content[:indent_span[0]] + reindented_replace
                               + content[indent_span[1]:])
                applied += 1
                detail["outcome"] = "applied_unique_indent_normalized"
                detail["output_sha256"] = fingerprint(files[path])
            else:
                skipped += 1
                detail["outcome"] = (
                    "skipped_ambiguous_indent_normalized"
                    if indent_status == "ambiguous" else "skipped_search_not_found"
                )
        if diagnostics is not None:
            diagnostics.append(detail)
    # build unified diff vs base
    patch = []
    for path, new in files.items():
        base = show(inst["repo"], inst["base_commit"], path) or ""
        if new == base:
            continue
        diff = difflib.unified_diff(base.splitlines(keepends=True),
                                    new.splitlines(keepends=True),
                                    fromfile=f"a/{path}", tofile=f"b/{path}")
        patch.append("".join(diff))
    return "".join(patch), applied, skipped

def default_sidecars(out: Path) -> tuple[Path, Path]:
    """Keep diagnostics adjacent to predictions, without changing legacy CLI use."""
    return (out.with_suffix(out.suffix + ".diagnostics.jsonl"),
            out.with_suffix(out.suffix + ".diagnostics.summary.json"))


def row_diagnostic(x, patch: str, block_diagnostics, expected_runner_sha256: str | None = None):
    response = x.get("response", "") or ""
    integrity = capture_integrity(x, response, expected_runner_sha256)
    parseable = len(block_diagnostics)
    skipped = sum(d["outcome"].startswith("skipped_") for d in block_diagnostics)
    applied = sum(d["outcome"].startswith("applied_") for d in block_diagnostics)
    if patch:
        empty_reason = None
    elif not response:
        empty_reason = "empty_response"
    elif not parseable:
        empty_reason = "no_parseable_search_replace_block"
    elif applied == 0:
        empty_reason = "all_parseable_blocks_skipped"
    else:
        empty_reason = "applied_blocks_produced_no_diff"
    return {
        "schema_version": 1,
        "instance_id": x["id"],
        "source_record_sha256": record_fingerprint(x),
        "finish_reason": x.get("finish_reason"),
        "completion_tokens": x.get("completion_tokens"),
        "prompt_tokens": x.get("prompt_tokens"),
        "truncated": x.get("truncated"),
        "response_chars": integrity["response_fingerprint"]["chars"],
        "response_utf8_bytes": integrity["response_fingerprint"]["utf8_bytes"],
        "response_sha256": integrity["response_fingerprint"]["sha256"],
        "parseable_block_count": parseable,
        "applied_block_count": applied,
        "skipped_block_count": skipped,
        "empty_patch": not bool(patch),
        "empty_patch_reason": empty_reason,
        "conversion_disposition": (
            "model_truncation_empty_patch"
            if x.get("finish_reason") == "length"
            else "converted"
        ),
        "patch_chars": len(patch),
        "patch_sha256": fingerprint(patch),
        "blocks": block_diagnostics,
        **integrity,
    }


def summary_status(diagnostics):
    """Classify converter trust without changing prediction or verdict semantics."""
    skipped_rows = [d for d in diagnostics if d["skipped_block_count"]]
    stopped_zero = [
        d for d in diagnostics
        if d["finish_reason"] == "stop" and not d["parseable_block_count"]
    ]
    length_zero = [
        d for d in diagnostics
        if d["finish_reason"] == "length" and not d["parseable_block_count"]
    ]
    fingerprint_verified = [
        d for d in diagnostics if d["response_fingerprint_status"] == "verified"
    ]
    fingerprint_mismatch = [
        d for d in diagnostics if d["response_fingerprint_status"] == "mismatch"
    ]
    fingerprint_missing = [
        d for d in diagnostics if d["response_fingerprint_status"] == "legacy_missing"
    ]
    capture_ineligible = [d for d in diagnostics if not d["scoring_eligible"]]
    if capture_ineligible:
        integrity_status = "fail_closed"
    elif fingerprint_missing:
        integrity_status = "legacy_unverified"
    else:
        integrity_status = "verified"
    if skipped_rows or stopped_zero:
        status = "provisional_converter_or_contract"
    elif length_zero:
        # Exhausted generation is a terminal model failure, not a harness ambiguity.
        status = "terminal_model_length_failure"
    else:
        status = "complete"
    scoring_ineligible_ids = list(dict.fromkeys(
        [d["instance_id"] for d in capture_ineligible]
        + [d["instance_id"] for d in skipped_rows]
        + [d["instance_id"] for d in stopped_zero]
    ))
    return {
        "conversion_status": status,
        "skipped_parseable_block_count": sum(
            d["skipped_block_count"] for d in skipped_rows),
        "skipped_parseable_block_instance_ids": [d["instance_id"] for d in skipped_rows],
        "stopped_zero_parseable_row_count": len(stopped_zero),
        "stopped_zero_parseable_instance_ids": [d["instance_id"] for d in stopped_zero],
        "length_zero_parseable_row_count": len(length_zero),
        "length_zero_parseable_instance_ids": [d["instance_id"] for d in length_zero],
        "response_fingerprint_verified_row_count": len(fingerprint_verified),
        "response_fingerprint_verified_instance_ids": [
            d["instance_id"] for d in fingerprint_verified
        ],
        "response_fingerprint_mismatch_row_count": len(fingerprint_mismatch),
        "response_fingerprint_mismatch_instance_ids": [
            d["instance_id"] for d in fingerprint_mismatch
        ],
        "response_fingerprint_legacy_missing_row_count": len(fingerprint_missing),
        "response_fingerprint_legacy_missing_instance_ids": [
            d["instance_id"] for d in fingerprint_missing
        ],
        "capture_integrity_eligible": not capture_ineligible,
        "capture_integrity_ineligible_instance_ids": [
            d["instance_id"] for d in capture_ineligible
        ],
        "scoring_eligible": not scoring_ineligible_ids,
        "ineligible_instance_ids": scoring_ineligible_ids,
        "artifact_integrity_status": integrity_status,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pq_path")
    parser.add_argument("arm")
    parser.add_argument("out")
    parser.add_argument("--diagnostics-jsonl", type=Path)
    parser.add_argument("--diagnostics-summary", type=Path)
    parser.add_argument("--runner-source", type=Path,
                        help="Reviewed runner source; required for v4 captures")
    return parser.parse_args(argv)


def atomic_write_json(path: Path, value) -> None:
    """Durably publish a completed prediction artifact in one replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        json.dump(value, tmp)
        tmp.flush()
        os.fsync(tmp.fileno())
        temporary_path = Path(tmp.name)
    try:
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main(argv=None):
    args = parse_args(argv)
    pq_path, arm, out = Path(args.pq_path), args.arm, Path(args.out)
    # A stale requested artifact must not remain consumable after this attempt.
    if out.exists():
        out.unlink()
    default_jsonl, default_summary = default_sidecars(out)
    diagnostics_jsonl = args.diagnostics_jsonl or default_jsonl
    diagnostics_summary = args.diagnostics_summary or default_summary
    source_rows = [json.loads(line) for line in pq_path.read_text().splitlines() if line.strip()]
    has_current_capture = any(x.get("capture_schema_version") == CURRENT_CAPTURE_SCHEMA for x in source_rows)
    if has_current_capture and args.runner_source is None:
        print("FAIL --runner-source is required for v4 capture conversion", file=sys.stderr)
        return 1
    if args.runner_source is not None and not args.runner_source.is_file():
        print(f"FAIL reviewed runner source is unreadable: {args.runner_source}", file=sys.stderr)
        return 1
    expected_runner_sha256 = (
        hashlib.sha256(args.runner_source.read_bytes()).hexdigest()
        if args.runner_source is not None else None
    )
    # Integrity is established before any repository content is read or any
    # patch is constructed.  Invalid capture evidence gets diagnostics only.
    preflight = [capture_integrity(x, x.get("response", "") or "", expected_runner_sha256)
                 for x in source_rows]
    if any(not integrity["scoring_eligible"] for integrity in preflight):
        diagnostic_rows = [row_diagnostic(x, "", [], expected_runner_sha256) for x in source_rows]
        summary = {
            "schema_version": 1, "arm": arm, "prediction_count": 0,
            "empty_patches": 0, "blocks_applied": 0, "blocks_skipped": 0,
            "diagnostics_jsonl": str(diagnostics_jsonl),
            "input_pq_sha256": hashlib.sha256(pq_path.read_bytes()).hexdigest(),
            "predictions_sha256": "", "prediction_artifact_written": False,
            "runner_source": str(args.runner_source) if args.runner_source else "",
            "runner_source_sha256": expected_runner_sha256 or "",
        }
        summary.update(summary_status(diagnostic_rows))
        with open(diagnostics_jsonl, "w") as f:
            for diagnostic in diagnostic_rows:
                f.write(json.dumps(diagnostic, sort_keys=True, separators=(",", ":")) + "\n")
        with open(diagnostics_summary, "w") as f:
            json.dump(summary, f, sort_keys=True, indent=2)
            f.write("\n")
        print("FAIL capture integrity preflight; predictions were not written", file=sys.stderr)
        return 1

    preds, stats = [], {"empty": 0, "applied": 0, "skipped": 0}
    diagnostic_rows = []
    for x in source_rows:
        inst = rows[x["id"]]
        block_diagnostics = []
        if x.get("finish_reason") == "length":
            # A token-cap/looping model outcome is a terminal failed draw.
            # Never recover an early partial block into a patch: the fixed
            # denominator receives an explicit empty prediction instead.
            patch, a, s = "", 0, 0
        else:
            patch, a, s = apply_blocks(inst, x.get("response", ""), block_diagnostics)
        stats["applied"] += a
        stats["skipped"] += s
        if not patch:
            stats["empty"] += 1
        preds.append({"instance_id": x["id"], "model_name_or_path": arm,
                      "model_patch": patch})
        diagnostic = row_diagnostic(x, patch, block_diagnostics, expected_runner_sha256)
        diagnostic_rows.append(diagnostic)
        if diagnostic["skipped_block_count"]:
            print(f"WARNING converter skipped {diagnostic['skipped_block_count']} parseable "
                  f"SEARCH/REPLACE block(s) for {x['id']}", file=sys.stderr)
        if x.get("finish_reason") == "stop" and not diagnostic["parseable_block_count"]:
            print(f"WARNING contract-ineligible stopped row for {x['id']}: "
                  "no parseable SEARCH/REPLACE block", file=sys.stderr)
        if x.get("finish_reason") == "length" and not diagnostic["parseable_block_count"]:
            print(f"WARNING terminal model length row for {x['id']}: "
                      "no parseable SEARCH/REPLACE block", file=sys.stderr)
            if diagnostic["response_fingerprint_status"] == "mismatch":
                print(f"WARNING capture fingerprint mismatch for {x['id']}; "
                      "scoring is ineligible", file=sys.stderr)
    summary_status_data = summary_status(diagnostic_rows)
    if not summary_status_data["scoring_eligible"]:
        print("FAIL converter output is not scoring-eligible; predictions were not written", file=sys.stderr)
        return_code = 1
    else:
        atomic_write_json(out, preds)
        return_code = 0
    with open(diagnostics_jsonl, "w") as f:
        for diagnostic in diagnostic_rows:
            f.write(json.dumps(diagnostic, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": 1,
        "arm": arm,
        "prediction_count": len(preds),
        "empty_patches": stats["empty"],
        "blocks_applied": stats["applied"],
        "blocks_skipped": stats["skipped"],
        "diagnostics_jsonl": str(diagnostics_jsonl),
        "input_pq_sha256": hashlib.sha256(pq_path.read_bytes()).hexdigest(),
        "predictions_sha256": fingerprint(json.dumps(preds)) if return_code == 0 else "",
        "prediction_artifact_written": return_code == 0,
        "runner_source": str(args.runner_source) if args.runner_source else "",
        "runner_source_sha256": expected_runner_sha256 or "",
    }
    summary.update(summary_status_data)
    with open(diagnostics_summary, "w") as f:
        json.dump(summary, f, sort_keys=True, indent=2)
        f.write("\n")
    print(f"{arm}: {len(preds)} predictions -> {out} | blocks applied={stats['applied']} "
          f"skipped={stats['skipped']} | empty patches={stats['empty']} | "
          f"diagnostics={diagnostics_jsonl}")
    return return_code

if __name__ == "__main__":
    raise SystemExit(main())
