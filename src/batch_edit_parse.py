"""BEP-1: parse a model-emitted patch set + the batch-edit prompt rider (intake-605, P23).

Think-then-act batch editing: the coder/architect reasons, then emits ONE structured patch set
in a fenced ```patchset block (JSON) instead of interleaving REPL tool calls. This module turns
that block into a typed `PatchSet` (from `src.batch_edit`) and supplies the instruction rider.

Pure (no disk/model/REPL imports) → unit-testable. The LIVE wiring into `_execute_turn`
(flag `batch_edit_mode`) that diverts from the REPL loop to apply the patch set is a separate,
reviewed step (BEP-4 runner + BEP-5 sandbox) — see module-level WIRING NOTE below.

WIRING NOTE (deferred, for review):
    In `src/graph/helpers.py:_execute_turn`, after `code = auto_wrap_final(code)`:
        if features().batch_edit_mode:
            ps = parse_patchset_from_model_output(raw_llm_output)   # None if no block
            if ps is not None:
                # BEP-4 runner: stage in a sandbox/worktree (BEP-5), apply via
                # src.batch_edit.apply_file_patch_to_text, run independent verify, then
                # synthesize FINAL(summary). Falls back to normal REPL if ps is None.
    This bypasses the REPL execution core, so it must land with the sandbox-apply + verify
    (BEP-5) and be validated under the J8 A/B before default-on.
"""

from __future__ import annotations

import json
import re

from src.batch_edit import PatchSet, FilePatch, Hunk, validate_patchset

# Matches a ```patchset ... ``` fenced block (case-insensitive language tag).
_FENCE = re.compile(r"```patchset\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


BATCH_EDIT_INSTRUCTIONS = """\
BATCH-EDIT MODE. Do all your reasoning first, then emit EXACTLY ONE patch set — do not call \
tools or run code incrementally. Output a single fenced block:

```patchset
{
  "base_repo_sha": "<repo sha you planned against, if known>",
  "files": [
    {"path": "pkg/mod.py", "operation": "modify",
     "base_content_sha256": "<sha256 of the file you read>",
     "hunks": [{"start_line": 12, "end_line": 14, "replacement": "new line A\\nnew line B\\n"}],
     "depends_on": [], "postconditions": ["pytest tests/unit/test_mod.py passes"]},
    {"path": "pkg/new.py", "operation": "create", "new_content": "full file body\\n"}
  ]
}
```

Rules: every `modify`/`delete`/`rename` MUST include the `base_content_sha256` of the file you \
read (stale-base protection); `create` sets `new_content` only; for an insertion before line N \
use start_line=N, end_line=N-1; declare cross-file ordering in `depends_on`. Emit the block and \
nothing after it. `postconditions` are descriptive declarations only, never executable commands; \
the runner does not execute strings from the patchset. The staged sandbox verifier is the only \
batch acceptance gate."""


# BEP-2 baseline rider (symmetric to BATCH_EDIT_INSTRUCTIONS): puts the coder into the
# INTERLEAVED editing mechanism so the A/B compares interleaved-vs-batched EDIT latency, not
# "edits vs answers-in-prose". Without it the coder tends to return the code as a FINAL() text
# answer and never writes the file (observed 2026-05-26). Default-off; experiment-only.
INTERLEAVED_EDIT_INSTRUCTIONS = """\
INTERLEAVED EDIT MODE. This is a file-editing task — judged by the actual files on disk. Do exactly \
ONE action per turn, as a single CLOSED ```python code block the REPL will run:
 - WRITE turns: emit a block containing only `file_write_safe(path, content)` call(s), then close \
the ``` fence. Use file_write_safe(...), NOT open() (open is blocked). Do NOT call FINAL() in a \
write turn — the REPL runs your writes and reports the result back to you next turn.
 - FINISH turn: only AFTER you have written every file and seen each write succeed, run the task's \
stated acceptance check through `run_shell(cmd=...)` or `run_python_code(code=...)`, as appropriate. \
Inspect the result and repair/recheck any failure. Do not call a missing or failing check a pass. Emit \
a separate `FINAL(...)` block with the result; if no runnable check exists or a failure remains, \
state that explicitly and mark the work unverified or failed rather than claiming acceptance.
Never put file_write_safe(...) and FINAL() in the same block (FINAL ends the turn before the write \
runs). Relative paths resolve inside the working directory; you may read files first with the read \
tools."""


def extract_patchset_json(text: str | None) -> dict | None:
    """Return the parsed JSON from the first ```patchset block, or None if absent/unparseable."""
    if not text:
        return None
    m = _FENCE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def parse_patchset_from_model_output(text: str | None, *, validate: bool = True) -> PatchSet | None:
    """Parse a model output into a PatchSet.

    Returns None when there is no ```patchset block (caller falls back to the normal REPL loop).
    Raises ValueError when a block IS present but malformed/invalid (caller logs + falls back),
    so a bad patch set never silently applies.
    """
    data = extract_patchset_json(text)
    if data is None:
        return None

    files: list[FilePatch] = []
    for fd in data.get("files", []):
        if not isinstance(fd, dict) or "path" not in fd:
            raise ValueError(f"patchset file entry missing 'path': {fd!r}")
        hunks = [
            Hunk(
                start_line=h.get("start_line"),
                end_line=h.get("end_line"),
                replacement=h.get("replacement"),
                unified_diff=h.get("unified_diff"),
            )
            for h in fd.get("hunks", [])
        ]
        files.append(FilePatch(
            path=fd["path"],
            operation=fd.get("operation", "modify"),
            base_content_sha256=fd.get("base_content_sha256"),
            hunks=hunks,
            new_content=fd.get("new_content"),
            rename_to=fd.get("rename_to"),
            postconditions=list(fd.get("postconditions", [])),
            depends_on=list(fd.get("depends_on", [])),
        ))

    ps = PatchSet(
        base_repo_sha=data.get("base_repo_sha"),
        files=files,
        bundle_id=data.get("bundle_id"),
        omitted_context_paths=list(data.get("omitted_context_paths", [])),
    )
    if validate:
        validate_patchset(ps)  # raises ValueError on invalid
    return ps


def build_batch_edit_instructions() -> str:
    """The system-prompt rider that puts a coder/architect into batch-edit mode."""
    return BATCH_EDIT_INSTRUCTIONS


def build_interleaved_edit_instructions() -> str:
    """The system-prompt rider that makes a coder/architect do interleaved file edits (BEP-2
    baseline arm). Symmetric counterpart to build_batch_edit_instructions()."""
    return INTERLEAVED_EDIT_INSTRUCTIONS
