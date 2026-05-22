<!-- gitnexus:start -->
<!-- gitnexus:keep -->
# GitNexus — Code Intelligence

Indexed as **epyc-inference-research** (11232 symbols, 15814 relationships, 263 execution flows). Use the `gitnexus` CLI.

**Re-index when stale:** `scripts/gitnexus-analyze.sh` — NOT bare `gitnexus analyze` (re-installs skills into a nested subdir).

## Required before editing

- Run `gitnexus impact <symbol> --direction upstream`. Report blast radius + risk. STOP and warn if HIGH or CRITICAL.
- Run `gitnexus status` once per session; re-analyze via wrapper if stale.

## Required for renames / refactors

- Run `gitnexus context <symbol>` to enumerate every caller/file BEFORE editing. Find-and-replace alone is unsafe.

## Additional CLI

`gitnexus query <concept>` (execution flows) · `gitnexus cypher <query>` (graph) · `gitnexus wiki` (docs)
<!-- gitnexus:end -->
