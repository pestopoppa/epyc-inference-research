#!/bin/bash
# Re-index this repo via gitnexus, preserving project conventions.
#
# gitnexus 1.6.5 supports --skip-skills. Always pass it: this project keeps
# GitNexus skills flat at .claude/skills/<name>/, and bare analyze otherwise
# regenerates a nested .claude/skills/gitnexus/<name>/ tree.
#
# --skip-agents-md: do NOT let analyze rewrite the gitnexus section of CLAUDE.md
# / AGENTS.md at all. Protects the lean keep-markered block + avoids re-bloat
# (see feedback_gitnexus_bloat_protection). Re-run this wrapper, never bare
# `gitnexus analyze`.
set -euo pipefail
HERE="$(dirname "$(readlink -f "$0")")"
exec /usr/bin/python3 -I "$HERE/gitnexus-writer.py" --repo app -- "$@"
