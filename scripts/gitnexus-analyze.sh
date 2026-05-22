#!/bin/bash
# Re-index this repo via gitnexus, preserving project conventions.
#
# --skip-skills: keeps gitnexus from re-installing skill files into
#   .claude/skills/gitnexus/<name>/. This project flattens them to
#   .claude/skills/<name>/ so they auto-surface in the Skill tool list.
#
# The CLAUDE.md auto-managed section is protected separately by the
# <!-- gitnexus:keep --> marker inside the gitnexus:start..end block,
# so the prose stays lean; only the stats line refreshes.
set -euo pipefail
exec gitnexus analyze --skip-skills "$@"
