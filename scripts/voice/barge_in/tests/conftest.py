import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
# CS-45 pinned modules: use the repo's once committed; DEV fallback to the CS-45 lane worktree while it is unmerged.
REPO = HERE.parent.parent.parent
if not (REPO / "scripts/voice/turn_taking/tt_silero.py").exists():
    dev = Path("/mnt/raid0/llm/worktrees/cs45-research")
    os.environ.setdefault("TT_PINS_DIR", str(dev / "scripts/voice/turn_taking"))
    os.environ.setdefault("CS45_DATA", str(dev / "data/turn-taking-cs45-v1"))
