import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
# CS-45 pins (scripts/voice/turn_taking, data/turn-taking-cs45-v1) resolve repo-relative; no other location is consulted.
