#!/usr/bin/env python3
"""Print the exact launch command for an M-12 serving recipe. It never launches anything.

Recipes are imported, never transcribed: the argv comes from
``autokernel.loop.serving.Recipe.server_argv`` and the loader environment from
``Recipe.server_env``, the same code the autokernel serving harness uses. Only the
two loader-owned variables are emitted, because every other inherited variable
passes through unchanged:

* ``LD_LIBRARY_PATH`` is pinned to the build's own ``bin``, which is the
  three-ggml-generations guard.
* ``HSA_OVERRIDE_GFX_VERSION`` is removed.

Usage::

    python scripts/benchmark/m12_launch_argv.py \\
        artifacts/serving-recipes/eval/qwen3.6-35b-a3b-q8-gpu-mtp-longctx196k-np4.json \\
        --port 8199 [--build /mnt/raid0/llm/tmp/build-fold-ef81196d5] [--json]
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for entry in (REPO / "scripts" / "kernel_rnd", REPO / "scripts", REPO):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from scripts.kernel_rnd.autokernel.loop.serving import Recipe  # noqa: E402

#: The autokernel champion the M-12 window runs on (MEMORY: champion is current).
DEFAULT_BUILD = Path("/mnt/raid0/llm/tmp/build-fold-ef81196d5")


def per_slot_ctx(recipe: Recipe) -> int:
    """``-c`` is the TOTAL context; with ``kv_unified`` false each slot gets ctx / np."""
    return recipe.ctx if recipe.kv_unified else recipe.ctx // recipe.np


def launch_spec(recipe_path: Path, *, port: int, build: Path = DEFAULT_BUILD) -> dict:
    recipe = Recipe.load(recipe_path)
    env = recipe.server_env(build, base={})
    return {
        "recipe": str(recipe_path),
        "name": recipe.name,
        "recipe_hash": recipe.recipe_hash,
        "build": str(build),
        "port": port,
        "np": recipe.np,
        "ctx_total": recipe.ctx,
        "ctx_per_slot": per_slot_ctx(recipe),
        "ubatch": recipe.ubatch,
        "env": env,
        "unset": ["HSA_OVERRIDE_GFX_VERSION"],
        "argv": recipe.server_argv(build, port),
    }


def shell_line(spec: dict) -> str:
    parts = ["env"] + [f"-u{name}" for name in spec["unset"]]
    parts += [f"{k}={v}" for k, v in sorted(spec["env"].items())]
    return shlex.join(parts + list(spec["argv"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--build", type=Path, default=DEFAULT_BUILD)
    parser.add_argument("--json", action="store_true", help="Emit the full launch spec as JSON")
    args = parser.parse_args()
    spec = launch_spec(args.recipe, port=args.port, build=args.build)
    if args.json:
        print(json.dumps(spec, indent=2))
    else:
        print(f"# {spec['name']}  recipe_hash={spec['recipe_hash']}  "
              f"np={spec['np']}  ctx/slot={spec['ctx_per_slot']}", file=sys.stderr)
        print(shell_line(spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
