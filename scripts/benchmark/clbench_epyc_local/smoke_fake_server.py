"""Smoke run of the epyc_local system against a FAKE server (CME-7). No model, no GPU.

Drives a multi-instance session (respond / observe / reset) the way CL-Bench's runner
does and prints a JSON summary; exit 0 only if every turn parsed and the request
prefixes grew append-only. Usage: python3 -I smoke_fake_server.py
"""

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tests"))
import fake_clbench_src  # noqa: E402
from stub_server import StubServer  # noqa: E402

fake_clbench_src.install()
spec = importlib.util.spec_from_file_location("epyc_local_system", HERE / "system.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from src.interface import Observation, Query  # noqa: E402


class Schema:
    def __init__(self, action):
        self.action = action

    @classmethod
    def model_json_schema(cls):
        return {"type": "object", "properties": {"action": {"type": "string"}}}

    @classmethod
    def model_validate_json(cls, s):
        return cls(json.loads(s)["action"])


def main() -> int:
    turns = 5
    with StubServer(['{"action": "call"}'] * (turns * 2)) as srv:
        s = mod.EpycLocalSystem(base_url=srv.url, model="fake", system_prompt="You play poker.")
        ok = True
        for run in range(2):
            s.reset()
            prev = None
            for t in range(turns):
                r = s.respond(Query(prompt=f"hand {t}", response_schema=Schema))
                ok &= r.action.action == "call"
                s.observe(Observation(content=f"reward {t}"))
                cur = [m["content"] for m in srv.requests[-1]["messages"][:-1]]
                if prev is not None:
                    ok &= cur[: len(prev)] == prev  # append-only prefix
                prev = [m["content"] for m in srv.requests[-1]["messages"][:-1]]
        ev = s.consume_usage_events()
    print(json.dumps({"requests": len(srv.requests), "usage_events_last_run": len(ev), "ok": bool(ok)}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
