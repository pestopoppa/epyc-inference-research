#!/usr/bin/env python3
"""Private source-visibility check through the real shared prepare() call.

The sync status below is a private-ledger fixture; this does not prove a live
loop's canonical freshness or its separate do-not-repeat history.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys

sys.path.insert(0, os.environ["EPYC_VIDYA_ROOT"])
import loop_beliefs  # noqa: E402


def main():
    root = Path(os.environ["KVU19A2_CANARY_OUT"])
    root.mkdir(exist_ok=False)
    asof = datetime.now(timezone.utc).isoformat()

    class PrivateSync:
        def status(self, *args):
            return {"fresh": True, "lag_entries": 0, "index_max_id": 0, "synced_max_id": 0,
                    "last_sync": asof}

    target = {"model_file": "diagnostic-placeholder.gguf", "backend": "gpu", "device": "gfx90a"}
    answer = loop_beliefs.prepare("kvu19a2.private-visibility", target,
                                  ledger=os.environ["KVU19A2_LEDGER"], index_path=root / "unused-index",
                                  as_of=asof, sync_module=PrivateSync(), known_good=root / "known-good.json",
                                  alarm_state=root / "alarm.json", alarm_config=root / "alarm.yaml",
                                  read_log=root / "reads.jsonl")
    (root / "context.json").write_text(json.dumps(answer, sort_keys=True, indent=2, default=str) + "\n")
    if answer.get("status") != "ready":
        raise RuntimeError("private prepare refused")
    text = json.dumps(answer)
    if "fa_output_nan_elements" not in text and "NaN output elements" not in text:
        raise RuntimeError("source claim absent from assembled planning context")
    print(json.dumps({"status": answer["status"], "claim_ids": answer.get("evidence", {}).get("claim_ids"),
                      "scope": "private source visibility only; freshness fixture, no do-not-repeat proof"}))


if __name__ == "__main__":
    main()
