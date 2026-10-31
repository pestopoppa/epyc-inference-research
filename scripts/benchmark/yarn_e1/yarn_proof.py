"""Shared YaRN launch visibility and log proof matching (offline, no server I/O)."""
import re

PROOF_LOG_FLAGS = ["--verbosity", "4"]
# A logger prefix may interrupt a single printf, even on the same physical line.
_PREFIX = re.compile(r"\b\d+\.\d{2}\.\d{3}\.\d{3} [A-Z] ")

def proof_log(log):
    cut = log.rfind("=== launch")
    if cut >= 0:
        log = log[cut:]
    log = _PREFIX.sub("", log)
    # Only join metadata assignment continuations, never unrelated proof lines.
    return re.sub(r"(Using metadata override[^\n]*=)[ \t]*\n[ \t]*", r"\1 ", log)

def check_proof(log, patterns, forbidden):
    log = proof_log(log)
    proofs = {p: bool(re.search(p, log, re.M)) for p in patterns}
    bad = {p: len(re.findall(p, log, re.I | re.M)) for p in forbidden}
    return {"proof": proofs, "forbidden": bad,
            "ok": all(proofs.values()) and not any(bad.values())}

if __name__ == "__main__":
    import sys
    if sys.argv[1:] != ["--log-flags"]:
        raise SystemExit("usage: yarn_proof.py --log-flags")
    print("\n".join(PROOF_LOG_FLAGS))
