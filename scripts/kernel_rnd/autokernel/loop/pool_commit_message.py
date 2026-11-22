"""Pool commit-message builder, split verbatim from pool.py (code-budget split)."""
from __future__ import annotations
import re
from . import gates


def commit_message(hypothesis, comparison, *, ppl_contract: bool = False,
                   bit_exact_oracle: "str | None" = None) -> str:
    """One spelling of the champion commit subject, shared by both run paths. A
    ppl_contract admission carries `gates.PPL_CONTRACT_TRAILER` (re-review 2026-10-06):
    git history is the durable record the fold-time quality gate reads."""
    def clean(value) -> str:
        # Round-7 resolution A: every interpolated field is escaped; a control
        # character (newline included) can never shape the message.
        return re.sub(r"[\x00-\x1f\x7f]", lambda m: f"\\x{ord(m.group()):02x}", str(value))
    subject = (f"{clean(hypothesis.mechanism_id)}: {comparison.effect * 100:+.3f}% "
               f"on {clean(comparison.surface)} over {clean(comparison.pairs)} pairs")
    if re.search(r"[\x00-\x1f\x7f]", subject):
        raise ValueError("commit subject carries a control character")
    if ppl_contract:
        return f"{subject}\n\n{gates.PPL_CONTRACT_TRAILER}"
    if bit_exact_oracle is not None:
        # Round-5 resolution A: written only by the loop, only after a bit-exact oracle
        # passed for THIS keep (sha256 of its passing verdicts).
        if not re.fullmatch(r"[0-9a-f]{64}", bit_exact_oracle):
            raise ValueError("bit_exact_oracle must be a sha256 hex digest")
        return (f"{subject}\n\n{gates.PPL_CONTRACT_BIT_EXACT_TRAILER}\n"
                f"{gates.PPL_CONTRACT_ORACLE_PREFIX}{bit_exact_oracle}")
    return subject
