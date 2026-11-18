"""Join frozen-prompt v2 seed/token-output requests to retained same-server responses.

Additive and hermetic. Input rows are the tuples `native_server_response.reopen_unit`
already returns (original request bytes parsed, original response bytes parsed, raw
capture row); nothing here reads a server, a model, a clock or the filesystem.

Evidence is derived only from bytes that were actually sent and retained. A v1 request
(no `seed`/`return_tokens`) yields `basis=v1_request_no_seed_token_request` with seed and
tokens None: absent evidence is never invented. The repeat witness compares token IDs of
same-prompt, same-seed greedy requests inside ONE server process; it is not an
anchor-versus-candidate comparison and is never a T0 PASS on its own.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

TOKEN_ROW_SCHEMA = "epyc.autokernel.server_token_evidence_row.v1"
REPEAT_WITNESS_SCHEMA = "epyc.autokernel.same_server_token_determinism.v1"
BASIS_V1 = "v1_request_no_seed_token_request"
BASIS_V2 = "v2_explicit_seed_token_request"
V2_REQUEST_KEYS = ("seed", "ignore_eos", "return_tokens", "stream")
MAX_TOKEN_ID = 2 ** 31
VERDICTS = ("identical", "diverged", "unknown")


class TokenEvidenceRefused(RuntimeError):
    """A retained row contradicts its own explicit v2 request."""


def _is_int(value: Any) -> bool:
    return type(value) is int


def _tokens(response: Any, request: Mapping[str, Any]) -> tuple[int, ...]:
    tokens = response.get("tokens") if isinstance(response, Mapping) else None
    if (not isinstance(tokens, list) or any(not _is_int(t) or not 0 <= t < MAX_TOKEN_ID for t in tokens)):
        raise TokenEvidenceRefused("return_tokens requested but response carries no valid token IDs")
    if len(tokens) > request["n_predict"]:
        raise TokenEvidenceRefused("response token count exceeds the requested n_predict")
    if request["ignore_eos"] is True and len(tokens) != request["n_predict"]:
        raise TokenEvidenceRefused("ignore_eos request did not deliver exactly n_predict tokens")
    timings = response.get("timings")
    predicted = timings.get("predicted_n") if isinstance(timings, Mapping) else None
    if predicted is not None and predicted != len(tokens):
        raise TokenEvidenceRefused("response timings.predicted_n contradicts token list")
    return tuple(tokens)


def join_row(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """One closed evidence record per retained request/response row."""
    request, response, raw = row["request"], row["response"], row["raw"]
    base = {"schema": TOKEN_ROW_SCHEMA, "prompt_id": raw["prompt_id"], "phase": raw["phase"],
            "slot_index": raw["slot_index"], "request_sha256": raw["request_sha256"],
            "response_sha256": raw["response_sha256"]}
    present = [key in request for key in V2_REQUEST_KEYS]
    if not any(present):
        return {**base, "basis": BASIS_V1, "seed": None, "ignore_eos": None,
                "token_output_requested": None, "tokens": None}
    if not all(present):
        raise TokenEvidenceRefused("request carries a partial v2 field set")
    seed, want_tokens = request["seed"], request["return_tokens"]
    if not _is_int(seed) or not 0 <= seed < 2 ** 32 or type(want_tokens) is not bool \
            or type(request["ignore_eos"]) is not bool or request["stream"] is not False:
        raise TokenEvidenceRefused("request v2 fields are malformed")
    out = {**base, "basis": BASIS_V2, "seed": seed, "ignore_eos": request["ignore_eos"],
           "token_output_requested": want_tokens, "tokens": None}
    if raw["error"] is not None or raw["response_sha256"] is None:
        return {**out, "failed": True}  # a failed request has no token evidence
    if want_tokens:
        out["tokens"] = list(_tokens(response, request))
    return out


def join_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[Mapping[str, Any], ...]:
    return tuple(join_row(row) for row in rows)


def _greedy(request: Mapping[str, Any]) -> bool:
    return request.get("temperature") == 0 or request.get("top_k") == 1


def repeat_witness(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Same-server multi-request determinism, derived from retained rows only.

    Groups by (prompt_id, seed) over greedy v2 rows that returned tokens. A group is
    `identical` only if it has >= 2 members and every token list equals the first; `diverged`
    reports the first differing index; anything lacking evidence is `unknown` with a reason.
    The overall verdict is `identical` only if every prompt in the input is covered by an
    identical group: no reduced witness list, no generic success."""
    joined = join_rows(rows)
    groups: dict[tuple[str, int], list[tuple[int, Mapping[str, Any]]]] = {}
    unknown_reasons: dict[str, str] = {}
    for index, (row, record) in enumerate(zip(rows, joined)):
        pid = record["prompt_id"]
        if record["basis"] == BASIS_V1:
            unknown_reasons.setdefault(pid, "v1 request carries no seed or token-output request")
        elif record.get("failed"):
            unknown_reasons.setdefault(pid, "request failed; no token evidence")
        elif record["tokens"] is None:
            unknown_reasons.setdefault(pid, "token output was not requested")
        elif not _greedy(row["request"]):
            unknown_reasons.setdefault(pid, "request is not greedy; repeat equality is not defined")
        else:
            groups.setdefault((pid, record["seed"]), []).append((index, record))
    results = []
    for (pid, seed), members in sorted(groups.items()):
        first = members[0][1]["tokens"]
        verdict, divergence, agree = "identical", None, 1.0
        if len(members) < 2:
            verdict, agree = "unknown", None
            unknown_reasons.setdefault(pid, "fewer than two same-seed requests")
        else:
            for _, record in members[1:]:
                other = record["tokens"]
                limit = max(len(first), len(other))
                for i in range(limit):
                    if i >= len(first) or i >= len(other) or first[i] != other[i]:
                        divergence = i if divergence is None else min(divergence, i)
                        break
            if divergence is not None:
                verdict = "diverged"
                agree = divergence / max(len(first), 1)
        results.append({"prompt_id": pid, "seed": seed, "requests": len(members),
                        "slots": [(m[1]["phase"], m[1]["slot_index"]) for m in members],
                        "verdict": verdict, "first_divergence_index": divergence,
                        "agreement_prefix_ratio": agree})
    covered = {item["prompt_id"] for item in results if item["verdict"] != "unknown"}
    all_prompts = {record["prompt_id"] for record in joined}
    for pid in sorted(all_prompts - covered):
        unknown_reasons.setdefault(pid, "no evidence group")
    if not joined:
        overall = "unknown"
    elif any(item["verdict"] == "diverged" for item in results):
        overall = "diverged"
    elif all_prompts <= covered and all(item["verdict"] == "identical" for item in results
                                        if item["prompt_id"] in covered) \
            and not any(pid in unknown_reasons for pid in covered):
        overall = "identical"
    else:
        overall = "unknown"
    return {"schema": REPEAT_WITNESS_SCHEMA, "evidence_class": "same_server_repeat_token_equality",
            "overall": overall, "groups": results,
            "unknown_prompts": {k: unknown_reasons[k] for k in sorted(unknown_reasons)
                                if k not in covered},
            "is_t0_pass": False, "rows": list(joined)}


__all__ = ["BASIS_V1", "BASIS_V2", "REPEAT_WITNESS_SCHEMA", "TOKEN_ROW_SCHEMA",
           "TokenEvidenceRefused", "join_row", "join_rows", "repeat_witness"]
