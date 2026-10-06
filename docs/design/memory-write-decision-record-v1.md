# M-19 memory-write record specification — `epyc.memory.write_decision.v1`

Design only for ROOT handoff `episodic-memory-integrity.md`, M-19. MAIN reviewed and applied 2026-10-06; this file does not implement a writer or gate. Source-reviewed against orchestrator `origin/main` `48a546e90fbc202a0c2ae103203621ba29175915` and ROOT integration lane `1f7ce096dd6a715b758e84368f5034b8fb707837`.

## Contract

The record is an append-only diagnostic/dependency record produced by the same writer that performs or refuses the memory write. It captures the typed decision and exact write outcome. It is not itself a calibration receipt, measurement claim, authorization, or grading input. There is no generic `ClaimTuple` projection from this record. A future calibration study, if filed as measurement, must have its own producer-native calibration record and use the existing measurement ladder.

`record_sha256` is SHA-256 of RFC 8785 canonical JSON UTF-8 with the `record_sha256` member omitted. Only finite numeric values are accepted. The digest covers the complete remaining record, including all IDs, status and the write receipt. No mutable post-hash fields.

Privacy default: never copy prompt text, transcript, model raw text, question text, or the persisted fact into this record. Store opaque source/subject/target references and catalogue/question IDs; keep candidate labels only when closed-vocabulary labels are part of the pinned catalogue, otherwise store candidate ordinals. Store no serving content by default. Do not add private content in error messages. The native profile store remains the content source of truth; this record only references its write receipt.

```json
{
  "schema": "epyc.memory.write_decision.v1",
  "record_id": "mwd-synthetic-0001",
  "created_at": "2026-10-06T00:00:00Z",
  "source_kind": "typed_decision_memory_write",
  "source_ref": "event-synthetic-0001",
  "subject_ref": "subject-synthetic-0001",
  "decision_status": "resolved",
  "decision_context": {
    "model_pin": "model-synthetic@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "question_catalogue_id": "memory-preference-v1",
    "question_catalogue_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
    "prompt_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
    "mode": "json"
  },
  "questions": [
    {
      "question_id": "preference_category",
      "kind": "choice",
      "answer_status": "resolved",
      "value_ref": "profile-write-value-synthetic-0001",
      "candidate_probabilities": [0.8, 0.2],
      "confidence": {
        "name": "choice_confidence.v1",
        "value": 0.6,
        "semantics": "uncalibrated_local_statistic"
      },
      "failure": null
    }
  ],
  "write": {
    "status": "accepted",
    "target_ref": "profile-entry-synthetic-0001",
    "reason_code": null
  },
  "calibration_ref": null,
  "record_sha256": "<digest over RFC8785 canonical object without this member>"
}
```

The example is synthetic; the confidence number illustrates a field, not a proposed threshold. `candidate_probabilities` is ordered exactly as the catalogue's candidate sequence and bound by its digest; it contains no candidate strings. `value_ref` is an opaque writer receipt/reference, not a digest or the value itself. The outer record's self-hash excludes its own digest member as defined above.

### Closed field semantics

- `source_kind`: `typed_decision_memory_write`, `direct_user_memory_write`, or `untyped_memory_write`. Direct-user provenance requires a producer-known human source; a tool named `user_conclude` is not proof of that provenance because agents can call it. Untyped writes preserve unknown provenance and never invent model/catalogue fields.
- `decision_status`: `resolved`, `no_answer`, `parse_failure`, `not_applicable`. Direct-user and untyped writes have `not_applicable`; the other statuses describe a typed decision. No absent answer may be silently treated as resolved.
- `decision_context`: for typed model decisions, all five fields are required: exact `model_pin`, stable catalogue ID + canonical catalogue SHA-256, prompt SHA-256, and `mode`. Direct-user and untyped writes use `decision_context: null`, not an object of five null fields. The model pin records observed serving identity; it is not a claim about quality.
- `questions`: one row per catalogue question for typed-decision writes. Each row binds question ID, kind (`choice|score|noul`), `answer_status` (`resolved|no_answer|parse_failure`), candidate probability sequence when resolved, confidence statistic `{name,value,semantics}` when resolved, and explicit failure code/detail-ref when unresolved. No question text, raw answer, or model text. `value_ref` is an opaque native writer receipt/reference, never a content digest or raw content.
- Direct-user and untyped writes use `questions: []`, `decision_status: not_applicable`, `decision_context: null`; their native write result remains recorded in `write`.
- `write.status`: `accepted`, `rejected`, `not_attempted`; `target_ref` is the native write receipt/reference on accepted writes; `reason_code` is an allowlisted non-content code for rejection/abstention.
- `calibration_ref`: null for ordinary decision records. An exact calibration reference may be present only if natively known at initial append. Later analysis may join a separate operator-ratified calibration record externally by exact record ID, model pin and catalogue digest; it must not modify the original record or fill its null field after hashing. It does not authorize a gate. There is intentionally no confidence threshold, gate verdict, promotion status, `allow` or `reject-by-confidence` field.
- For a typed result, outer status is `parse_failure` if any question has that status, otherwise `no_answer` if any question is unresolved, otherwise `resolved`. This describes observed decision completeness independently of the actual write outcome; it introduces no enforcement.
- Empty/partial model results preserve each unresolved question's failure status; write may be `not_attempted` or native `rejected` as actually observed. Do not fabricate a complete set.

### Truthful illustrative no-answer record

A synthetic no-answer write record has `source_kind=typed_decision_memory_write`, model/catalogue context present, `decision_status=no_answer`, a question row with `answer_status=no_answer`, `candidate_probabilities=null`, `confidence=null`, and `failure={"code":"no_json","detail_ref":"failure-synthetic-0001"}`. `write.status=not_attempted`, `target_ref=null`, and `reason_code="decision_unresolved"`. It carries no raw model output or error text.

### Source grounding and later boundary

At APP `48a546e90`, `src/typed_decisions/types.py` defines frozen Question/Decision/ParseFailure/DecisionResult; Decision includes per-question probability mapping and local confidence, DecisionResult includes typed failures and prompt SHA. `src/user_modeling/tools.py::user_conclude` currently accepts fact/category/user_id and writes UserFact directly; `src/user_modeling/deriver.py` extracts plain preferences. Neither persists typed-decision/catalogue/model provenance or a linked write receipt. The scoped M19-RECORD task closes only this record-spec design; parent M-19 remains open for implementation and calibrated gate design. A future writer hook/adaptor must preserve producer bytes prospectively; no backfill.

Only live gate blockers are calibration of the specific question catalogue on the exact model pin and the TD-5 operator approval for enforcement. They do not block this spec. This record is explicitly ungraded metadata; any future decision-bearing calibration measurements require their separate source row/producer and the existing shared measurement ladder.
