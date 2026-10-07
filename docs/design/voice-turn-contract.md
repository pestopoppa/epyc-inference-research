# Voice-turn API contract

This note defines the first-party HTTP boundary for a single voice turn. It is
an additive route contract; the OpenAI-compatible `/v1/chat/completions` and
existing `/chat` response contracts remain unchanged. The route uses the
existing direct-answer (`disable_repl`) path. It does not add a speech model,
router, or inference policy.

## Request

`POST /v1/voice/turn` accepts `application/json` with:

| Field | Contract |
| --- | --- |
| `session_id` | Required non-empty session key. If `x_session_id` is also sent, the values must match; a mismatch is `400`. |
| `user_request` | Required non-empty user text for this turn. |
| `conversation_context` | Optional caller-supplied context text; absent means no additional context. |
| `response_goal` | `spoken` or `display`; defaults to `spoken`. |
| `language` | Optional BCP-47 language hint; absent leaves language selection to the existing direct-answer prompt. |
| `max_spoken_seconds` | Optional positive finite cap used by the spoken-answer profile; it never extends the request deadline. |
| `cancel_token` | Optional opaque per-turn correlation value, returned unchanged in terminal events. It is not an authorization credential or a persistent server-side cancellation handle. |

The route creates one `turn_id` before generation and uses it for all emitted
events and persisted rows. It records the accepted user message and persists a
complete assistant message only after successful completion. A cancelled or
failed partial answer is not written as a completed message.

## Streaming response

The successful response is `text/event-stream`. Each SSE `data` value is JSON
with `turn_id` and a monotonically increasing `sequence`:

| Event | Data |
| --- | --- |
| `answer.delta` | `text`: the next answer fragment. |
| `display` | `payload`: a structured display object for code, commands, URLs, or tables; it is shown without being spoken. |
| `preserve` | `values`: exact strings the speech plane must retain, such as numbers, proper nouns, and caveats. |
| `done` | Terminal completion metadata and the optional `cancel_token`. Emitted once, after persistence succeeds. |
| `error` | Terminal `{code, message, retryable, cancel_token}` envelope. Emitted once if a failure occurs after SSE headers are sent. |

Errors discovered before the stream starts use ordinary HTTP status responses.
After headers, exactly one terminal event (`done` or `error`) closes the stream.
The route does not replay accumulated answer characters after real backend
chunks have started. The acceptance check for the stream boundary is that the
client's first `answer.delta` arrives within 50 ms of the backend's first
token, as required by CS-14.

## Cancellation and deadlines

Client disconnect and the controller's per-turn cancel signal share the
request-scoped cancellation callback already carried through `LLMPrimitives`.
The route checks disconnection while waiting for events and passes the same
callback to the direct-answer call; cancellation terminates generation and
closes the stream without emitting `done`. `cancel_token` correlates the
client's turn and terminal event only; it cannot cancel another session or
create cross-request cancellation state.

The route uses the existing frontdoor role timeout and request-deadline
calculation. `max_spoken_seconds` is an answer-length constraint, not a model
timeout. A deadline before stream start returns the existing timeout response;
a deadline after streaming begins emits `error` with code `timeout` and closes
the stream. No new timeout tier or retry loop is introduced.

## Persistence and retention

Conversation rows live in `session_messages`, separate from trace events.
Each row carries `session_id`, `turn_id`, `role`, `text`, `spoken_text`, a
structured `display` payload, and `created_at`/`updated_at` timestamps. Rows
are append-only in this phase. The store returns at most the newest 200 rows in
chronological order for context assembly; that read cap does not delete older
history. An explicit summary-refresh call accepts a caller-supplied summarizer,
passes it the previous compact state plus the ordered messages after that
state's `through_message_id`, and stores the returned state with the new
frontier. If the unseen suffix exceeds the 200-message cap, refresh refuses
instead of silently omitting context. The store never generates semantic
summaries itself; a failed or empty refresh leaves the prior summary and source
messages intact.

Retention follows the existing session lifecycle: ordinary and archived
sessions retain their message history and latest summary; explicit session
deletion removes both in the same transaction. No time-based expiry or
background pruning is introduced. Transcript writes do not change the legacy `sessions.message_count`,
which remains the REPL checkpoint/turn counter.

Transcript append and summary-refresh writes accept the current session lease
fencing token. When a live lease exists, an absent or stale token is refused;
the lease check and each SQLite write share the same `BEGIN IMMEDIATE`
transaction. Unleased sessions keep the existing no-token call shape.

## Boundaries

- CS-13 defines this route contract; CS-14 implements chunk propagation;
  CS-15 implements the store; CS-16 supplies speech/display/preserve formatting;
  CS-18 wires controller cancellation; CS-19 tests the integrated path.
- Until CS-16 is integrated, `spoken_text` may equal the complete assistant
  answer and `display` may be absent. The store does not derive either field.
- Speech measurements remain under the existing prospective
  `VB-SPEECH-CONV-1` write-side task and human-ratified Annex S protocol. This
  API contract adds no measurement or grading rule.
