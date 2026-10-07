"""Conversation-message persistence controls for the session SQLite store."""

from datetime import datetime, timezone
import sqlite3

import pytest

from src.session import Session, SessionConversationSummary, SessionMessage, SQLiteSessionStore
from src.session.sqlite_store import SummaryBacklogExceeded
from src.session.lease import StaleFencingToken


@pytest.fixture
def store(tmp_path):
    value = SQLiteSessionStore(
        db_path=tmp_path / "sessions.db",
        embeddings_path=tmp_path / "embeddings.npy",
    )
    yield value
    value.close()


def test_messages_round_trip_in_order_and_are_scoped_to_session(store):
    first = Session.create(name="voice one")
    other = Session.create(name="voice other")
    store.create_session(first)
    store.create_session(other)
    t0 = datetime(2026, 10, 7, 1, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 10, 7, 1, 1, tzinfo=timezone.utc)

    user = store.append_message(
        first.id, "turn-1", "user", "Read this command.", created_at=t0
    )
    assistant = store.append_message(
        first.id,
        "turn-1",
        "assistant",
        "Run `pytest -q`.",
        spoken_text="Run the tests quietly.",
        display={"code": "pytest -q", "kind": "command"},
        created_at=t1,
    )
    store.append_message(other.id, "turn-2", "user", "Other session")

    assert isinstance(user, SessionMessage)
    assert [m.id for m in store.get_messages(first.id)] == [user.id, assistant.id]
    assert store.get_messages(first.id)[1].display == {
        "code": "pytest -q", "kind": "command"
    }
    assert store.get_messages(first.id)[1].spoken_text == "Run the tests quietly."
    assert store.get_messages(first.id)[1].updated_at == t1
    assert [m.text for m in store.get_messages(other.id)] == ["Other session"]
    # The old field counts REPL turns; transcript writes do not redefine it.
    assert store.get_session(first.id).message_count == 0


def test_message_window_returns_newest_rows_in_chronological_order(store):
    session = Session.create(name="bounded history")
    store.create_session(session)
    for index in range(3):
        store.append_message(
            session.id,
            f"turn-{index}",
            "user",
            f"message-{index}",
            created_at=datetime(2026, 10, 7, 1, index, tzinfo=timezone.utc),
        )

    assert [row.text for row in store.get_messages(session.id, limit=2)] == [
        "message-1", "message-2"
    ]
    with pytest.raises(ValueError, match="limit"):
        store.get_messages(session.id, limit=0)


def test_archive_retains_messages_and_explicit_delete_removes_them(store):
    session = Session.create(name="retention")
    store.create_session(session)
    stored = store.append_message(session.id, "turn-1", "user", "retain me")
    summary = store.save_conversation_summary(session.id, stored.id, "The user asked for retention.")

    assert store.archive_session(session.id)
    assert [row.id for row in store.get_messages(session.id)] == [stored.id]
    assert store.get_conversation_summary(session.id) == summary
    assert store.delete_session(session.id)
    assert store.get_messages(session.id) == []
    assert store.get_conversation_summary(session.id) is None


def test_summary_refresh_uses_ordered_messages_and_records_its_frontier(store):
    session = Session.create(name="summary refresh")
    store.create_session(session)
    first = store.append_message(session.id, "turn-1", "user", "I prefer metric units.")
    second = store.append_message(session.id, "turn-1", "assistant", "I will use metric units.")
    seen = []

    def summarize(previous, messages):
        assert previous is None
        seen.extend(message.id for message in messages)
        return "Preference: metric units."

    summary = store.refresh_conversation_summary(session.id, summarize)

    assert isinstance(summary, SessionConversationSummary)
    assert seen == [first.id, second.id]
    assert summary.session_id == session.id
    assert summary.through_message_id == second.id
    assert summary.summary == "Preference: metric units."
    assert store.get_conversation_summary(session.id) == summary
    assert [message.text for message in store.get_messages(session.id)] == [
        "I prefer metric units.", "I will use metric units."
    ]


def test_summary_refresh_passes_prior_state_and_only_unseen_suffix(store):
    session = Session.create(name="incremental summary")
    store.create_session(session)
    first = store.append_message(session.id, "turn-1", "user", "I use metric units.")
    seen = []
    initial = store.refresh_conversation_summary(
        session.id,
        lambda previous, messages: (
            seen.extend((previous, [m.id for m in messages])) or "User prefers metric units."
        ),
    )
    second = store.append_message(session.id, "turn-2", "user", "My car is electric.")
    seen.clear()

    incremental = store.refresh_conversation_summary(
        session.id,
        lambda previous, messages: (
            seen.extend((previous, [m.id for m in messages]))
            or f"{previous.summary} They drive an electric car."
        ),
    )

    assert seen == [initial, [second.id]]
    assert first.id < second.id
    assert incremental.through_message_id == second.id
    assert incremental.summary == "User prefers metric units. They drive an electric car."


def test_summary_frontier_tracks_highest_appended_id_when_timestamps_are_out_of_order(store):
    session = Session.create(name="out of order timestamps")
    store.create_session(session)
    first = store.append_message(
        session.id, "turn-1", "user", "First inserted.",
        created_at=datetime(2026, 10, 7, 1, 2, tzinfo=timezone.utc),
    )
    saved = store.refresh_conversation_summary(
        session.id, lambda _previous, _new: "State through first insert."
    )
    second = store.append_message(
        session.id, "turn-2", "user", "Second inserted, earlier timestamp.",
        created_at=datetime(2026, 10, 7, 1, 0, tzinfo=timezone.utc),
    )
    third = store.append_message(
        session.id, "turn-3", "user", "Third inserted.",
        created_at=datetime(2026, 10, 7, 1, 1, tzinfo=timezone.utc),
    )
    observed = []

    result = store.refresh_conversation_summary(
        session.id,
        lambda previous, new: (
            observed.extend((previous, [message.id for message in new]))
            or "State through three inserts."
        ),
    )

    assert observed == [saved, [second.id, third.id]]
    assert result.through_message_id == third.id
    assert first.id < second.id < third.id


def test_equal_frontier_does_not_call_summary_callback(store):
    session = Session.create(name="equal frontier")
    store.create_session(session)
    message = store.append_message(session.id, "turn-1", "user", "Already summarized.")
    saved = store.save_conversation_summary(session.id, message.id, "Prior compact state.")

    result = store.refresh_conversation_summary(
        session.id, lambda _prior, _new: (_ for _ in ()).throw(AssertionError("must not run"))
    )

    assert result == saved


def test_summary_refresh_is_injected_and_preserves_previous_state_on_failure(store):
    session = Session.create(name="summary failure")
    store.create_session(session)
    first = store.append_message(session.id, "turn-1", "user", "Keep this state.")
    saved = store.save_conversation_summary(session.id, first.id, "Existing compact state.")
    store.append_message(session.id, "turn-2", "user", "New content.")

    def fail_without_inference(_previous, _messages):
        raise RuntimeError("synthetic summarizer refusal")

    with pytest.raises(RuntimeError, match="synthetic summarizer refusal"):
        store.refresh_conversation_summary(session.id, fail_without_inference)

    assert store.get_conversation_summary(session.id) == saved


def test_empty_summary_result_does_not_replace_previous_state(store):
    session = Session.create(name="empty summary callback")
    store.create_session(session)
    first = store.append_message(session.id, "turn-1", "user", "State stays.")
    saved = store.save_conversation_summary(session.id, first.id, "Existing state.")
    store.append_message(session.id, "turn-2", "user", "New row.")

    with pytest.raises(ValueError, match="non-empty"):
        store.refresh_conversation_summary(session.id, lambda _previous, _new: "  ")

    assert store.get_conversation_summary(session.id) == saved


def test_summary_frontier_cannot_move_backwards(store):
    session = Session.create(name="summary frontier")
    store.create_session(session)
    first = store.append_message(session.id, "turn-1", "user", "One.")
    second = store.append_message(session.id, "turn-2", "user", "Two.")
    latest = store.save_conversation_summary(session.id, second.id, "One and two.")

    older = store.save_conversation_summary(session.id, first.id, "Only one.")

    assert older == latest
    assert store.get_conversation_summary(session.id) == latest


def test_summary_refresh_empty_window_returns_existing_summary_without_callback(store):
    session = Session.create(name="empty summary refresh")
    store.create_session(session)
    assert store.refresh_conversation_summary(
        session.id, lambda _previous, _messages: (_ for _ in ()).throw(AssertionError("must not run"))
    ) is None

    message = store.append_message(session.id, "turn-1", "user", "Context.")
    saved = store.save_conversation_summary(session.id, message.id, "Context summary.")
    assert store.refresh_conversation_summary(
        session.id, lambda _previous, _messages: (_ for _ in ()).throw(AssertionError("must not run"))
    ) == saved


def test_summary_refresh_refuses_unseen_backlog_over_cap_without_callback(store):
    session = Session.create(name="summary backlog cap")
    store.create_session(session)
    for index in range(201):
        store.append_message(session.id, f"turn-{index}", "user", f"message-{index}")

    with pytest.raises(SummaryBacklogExceeded, match="exceed refresh cap"):
        store.refresh_conversation_summary(
            session.id,
            lambda _previous, _messages: (_ for _ in ()).throw(AssertionError("must not run")),
        )
    assert store.get_conversation_summary(session.id) is None


def test_newer_concurrent_frontier_wins_over_slow_older_summary(store):
    session = Session.create(name="concurrent frontier")
    store.create_session(session)
    first = store.append_message(session.id, "turn-1", "user", "First.")
    second = store.append_message(session.id, "turn-2", "user", "Second.")

    def slow_summary(previous, unseen):
        assert previous is None
        assert [message.id for message in unseen] == [first.id, second.id]
        third = store.append_message(session.id, "turn-3", "user", "Third.")
        newest = store.save_conversation_summary(session.id, third.id, "Newer state.")
        assert newest.through_message_id == third.id
        return "Stale state through second."

    result = store.refresh_conversation_summary(session.id, slow_summary)

    assert result.summary == "Newer state."
    assert result.through_message_id > second.id
    assert store.get_conversation_summary(session.id) == result


def test_delete_winning_during_summary_callback_does_not_recreate_session(store):
    session = Session.create(name="delete wins")
    store.create_session(session)
    store.append_message(session.id, "turn-1", "user", "Will be deleted.")

    def delete_before_save(_previous, _unseen):
        assert store.delete_session(session.id)
        return "Must not be persisted."

    with pytest.raises(ValueError, match="does not exist"):
        store.refresh_conversation_summary(session.id, delete_before_save)
    assert store.get_session(session.id) is None
    assert store.get_conversation_summary(session.id) is None


def test_existing_database_recreates_additive_message_table(store, tmp_path):
    session = Session.create(name="pre-message schema")
    store.create_session(session)
    store.close()
    with sqlite3.connect(tmp_path / "sessions.db") as conn:
        conn.execute("DROP TABLE session_messages")
        conn.commit()

    reopened = SQLiteSessionStore(
        db_path=tmp_path / "sessions.db",
        embeddings_path=tmp_path / "embeddings.npy",
    )
    try:
        saved = reopened.append_message(session.id, "turn-1", "user", "after migration")
        assert [row.id for row in reopened.get_messages(session.id)] == [saved.id]
    finally:
        reopened.close()


def test_message_write_requires_existing_session_and_json_display(store):
    with pytest.raises(ValueError, match="does not exist"):
        store.append_message("missing", "turn-1", "user", "text")

    session = Session.create(name="invalid display")
    store.create_session(session)
    with pytest.raises(ValueError, match="display"):
        store.append_message(
            session.id, "turn-1", "assistant", "text", display=["not", "object"]
        )
    with pytest.raises(ValueError, match="Out of range float"):
        store.append_message(
            session.id, "turn-1", "assistant", "text", display={"value": float("nan")}
        )


def test_transcript_writes_require_the_current_token_for_a_leased_session(store):
    session = Session.create(name="fenced transcript writes")
    store.create_session(session)
    seed = store.append_message(session.id, "turn-seed", "user", "before lease")
    lease = store.leases.try_acquire(session.id, ttl_s=3600)
    assert lease is not None
    token = lease.fencing_token

    append = lambda **kwargs: store.append_message(
        session.id, "turn-1", "user", "fenced append", **kwargs
    )
    with pytest.raises(StaleFencingToken, match="unfenced write is refused"):
        append()
    with pytest.raises(StaleFencingToken, match="stale"):
        append(fencing_token=token - 1)
    appended = append(fencing_token=token)
    assert appended.text == "fenced append"

    save = lambda **kwargs: store.save_conversation_summary(
        session.id, appended.id, "fenced summary", **kwargs
    )
    with pytest.raises(StaleFencingToken, match="unfenced write is refused"):
        save()
    with pytest.raises(StaleFencingToken, match="stale"):
        save(fencing_token=token - 1)
    assert save(fencing_token=token).summary == "fenced summary"

    newest = store.append_message(
        session.id, "turn-2", "user", "refresh suffix", fencing_token=token
    )
    refresh = lambda **kwargs: store.refresh_conversation_summary(
        session.id, lambda _prior, unseen: f"through {unseen[-1].text}", **kwargs
    )
    with pytest.raises(StaleFencingToken, match="unfenced write is refused"):
        refresh()
    with pytest.raises(StaleFencingToken, match="stale"):
        refresh(fencing_token=token - 1)
    result = refresh(fencing_token=token)
    assert result.through_message_id == newest.id
    assert result.summary == "through refresh suffix"
    assert seed.id < appended.id < newest.id
