"""Conversation-message persistence controls for the session SQLite store."""

from datetime import datetime, timezone
import sqlite3

import pytest

from src.session import Session, SessionMessage, SQLiteSessionStore


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

    assert store.archive_session(session.id)
    assert [row.id for row in store.get_messages(session.id)] == [stored.id]
    assert store.delete_session(session.id)
    assert store.get_messages(session.id) == []


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
