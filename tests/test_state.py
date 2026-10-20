"""A bounded admission cursor must stay atomic through failures and cancellation."""

import asyncio
import json
import os
import stat
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from hermes_vk_workspace import state
from hermes_vk_workspace.state import CursorStore, StateError


def test_missing_cursor_starts_at_zero(tmp_path):
    assert CursorStore(tmp_path / "missing" / "cursor.json").load() == 0


@pytest.mark.asyncio
async def test_roundtrip_large_cursor_uses_private_file_and_directory(tmp_path):
    store = CursorStore(tmp_path / "private" / "cursor.json")
    value = 10**29 + 777
    await store.save(value)
    assert store.load() == value
    assert json.loads(store.path.read_text()) == {"version": 1, "last_event_id": value}
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700
    assert list(store.path.parent.iterdir()) == [store.path]


@pytest.mark.parametrize(
    "payload",
    [
        [],
        None,
        {},
        {"version": 1},
        {"version": 1, "last_event_id": 1, "extra": "field"},
        {"version": True, "last_event_id": 1},
        {"version": 1.0, "last_event_id": 1},
        {"version": "1", "last_event_id": 1},
        {"version": 2, "last_event_id": 1},
        {"version": 1, "last_event_id": True},
        {"version": 1, "last_event_id": -1},
        {"version": 1, "last_event_id": 1.5},
        {"version": 1, "last_event_id": "-1"},
        {"version": 1, "last_event_id": 10**30},
        {"version": 1, "last_event_id": None},
    ],
)
def test_invalid_schema_or_version_fails_closed(tmp_path, payload):
    store = CursorStore(tmp_path / "private-secret-name.json")
    store.path.write_text(json.dumps(payload))
    with pytest.raises(StateError) as exc:
        store.load()
    assert str(tmp_path) not in str(exc.value) and store.path.name not in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__


@pytest.mark.parametrize(
    "body", [b"not json", b"{\xff}", b'{"version": 1, "last_event_id": 1}' + b" " * 1024]
)
def test_malformed_or_oversized_file_is_rejected(tmp_path, body):
    store = CursorStore(tmp_path / "cursor.json")
    store.path.write_bytes(body)
    with pytest.raises(StateError):
        store.load()


def test_growth_after_fstat_cannot_bypass_read_limit(tmp_path, monkeypatch):
    store = CursorStore(tmp_path / "cursor.json")
    body = b'{"version": 1, "last_event_id": 1}'
    store.path.write_bytes(body + b" " * (1025 - len(body)))
    real_fstat = state.os.fstat

    def underreported(fd):
        info = real_fstat(fd)
        return SimpleNamespace(st_mode=info.st_mode, st_size=100)

    monkeypatch.setattr(state.os, "fstat", underreported)
    with pytest.raises(StateError):
        store.load()


def test_numeric_string_cursor_alias_is_supported(tmp_path):
    store = CursorStore(tmp_path / "cursor.json")
    store.path.write_text('{"version": 1, "last_event_id": "000000123"}')
    assert store.load() == 123


@pytest.mark.parametrize("dangling", [False, True])
def test_cursor_symlink_is_never_followed(tmp_path, dangling):
    target = tmp_path / "secret.json"
    if not dangling:
        target.write_text('{"version": 1, "last_event_id": 123}')
    store = CursorStore(tmp_path / "cursor.json")
    store.path.symlink_to(target)
    with pytest.raises(StateError) as exc:
        store.load()
    assert str(target) not in str(exc.value)


def test_fifo_is_rejected_without_waiting_for_a_writer(tmp_path):
    store = CursorStore(tmp_path / "cursor.fifo")
    os.mkfifo(store.path)
    with pytest.raises(StateError):
        store.load()


def test_directory_is_not_a_cursor(tmp_path):
    with pytest.raises(StateError):
        CursorStore(tmp_path).load()


def test_read_errors_are_sanitized(tmp_path, monkeypatch):
    store = CursorStore(tmp_path / "private-token-should-not-appear.json")

    def fail_open(*args, **kwargs):
        raise PermissionError(f"private-token at {store.path}")

    monkeypatch.setattr(state.os, "open", fail_open)
    with pytest.raises(StateError) as exc:
        store.load()
    assert "private-token" not in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__


@pytest.mark.asyncio
async def test_atomic_replace_observes_complete_private_temp_file(tmp_path, monkeypatch):
    store = CursorStore(tmp_path / "cursor.json")
    await store.save(1)
    original_replace = state.os.replace
    observed = []

    def checked_replace(source, destination):
        assert Path(source).parent == store.path.parent
        assert stat.S_IMODE(Path(source).stat().st_mode) == 0o600
        observed.append(json.loads(Path(source).read_text()))
        assert store.load() == 1
        original_replace(source, destination)

    monkeypatch.setattr(state.os, "replace", checked_replace)
    await store.save(2)
    assert observed == [{"version": 1, "last_event_id": 2}]
    assert store.load() == 2
    assert list(tmp_path.iterdir()) == [store.path]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["fsync", "replace"])
async def test_failed_write_retains_old_cursor_and_removes_temporary_file(
    tmp_path, monkeypatch, operation
):
    store = CursorStore(tmp_path / "cursor.json")
    await store.save(11)

    def fail(*args):
        raise OSError(f"secret filesystem detail: {tmp_path}")

    monkeypatch.setattr(state.os, operation, fail)
    with pytest.raises(StateError) as exc:
        await store.save(12)
    assert "secret" not in str(exc.value) and str(tmp_path) not in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__
    assert store.load() == 11
    assert list(tmp_path.iterdir()) == [store.path]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [True, -1, 1.0, None, "invalid", 10**30])
async def test_invalid_cursor_does_not_create_state_directory(tmp_path, invalid):
    store = CursorStore(tmp_path / "not-created" / "cursor.json")
    with pytest.raises(StateError):
        await store.save(invalid)
    assert not store.path.parent.exists()


@pytest.mark.asyncio
async def test_save_replaces_final_symlink_without_writing_its_target(tmp_path):
    target = tmp_path / "unrelated.json"
    target.write_text("Do not change")
    store = CursorStore(tmp_path / "cursor.json")
    store.path.symlink_to(target)
    await store.save(17)
    assert not store.path.is_symlink() and store.load() == 17
    assert target.read_text() == "Do not change"


@pytest.mark.asyncio
async def test_temporary_cleanup_error_is_sanitized(tmp_path, monkeypatch):
    store = CursorStore(tmp_path / "cursor.json")
    original_unlink = Path.unlink

    def fail_cleanup(path, *args, **kwargs):
        if path.name.startswith(".cursor-"):
            raise PermissionError(f"private-token at {path}")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_cleanup)
    with pytest.raises(StateError) as exc:
        await store.save(2)
    assert "private-token" not in str(exc.value) and str(tmp_path) not in str(exc.value)
    assert exc.value.__cause__ is None and exc.value.__suppress_context__


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellations", [1, 2, 3])
async def test_cancelled_save_waits_for_writer_even_after_repeated_cancellation(
    tmp_path, monkeypatch, cancellations
):
    store = CursorStore(tmp_path / "cursor.json")
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    original_save = store._save

    def blocked_save(value):
        entered.set()
        assert release.wait(5), "Test failed to release cursor writer"
        original_save(value)
        finished.set()

    monkeypatch.setattr(store, "_save", blocked_save)
    task = asyncio.create_task(store.save(41))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        for _ in range(cancellations):
            task.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 2)
        assert finished.is_set() and store.load() == 41
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_writer_failure_after_cancellation_is_observed(tmp_path, monkeypatch):
    store = CursorStore(tmp_path / "cursor.json")
    entered = threading.Event()
    release = threading.Event()

    def failed_save(value):
        entered.set()
        assert release.wait(5)
        raise StateError("VK polling cursor could not be saved; polling stopped")

    monkeypatch.setattr(store, "_save", failed_save)
    task = asyncio.create_task(store.save(41))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(StateError, match="polling stopped"):
            await asyncio.wait_for(task, 2)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
