"""Gateway behavior against real Hermes classes; only the remote VK API is mocked."""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from gateway.platforms.event import MessageEvent, MessageType
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from hermes_vk_workspace.adapter import standalone_send
from hermes_vk_workspace.api import ApiError
from hermes_vk_workspace.formatting import render_chunks, utf16_length
from hermes_vk_workspace.media import Media, local_file
from hermes_vk_workspace.state import StateError


def wire(
    *,
    chat="alice@example",
    user="alice@example",
    kind="private",
    message="hi",
    msg_id="10",
    update=1,
    parts=None,
    parent=None,
):
    payload = {
        "msgId": msg_id,
        "chat": {"chatId": chat, "type": kind, "title": "Test chat"},
        "from": {"userId": user, "firstName": "Alice"},
        "text": message,
        "parts": parts or [],
        "timestamp": 1730000000,
    }
    if parent:
        payload["parent_topic"] = {"chatId": parent, "messageId": "root-1"}
    return {"eventId": update, "type": "newMessage", "payload": payload}


def inbound(adapter, *, chat="alice@example", user="alice@example", msg_id="1"):
    return MessageEvent(
        text="hello",
        source=adapter.build_source(chat, user_id=user, message_id=msg_id),
        message_id=msg_id,
    )


async def test_gateway_admission_processes_new_message_without_reimplementing_sessions(adapter):
    handled = asyncio.Event()
    seen = []

    async def handler(event):
        seen.append(event)
        assert adapter.current_event() is event
        handled.set()
        return "**Hello** from Hermes"

    adapter.set_message_handler(handler)
    await adapter._process_batch([wire()])
    await asyncio.wait_for(handled.wait(), 2)
    await asyncio.gather(*list(adapter._background_tasks))
    assert seen[0].source.platform.value == "vk-workspace"
    assert seen[0].message_id == "10"
    assert adapter._cursor == 1
    assert adapter._state.load() == 1
    assert adapter.api.send_text.await_count == 1
    assert "<b>Hello</b>" in adapter.api.send_text.call_args.args[1]
    assert not adapter._active_sessions
    await adapter.disconnect()


@pytest.mark.parametrize("authorized", [False, None, "yes"])
async def test_unapproved_dm_reaches_pairing_without_downloading(adapter, authorized):
    adapter.set_authorization_check(lambda *args, **kwargs: authorized)
    adapter.handle_message = AsyncMock()
    await adapter._dispatch(wire(parts=[{"type": "file", "payload": {"fileId": "secret-file"}}]))
    adapter.api.file_info.assert_not_awaited()
    adapter.api.download.assert_not_awaited()
    event = adapter.handle_message.call_args.args[0]
    assert event.media_urls == []
    assert event.source.user_id == "alice@example"


@pytest.mark.parametrize(
    "allowed,mentioned,authorized,delivered",
    [
        (False, True, True, False),
        (True, False, True, False),
        (True, True, False, False),
        (True, True, True, True),
    ],
)
async def test_group_and_mention_gates_precede_media(
    adapter, allowed, mentioned, authorized, delivered
):
    from dataclasses import replace

    adapter.settings = replace(adapter.settings, allowed_chats=("team@chat",) if allowed else ())
    adapter.set_authorization_check(lambda *args, **kwargs: authorized)
    adapter.handle_message = AsyncMock()
    adapter.api.file_info.side_effect = ApiError("files/getInfo", "api_rejected")
    event = wire(
        chat="team@chat",
        kind="group",
        message="@[bot@example] hi" if mentioned else "hi",
        parts=[{"type": "file", "payload": {"fileId": "private-file"}}],
    )
    await adapter._dispatch(event)
    assert adapter.handle_message.await_count == int(delivered)
    assert adapter.api.file_info.await_count == int(delivered)


async def test_native_thread_is_recipient_and_session_with_parent_gate(adapter):
    from dataclasses import replace

    adapter.settings = replace(adapter.settings, allowed_chats=("parent@chat",))
    adapter.handle_message = AsyncMock()
    await adapter._dispatch(
        wire(chat="child@thread", parent="parent@chat", kind="group", message="@[bot@example] hi")
    )
    source = adapter.handle_message.call_args.args[0].source
    assert source.chat_id == source.thread_id == "child@thread"
    assert source.parent_chat_id == "parent@chat"
    result = await adapter.send(source.chat_id, "answer", metadata={"thread_id": source.thread_id})
    assert result.success
    assert adapter.api.send_text.call_args.args[0] == "child@thread"
    assert not (
        await adapter.send("parent@chat", "answer", metadata={"thread_id": "child@thread"})
    ).success
    assert adapter.api.send_text.await_count == 1


async def test_threads_and_profiles_have_distinct_real_hermes_session_keys(adapter):
    first = adapter.build_source(
        "one@thread",
        chat_type="group",
        user_id="alice@example",
        parent_chat_id="team@chat",
        thread_id="one@thread",
    )
    second = adapter.build_source(
        "two@thread",
        chat_type="group",
        user_id="alice@example",
        parent_chat_id="team@chat",
        thread_id="two@thread",
    )
    assert adapter._source_session_key(first) != adapter._source_session_key(second)
    original = adapter._source_session_key(first)
    adapter.set_owner_profile("work")
    assert adapter._source_session_key(first) != original


async def test_incoming_document_is_cached_as_local_path_with_explicit_text_contract(adapter):
    adapter.handle_message = AsyncMock()
    adapter.api.file_info.return_value = {
        "url": "https://vk.example/file?signed=secret",
        "filename": "../memo.txt",
    }
    adapter.api.download.return_value = Media(b"real document", "memo.txt", "text/plain")
    raw = wire(parts=[{"type": "file", "payload": {"fileId": "F1", "type": "file"}}])
    await adapter._dispatch(raw)
    event = adapter.handle_message.call_args.args[0]
    assert len(event.media_urls) == 1
    assert Path(event.media_urls[0]).read_bytes() == b"real document"
    assert event.media_types == ["text/plain"]
    assert event.media_text_inlined == [False]
    assert event.message_type == MessageType.DOCUMENT
    assert "signed=" not in event.text
    assert "https://" not in event.media_urls[0]


async def test_media_failure_is_visible_context_and_does_not_poison_cursor(adapter):
    adapter.handle_message = AsyncMock()
    adapter.api.file_info.side_effect = ApiError("files/getInfo", "media_size_limit")
    await adapter._process_batch([wire(parts=[{"type": "file", "payload": {"fileId": "F1"}}])])
    event = adapter.handle_message.call_args.args[0]
    assert "attachment could not" in event.text
    assert event.media_urls == []
    assert adapter._cursor == 1


async def test_replay_duplicate_message_and_out_of_order_batch_are_admitted_once(adapter):
    adapter.handle_message = AsyncMock()
    await adapter._process_batch(
        [wire(update=3, msg_id="30"), wire(update=1), wire(update=2, msg_id="20")]
    )
    assert [call.args[0].message_id for call in adapter.handle_message.call_args_list] == [
        "10",
        "20",
        "30",
    ]
    await adapter._process_batch([wire(update=3, msg_id="30"), wire(update=4, msg_id="30")])
    assert adapter.handle_message.await_count == 3
    assert adapter._cursor == 4


async def test_malformed_cursor_validated_before_any_handoff(adapter):
    adapter.handle_message = AsyncMock()
    malformed = wire(update=True, msg_id="20")
    with pytest.raises(ValueError):
        await adapter._process_batch([wire(), malformed])
    adapter.handle_message.assert_not_awaited()
    assert adapter._cursor == 0


async def test_failed_handoff_releases_dedup_but_keeps_successful_prefix(adapter):
    adapter.handle_message = AsyncMock(side_effect=[None, RuntimeError("host failed"), None])
    batch = [wire(), wire(update=2, msg_id="20")]
    with pytest.raises(RuntimeError):
        await adapter._process_batch(batch)
    assert adapter._cursor == 0
    await adapter._process_batch(batch)
    assert adapter.handle_message.await_count == 3
    assert adapter._cursor == 2


async def test_failed_cursor_save_never_claims_persistence(adapter):
    adapter.handle_message = AsyncMock()
    adapter._state.save = AsyncMock(side_effect=StateError("write failed"))
    with pytest.raises(StateError):
        await adapter._process_batch([wire()])
    assert adapter._cursor == 0


async def test_long_html_send_has_balanced_wire_chunks_and_last_id_contract(adapter):
    result = await adapter.send("team@chat", "**" + "<&😀" * 1600 + "**", reply_to="original")
    assert result.success
    calls = adapter.api.send_text.call_args_list
    assert len(calls) > 1
    assert all(utf16_length(call.args[1]) <= 4096 for call in calls)
    assert calls[0].kwargs["reply_to"] == "original"
    assert all(call.kwargs["reply_to"] is None for call in calls[1:])
    assert result.message_id == str(len(calls))
    assert result.continuation_message_ids == tuple(str(i) for i in range(1, len(calls)))


async def test_ambiguous_send_blocks_same_turn_including_child_stream_tasks_only(adapter):
    event = inbound(adapter)
    await adapter.on_processing_start(event)
    adapter.api.send_text.side_effect = ApiError("messages/sendText", "timeout")
    result = await asyncio.create_task(adapter.send("alice@example", "first"))
    assert not result.success and not result.retryable
    result = await adapter.send("alice@example", "automatic repeat")
    assert not result.success and result.raw_response["delivery_uncertain"]
    assert adapter.api.send_text.await_count == 1
    adapter.api.send_text.side_effect = None
    adapter.api.send_text.return_value = "2"
    assert (await adapter.send("other@example", "independent")).success
    await adapter.on_processing_start(inbound(adapter, msg_id="2"))
    assert (await adapter.send("alice@example", "new turn")).success


async def test_only_confirmed_429_is_retryable_and_partial_send_never_is(adapter):
    adapter.api.send_text.side_effect = ApiError(
        "messages/sendText", "rate_limited", status=429, retry_after=3
    )
    result = await adapter.send("alice@example", "hello")
    assert result.retryable and result.retry_after == 3
    adapter.api.send_text.side_effect = [
        "first",
        ApiError("messages/sendText", "rate_limited", status=429),
    ]
    result = await adapter.send("alice@example", "x" * 8000)
    assert not result.retryable and not result.success
    assert result.message_id == "first"


async def test_edits_and_deletes_require_owned_message_and_overflow_does_not_mutate(adapter):
    assert not (await adapter.edit_message("alice@example", "foreign", "overwrite")).success
    assert not await adapter.delete_message("alice@example", "foreign")
    sent = await adapter.send("alice@example", "initial")
    assert (
        await adapter.edit_message("alice@example", sent.message_id, "**updated**", finalize=True)
    ).success
    count = adapter.api.edit_text.await_count
    assert not (await adapter.edit_message("alice@example", sent.message_id, "&" * 4000)).success
    assert adapter.api.edit_text.await_count == count
    assert await adapter.delete_message("alice@example", sent.message_id)
    assert not await adapter.delete_message("alice@example", sent.message_id)


async def test_file_policy_rejection_happens_before_upload(adapter):
    result = await adapter.send_document("alice@example", "/etc/passwd")
    assert not result.success
    adapter.api.send_file.assert_not_awaited()


@pytest.mark.skipif(not hasattr(os, "O_NOFOLLOW"), reason="POSIX no-follow contract")
async def test_file_replaced_with_symlink_after_host_validation_is_not_uploaded(
    adapter, tmp_path, monkeypatch
):
    allowed = tmp_path / "report.txt"
    allowed.write_text("public report")
    secret = Path(os.environ["HERMES_HOME"]) / ".env"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("private test credential")
    assert adapter.validate_media_delivery_path(str(secret)) is None

    async def replaced_after_validation(path, limit):
        assert Path(path) == allowed
        allowed.unlink()
        allowed.symlink_to(secret)
        return await local_file(path, limit)

    monkeypatch.setattr("hermes_vk_workspace.adapter.local_file", replaced_after_validation)
    result = await adapter.send_document("alice@example", str(allowed))
    assert not result.success
    adapter.api.send_file.assert_not_awaited()


async def test_long_caption_and_native_voice_keep_all_text_without_wrong_voice_caption(
    adapter, tmp_path
):
    text = "caption & " * 200
    document = tmp_path / "note.txt"
    document.write_text("hello")
    result = await adapter.send_document(
        "alice@example", str(document), caption=text, file_name="../safe.txt"
    )
    assert result.success
    assert adapter.api.send_file.call_args.args[2] == "safe.txt"
    assert adapter.api.send_file.call_args.kwargs["caption"] == ""
    assert [call.args[1] for call in adapter.api.send_text.call_args_list] == render_chunks(text)
    adapter.api.send_text.reset_mock()
    audio = tmp_path / "voice.dat"
    audio.write_bytes(b"OggS" + b"\0" * 64)
    result = await adapter.send_voice("alice@example", str(audio), caption="voice caption")
    assert result.success
    assert "caption" not in adapter.api.send_voice.call_args.kwargs
    assert "voice caption" in adapter.api.send_text.call_args.args[1]


@pytest.mark.parametrize("mode", ["first", "all", "off"])
@pytest.mark.parametrize("voice", [False, True])
async def test_separate_captions_follow_reply_mode(adapter, mode, voice):
    adapter.config.reply_to_mode = mode
    media = Media(b"OggS" if voice else b"hello", "attachment.bin", "application/octet-stream")
    result = await adapter._upload(
        "team@chat", media, caption="x" * 5000, reply_to="original", voice=voice
    )
    assert result.success
    sender = adapter.api.send_voice if voice else adapter.api.send_file
    assert sender.call_args.kwargs["reply_to"] == (None if mode == "off" else "original")
    assert adapter.api.send_text.await_count == 2
    assert all(
        call.kwargs["reply_to"] == ("original" if mode == "all" else None)
        for call in adapter.api.send_text.call_args_list
    )


async def test_voice_conversion_failure_does_not_send_document(adapter, tmp_path, monkeypatch):
    path = tmp_path / "tts.mp3"
    path.write_bytes(b"ID3" + b"\0" * 40)
    monkeypatch.setattr("gateway.platforms.base.transcode_to_ogg_opus", Mock(return_value=None))
    result = await adapter.send_voice("alice@example", str(path))
    assert not result.success and "ffmpeg" in result.error
    adapter.api.send_voice.assert_not_awaited()
    adapter.api.send_file.assert_not_awaited()


async def test_typing_clear_and_disconnect_order(adapter, monkeypatch):
    order = []
    await adapter.send_typing("alice@example")
    await adapter.stop_typing("alice@example")
    assert [call.args[1] for call in adapter.api.typing.call_args_list] == [True, False]
    monkeypatch.setattr(
        adapter, "cancel_background_tasks", AsyncMock(side_effect=lambda: order.append("tasks"))
    )
    adapter.api.close.side_effect = lambda: order.append("http")
    monkeypatch.setattr(adapter, "_release_platform_lock", lambda: order.append("lock"))
    await adapter.disconnect()
    assert order == ["tasks", "http", "lock"]
    assert not (await adapter.send("alice@example", "late")).success


async def test_connect_retains_cursor_and_cleans_failed_identity_probe(adapter, monkeypatch):
    adapter._running = False
    monkeypatch.setattr(adapter, "_acquire_platform_lock", Mock(return_value=True))
    release = Mock()
    monkeypatch.setattr(adapter, "_release_platform_lock", release)
    adapter.api.self.side_effect = ApiError("self/get", "http_error", status=401)
    assert not await adapter.connect()
    release.assert_called_once()
    adapter.api.close.assert_awaited_once()
    assert not adapter._fatal_error_retryable
    adapter.api.self.side_effect = None
    adapter.api.self.return_value = {"userId": "bot@example"}
    await adapter._state.save(9876543210123456789)
    monkeypatch.setattr(adapter, "_wire_plugin_handlers", Mock())
    monkeypatch.setattr(adapter, "_poll", AsyncMock())
    assert await adapter.connect(is_reconnect=True)
    assert adapter._cursor == 9876543210123456789
    await adapter.disconnect()


async def test_real_hermes_stream_finishes_without_duplicate_prefix_on_html_overflow(adapter):
    await adapter.on_processing_start(inbound(adapter))
    visible = {}

    async def send(chat, text, **kwargs):
        message_id = str(len(visible) + 1)
        visible[message_id] = text
        return message_id

    async def edit(chat, message_id, text, **kwargs):
        visible[message_id] = text

    adapter.api.send_text.side_effect = send
    adapter.api.edit_text.side_effect = edit
    cfg = StreamConsumerConfig(edit_interval=0.01, buffer_threshold=20, cursor="", transport="edit")
    consumer = GatewayStreamConsumer(adapter, "alice@example", cfg)
    prefix = "The beginning " + "x" * 200
    consumer.on_delta(prefix)
    task = asyncio.create_task(consumer.run())
    for _ in range(100):
        if visible:
            break
        await asyncio.sleep(0.005)
    assert visible
    final = prefix + " &" * 2200 + " END"
    consumer.on_delta(final[len(prefix) :])
    consumer.finish(final_text=final)
    await asyncio.wait_for(task, 3)
    import re
    from html import unescape

    actual = "".join(unescape(re.sub(r"<[^>]*>", "", part)) for part in visible.values())
    # Hermes trims boundary whitespace between independently displayed bubbles.
    # Every content character, including escaped entities, must appear exactly once.
    assert re.sub(r"\s+", "", actual) == re.sub(r"\s+", "", final)
    assert all(utf16_length(part) <= 4096 for part in visible.values())


async def test_standalone_sender_never_acquires_consumer_lock_or_polls(adapter, monkeypatch):
    monkeypatch.setattr(
        "hermes_vk_workspace.adapter.VKWorkspaceAdapter", Mock(return_value=adapter)
    )
    lock = Mock(side_effect=AssertionError("standalone must not acquire lock"))
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lock)
    result = await standalone_send(adapter.config, "alice@example", "scheduled message")
    assert result == {"success": True, "message_id": "1"}
    adapter.api.self.assert_not_awaited()
    adapter.api.events.assert_not_awaited()
    adapter.api.close.assert_awaited_once()


@pytest.mark.parametrize(
    "code,status",
    [("timeout", None), ("network_error", None), ("http_error", 503), ("api_rejected", None)],
)
async def test_core_send_retry_never_replays_uncertain_or_rejected_send_without_turn_scope(
    adapter, code, status
):
    adapter.api.send_text.side_effect = [
        ApiError("messages/sendText", code, status=status),
        "wrong-retry",
    ]
    result = await adapter._send_with_retry("alice@example", "hello", max_retries=0)
    assert not result.success
    assert adapter.api.send_text.await_count == 1


async def test_core_send_retry_stops_after_partial_429_and_can_retry_initial_429(
    adapter, monkeypatch
):
    adapter.api.send_text.side_effect = [
        "first",
        ApiError("messages/sendText", "rate_limited", status=429),
        "duplicate",
    ]
    result = await adapter._send_with_retry(
        "alice@example", "x" * 8000, max_retries=1, base_delay=0
    )
    assert not result.success and result.message_id == "first"
    assert adapter.api.send_text.await_count == 2
    adapter.api.send_text.reset_mock()
    adapter.api.send_text.side_effect = [
        ApiError("messages/sendText", "rate_limited", status=429),
        "sent",
    ]
    monkeypatch.setattr("gateway.platforms.base.random.uniform", lambda *_: 0)
    result = await adapter._send_with_retry("alice@example", "hello", max_retries=1, base_delay=0)
    assert result.success and result.message_id == "sent"
    assert adapter.api.send_text.await_count == 2


async def test_uncertain_native_picker_blocks_host_text_continuation(adapter):
    event = inbound(adapter)
    event.text = "/reasoning"
    await adapter.on_processing_start(event)
    adapter.api.send_text.side_effect = [
        ApiError("messages/sendText", "network_error"),
        "wrong-continuation",
    ]
    picker = await adapter.send_choice_picker(
        "alice@example",
        "Reasoning",
        [{"value": "high", "label": "High"}],
        adapter._event_session_key(event),
        AsyncMock(return_value="done"),
    )
    assert not picker.success
    result = await adapter._send_with_retry(
        "alice@example", "Current reasoning: medium", max_retries=0
    )
    assert not result.success
    assert adapter.api.send_text.await_count == 1


@pytest.mark.parametrize("kind", ["text", "edit", "file", "voice", "picker"])
async def test_cancellation_during_mutation_blocks_following_send(adapter, kind):
    event = inbound(adapter)
    await adapter.on_processing_start(event)
    entered = asyncio.Event()

    async def cancelled_request(*args, **kwargs):
        entered.set()
        await asyncio.Future()

    adapter.api.send_text.side_effect = cancelled_request
    adapter.api.edit_text.side_effect = cancelled_request
    adapter.api.send_file.side_effect = cancelled_request
    adapter.api.send_voice.side_effect = cancelled_request
    if kind == "edit":
        adapter.remember_message("alice@example", "owned")
        operation = adapter.edit_message("alice@example", "owned", "replacement")
    elif kind in {"file", "voice"}:
        operation = adapter._upload(
            "alice@example", Media(b"OggSdata", "voice.ogg", "audio/ogg"), voice=kind == "voice"
        )
    elif kind == "picker":
        operation = adapter.send_choice_picker(
            "alice@example",
            "Pick",
            [{"value": "a"}],
            adapter._event_session_key(event),
            AsyncMock(return_value="done"),
        )
    else:
        operation = adapter.send("alice@example", "hello")
    task = asyncio.create_task(operation)
    await asyncio.wait_for(entered.wait(), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    adapter.api.send_text.side_effect = None
    adapter.api.send_text.return_value = "duplicate"
    calls = adapter.api.send_text.await_count
    assert not (await adapter.send("alice@example", "automatic repeat")).success
    assert adapter.api.send_text.await_count == calls


async def test_disconnect_during_first_chunk_prevents_late_continuations(adapter):
    async def first_chunk(*args, **kwargs):
        adapter._running = False
        return "already-in-flight"

    adapter.api.send_text.side_effect = first_chunk
    result = await adapter.send("alice@example", "x" * 9000)
    assert not result.success
    assert result.message_id == "already-in-flight"
    assert adapter.api.send_text.await_count == 1


async def test_callback_execution_does_not_hold_poll_and_is_cancelled_at_disconnect(adapter):
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def callback(payload):
        entered.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    adapter.pickers.handle = callback
    await adapter._process_batch(
        [{"eventId": 1, "type": "callbackQuery", "payload": {"queryId": "Q"}}]
    )
    await asyncio.wait_for(entered.wait(), 1)
    assert adapter._cursor == 1
    assert len(adapter._callback_tasks) == 1
    await adapter.disconnect()
    assert cancelled.is_set()
    assert not adapter._callback_tasks


@pytest.mark.parametrize("caption_prefix", [[], ["caption-1"]])
async def test_file_receipt_survives_failed_separate_caption(adapter, caption_prefix):
    adapter.api.send_text.side_effect = [*caption_prefix, ApiError("messages/sendText", "timeout")]
    result = await adapter._upload(
        "alice@example", Media(b"hello", "note.txt", "text/plain"), caption="x" * 9000
    )
    expected = ["file-1", *caption_prefix]
    assert not result.success and not result.retryable
    assert result.message_id == expected[-1]
    assert result.continuation_message_ids == tuple(expected[:-1])
    assert result.raw_response["delivered_message_ids"] == expected


async def test_generic_audio_attachment_is_not_limited_to_native_voice_containers(adapter):
    adapter.handle_message = AsyncMock()
    adapter.api.file_info.return_value = {
        "url": "https://vk.example/file",
        "filename": "recording.flac",
    }
    adapter.api.download.return_value = Media(b"fLaC" + b"\0" * 64, "recording.flac", "audio/flac")
    await adapter._dispatch(
        wire(parts=[{"type": "file", "payload": {"fileId": "F1", "type": "audio"}}])
    )
    event = adapter.handle_message.call_args.args[0]
    assert len(event.media_urls) == 1
    assert event.media_types == ["audio/flac"]
    assert event.message_type == MessageType.AUDIO


async def test_flac_voice_uses_conversion_then_native_voice(adapter, tmp_path, monkeypatch):
    path = tmp_path / "recording.flac"
    path.write_bytes(b"fLaC" + b"\0" * 64)

    def transcode(source, *, output_path):
        assert Path(source).read_bytes().startswith(b"fLaC")
        Path(output_path).write_bytes(b"OggS" + b"\0" * 64)
        return output_path

    monkeypatch.setattr("gateway.platforms.base.transcode_to_ogg_opus", transcode)
    result = await adapter.send_voice("alice@example", str(path))
    assert result.success
    assert adapter.api.send_voice.call_args.args[1].startswith(b"OggS")
    adapter.api.send_file.assert_not_awaited()
