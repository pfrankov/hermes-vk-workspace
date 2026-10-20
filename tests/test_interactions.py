"""Native pickers use real Hermes events, session keys, authorization and provider shapes."""

import asyncio
from contextvars import ContextVar
from copy import deepcopy
from unittest.mock import AsyncMock

import hermes_vk_workspace.interactions as interactions
import pytest
from gateway.platforms.event import MessageEvent
from hermes_cli.model_switch_providers import list_picker_providers
from hermes_vk_workspace.api import ApiError
from hermes_vk_workspace.interactions import MAX_MENUS, MENU_TTL

CHAT = "engineering@chat.agent"
USER = "alice@example.com"
BOT = "hermes@example.com"


@pytest.fixture
async def adapter(adapter):
    """Extend the shared real-adapter fixture with an active native-menu request."""
    adapter.self_id = BOT
    adapter.api.send_text.side_effect = lambda *_args, **_kwargs: (
        f"menu-{adapter.api.send_text.call_count}"
    )
    adapter.allowed = True
    adapter.set_authorization_check(lambda *_args, **_kwargs: adapter.allowed)
    adapter.event = MessageEvent(
        text="/model",
        source=adapter.build_source(chat_id=CHAT, chat_type="group", user_id=USER),
        message_id="request-1",
    )
    await adapter.on_processing_start(adapter.event)
    yield adapter
    adapter.pickers.clear()


def callback_payload(token, message_id="menu-1", query_id="query-1"):
    return {
        "queryId": query_id,
        "callbackData": token,
        "from": {"userId": USER},
        "message": {
            "msgId": message_id,
            "from": {"userId": BOT},
            "chat": {"chatId": CHAT, "type": "group"},
        },
    }


def keyboard(adapter, *, edited=False):
    method = adapter.api.edit_text if edited else adapter.api.send_text
    return method.call_args.kwargs["keyboard"]


def token_named(rows, text):
    return next(button["callbackData"] for row in rows for button in row if button["text"] == text)


async def choice(adapter, count=2, callback=None, title="Choose **reasoning**"):
    callback = callback or AsyncMock(return_value="**Reasoning updated.**")
    result = await adapter.pickers.choice(
        CHAT,
        title,
        [
            {"value": f"level-{i}", "label": f"Level {i}", "is_current": i == 0}
            for i in range(count)
        ],
        adapter._event_session_key(adapter.event),
        callback,
    )
    return result, callback


async def test_choice_invokes_public_callback_once_with_exact_value_and_retires_siblings(adapter):
    result, callback = await choice(adapter)
    assert result.success
    rows = keyboard(adapter)
    chosen = rows[0][0]["callbackData"]
    sibling = rows[1][0]["callbackData"]
    assert chosen.startswith("vkp:") and len(chosen) < 256
    assert all(value not in chosen for value in (USER, CHAT, "level-0", result.message_id))
    assert rows[0][0]["style"] == "primary"
    assert adapter.api.send_text.call_args.kwargs["reply_to"] == "request-1"
    assert adapter.api.send_text.call_args.kwargs["parse_mode"] == "HTML"
    await adapter.pickers.handle(callback_payload(chosen))
    callback.assert_awaited_once_with(CHAT, "level-0")
    assert adapter.api.edit_text.call_args.kwargs["keyboard"] == []
    assert "<b>Reasoning updated.</b>" in adapter.api.edit_text.call_args.args[2]
    await adapter.pickers.handle(callback_payload(chosen, query_id="repeat"))
    await adapter.pickers.handle(callback_payload(sibling, query_id="sibling"))
    callback.assert_awaited_once()
    assert adapter.api.answer_callback.call_args.kwargs["show_alert"] is True


async def test_concurrent_taps_are_consumed_before_first_network_await(adapter):
    _, callback = await choice(adapter)
    token = keyboard(adapter)[0][0]["callbackData"]
    entered, release = asyncio.Event(), asyncio.Event()

    async def ack(_query_id, text, show_alert):
        if text == "Selection received":
            entered.set()
            await release.wait()

    adapter.api.answer_callback.side_effect = ack
    first = asyncio.create_task(adapter.pickers.handle(callback_payload(token)))
    await entered.wait()
    await adapter.pickers.handle(callback_payload(token, query_id="concurrent"))
    callback.assert_not_awaited()
    release.set()
    await first
    callback.assert_awaited_once()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("from", "userId"), "mallory@example.com"),
        (("from", "userId"), None),
        (("from", "isBot"), True),
        (("message", "from", "userId"), USER),
        (("message", "from", "userId"), None),
        (("message", "chat", "chatId"), "other@chat.agent"),
        (("message", "chat", "type"), "private"),
        (("message", "msgId"), "other-message"),
        (("message", "msgId"), True),
        (("callbackData",), "level-0"),
        (("message", "from"), []),
        (("message",), None),
    ],
)
async def test_untrusted_callback_fields_cannot_retarget_or_authorize_a_menu(adapter, path, value):
    _, callback = await choice(adapter)
    original = callback_payload(keyboard(adapter)[0][0]["callbackData"])
    tampered = deepcopy(original)
    node = tampered
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    await adapter.pickers.handle(tampered)
    callback.assert_not_awaited()
    assert adapter.api.answer_callback.call_args.kwargs["show_alert"] is True
    # An attack cannot consume the real requester's menu.
    await adapter.pickers.handle(original)
    callback.assert_awaited_once_with(CHAT, "level-0")


@pytest.mark.parametrize("authorized", [False, None, 1, "yes"])
async def test_click_requires_current_literal_true_hermes_authorization(adapter, authorized):
    _, callback = await choice(adapter)
    token = keyboard(adapter)[0][0]["callbackData"]
    adapter.allowed = authorized
    await adapter.pickers.handle(callback_payload(token))
    callback.assert_not_awaited()
    assert adapter.api.answer_callback.call_args.kwargs["show_alert"] is True


async def test_authorization_is_checked_again_after_acknowledgement_await(adapter):
    _, callback = await choice(adapter)
    token = keyboard(adapter)[0][0]["callbackData"]

    async def revoke(*_args, **_kwargs):
        adapter.allowed = False

    adapter.api.answer_callback.side_effect = revoke
    await adapter.pickers.handle(callback_payload(token))
    callback.assert_not_awaited()
    assert "unavailable" in adapter.api.edit_text.call_args.args[2]


@pytest.mark.parametrize("invalidation", ["disconnect", "expiry"])
async def test_consumed_selection_rechecks_disconnect_and_expiry_after_ack(
    adapter, monkeypatch, invalidation
):
    now = [100.0]
    monkeypatch.setattr(interactions, "monotonic", lambda: now[0])
    _, callback = await choice(adapter)
    token = keyboard(adapter)[0][0]["callbackData"]

    async def invalidate(*_args, **_kwargs):
        if invalidation == "disconnect":
            adapter.pickers.clear()
        else:
            now[0] += MENU_TTL

    adapter.api.answer_callback.side_effect = invalidate
    await adapter.pickers.handle(callback_payload(token))
    callback.assert_not_awaited()
    assert "unavailable" in adapter.api.edit_text.call_args.args[2]


async def test_session_and_bot_replacement_invalidate_bound_callbacks(adapter):
    _, callback = await choice(adapter)
    first_token = keyboard(adapter)[0][0]["callbackData"]
    adapter.config.extra["group_sessions_per_user"] = False
    await adapter.pickers.handle(callback_payload(first_token))
    callback.assert_not_awaited()
    _, callback = await choice(adapter)
    second_token = keyboard(adapter)[0][0]["callbackData"]
    adapter.self_id = "replacement-bot@example.com"
    await adapter.pickers.handle(callback_payload(second_token, "menu-2"))
    callback.assert_not_awaited()


async def test_callback_runs_in_captured_context_and_original_requester_is_bound_in_shared_group(
    adapter,
):
    adapter.config.extra["group_sessions_per_user"] = False
    marker = ContextVar("request_profile", default="default-profile")
    binding = marker.set("alice-profile")
    seen = []

    async def selected(chat_id, value):
        seen.append((marker.get(), adapter.current_event().source.user_id, chat_id, value))
        return "Changed."

    _, _ = await choice(adapter, callback=selected)
    token = keyboard(adapter)[0][0]["callbackData"]
    marker.reset(binding)
    await adapter.on_processing_start(
        MessageEvent(
            text="a different request",
            source=adapter.build_source(chat_id=CHAT, chat_type="group", user_id="bob@example.com"),
            message_id="request-2",
        )
    )
    await adapter.pickers.handle(callback_payload(token))
    assert seen == [("alice-profile", USER, CHAT, "level-0")]


async def test_choice_starts_on_page_with_current_value(adapter):
    callback = AsyncMock(return_value="Changed")
    result = await adapter.pickers.choice(
        CHAT,
        "Choose reasoning",
        [{"value": f"r-{i}", "label": f"Level {i}", "is_current": i == 13} for i in range(14)],
        adapter._event_session_key(adapter.event),
        callback,
    )
    assert result.success
    rows = keyboard(adapter)
    assert len(rows[:-1]) == 2
    assert rows[1][0]["text"].startswith("14.")
    assert rows[1][0]["style"] == "primary"
    await adapter.pickers.handle(callback_payload(rows[1][0]["callbackData"]))
    callback.assert_awaited_once_with(CHAT, "r-13")


async def test_gateway_stopping_during_ack_rejects_old_picker_action(adapter):
    _, callback = await choice(adapter)
    token = keyboard(adapter)[0][0]["callbackData"]
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed_ack(*_args, **_kwargs):
        entered.set()
        await release.wait()

    adapter.api.answer_callback.side_effect = delayed_ack
    task = asyncio.create_task(adapter.pickers.handle(callback_payload(token)))
    await entered.wait()
    # disconnect marks the platform stopped before callbacks and socket teardown complete.
    adapter._running = False
    release.set()
    await task
    callback.assert_not_awaited()


async def test_choices_paginate_six_rows_and_retire_old_page_tokens(adapter):
    _, callback = await choice(adapter, count=14)
    first_rows = keyboard(adapter)
    assert len(first_rows[:-1]) == 6
    stale = first_rows[0][0]["callbackData"]
    await adapter.pickers.handle(callback_payload(token_named(first_rows, "Next")))
    second_rows = keyboard(adapter, edited=True)
    assert len(second_rows[:-1]) == 6
    assert second_rows[0][0]["text"].startswith("7.")
    assert {button["text"] for button in second_rows[-1]} == {"Previous", "Next", "Cancel"}
    await adapter.pickers.handle(callback_payload(stale, query_id="old-page"))
    callback.assert_not_awaited()
    await adapter.pickers.handle(
        callback_payload(token_named(second_rows, "Next"), query_id="last-page")
    )
    last_rows = keyboard(adapter, edited=True)
    assert len(last_rows[:-1]) == 2
    await adapter.pickers.handle(
        callback_payload(last_rows[1][0]["callbackData"], query_id="select-last")
    )
    callback.assert_awaited_once_with(CHAT, "level-13")


async def test_provider_model_drilldown_uses_latest_hermes_provider_shape(adapter, monkeypatch):
    rows = [
        {
            "slug": f"provider-{i}",
            "name": f"Provider {i}",
            "is_current": i == 6,
            "models": [f"namespace/model-{n}" for n in range(8)],
            "total_models": 8,
        }
        for i in range(8)
    ]
    rows += [
        {
            "slug": "custom:local",
            "name": "Local",
            "models": [],
            "total_models": 0,
            "is_user_defined": True,
            "api_url": "https://models.example.com/v1",
        },
        {"slug": "unconfigured", "name": "Empty", "models": [], "total_models": 0},
    ]
    monkeypatch.setattr(
        "hermes_cli.model_switch.list_authenticated_providers", lambda **_kwargs: deepcopy(rows)
    )
    # Exercise the public host builder, including retention of custom empty catalogs.
    providers = list_picker_providers(
        current_provider="provider-6", current_model="namespace/model-7", non_blocking_catalogs=True
    )
    assert providers[-1]["slug"] == "custom:local"
    callback = AsyncMock(return_value="Model changed.")
    result = await adapter.pickers.model(
        CHAT,
        providers,
        "namespace/model-7",
        "provider-6",
        adapter._event_session_key(adapter.event),
        callback,
    )
    assert result.success
    # The active provider/model is immediately visible on its own page.
    initial_models = keyboard(adapter)
    assert len(initial_models[:-1]) == 2
    assert initial_models[1][0]["style"] == "primary"
    await adapter.pickers.handle(
        callback_payload(token_named(initial_models, "Providers"), query_id="back-providers")
    )
    provider_page = keyboard(adapter, edited=True)
    assert provider_page[0][0]["text"].startswith("7.")
    assert provider_page[0][0]["style"] == "primary"
    await adapter.pickers.handle(
        callback_payload(provider_page[0][0]["callbackData"], query_id="reopen-provider")
    )
    reopened_models = keyboard(adapter, edited=True)
    assert len(reopened_models[:-1]) == 2
    assert reopened_models[1][0]["style"] == "primary"
    await adapter.pickers.handle(
        callback_payload(token_named(reopened_models, "Previous"), query_id="model-prev")
    )
    first_models = keyboard(adapter, edited=True)
    assert len(first_models[:-1]) == 6
    await adapter.pickers.handle(
        callback_payload(token_named(first_models, "Next"), query_id="model-next")
    )
    await adapter.pickers.handle(
        callback_payload(
            keyboard(adapter, edited=True)[1][0]["callbackData"], query_id="choose-model"
        )
    )
    callback.assert_awaited_once_with(CHAT, "namespace/model-7", "provider-6")


async def test_empty_custom_catalog_shows_instruction_and_back_button(adapter):
    callback = AsyncMock()
    result = await adapter.pickers.model(
        CHAT,
        [{"slug": "custom:local", "name": "Local", "models": []}],
        "",
        "",
        adapter._event_session_key(adapter.event),
        callback,
    )
    assert result.success
    await adapter.pickers.handle(callback_payload(keyboard(adapter)[0][0]["callbackData"]))
    assert "No catalog models" in adapter.api.edit_text.call_args.args[2]
    assert {button["text"] for button in keyboard(adapter, edited=True)[0]} == {
        "Providers",
        "Cancel",
    }
    callback.assert_not_awaited()


async def test_cancel_expiry_delete_and_disconnect_invalidate_tokens(adapter, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(interactions, "monotonic", lambda: now[0])
    _, callback = await choice(adapter)
    token = keyboard(adapter)[0][0]["callbackData"]
    await adapter.pickers.handle(callback_payload(token_named(keyboard(adapter), "Cancel")))
    callback.assert_not_awaited()
    assert adapter.api.edit_text.call_args.kwargs["keyboard"] == []
    await adapter.pickers.handle(callback_payload(token, query_id="after-cancel"))
    callback.assert_not_awaited()
    _, callback = await choice(adapter)
    token = keyboard(adapter)[0][0]["callbackData"]
    now[0] += MENU_TTL
    await adapter.pickers.handle(callback_payload(token, "menu-2", query_id="expired"))
    callback.assert_not_awaited()
    _, callback = await choice(adapter)
    token = keyboard(adapter)[0][0]["callbackData"]
    adapter.pickers.discard(CHAT, "menu-3")
    await adapter.pickers.handle(callback_payload(token, "menu-3", query_id="deleted"))
    callback.assert_not_awaited()
    _, callback = await choice(adapter)
    token = keyboard(adapter)[0][0]["callbackData"]
    adapter.pickers.clear()
    await adapter.pickers.handle(callback_payload(token, "menu-4", query_id="disconnected"))
    callback.assert_not_awaited()


async def test_menu_cache_is_bounded_and_evicted_tokens_cannot_execute(adapter):
    _, first_callback = await choice(adapter)
    first = keyboard(adapter)[0][0]["callbackData"]
    for _ in range(MAX_MENUS):
        await choice(adapter)
    assert len(adapter.pickers._menus) == MAX_MENUS
    assert len(adapter.pickers._tokens) == MAX_MENUS * 3
    await adapter.pickers.handle(callback_payload(first, query_id="evicted"))
    first_callback.assert_not_awaited()


async def test_pending_menu_send_cannot_reactivate_after_disconnect(adapter):
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed_send(*_args, **_kwargs):
        entered.set()
        await release.wait()
        return "late-message"

    adapter.api.send_text.side_effect = delayed_send
    pending = asyncio.create_task(choice(adapter))
    await entered.wait()
    adapter.pickers.clear()
    release.set()
    result, _ = await pending
    assert result.success is False
    assert not adapter.pickers._menus
    assert not adapter.pickers._tokens


async def test_send_and_edit_failures_do_not_retry_or_leave_actionable_menus(adapter):
    adapter.api.send_text.side_effect = ApiError("messages/sendText", "read_timeout")
    result, callback = await choice(adapter)
    assert result.success is False and result.retryable is False
    assert adapter.api.send_text.call_count == 1
    assert not adapter.pickers._menus
    callback.assert_not_awaited()
    # An ambiguous send blocks this turn; a new user request gets a fresh scope.
    await adapter.on_processing_start(adapter.event)
    adapter.api.send_text.side_effect = lambda *_args, **_kwargs: "menu-nav"
    _, callback = await choice(adapter, count=8)
    rows = keyboard(adapter)
    adapter.api.edit_text.side_effect = ApiError("messages/editText", "read_timeout")
    await adapter.pickers.handle(callback_payload(token_named(rows, "Next"), "menu-nav"))
    assert adapter.api.edit_text.call_count == 1
    assert not adapter.pickers._menus
    assert not adapter.pickers._tokens
    callback.assert_not_awaited()


async def test_ack_failure_does_not_replay_or_lose_an_already_consumed_selection(adapter):
    _, callback = await choice(adapter)
    token = keyboard(adapter)[0][0]["callbackData"]
    adapter.api.answer_callback.side_effect = ApiError(
        "messages/answerCallbackQuery", "read_timeout"
    )
    await adapter.pickers.handle(callback_payload(token))
    await adapter.pickers.handle(callback_payload(token, query_id="repeat"))
    callback.assert_awaited_once()


@pytest.mark.parametrize("failure", [RuntimeError("secret-bot-token"), "x" * 5000, None])
async def test_host_errors_and_oversized_results_are_explicit_without_leaking_or_truncating(
    adapter, caplog, failure
):
    callback = (
        AsyncMock(side_effect=failure)
        if isinstance(failure, Exception)
        else AsyncMock(return_value=failure)
    )
    _, _ = await choice(adapter, callback=callback)
    await adapter.pickers.handle(callback_payload(keyboard(adapter)[0][0]["callbackData"]))
    text = adapter.api.edit_text.call_args.args[2]
    assert (
        "command failed" in text if isinstance(failure, Exception) else "native menu limit" in text
    )
    assert "secret-bot-token" not in text + caplog.text
    assert "xxxx" not in text
    callback.assert_awaited_once()


async def test_long_labels_remain_complete_and_untrusted_markup_is_literal(adapter):
    label = "<script>not code</script> " + "model-name-" * 15
    callback = AsyncMock(return_value="Done.")
    result = await adapter.pickers.choice(
        CHAT,
        "**Choose**",
        [{"value": "safe-choice", "label": label}],
        adapter._event_session_key(adapter.event),
        callback,
    )
    assert result.success
    text = adapter.api.send_text.call_args.args[1]
    assert "&lt;script&gt;not code&lt;/script&gt;" in text
    assert "model-name-" * 15 in text
    assert "<script>" not in text
    assert keyboard(adapter)[0][0]["text"] == "1"


@pytest.mark.parametrize(
    "case",
    [
        "missing_context",
        "wrong_chat",
        "wrong_session",
        "unauthorized",
        "oversized_title",
        "bad_choice",
        "oversized_catalog",
    ],
)
async def test_invalid_menu_requests_fail_before_network_io(adapter, monkeypatch, case):
    chat_id, session_key, title = CHAT, adapter._event_session_key(adapter.event), "Choose"
    choices = [{"value": "one", "label": "One"}]
    if case == "missing_context":
        monkeypatch.setattr(adapter, "current_event", lambda: None)
    elif case == "wrong_chat":
        chat_id = "other@chat.agent"
    elif case == "wrong_session":
        session_key += ":other"
    elif case == "unauthorized":
        adapter.allowed = None
    elif case == "oversized_title":
        title = "x" * 5000
    elif case == "bad_choice":
        choices = [{"value": True}]
    else:
        choices = [{"value": "one"}] * (interactions.MAX_OPTIONS + 1)
    result = await adapter.pickers.choice(chat_id, title, choices, session_key, AsyncMock())
    assert result.success is False
    adapter.api.send_text.assert_not_awaited()


async def test_reply_to_mode_off_is_respected(adapter):
    adapter.config.reply_to_mode = "off"
    result, _ = await choice(adapter)
    assert result.success
    assert adapter.api.send_text.call_args.kwargs["reply_to"] is None
