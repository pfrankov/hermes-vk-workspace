"""Requester-bound native pickers for Hermes' public model and choice callbacks."""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import unicodedata
from collections import OrderedDict
from contextvars import Context, copy_context
from dataclasses import dataclass, field, replace
from time import monotonic
from typing import TYPE_CHECKING, Any

from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent

from .api import ApiError
from .config import PLATFORM, ConfigError, identifier
from .formatting import MAX_TEXT, render_chunks, utf16_length

if TYPE_CHECKING:
    from gateway.session import SessionSource

logger = logging.getLogger(__name__)
PAGE_SIZE = 6
MAX_MENUS = 128
MENU_TTL = 15 * 60
MAX_OPTIONS = 4096
_UNAVAILABLE = "This menu is unavailable. Run the command again."


@dataclass(frozen=True, slots=True)
class _Item:
    label: str
    values: tuple[str, ...] = ()
    current: bool = False
    children: tuple[_Item, ...] | None = None


@dataclass(slots=True)
class _Menu:
    chat_id: str
    bot_id: str
    source: SessionSource
    session_key: str
    context: Context
    callback: Any
    title: str
    items: tuple[_Item, ...]
    expires_at: float
    generation: int
    message_id: str | None = None
    page: int = 0
    provider: int | None = None
    provider_page: int = 0
    tokens: dict[str, tuple[str, int]] = field(default_factory=dict)


def _text(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 2048
        or any(unicodedata.category(char).startswith("C") for char in value)
    ):
        raise ValueError("Picker values must contain 1–2048 visible characters")
    return value


def _card(content: str) -> str:
    chunks = render_chunks(content)
    if len(chunks) != 1:
        raise ValueError("Picker content does not fit one message; use the text command")
    return chunks[0]


def _literal(text: str) -> str:
    # Delimiters must be longer than any literal run in the configured model/label.
    delimiter = "`" * (max((len(run) for run in re.findall(r"`+", text)), default=0) + 1)
    return f"{delimiter} {text} {delimiter}"


class PickerManager:
    def __init__(self, adapter):
        self.adapter = adapter
        self._menus: OrderedDict[int, _Menu] = OrderedDict()
        self._tokens: dict[str, tuple[_Menu, str, int]] = {}
        self._generation = 0

    def _drop(self, menu: _Menu) -> None:
        self._menus.pop(id(menu), None)
        self._retire_tokens(menu)

    def _retire_tokens(self, menu: _Menu) -> None:
        for token in menu.tokens:
            self._tokens.pop(token, None)
        menu.tokens.clear()

    def _prune(self) -> None:
        now = monotonic()
        for menu in tuple(self._menus.values()):
            if menu.expires_at <= now:
                self._drop(menu)

    def discard(self, chat_id: str, message_id: str) -> None:
        for menu in tuple(self._menus.values()):
            if (menu.chat_id, menu.message_id) == (chat_id, message_id):
                self._drop(menu)

    def clear(self) -> None:
        self._generation += 1
        self._menus.clear()
        self._tokens.clear()

    def _authorized(self, menu: _Menu) -> bool:
        source = menu.source
        return (
            self.adapter._running
            and menu.expires_at > monotonic()
            and menu.generation == self._generation
            and self.adapter.self_id == menu.bot_id
            and self.adapter._source_session_key(source) == menu.session_key
            and self.adapter._is_sender_authorized(
                source.user_id,
                source.chat_type,
                source.chat_id,
                is_bot=source.is_bot,
                thread_id=source.thread_id,
            )
            is True
        )

    def _make(self, chat_id, title, items, session_key, callback) -> _Menu:
        event = self.adapter.current_event()
        if not isinstance(event, MessageEvent) or event.source is None:
            raise ValueError("A native picker requires the requesting message context")
        source = event.source
        chat_id = identifier(chat_id)
        bot_id = identifier(self.adapter.self_id)
        user_id = identifier(source.user_id)
        if (
            source.platform.value != PLATFORM
            or source.chat_id != chat_id
            or source.is_bot
            or user_id == bot_id
            or event.internal
            or not event.allow_gateway_control
            or not callable(callback)
            or not isinstance(title, str)
            or not title.strip()
            or self.adapter._event_session_key(event) != session_key
        ):
            raise ValueError("Picker context does not match the requesting chat and session")
        menu = _Menu(
            chat_id,
            bot_id,
            replace(source),
            session_key,
            copy_context(),
            callback,
            title,
            items,
            monotonic() + MENU_TTL,
            self._generation,
        )
        if not self._authorized(menu):
            raise ValueError("The requesting user is not authorized")
        count, size = 0, len(title)
        for item in items:
            for entry in (item, *(item.children or ())):
                count += 1
                size += len(entry.label) + sum(map(len, entry.values))
        if not items or count > MAX_OPTIONS or size > MAX_TEXT:
            raise ValueError("Picker catalog is empty or exceeds the plugin resource limit")
        return menu

    def _render(self, menu: _Menu) -> tuple[str, list, dict[str, tuple[str, int]]]:
        items = menu.items
        lines = [menu.title]
        if menu.provider is not None:
            provider = items[menu.provider]
            lines += ["", "Provider: " + _literal(provider.label)]
            items = provider.children or ()
        pages = max(1, (len(items) + PAGE_SIZE - 1) // PAGE_SIZE)
        lines += ["", f"Page {menu.page + 1}/{pages}"]
        keyboard, tokens = [], {}

        def button(label: str, kind: str, index: int = 0, current: bool = False) -> dict:
            token = "vkp:" + secrets.token_urlsafe(18)
            tokens[token] = (kind, index)
            return {"text": label, "callbackData": token, "style": "primary" if current else "base"}

        start = menu.page * PAGE_SIZE
        for index, item in enumerate(items[start : start + PAGE_SIZE], start):
            number = index + 1
            marker = "✓ " if item.current else ""
            lines.append(f"{number}. {marker}{_literal(item.label)}")
            label = f"{number}. {marker}{item.label}"
            # Full text always appears above; long model IDs use a numbered button.
            if utf16_length(label) > 128:
                label = str(number)
            kind = "open" if item.children is not None else "select"
            keyboard.append([button(label, kind, index, item.current)])
        if not items:
            lines += ["", "No catalog models are available. Specify the model with /model."]
        navigation = []
        if menu.page:
            navigation.append(button("Previous", "page", menu.page - 1))
        if menu.page + 1 < pages:
            navigation.append(button("Next", "page", menu.page + 1))
        if menu.provider is not None:
            navigation.append(button("Providers", "back"))
        navigation.append(button("Cancel", "cancel"))
        keyboard.append(navigation)
        return _card("\n".join(lines)), keyboard, tokens

    def _activate(self, menu: _Menu, tokens: dict[str, tuple[str, int]]) -> None:
        menu.tokens = tokens
        self._tokens.update((token, (menu, *action)) for token, action in tokens.items())

    async def _send(self, menu: _Menu) -> SendResult:
        if failure := self.adapter._ready(menu.chat_id):
            return failure
        text, keyboard, tokens = self._render(menu)
        event = self.adapter.current_event()
        reply_to = None
        if menu.source.chat_type != "dm" and self.adapter.config.reply_to_mode != "off":
            reply_to = event.reply_anchor_override or event.message_id
            reply_to = identifier(reply_to) if reply_to is not None else None
        generation = self._generation
        try:
            message_id = await self.adapter.api.send_text(
                menu.chat_id,
                text,
                reply_to=reply_to,
                parse_mode="HTML",
                keyboard=keyboard,
            )
        except asyncio.CancelledError:
            self.adapter._cancelled_delivery(menu.chat_id)
            raise
        except ApiError as exc:
            return self.adapter._failure(menu.chat_id, exc)
        if generation != self._generation:
            return SendResult(success=False, error="Picker was cancelled during disconnect")
        menu.message_id = identifier(message_id)
        self.adapter.remember_message(menu.chat_id, menu.message_id)
        self._prune()
        while len(self._menus) >= MAX_MENUS:
            self._drop(next(iter(self._menus.values())))
        self._menus[id(menu)] = menu
        self._activate(menu, tokens)
        return SendResult(success=True, message_id=menu.message_id)

    async def choice(
        self, chat_id, title, choices, session_key, on_choice_selected, metadata=None
    ) -> SendResult:
        try:
            if not isinstance(choices, list) or len(choices) > MAX_OPTIONS:
                raise ValueError("Picker choices must be a bounded list")
            items = []
            for choice in choices:
                if not isinstance(choice, dict) or not isinstance(
                    choice.get("is_current", False), bool
                ):
                    raise ValueError("Invalid picker choice")
                value = _text(choice.get("value"))
                items.append(
                    _Item(
                        _text(choice.get("label", value)), (value,), choice.get("is_current", False)
                    )
                )
            menu = self._make(chat_id, title, tuple(items), session_key, on_choice_selected)
            menu.page = next((i // PAGE_SIZE for i, item in enumerate(items) if item.current), 0)
            return await self._send(menu)
        except ConfigError, ValueError, UnicodeError:
            return SendResult(
                success=False, error="Invalid or oversized picker; use the text command"
            )

    async def model(
        self,
        chat_id,
        providers,
        current_model,
        current_provider,
        session_key,
        on_model_selected,
        metadata=None,
    ) -> SendResult:
        try:
            if not isinstance(providers, list) or len(providers) > MAX_OPTIONS:
                raise ValueError("Picker providers must be a bounded list")
            if not isinstance(current_model, str) or not isinstance(current_provider, str):
                raise ValueError("Current model and provider must be strings")
            items = []
            count = 0
            for provider in providers:
                if not isinstance(provider, dict) or not isinstance(provider.get("models"), list):
                    raise ValueError("Invalid picker provider")
                slug = _text(provider.get("slug"))
                name = _text(provider.get("name", slug))
                models = provider["models"]
                count += len(models) + 1
                if count > MAX_OPTIONS:
                    raise ValueError("Picker catalog exceeds the plugin resource limit")
                is_current = provider.get("is_current", slug == current_provider) is True
                children = tuple(
                    _Item(_text(model), (model, slug), is_current and model == current_model)
                    for model in models
                )
                items.append(_Item(name, current=is_current, children=children))
            current = _literal(current_model or "unknown")
            provider_label = _literal(current_provider or "unknown")
            title = f"Choose a model\n\nCurrent: {current}\nProvider: {provider_label}"
            menu = self._make(chat_id, title, tuple(items), session_key, on_model_selected)
            for index, item in enumerate(items):
                if item.current:
                    menu.provider_page = index // PAGE_SIZE
                    menu.page = menu.provider_page
                    selected = next(
                        (i for i, child in enumerate(item.children or ()) if child.current), None
                    )
                    if selected is not None:
                        menu.provider, menu.page = index, selected // PAGE_SIZE
                    break
            return await self._send(menu)
        except ConfigError, ValueError, UnicodeError:
            return SendResult(success=False, error="Invalid or oversized model picker; use /model")

    async def _ack(self, query_id: str, text: str, *, rejected: bool = False) -> None:
        try:
            await self.adapter.api.answer_callback(query_id, text=text, show_alert=rejected)
        except ApiError as exc:
            logger.debug("VK picker acknowledgement failed: %s", exc)

    def _lookup(self, payload: dict) -> tuple[_Menu, str, int] | None:
        token = payload.get("callbackData")
        if not isinstance(token, str) or len(token) > 256:
            return None
        record = self._tokens.get(token)
        if record is None:
            return None
        menu, _, _ = record
        sender, message = payload.get("from"), payload.get("message")
        if not isinstance(sender, dict) or not isinstance(message, dict):
            return None
        author, chat = message.get("from"), message.get("chat")
        if not isinstance(author, dict) or not isinstance(chat, dict):
            return None
        native_type = chat.get("type")
        chat_type = (
            {"private": "dm", "group": "group", "channel": "channel"}.get(native_type)
            if isinstance(native_type, str)
            else None
        )
        try:
            if (
                identifier(sender.get("userId")) != menu.source.user_id
                or ("isBot" in sender and sender["isBot"] is not False)
                or identifier(author.get("userId")) != menu.bot_id
                or identifier(chat.get("chatId")) != menu.chat_id
                or identifier(message.get("msgId")) != menu.message_id
                or chat_type != menu.source.chat_type
                or not menu.context.run(self._authorized, menu)
            ):
                return None
        except ConfigError, ValueError:
            return None
        return record

    async def _edit(self, menu: _Menu, text: str, keyboard: list) -> bool:
        if self.adapter._ready(menu.chat_id):
            self._drop(menu)
            return False
        try:
            await self.adapter.api.edit_text(
                menu.chat_id,
                menu.message_id,
                text,
                parse_mode="HTML",
                keyboard=keyboard,
            )
        except asyncio.CancelledError:
            self.adapter._cancelled_delivery(menu.chat_id)
            raise
        except ApiError as exc:
            self._drop(menu)
            self.adapter._failure(menu.chat_id, exc)
            logger.debug("VK picker update failed: %s", exc)
            return False
        return True

    async def _select(self, menu: _Menu, values: tuple[str, ...]) -> None:
        try:
            if not self._authorized(menu):
                await self._edit(menu, _card(_UNAVAILABLE), [])
                return
            result = await menu.callback(menu.chat_id, *values)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Host/provider exceptions may contain secrets. Never log their text or traceback.
            text = _card("The command failed. Check the current setting with the text command.")
        else:
            try:
                text = _card(result)
            except ValueError, UnicodeError:
                text = _card(
                    "The command finished, but its confirmation is empty or exceeds the native menu limit. "
                    "Check the current setting with the text command."
                )
        await self._edit(menu, text, [])

    async def handle(self, payload) -> None:
        if not isinstance(payload, dict):
            return
        try:
            query_id = identifier(payload.get("queryId"))
        except ConfigError:
            return
        self._prune()
        record = self._lookup(payload)
        if record is None:
            await self._ack(query_id, _UNAVAILABLE, rejected=True)
            return
        menu, action, index = record
        # No await before retiring the whole page: concurrent/replayed taps cannot act twice.
        self._retire_tokens(menu)
        if action in {"select", "cancel"}:
            self._drop(menu)
            await self._ack(query_id, "Cancelled" if action == "cancel" else "Selection received")
            if action == "cancel":
                await self._edit(menu, _card("Selection cancelled."), [])
                return
            items = menu.items if menu.provider is None else menu.items[menu.provider].children
            task = asyncio.create_task(
                self._select(menu, items[index].values), context=menu.context.copy()
            )
            await task
            return
        if action == "open":
            menu.provider_page, menu.provider = menu.page, index
            menu.page = next(
                (
                    i // PAGE_SIZE
                    for i, item in enumerate(menu.items[index].children or ())
                    if item.current
                ),
                0,
            )
        elif action == "back":
            menu.provider, menu.page = None, menu.provider_page
        else:
            menu.page = index
        try:
            text, keyboard, tokens = self._render(menu)
        except ValueError, UnicodeError:
            self._drop(menu)
            await self._ack(
                query_id, "This menu is too large. Use the text command.", rejected=True
            )
            return
        await self._ack(query_id, "Opening menu")
        if await self._edit(menu, text, keyboard) and id(menu) in self._menus:
            self._activate(menu, tokens)
