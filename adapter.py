"""VK transport for the current Hermes gateway; Hermes owns conversation execution."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
    cache_image_from_url,
    cache_media_bytes_async,
)
from gateway.platforms.event import MessageEvent, MessageType
from gateway.platforms.helpers import MessageDeduplicator, bounded_put, cancel_task
from hermes_constants import get_hermes_home

from .api import ApiError, TeamsAPI
from .config import PLATFORM, ConfigError, Settings, identifier
from .events import Attachment, coalesce_batch, event_id, parse_message
from .formatting import CAPTION_LIMIT, TEXT_LIMIT, render_chunks, utf16_length
from .media import (
    Media,
    MediaError,
    convert_voice,
    local_file,
    native_voice_supported,
    normalize_voice,
    safe_filename,
    sniff_image,
)
from .state import CursorStore, StateError

logger = logging.getLogger(__name__)


@dataclass
class DeliveryScope:
    adapter: VKWorkspaceAdapter
    event: MessageEvent
    blocked: bool = False


_TURN: ContextVar[DeliveryScope | None] = ContextVar("vk_workspace_turn", default=None)


class VKWorkspaceAdapter(BasePlatformAdapter):
    MAX_MESSAGE_LENGTH = TEXT_LIMIT
    supports_code_blocks = True
    splits_long_messages = True

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform(PLATFORM))
        from .interactions import PickerManager

        self.settings = Settings.from_platform(config)
        self.api = TeamsAPI(self.settings)
        self.self_id = ""
        self._identity = hashlib.sha256(
            f"{self.settings.base_url}\0{self.settings.token}".encode()
        ).hexdigest()
        self._state = CursorStore(
            Path(get_hermes_home()) / "state" / PLATFORM / f"{self._identity}.json"
        )
        self._cursor = 0
        self._poll_task: asyncio.Task | None = None
        self._callback_tasks: set[asyncio.Task] = set()
        self._dedup = MessageDeduplicator(max_size=4096, ttl_seconds=3600)
        self._known_chats: dict[str, dict] = {}
        self._owned_messages: dict[tuple[str, str], None] = {}
        self.pickers = PickerManager(self)

    @property
    def message_len_fn(self):
        return utf16_length

    def prefers_fresh_final_streaming(self, text, *, metadata=None, chat_id=None) -> bool:
        # An edit has the same renderer as a send; replacing a preview adds no value.
        return False

    def _send_retry_is_final(self, result: SendResult) -> bool:
        # The core otherwise also tries a plain-text send after permanent errors.
        # VK permits another attempt only after an explicit rejection with no
        # delivered prefix; formatting changes cannot repair uncertain delivery.
        return result.error_kind != "rate_limited" or not result.retryable

    def current_event(self) -> MessageEvent | None:
        scope = _TURN.get()
        return scope.event if scope is not None and scope.adapter is self else None

    def _scope(self, chat_id: str) -> DeliveryScope | None:
        scope = _TURN.get()
        if scope is not None and scope.adapter is self and scope.event.source.chat_id == chat_id:
            return scope
        return None

    async def handle_message(self, event: MessageEvent) -> None:
        token = _TURN.set(DeliveryScope(self, event))
        try:
            await super().handle_message(event)
        finally:
            _TURN.reset(token)

    async def on_processing_start(self, event: MessageEvent) -> None:
        # A queued event inherits the prior task's context. Establish its own scope;
        # stream child tasks share this mutable object, including the uncertain flag.
        _TURN.set(DeliveryScope(self, event))

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        if self._running:
            return True
        if not self._acquire_platform_lock(PLATFORM, self._identity, "VK Workspace bot polling"):
            return False
        try:
            self._cursor = await asyncio.to_thread(self._state.load)
            await self.api.open()
            self.self_id = identifier((await self.api.self())["userId"])
            self._wire_plugin_handlers(self.api)
            self._mark_connected()
            self._poll_task = asyncio.create_task(self._poll(), name="vk-workspace-poll")
            return True
        except asyncio.CancelledError:
            await self.api.close()
            self._release_platform_lock()
            raise
        except (ApiError, StateError, ConfigError) as exc:
            await self.api.close()
            self._release_platform_lock()
            retryable = isinstance(exc, ApiError) and self._poll_retryable(exc)
            self._set_fatal_error("vk_workspace_connect", str(exc), retryable=retryable)
            return False

    async def disconnect(self) -> None:
        self._mark_disconnected()
        task, self._poll_task = self._poll_task, None
        try:
            await cancel_task(task)
            await self.cancel_background_tasks()
        finally:
            self.pickers.clear()
            try:
                await self.api.close()
            finally:
                self._release_platform_lock()

    @staticmethod
    def _poll_retryable(exc: ApiError) -> bool:
        return (
            exc.code in {"network_error", "timeout"}
            or exc.status == 429
            or (exc.status is not None and exc.status >= 500)
        )

    async def _poll(self) -> None:
        failures = 0
        while self._running:
            started = time.monotonic()
            try:
                events = await self.api.events(self._cursor, self.settings.poll_time)
                cursor_before = self._cursor
                await self._process_batch(events)
                if failures:
                    self._mark_connected()
                failures = 0
                # A proxy returning empty or already-acknowledged batches must not spin.
                if self._cursor == cursor_before:
                    await asyncio.sleep(max(0, 0.25 - (time.monotonic() - started)))
            except asyncio.CancelledError:
                raise
            except ApiError as exc:
                if not self._poll_retryable(exc):
                    await self._fatal("vk_workspace_poll", str(exc))
                    return
                failures += 1
                self._mark_degraded()
                delay = min(30.0, 2 ** min(failures - 1, 5)) * random.uniform(0.8, 1.2)
                await asyncio.sleep(max(delay, exc.retry_after or 0))
            except StateError, ValueError:
                await self._fatal(
                    "vk_workspace_cursor", "VK Workspace cursor or event batch is invalid"
                )
                return
            except Exception:
                # Incoming payloads and vendor exceptions must never enter a traceback
                # containing request URLs. The gateway can restart after code repair.
                await self._fatal("vk_workspace_ingress", "VK Workspace event processing failed")
                return

    async def _fatal(self, code: str, message: str) -> None:
        self._set_fatal_error(code, message, retryable=False)
        logger.error("%s", message)
        await self._notify_fatal_error()

    @staticmethod
    def _dedup_key(raw: dict) -> str:
        payload = raw.get("payload")
        if raw.get("type") == "newMessage" and isinstance(payload, dict):
            chat = payload.get("chat")
            if isinstance(chat, dict):
                try:
                    return json.dumps(
                        [identifier(chat.get("chatId")), identifier(payload.get("msgId"))]
                    )
                except ConfigError:
                    pass
        return f"event:{event_id(raw.get('eventId'))}"

    async def _process_batch(self, events: list[dict]) -> None:
        if not events:
            return
        # Validate every cursor before any handoff. Python integers preserve VK IDs.
        ordered = sorted(events, key=lambda raw: event_id(raw.get("eventId")))
        newest = event_id(ordered[-1].get("eventId"))
        pending = [raw for raw in ordered if event_id(raw.get("eventId")) > self._cursor]
        for raw in coalesce_batch(pending):
            key = self._dedup_key(raw)
            if self._dedup.is_duplicate(key):
                continue
            try:
                await self._dispatch(raw)
            except BaseException:
                self._dedup.discard(key)
                raise
        if newest > self._cursor:
            # This acknowledges gateway admission, NOT durable execution/delivery.
            await self._state.save(newest)
            self._cursor = newest

    async def _dispatch(self, raw: dict) -> None:
        if raw.get("type") == "callbackQuery":
            await self._queue_callback(raw.get("payload"))
            return
        incoming = parse_message(raw, self.self_id)
        if incoming is None:
            return
        if incoming.chat_type != "dm":
            group_id = incoming.parent_chat_id or incoming.chat_id
            if group_id not in self.settings.allowed_chats:
                return
            if self.settings.require_mention and not incoming.mentioned:
                return
        source = self.build_source(
            chat_id=incoming.chat_id,
            chat_type=incoming.chat_type,
            chat_name=incoming.chat_name,
            user_id=incoming.sender_id,
            user_name=incoming.sender_name,
            # VK supplies a native child chat. It is BOTH recipient and thread identity.
            thread_id=incoming.chat_id if incoming.parent_chat_id else None,
            parent_chat_id=incoming.parent_chat_id,
            message_id=incoming.message_id,
        )
        event = MessageEvent(
            text=incoming.text,
            source=source,
            user_id=incoming.sender_id,
            user_name=incoming.sender_name,
            raw_message=raw,
            message_id=incoming.message_id,
            platform_update_id=event_id(raw.get("eventId")),
            timestamp=incoming.timestamp,
            reply_to_message_id=incoming.reply_id,
            reply_to_text=incoming.reply_text,
            reply_to_author_id=incoming.reply_author_id,
            reply_to_author_name=incoming.reply_author_name,
            reply_to_is_own_message=incoming.reply_is_own,
            channel_context=incoming.context,
            reply_expected=True,
        )
        if self._drop_unresolved(event):
            return
        authorized = (
            self._is_sender_authorized(
                incoming.sender_id, incoming.chat_type, incoming.chat_id, thread_id=source.thread_id
            )
            is True
        )
        if not authorized and incoming.chat_type != "dm":
            return
        bounded_put(
            self._known_chats,
            incoming.chat_id,
            {"name": incoming.chat_name, "type": incoming.chat_type},
            1024,
        )
        # Unapproved DMs still reach Hermes pairing, with no credentialed media fetch.
        if authorized:
            await self._receive_attachments(event, incoming.attachments)
        await self.handle_message(event)

    async def _receive_attachments(
        self, event: MessageEvent, attachments: tuple[Attachment, ...]
    ) -> None:
        """Credentialed media I/O, called only after the sender and chat gates."""
        for attachment in attachments:
            try:
                info = await self.api.file_info(attachment.file_id)
                media = await self.api.download(info["url"], filename=info.get("filename"))
                if attachment.kind == "voice":
                    media = normalize_voice(media)
                cached = await cache_media_bytes_async(
                    media.data,
                    filename=media.filename,
                    mime_type=media.mime,
                    default_kind={"voice": "audio", "sticker": "image"}.get(attachment.kind),
                )
                if cached is None:
                    raise MediaError("invalid_image")
                event.media_urls.append(cached.path)
                event.media_types.append(cached.media_type)
                event.media_text_inlined.append(False)
                if len(attachments) == 1:
                    event.message_type = {
                        "voice": MessageType.VOICE,
                        "image": MessageType.PHOTO,
                        "audio": MessageType.AUDIO,
                        "video": MessageType.VIDEO,
                        "sticker": MessageType.STICKER,
                    }.get(attachment.kind, MessageType.DOCUMENT)
            except ApiError, MediaError, OSError, ValueError:
                event.text += "\n[An attachment could not be downloaded or validated.]"

    async def _queue_callback(self, payload) -> None:
        """Keep slow host callbacks off polling, bounded and owned by gateway shutdown."""
        if not isinstance(payload, dict):
            return
        if len(self._callback_tasks) >= 32:
            await asyncio.wait(self._callback_tasks, return_when=asyncio.FIRST_COMPLETED)
        task = asyncio.create_task(self._handle_callback(payload))
        self._callback_tasks.add(task)
        self._background_tasks.add(task)
        task.add_done_callback(self._callback_tasks.discard)
        task.add_done_callback(self._background_tasks.discard)

    async def _handle_callback(self, payload: dict) -> None:
        try:
            await self.pickers.handle(payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("VK Workspace callback processing failed")

    def _target(self, chat_id, metadata=None) -> str:
        target = identifier(chat_id)
        if metadata and metadata.get("thread_id") is not None:
            if identifier(metadata["thread_id"]) != target:
                raise ConfigError("VK thread delivery requires the native child chat ID as target")
        return target

    def _reply(self, chat_id: str, reply_to) -> str | None:
        if (
            not reply_to
            or self.config.reply_to_mode == "off"
            or self._known_chats.get(chat_id, {}).get("type") == "dm"
        ):
            return None
        return identifier(reply_to)

    def remember_message(self, chat_id: str, message_id: str) -> None:
        bounded_put(self._owned_messages, (chat_id, message_id), None, 4096)

    def _ready(self, chat_id: str) -> SendResult | None:
        if not self._running:
            return SendResult(False, error="VK Workspace is not connected")
        scope = self._scope(chat_id)
        if scope is not None and scope.blocked:
            return SendResult(
                False,
                error="VK delivery is uncertain; automatic continuation stopped",
                raw_response={"delivery_uncertain": True},
                retryable=False,
            )
        return None

    def _cancelled_delivery(self, chat_id: str) -> None:
        if scope := self._scope(chat_id):
            scope.blocked = True

    def _failure(self, chat_id: str, exc: ApiError, delivered=()) -> SendResult:
        # Even a confirmed rejection of a later chunk must not replay earlier chunks.
        uncertain = not (
            exc.code == "api_rejected" or (exc.status is not None and 400 <= exc.status < 500)
        )
        scope = self._scope(chat_id)
        if scope is not None and (uncertain or delivered):
            scope.blocked = True
        kind = {429: "rate_limited", 401: "forbidden", 403: "forbidden", 404: "not_found"}.get(
            exc.status, "unknown"
        )
        return SendResult(
            False,
            message_id=delivered[-1] if delivered else None,
            error=str(exc),
            continuation_message_ids=tuple(delivered[:-1]),
            raw_response={
                "delivery_uncertain": uncertain,
                "delivered_message_ids": list(delivered),
            },
            retryable=exc.retryable and not delivered,
            retry_after=exc.retry_after,
            error_kind=kind,
        )

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        try:
            chat_id = self._target(chat_id, metadata)
            reply = self._reply(chat_id, reply_to)
            chunks = render_chunks(content)
        except (ConfigError, ValueError, UnicodeError) as exc:
            return SendResult(False, error=str(exc), error_kind="bad_format")
        if not chunks:
            return SendResult(False, error="No text to send")
        delivered = []
        try:
            for index, chunk in enumerate(chunks):
                if failure := self._ready(chat_id):
                    failure.message_id = delivered[-1] if delivered else None
                    failure.continuation_message_ids = tuple(delivered[:-1])
                    if delivered:
                        failure.raw_response = {
                            **(failure.raw_response or {}),
                            "delivered_message_ids": list(delivered),
                        }
                    return failure
                msg_id = await self.api.send_text(
                    chat_id,
                    chunk,
                    reply_to=reply if index == 0 or self.config.reply_to_mode == "all" else None,
                    parse_mode="HTML",
                )
                self.remember_message(chat_id, msg_id)
                delivered.append(msg_id)
        except asyncio.CancelledError:
            self._cancelled_delivery(chat_id)
            raise
        except ApiError as exc:
            return self._failure(chat_id, exc, delivered)
        return SendResult(
            True, message_id=delivered[-1], continuation_message_ids=tuple(delivered[:-1])
        )

    async def edit_message(self, chat_id, message_id, content, *, finalize=False) -> SendResult:
        try:
            chat_id, message_id = self._target(chat_id), identifier(message_id)
            chunks = render_chunks(content)
        except (ConfigError, ValueError, UnicodeError) as exc:
            return SendResult(False, error=str(exc), error_kind="bad_format")
        if (chat_id, message_id) not in self._owned_messages:
            return SendResult(
                False, error="Message is not owned by this adapter", error_kind="forbidden"
            )
        if len(chunks) != 1:
            # No mutation. Hermes retains its known preview and sends the missing tail.
            return SendResult(
                False, error="Rendered edit exceeds one VK message", error_kind="too_long"
            )
        if failure := self._ready(chat_id):
            return failure
        self.pickers.discard(chat_id, message_id)
        try:
            await self.api.edit_text(chat_id, message_id, chunks[0], parse_mode="HTML", keyboard=[])
        except asyncio.CancelledError:
            self._cancelled_delivery(chat_id)
            raise
        except ApiError as exc:
            return self._failure(chat_id, exc)
        return SendResult(True, message_id=message_id)

    async def delete_message(self, chat_id, message_id) -> bool:
        try:
            chat_id, message_id = self._target(chat_id), identifier(message_id)
        except ConfigError:
            return False
        if (chat_id, message_id) not in self._owned_messages or self._ready(chat_id):
            return False
        self.pickers.discard(chat_id, message_id)
        self._owned_messages.pop((chat_id, message_id), None)
        try:
            await self.api.delete_message(chat_id, message_id)
        except ApiError:
            return False
        return True

    async def send_typing(self, chat_id, metadata=None) -> None:
        if not self._running or not self.config.typing_indicator:
            return
        try:
            await self.api.typing(self._target(chat_id, metadata), True)
        except ConfigError, ApiError:
            pass

    async def stop_typing(self, chat_id) -> None:
        if not self._running or not self.config.typing_indicator:
            return
        try:
            await self.api.typing(self._target(chat_id), False)
        except ConfigError, ApiError:
            pass

    async def get_chat_info(self, chat_id) -> dict:
        chat_id = self._target(chat_id)
        if chat_id in self._known_chats:
            return dict(self._known_chats[chat_id])
        info = await self.api.chat_info(chat_id)
        name = info.get("title") or info.get("firstName") or chat_id
        result = {
            "name": name[:256] if isinstance(name, str) else chat_id,
            "type": {"private": "dm", "group": "group", "channel": "channel"}[info["type"]],
        }
        bounded_put(self._known_chats, chat_id, result, 1024)
        return dict(result)

    def _media_ready(self, chat_id, metadata=None) -> SendResult | None:
        """Reject invalid/blocked destinations before reading, fetching or transcoding media."""
        try:
            chat_id = self._target(chat_id, metadata)
        except ConfigError as exc:
            return SendResult(False, error=str(exc))
        return self._ready(chat_id)

    async def _local_media(self, path) -> Media:
        scope = self.current_event()
        session_key = self._event_session_key(scope) if scope is not None else ""
        safe_path = await asyncio.to_thread(self.validate_media_delivery_path, path, session_key)
        if safe_path is None:
            raise MediaError("path_denied_or_missing")
        return await local_file(safe_path, self.settings.media_max_bytes)

    async def _upload(
        self, chat_id, media: Media, caption=None, reply_to=None, metadata=None, *, voice=False
    ) -> SendResult:
        try:
            chat_id = self._target(chat_id, metadata)
            reply = self._reply(chat_id, reply_to)
            caption_chunks = render_chunks(caption or "", CAPTION_LIMIT)
        except (ConfigError, ValueError, UnicodeError) as exc:
            return SendResult(False, error=str(exc))
        if failure := self._ready(chat_id):
            return failure
        separate_caption = bool(caption_chunks) and (voice or len(caption_chunks) > 1)
        try:
            if voice:
                msg_id = await self.api.send_voice(
                    chat_id, media.data, media.filename, media.mime, reply_to=reply
                )
            else:
                msg_id = await self.api.send_file(
                    chat_id,
                    media.data,
                    media.filename,
                    media.mime,
                    caption=caption_chunks[0] if caption_chunks and not separate_caption else "",
                    reply_to=reply,
                    parse_mode="HTML",
                )
            self.remember_message(chat_id, msg_id)
        except asyncio.CancelledError:
            self._cancelled_delivery(chat_id)
            raise
        except ApiError as exc:
            return self._failure(chat_id, exc)
        if separate_caption:
            result = await self.send(
                chat_id,
                caption,
                reply_to=reply if self.config.reply_to_mode == "all" else None,
                metadata=metadata,
            )
            delivered = [msg_id, *result.continuation_message_ids]
            if result.message_id is not None:
                delivered.append(result.message_id)
            result.message_id = delivered[-1]
            result.continuation_message_ids = tuple(delivered[:-1])
            if not result.success:
                result.retryable = False
                result.raw_response = {
                    **(result.raw_response or {}),
                    "delivered_message_ids": delivered,
                }
                if scope := self._scope(chat_id):
                    scope.blocked = True
            return result
        return SendResult(True, message_id=msg_id)

    async def send_document(
        self,
        chat_id,
        file_path,
        caption=None,
        file_name=None,
        reply_to=None,
        metadata=None,
        **kwargs,
    ) -> SendResult:
        if failure := self._media_ready(chat_id, metadata):
            return failure
        try:
            media = await self._local_media(file_path)
            if file_name:
                media = Media(media.data, safe_filename(file_name), media.mime)
        except MediaError, OSError, ValueError:
            return SendResult(
                False, error="VK file is unavailable, unsafe, or exceeds the media limit"
            )
        return await self._upload(chat_id, media, caption, reply_to, metadata)

    async def send_image_file(
        self, chat_id, image_path, caption=None, reply_to=None, metadata=None, **kwargs
    ) -> SendResult:
        if failure := self._media_ready(chat_id, metadata):
            return failure
        try:
            media = await self._local_media(image_path)
            mime = sniff_image(media.data)
            if mime is None:
                raise MediaError("invalid_image")
            media = Media(media.data, media.filename, mime)
        except MediaError, OSError, ValueError:
            return SendResult(
                False, error="VK image is unavailable, invalid, or exceeds the media limit"
            )
        return await self._upload(chat_id, media, caption, reply_to, metadata)

    async def send_image(
        self, chat_id, image_url, caption=None, reply_to=None, metadata=None
    ) -> SendResult:
        if failure := self._media_ready(chat_id, metadata):
            return failure
        # Public model-generated images use Hermes' maintained DNS/redirect SSRF guard
        # and image validation. VK incoming downloads use the explicit origin policy.
        try:
            cached_path = await cache_image_from_url(image_url, retries=0)
        except asyncio.CancelledError:
            raise
        except Exception:
            return SendResult(False, error="Remote image could not be safely downloaded")
        return await self.send_image_file(chat_id, cached_path, caption, reply_to, metadata)

    async def send_video(
        self, chat_id, video_path, caption=None, reply_to=None, metadata=None, **kwargs
    ) -> SendResult:
        return await self.send_document(
            chat_id, video_path, caption, reply_to=reply_to, metadata=metadata
        )

    async def send_voice(
        self, chat_id, audio_path, caption=None, reply_to=None, metadata=None, **kwargs
    ) -> SendResult:
        if failure := self._media_ready(chat_id, metadata):
            return failure
        try:
            media = normalize_voice(await self._local_media(audio_path))
            if not native_voice_supported(media):
                # Hermes' TTS commonly produces MP3; VK native voice needs OGG/AAC/M4A.
                # Unknown files are rejected before ffmpeg's auto-detection runs.
                media = await asyncio.to_thread(convert_voice, media, self.settings.media_max_bytes)
            return await self._upload(chat_id, media, caption, reply_to, metadata, voice=True)
        except MediaError as exc:
            if exc.code == "voice_conversion_requires_ffmpeg":
                return SendResult(False, error="Native VK voice conversion requires working ffmpeg")
            return SendResult(
                False, error="VK voice is unavailable, invalid, or exceeds the media limit"
            )
        except OSError, ValueError:
            return SendResult(
                False, error="VK voice is unavailable, invalid, or exceeds the media limit"
            )

    async def send_choice_picker(
        self, chat_id, title, choices, session_key, on_choice_selected, metadata=None
    ) -> SendResult:
        return await self.pickers.choice(
            chat_id, title, choices, session_key, on_choice_selected, metadata
        )

    async def send_model_picker(
        self,
        chat_id,
        providers,
        current_model,
        current_provider,
        session_key,
        on_model_selected,
        metadata=None,
    ) -> SendResult:
        return await self.pickers.model(
            chat_id,
            providers,
            current_model,
            current_provider,
            session_key,
            on_model_selected,
            metadata,
        )


async def standalone_send(
    config: PlatformConfig,
    chat_id,
    message,
    *,
    thread_id=None,
    media_files=None,
    force_document=False,
) -> dict:
    """Hermes cron/CLI contract. No listener, polling cursor, or bot-consumer lock."""
    try:
        adapter = VKWorkspaceAdapter(config)
        metadata = {"thread_id": thread_id} if thread_id is not None else None
        adapter._target(chat_id, metadata)
    except ConfigError, ValueError:
        return {"error": "VK Workspace configuration or target is invalid"}
    try:
        await adapter.api.open()
        # Outbound-only clients must not overwrite a live gateway's runtime status.
        adapter._running = True
        result = None
        if message and message.strip():
            result = await adapter.send(chat_id, message, metadata=metadata)
            if not result.success:
                return {"error": result.error}
        for media_path, is_voice in media_files or []:
            sender = (
                adapter.send_voice if is_voice and not force_document else adapter.send_document
            )
            result = await sender(chat_id, media_path, metadata=metadata)
            if not result.success:
                return {"error": result.error}
        if result is None:
            return {"error": "No text or media to send"}
        return {"success": True, "message_id": result.message_id}
    except ApiError as exc:
        return {"error": str(exc)}
    finally:
        adapter._running = False
        await adapter.api.close()
