"""Async VK Teams Bot API transport with bounded, credential-safe responses."""

from __future__ import annotations

import asyncio
import json
import math
import zlib
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlsplit

import aiohttp
from yarl import URL

from .media import (
    Media,
    content_type,
    native_voice_supported,
    normalize_voice,
    safe_filename,
    sniff_image,
)

if TYPE_CHECKING:
    from .config import Settings


MAX_API_BYTES = 8 * 1024 * 1024
_ENDPOINTS = frozenset(
    {
        "self/get",
        "events/get",
        "messages/sendText",
        "messages/editText",
        "messages/sendFile",
        "messages/sendVoice",
        "messages/deleteMessages",
        "chats/sendActions",
        "messages/answerCallbackQuery",
        "chats/getInfo",
        "files/getInfo",
    }
)


class ApiError(Exception):
    """Never retain provider descriptions or exception causes containing token URLs."""

    def __init__(
        self,
        endpoint: str,
        code: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
    ):
        self.endpoint = endpoint if endpoint in _ENDPOINTS or endpoint == "download" else "request"
        self.code = code
        self.status = status
        self.retryable = status == 429
        self.retry_after = retry_after
        suffix = f" (HTTP {status})" if status is not None else ""
        super().__init__(f"VK Teams {self.endpoint}: {code.replace('_', ' ')}{suffix}")


def _identifier(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 1024
        or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ApiError("request", "invalid_identifier")
    return value


def _origin(value: str, allow_http: bool) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ApiError("download", "untrusted_origin")
    try:
        parsed = URL(value)
        if parsed.scheme not in ({"http", "https"} if allow_http else {"https"}):
            raise ValueError
        if (
            not parsed.raw_host
            or parsed.user is not None
            or parsed.password is not None
            or parsed.fragment
        ):
            raise ValueError
        # Check the exact IDNA/IP normalization aiohttp uses for the request.
        return str(parsed.origin())
    except ValueError, UnicodeError:
        raise ApiError("download", "untrusted_origin") from None


def _format_params(parse_mode: str | None) -> dict[str, str]:
    if parse_mode is None:
        return {}
    if not isinstance(parse_mode, str) or parse_mode not in {"HTML", "MarkdownV2"}:
        raise ApiError("request", "invalid_parse_mode")
    return {"parseMode": parse_mode}


def _validate_button(button: object) -> None:
    """Validate one button's label, action and style independently of keyboard layout."""
    if not isinstance(button, dict) or set(button) - {"text", "url", "callbackData", "style"}:
        raise ApiError("request", "invalid_keyboard")
    text = button.get("text")
    if (
        not isinstance(text, str)
        or not text.strip()
        or len(text) > 128
        or any(ord(c) < 32 for c in text)
    ):
        raise ApiError("request", "invalid_keyboard")
    if ("url" in button) == ("callbackData" in button):
        raise ApiError("request", "invalid_keyboard")
    style = button.get("style", "base")
    if not isinstance(style, str) or style not in {"base", "primary", "attention"}:
        raise ApiError("request", "invalid_keyboard")
    target = button.get("url", button.get("callbackData"))
    if (
        not isinstance(target, str)
        or not target
        or any(ord(c) < 32 or ord(c) == 127 for c in target)
    ):
        raise ApiError("request", "invalid_keyboard")
    if "url" in button:
        _origin(target.split("#", 1)[0], True)
    elif len(target.encode("utf-8", errors="replace")) > 256:
        raise ApiError("request", "invalid_keyboard")


def _keyboard(value: list[list[dict[str, str]]] | None) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, list) or len(value) > 10:
        raise ApiError("request", "invalid_keyboard")
    for row in value:
        if not isinstance(row, list) or not 1 <= len(row) <= 8:
            raise ApiError("request", "invalid_keyboard")
        for button in row:
            _validate_button(button)
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > 16 * 1024:
            raise ApiError("request", "invalid_keyboard")
    except TypeError, UnicodeError:
        raise ApiError("request", "invalid_keyboard") from None
    return {"inlineKeyboardMarkup": encoded}


def _retry_after(value: str | None) -> float | None:
    try:
        seconds = float(value) if value is not None else None
        return seconds if seconds is not None and math.isfinite(seconds) and seconds >= 0 else None
    except ValueError:
        return None


async def _bounded(response: aiohttp.ClientResponse, limit: int, endpoint: str) -> bytes:
    if response.content_length is not None and response.content_length > limit:
        raise ApiError(endpoint, "response_size_limit")
    encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
    if encoding not in {"", "identity", "gzip", "deflate"}:
        raise ApiError(endpoint, "unsupported_content_encoding")
    decoder = (
        zlib.decompressobj(16 + zlib.MAX_WBITS if encoding == "gzip" else zlib.MAX_WBITS)
        if encoding in {"gzip", "deflate"}
        else None
    )
    data = bytearray()
    wire_size = 0
    try:
        async for chunk in response.content.iter_chunked(64 * 1024):
            wire_size += len(chunk)
            if wire_size > limit:
                raise ApiError(endpoint, "response_size_limit")
            # Bound decompression itself, not just an already expanded body.
            decoded = decoder.decompress(chunk, limit - len(data) + 1) if decoder else chunk
            if len(data) + len(decoded) > limit:
                raise ApiError(endpoint, "response_size_limit")
            data.extend(decoded)
        if decoder and (not decoder.eof or decoder.unused_data):
            raise ApiError(endpoint, "invalid_compressed_response")
    except zlib.error:
        raise ApiError(endpoint, "invalid_compressed_response") from None
    return bytes(data)


def _single_attempt(endpoint: str):
    async def middleware(request, handler):
        try:
            return await handler(request)
        except TimeoutError:
            raise ApiError(endpoint, "timeout") from None
        except aiohttp.ClientError, OSError:
            # VK mutates via GET. Prevent aiohttp's automatic retry of a GET
            # after a disconnect by raising outside its transport-error family.
            raise ApiError(endpoint, "network_error") from None

    return middleware


class TeamsAPI:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._session: aiohttp.ClientSession | None = None

    async def open(self) -> None:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._settings.request_timeout),
                connector=aiohttp.TCPConnector(limit=8),
                cookie_jar=aiohttp.DummyCookieJar(),
                trust_env=False,
                auto_decompress=False,
                headers={"Accept-Encoding": "identity"},
            )

    async def close(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            await session.close()

    def _client(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            raise ApiError("request", "not_connected")
        return self._session

    async def _request(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        *,
        file: Media | None = None,
        request_seconds: float | None = None,
    ) -> dict[str, Any]:
        if endpoint not in _ENDPOINTS:
            raise ApiError("request", "unsupported_endpoint")
        query = [
            (key, str(value).lower() if isinstance(value, bool) else str(value))
            for key, value in (params or {}).items()
            if value is not None and key != "token"
        ]
        query.append(("token", self._settings.token))
        body = None
        if file is not None:
            if len(file.data) > self._settings.media_max_bytes:
                raise ApiError(endpoint, "media_size_limit")
            body = aiohttp.FormData()
            name = safe_filename(file.filename)
            body.add_field(
                "file", file.data, filename=name, content_type=content_type(file.mime, name)
            )
        client = self._client()
        try:
            async with client.request(
                "POST" if file is not None else "GET",
                f"{self._settings.base_url}/{endpoint}",
                params=query,
                data=body,
                allow_redirects=False,
                timeout=aiohttp.ClientTimeout(
                    total=request_seconds or self._settings.request_timeout
                ),
                headers={"Accept": "application/json"},
                middlewares=(_single_attempt(endpoint),),
            ) as response:
                if not 200 <= response.status < 300:
                    raise ApiError(
                        endpoint,
                        "rate_limited" if response.status == 429 else "http_error",
                        status=response.status,
                        retry_after=_retry_after(response.headers.get("Retry-After")),
                    )
                raw = await _bounded(response, MAX_API_BYTES, endpoint)
                try:
                    payload = json.loads(raw)
                except ValueError, UnicodeError, RecursionError:
                    raise ApiError(endpoint, "invalid_response") from None
                if not isinstance(payload, dict):
                    raise ApiError(endpoint, "invalid_response")
                if (
                    ("ok" in payload and payload["ok"] is not True)
                    or "error" in payload
                    or (payload.get("ok") is not True and "description" in payload)
                ):
                    raise ApiError(endpoint, "api_rejected")
                return payload
        except TimeoutError:
            raise ApiError(endpoint, "timeout") from None
        except aiohttp.ClientError, OSError, ValueError:
            raise ApiError(endpoint, "network_error") from None

    @staticmethod
    def _confirmed(payload: dict[str, Any], endpoint: str) -> None:
        if payload.get("ok") is not True:
            raise ApiError(endpoint, "missing_confirmation")

    @classmethod
    def _message_id(cls, payload: dict[str, Any], endpoint: str) -> str:
        cls._confirmed(payload, endpoint)
        identifier = payload.get("msgId")
        if isinstance(identifier, int) and not isinstance(identifier, bool) and identifier >= 0:
            return str(identifier)
        if isinstance(identifier, str):
            return _identifier(identifier)
        raise ApiError(endpoint, "missing_message_id")

    async def self(self) -> dict[str, Any]:
        payload = await self._request("self/get")
        if not isinstance(payload.get("userId"), str):
            raise ApiError("self/get", "missing_bot_id")
        _identifier(payload["userId"])
        return payload

    async def events(self, last_event_id: int, poll_time: int) -> list[dict[str, Any]]:
        if (
            isinstance(last_event_id, bool)
            or not isinstance(last_event_id, int)
            or last_event_id < 0
        ):
            raise ApiError("events/get", "invalid_cursor")
        if (
            isinstance(poll_time, bool)
            or not isinstance(poll_time, int)
            or not 1 <= poll_time <= 60
        ):
            raise ApiError("events/get", "invalid_poll_time")
        payload = await self._request(
            "events/get",
            {"lastEventId": last_event_id, "pollTime": poll_time},
            request_seconds=poll_time + self._settings.request_timeout,
        )
        self._confirmed(payload, "events/get")
        events = payload.get("events")
        if not isinstance(events, list) or not all(isinstance(event, dict) for event in events):
            raise ApiError("events/get", "invalid_events")
        return events

    async def send_text(
        self,
        chat_id: str,
        text: str,
        *,
        reply_to: str | None = None,
        parse_mode: str | None = None,
        keyboard: list | None = None,
    ) -> str:
        endpoint = "messages/sendText"
        if not isinstance(text, str) or not text or len(text) > 65536:
            raise ApiError(endpoint, "invalid_text")
        payload = await self._request(
            endpoint,
            {
                "chatId": _identifier(chat_id),
                "text": text,
                "replyMsgId": _identifier(reply_to) if reply_to is not None else None,
                **_format_params(parse_mode),
                **_keyboard(keyboard),
            },
        )
        return self._message_id(payload, endpoint)

    async def edit_text(
        self,
        chat_id: str,
        msg_id: str,
        text: str,
        *,
        parse_mode: str | None = None,
        keyboard: list | None = None,
    ) -> None:
        endpoint = "messages/editText"
        if not isinstance(text, str) or not text or len(text) > 65536:
            raise ApiError(endpoint, "invalid_text")
        payload = await self._request(
            endpoint,
            {
                "chatId": _identifier(chat_id),
                "msgId": _identifier(msg_id),
                "text": text,
                **_format_params(parse_mode),
                **_keyboard(keyboard),
            },
        )
        self._confirmed(payload, endpoint)

    async def send_file(
        self,
        chat_id: str,
        data: bytes,
        filename: str,
        mime: str,
        *,
        caption: str = "",
        reply_to: str | None = None,
        parse_mode: str | None = None,
    ) -> str:
        endpoint = "messages/sendFile"
        if not isinstance(data, bytes) or not isinstance(caption, str) or len(caption) > 65536:
            raise ApiError(endpoint, "invalid_media")
        payload = await self._request(
            endpoint,
            {
                "chatId": _identifier(chat_id),
                "caption": caption,
                "replyMsgId": _identifier(reply_to) if reply_to is not None else None,
                **_format_params(parse_mode),
            },
            file=Media(data, filename, mime),
        )
        return self._message_id(payload, endpoint)

    async def send_voice(
        self, chat_id: str, data: bytes, filename: str, mime: str, *, reply_to: str | None = None
    ) -> str:
        endpoint = "messages/sendVoice"
        if not isinstance(data, bytes):
            raise ApiError(endpoint, "invalid_media")
        media = Media(data, filename, mime)
        if not native_voice_supported(media):
            raise ApiError(endpoint, "unsupported_native_voice")
        payload = await self._request(
            endpoint,
            {
                "chatId": _identifier(chat_id),
                "replyMsgId": _identifier(reply_to) if reply_to is not None else None,
            },
            file=normalize_voice(media),
        )
        return self._message_id(payload, endpoint)

    async def delete_message(self, chat_id: str, msg_id: str) -> None:
        endpoint = "messages/deleteMessages"
        self._confirmed(
            await self._request(
                endpoint, {"chatId": _identifier(chat_id), "msgId": _identifier(msg_id)}
            ),
            endpoint,
        )

    async def typing(self, chat_id: str, active: bool) -> None:
        endpoint = "chats/sendActions"
        if not isinstance(active, bool):
            raise ApiError(endpoint, "invalid_activity")
        self._confirmed(
            await self._request(
                endpoint, {"chatId": _identifier(chat_id), "actions": "typing" if active else ""}
            ),
            endpoint,
        )

    async def answer_callback(
        self, query_id: str, text: str = "", show_alert: bool = False
    ) -> None:
        endpoint = "messages/answerCallbackQuery"
        if not isinstance(text, str) or len(text) > 4096 or not isinstance(show_alert, bool):
            raise ApiError(endpoint, "invalid_callback")
        self._confirmed(
            await self._request(
                endpoint, {"queryId": _identifier(query_id), "text": text, "showAlert": show_alert}
            ),
            endpoint,
        )

    async def chat_info(self, chat_id: str) -> dict[str, Any]:
        payload = await self._request("chats/getInfo", {"chatId": _identifier(chat_id)})
        if not isinstance(payload.get("type"), str) or payload["type"] not in {
            "private",
            "group",
            "channel",
        }:
            raise ApiError("chats/getInfo", "invalid_chat_type")
        return payload

    async def file_info(self, file_id: str) -> dict[str, Any]:
        payload = await self._request("files/getInfo", {"fileId": _identifier(file_id)})
        if not isinstance(payload.get("url"), str) or not payload["url"]:
            raise ApiError("files/getInfo", "missing_download_url")
        size = payload.get("size")
        if size is not None and (isinstance(size, bool) or not isinstance(size, int) or size < 0):
            raise ApiError("files/getInfo", "invalid_file_size")
        if size is not None and size > self._settings.media_max_bytes:
            raise ApiError("files/getInfo", "media_size_limit")
        return payload

    async def download(self, url: str, *, filename: str | None = None) -> Media:
        allowed = {
            _origin(value, self._settings.allow_insecure_http)
            for value in self._settings.media_allowed_origins
        }
        allowed.add(_origin(self._settings.base_url, self._settings.allow_insecure_http))
        client = self._client()
        try:
            async with asyncio.timeout(self._settings.request_timeout):
                for hop in range(4):
                    if _origin(url, self._settings.allow_insecure_http) not in allowed:
                        raise ApiError("download", "untrusted_origin")
                    async with client.get(
                        url, allow_redirects=False, middlewares=(_single_attempt("download"),)
                    ) as response:
                        if response.status in {301, 302, 303, 307, 308}:
                            location = response.headers.get("Location")
                            if not location or hop == 3:
                                raise ApiError("download", "redirect_limit")
                            url = urljoin(url, location)
                            continue
                        if not 200 <= response.status < 300:
                            raise ApiError("download", "http_error", status=response.status)
                        data = await _bounded(response, self._settings.media_max_bytes, "download")
                        name = safe_filename(filename or urlsplit(url).path)
                        return Media(
                            data,
                            name,
                            sniff_image(data)
                            or content_type(response.headers.get("Content-Type"), name),
                        )
        except TimeoutError:
            raise ApiError("download", "timeout") from None
        except aiohttp.ClientError, OSError, ValueError:
            raise ApiError("download", "network_error") from None
        raise ApiError("download", "redirect_limit")
