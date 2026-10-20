"""Exercise the real HTTP wire contract, failure boundaries and download policy."""

import asyncio
import gzip
import json
import traceback
import unittest
import zlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aiohttp import web
from hermes_vk_workspace.api import ApiError, TeamsAPI, _origin
from hermes_vk_workspace.config import Settings


class APITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.seen = []
        self.runners = []
        self.handler = self.success
        origin = await self.server(self.dispatch)
        self.settings = SimpleNamespace(
            base_url=origin + "/bot/v1",
            token="test-only-token:never-real",
            request_timeout=2.0,
            poll_time=30,
            media_max_bytes=1024,
            media_allowed_origins=(origin,),
            allow_insecure_http=True,
        )
        self.origin = origin
        self.api = TeamsAPI(self.settings)
        await self.api.open()

    async def asyncTearDown(self):
        await self.api.close()
        for runner in reversed(self.runners):
            await runner.cleanup()

    async def server(self, handler):
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", handler)
        runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        self.runners.append(runner)
        return f"http://127.0.0.1:{runner.addresses[0][1]}"

    async def success(self, request):
        return web.json_response({"ok": True, "msgId": "message-1"})

    async def dispatch(self, request):
        self.seen.append(
            {
                "method": request.method,
                "path": request.path,
                "query": request.query.copy(),
                "headers": request.headers.copy(),
            }
        )
        return await self.handler(request)

    async def test_metadata_allows_absent_ok_but_rejects_explicit_error(self):
        async def metadata(request):
            if request.path.endswith("self/get"):
                return web.json_response({"userId": "bot@company.example"})
            if request.path.endswith("chats/getInfo"):
                return web.json_response({"type": "group", "title": "Group"})
            return web.json_response(
                {"url": self.origin + "/file", "filename": "report.txt", "size": 4}
            )

        self.handler = metadata
        self.assertEqual((await self.api.self())["userId"], "bot@company.example")
        self.assertEqual((await self.api.chat_info("group@example"))["type"], "group")
        self.assertEqual((await self.api.file_info("opaque-file"))["size"], 4)
        for request in self.seen:
            self.assertEqual(request["query"]["token"], self.settings.token)
            self.assertNotIn("Authorization", request["headers"])

        async def denied(request):
            return web.json_response(
                {"ok": False, "userId": "bot", "description": self.settings.token}
            )

        self.handler = denied
        with self.assertRaises(ApiError) as caught:
            await self.api.self()
        self.assertFalse(caught.exception.retryable)
        self.assertNotIn(self.settings.token, str(caught.exception))

    async def test_send_and_edit_use_query_parameters(self):
        keyboard = [
            [
                {"text": "Approve", "callbackData": "opaque-token", "style": "primary"},
                {"text": "Docs", "url": "https://example.com/path#reference"},
            ]
        ]
        result = await self.api.send_text(
            "user@example",
            "<b>Привет & мир</b>",
            reply_to="reply-123",
            parse_mode="HTML",
            keyboard=keyboard,
        )
        self.assertEqual(result, "message-1")
        request = self.seen[-1]
        self.assertEqual(request["method"], "GET")
        self.assertEqual(request["path"], "/bot/v1/messages/sendText")
        self.assertEqual(request["query"]["chatId"], "user@example")
        self.assertEqual(request["query"]["replyMsgId"], "reply-123")
        self.assertEqual(request["query"]["text"], "<b>Привет & мир</b>")
        self.assertEqual(request["query"]["parseMode"], "HTML")
        self.assertEqual(json.loads(request["query"]["inlineKeyboardMarkup"]), keyboard)
        await self.api.edit_text("user@example", result, "Edited", parse_mode="HTML", keyboard=[])
        request = self.seen[-1]
        self.assertEqual(request["path"], "/bot/v1/messages/editText")
        self.assertEqual(request["query"]["msgId"], result)
        self.assertEqual(request["query"]["inlineKeyboardMarkup"], "[]")

    async def test_upload_body_contains_only_file_all_other_fields_stay_query(self):
        parts = []

        async def upload(request):
            async for part in await request.multipart():
                parts.append(
                    (
                        part.name,
                        part.filename,
                        part.headers.get("Content-Type"),
                        bytes(await part.read()),
                    )
                )
            return web.json_response({"ok": True, "msgId": 12345678901234567890})

        self.handler = upload
        result = await self.api.send_file(
            "group@example",
            b"example bytes",
            "../report.txt",
            "text/plain",
            caption="<b>Caption</b>",
            reply_to="parent",
            parse_mode="HTML",
        )
        self.assertEqual(result, "12345678901234567890")
        self.assertEqual(parts, [("file", "report.txt", "text/plain", b"example bytes")])
        request = self.seen[-1]
        self.assertEqual(request["method"], "POST")
        self.assertEqual(request["query"]["caption"], "<b>Caption</b>")
        self.assertEqual(request["query"]["replyMsgId"], "parent")
        self.assertEqual(request["query"]["parseMode"], "HTML")
        self.assertEqual(request["query"]["token"], self.settings.token)

    async def test_voice_requires_supported_bytes_and_normalizes_metadata(self):
        parts = []

        async def upload(request):
            async for part in await request.multipart():
                parts.append(
                    (part.filename, part.headers.get("Content-Type"), bytes(await part.read()))
                )
            return web.json_response({"ok": True, "msgId": "voice-id"})

        self.handler = upload
        with self.assertRaises(ApiError):
            await self.api.send_voice("chat", b"not audio", "voice.aac", "audio/aac")
        with self.assertRaises(ApiError):
            await self.api.send_voice("chat", b"ID3mp3 audio", "voice.mp3", "audio/mpeg")
        self.assertEqual(self.seen, [])
        data = b"\xff\xf1\x50\x80\x01\x7f\xfc"
        self.assertEqual(
            await self.api.send_voice(
                "chat", data, "voice.bin", "application/octet-stream", reply_to="parent"
            ),
            "voice-id",
        )
        self.assertEqual(parts, [("voice.aac", "audio/aac", data)])
        query = self.seen[-1]["query"]
        self.assertEqual(query["replyMsgId"], "parent")
        self.assertNotIn("caption", query)
        self.assertNotIn("parseMode", query)

    async def test_activity_clear_callback_and_delete_contracts(self):
        await self.api.typing("chat", True)
        await self.api.typing("chat", False)
        self.assertEqual(self.seen[0]["query"].getall("actions"), ["typing"])
        self.assertEqual(self.seen[1]["query"].getall("actions"), [""])
        await self.api.answer_callback("query-opaque", "Accepted", True)
        query = self.seen[-1]["query"]
        self.assertEqual(query["text"], "Accepted")
        self.assertEqual(query["showAlert"], "true")
        self.assertNotIn("textAnswer", query)
        await self.api.delete_message("chat", "message-1")
        self.assertEqual(self.seen[-1]["query"].getall("msgId"), ["message-1"])

    async def test_events_preserve_large_ids_and_reject_missing_events(self):
        events = [{"eventId": 2**63 + 1, "type": "newMessage", "payload": {}}]

        async def poll(request):
            return web.json_response({"ok": True, "events": events})

        self.handler = poll
        self.assertEqual(await self.api.events(2**63, 30), events)
        self.assertEqual(self.seen[-1]["query"]["lastEventId"], str(2**63))
        self.assertEqual(self.seen[-1]["query"]["pollTime"], "30")
        self.handler = self.success
        with self.assertRaises(ApiError):
            await self.api.events(2**63, 30)
        with self.assertRaises(ApiError):
            await self.api.events(True, 30)

    async def test_only_confirmed_http_429_is_retryable_no_request_is_replayed(self):
        for status in (429, 500, 503, 401, 403):

            async def reject(request, response_status=status):
                return web.Response(
                    status=response_status, text=self.settings.token, headers={"Retry-After": "120"}
                )

            self.handler = reject
            before = len(self.seen)
            with self.assertRaises(ApiError) as caught:
                await self.api.send_text("chat", "hello")
            self.assertEqual(len(self.seen), before + 1)
            self.assertEqual(caught.exception.retryable, status == 429)
            self.assertEqual(caught.exception.status, status)
            self.assertNotIn(self.settings.token, str(caught.exception))
        self.assertEqual(caught.exception.retry_after, 120)

    async def test_disconnect_after_receiving_get_does_not_trigger_aiohttp_retry(self):
        async def disconnect(request):
            request.transport.close()
            return web.Response()

        self.handler = disconnect
        try:
            await self.api.send_text("chat", "must not duplicate")
        except ApiError as error:
            self.assertEqual(error.code, "network_error")
            self.assertFalse(error.retryable)
            self.assertNotIn(self.settings.token, traceback.format_exc())
            self.assertIsNone(error.__cause__)
        else:
            self.fail("Expected an ambiguous transport failure")
        self.assertEqual(len(self.seen), 1)

    async def test_api_redirect_is_not_followed(self):
        async def redirect(request):
            return web.Response(status=302, headers={"Location": self.origin + "/secret"})

        self.handler = redirect
        with self.assertRaises(ApiError) as caught:
            await self.api.send_text("chat", "hello")
        self.assertEqual(caught.exception.status, 302)
        self.assertEqual(len(self.seen), 1)

    async def test_mutations_require_explicit_confirmation_and_message_id(self):
        for payload in (
            {"msgId": "m"},
            {"ok": True},
            {"ok": True, "msgId": True},
            {"ok": 1, "msgId": "m"},
            {"ok": True, "error": "hidden failure"},
        ):

            async def answer(request, value=payload):
                return web.json_response(value)

            self.handler = answer
            with self.assertRaises(ApiError):
                await self.api.send_text("chat", "hello")

    async def test_api_response_limit_applies_during_decompression(self):
        async def oversized(request):
            data = json.dumps({"ok": True, "text": "x" * 2000}).encode()
            return web.Response(body=gzip.compress(data), headers={"Content-Encoding": "gzip"})

        self.handler = oversized
        with (
            patch("hermes_vk_workspace.api.MAX_API_BYTES", 512),
            self.assertRaises(ApiError) as caught,
        ):
            await self.api.self()
        self.assertEqual(caught.exception.code, "response_size_limit")

    async def test_small_compressed_api_response_is_accepted(self):
        async def compressed(request):
            return web.Response(
                body=gzip.compress(b'{"ok":true,"userId":"bot"}'),
                headers={"Content-Encoding": "gzip"},
            )

        self.handler = compressed
        self.assertEqual((await self.api.self())["userId"], "bot")

    async def test_timeout_and_cancellation_remain_distinct(self):
        entered = asyncio.Event()

        async def wait_forever(request):
            entered.set()
            await asyncio.Event().wait()

        self.handler = wait_forever
        self.settings.request_timeout = 0.03
        with self.assertRaises(ApiError) as caught:
            await self.api.send_text("chat", "hello")
        self.assertEqual(caught.exception.code, "timeout")
        self.assertFalse(caught.exception.retryable)
        self.settings.request_timeout = 2
        entered.clear()
        task = asyncio.create_task(self.api.events(0, 1))
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

    async def test_download_allows_explicit_origin_and_never_sends_token_or_cookies(self):
        seen_download = []

        async def download(request):
            seen_download.append((request.query.copy(), request.headers.copy()))
            return web.Response(body=b"file bytes", headers={"Content-Type": "text/plain"})

        other = await self.server(download)
        self.settings.media_allowed_origins += (other,)

        async def metadata(request):
            response = web.json_response({"userId": "bot"})
            response.set_cookie("sensitive", "must-not-forward")
            return response

        self.handler = metadata
        await self.api.self()
        media = await self.api.download(other + "/file?signature=opaque", filename="../report.txt")
        self.assertEqual(
            (media.data, media.filename, media.mime), (b"file bytes", "report.txt", "text/plain")
        )
        query, headers = seen_download[0]
        self.assertEqual(dict(query), {"signature": "opaque"})
        self.assertNotIn("Authorization", headers)
        self.assertNotIn("Cookie", headers)

    async def test_download_revalidates_redirect_origin_before_request(self):
        untrusted_requests = []

        async def untrusted(request):
            untrusted_requests.append(request.path)
            return web.Response(body=b"must not read")

        other = await self.server(untrusted)

        async def redirect(request):
            return web.Response(status=302, headers={"Location": other + "/file?token=secret"})

        self.handler = redirect
        with self.assertRaises(ApiError) as caught:
            await self.api.download(self.origin + "/file")
        self.assertEqual(caught.exception.code, "untrusted_origin")
        self.assertEqual(untrusted_requests, [])
        self.assertNotIn("secret", str(caught.exception))

    async def test_download_rejects_credentials_protocol_and_parser_ambiguity(self):
        invalid = [
            "file:///etc/passwd",
            "http://u:p@127.0.0.1/file",
            self.origin + "/x\n",
            self.origin + "/x\\evil",
            "https://127.0.0.1:wrong/file",
        ]
        for url in invalid:
            with self.subTest(url=url), self.assertRaises(ApiError):
                await self.api.download(url)
        self.assertEqual(self.seen, [])

    async def test_download_redirects_bounded_and_relative_redirect_supported(self):
        async def redirect(request):
            if request.path == "/initial":
                return web.Response(status=307, headers={"Location": "/final"})
            if request.path == "/final":
                return web.Response(body=b"result")
            return web.Response(status=302, headers={"Location": "/loop"})

        self.handler = redirect
        self.assertEqual((await self.api.download(self.origin + "/initial")).data, b"result")
        before = len(self.seen)
        with self.assertRaises(ApiError) as caught:
            await self.api.download(self.origin + "/loop")
        self.assertEqual(caught.exception.code, "redirect_limit")
        self.assertEqual(len(self.seen), before + 4)

    async def test_download_stream_and_compressed_body_cannot_exceed_limit(self):
        async def stream(request):
            response = web.StreamResponse()
            await response.prepare(request)
            await response.write(b"x" * 1025)
            await response.write_eof()
            return response

        self.handler = stream
        with self.assertRaises(ApiError) as caught:
            await self.api.download(self.origin + "/large")
        self.assertEqual(caught.exception.code, "response_size_limit")

        async def compressed(request):
            return web.Response(
                body=gzip.compress(b"x" * 1025), headers={"Content-Encoding": "gzip"}
            )

        self.handler = compressed
        with self.assertRaises(ApiError) as caught:
            await self.api.download(self.origin + "/compressed")
        self.assertEqual(caught.exception.code, "response_size_limit")

    async def test_chunked_compression_cannot_bypass_the_wire_byte_limit(self):
        # Valid deflate can consume arbitrarily many empty blocks without
        # producing decoded bytes, so both sides of decompression need a limit.
        wire = (
            b"\x78\x01"
            + b"\x00\x00\x00\xff\xff" * 400
            + b"\x01\x00\x00\xff\xff"
            + b"\x00\x00\x00\x01"
        )
        self.assertEqual(zlib.decompress(wire), b"")

        async def stream(request):
            response = web.StreamResponse(headers={"Content-Encoding": "deflate"})
            await response.prepare(request)
            await response.write(wire)
            await response.write_eof()
            return response

        self.handler = stream
        with self.assertRaises(ApiError) as caught:
            await self.api.download(self.origin + "/chunked-compressed")
        self.assertEqual(caught.exception.code, "response_size_limit")

    async def test_upload_limit_rejected_before_network_and_closed_client_is_safe(self):
        with self.assertRaises(ApiError):
            await self.api.send_file("chat", b"x" * 1025, "big.bin", "application/octet-stream")
        self.assertEqual(self.seen, [])
        await self.api.close()
        with self.assertRaises(ApiError) as caught:
            await self.api.self()
        self.assertEqual(caught.exception.code, "not_connected")

    async def test_invalid_provider_metadata_and_malformed_json_fail_safely(self):
        async def bad_chat(request):
            return web.json_response({"ok": True, "type": ["private"]})

        self.handler = bad_chat
        with self.assertRaises(ApiError):
            await self.api.chat_info("chat")
        for size in (True, -1, 1025):

            async def bad_size(request, value=size):
                return web.json_response({"url": self.origin + "/file", "size": value})

            self.handler = bad_size
            with self.assertRaises(ApiError):
                await self.api.file_info("file")

        async def malformed(request):
            return web.Response(text="<html>" + self.settings.token)

        self.handler = malformed
        with self.assertRaises(ApiError) as caught:
            await self.api.self()
        self.assertEqual(caught.exception.code, "invalid_response")
        self.assertNotIn(self.settings.token, str(caught.exception))

    async def test_invalid_parse_mode_does_not_reach_transport(self):
        with self.assertRaises(ApiError):
            await self.api.send_text("chat", "hello", parse_mode=[])
        self.assertEqual(self.seen, [])

    async def test_truncated_compressed_download_is_rejected(self):
        async def broken(request):
            return web.Response(
                body=gzip.compress(b"hello")[:-5], headers={"Content-Encoding": "gzip"}
            )

        self.handler = broken
        with self.assertRaises(ApiError) as caught:
            await self.api.download(self.origin + "/truncated")
        self.assertEqual(caught.exception.code, "invalid_compressed_response")


@pytest.mark.parametrize(
    "keyboard",
    [
        pytest.param({}, id="not-a-keyboard"),
        pytest.param([[]], id="empty-row"),
        pytest.param([[{"text": "Yes", "callbackData": "x"}]] * 11, id="too-many-rows"),
        pytest.param([[{"text": "Yes", "callbackData": "x"}] * 9], id="too-many-buttons"),
        pytest.param([[None]], id="not-a-button"),
        pytest.param([[{"text": "Yes", "callbackData": "x", "extra": True}]], id="unknown-field"),
        pytest.param([[{"text": "", "callbackData": "x"}]], id="empty-label"),
        pytest.param([[{"text": "x" * 129, "callbackData": "x"}]], id="long-label"),
        pytest.param([[{"text": "Yes\n", "callbackData": "x"}]], id="control-in-label"),
        pytest.param([[{"text": "Yes"}]], id="missing-action"),
        pytest.param(
            [[{"text": "Yes", "url": "https://example.com", "callbackData": "x"}]],
            id="two-actions",
        ),
        pytest.param([[{"text": "Yes", "callbackData": "x", "style": []}]], id="invalid-style"),
        pytest.param([[{"text": "Yes", "callbackData": []}]], id="invalid-callback-type"),
        pytest.param([[{"text": "Yes", "callbackData": "x\x00"}]], id="control-in-callback"),
        pytest.param([[{"text": "Yes", "callbackData": "😀" * 65}]], id="callback-byte-limit"),
        pytest.param([[{"text": "Yes", "url": "file:///private"}]], id="unsafe-url-scheme"),
        pytest.param(
            [[{"text": "Yes", "url": "https://secret@example.com"}]], id="url-credentials"
        ),
        pytest.param([[{"text": "\ud800", "callbackData": "x"}]], id="invalid-unicode"),
        pytest.param(
            [[{"text": "Yes", "url": "https://example.com/" + "x" * (16 * 1024)}]],
            id="keyboard-byte-limit",
        ),
    ],
)
async def test_invalid_keyboard_does_not_reach_transport(keyboard, monkeypatch):
    api = TeamsAPI(SimpleNamespace())
    request = AsyncMock(side_effect=AssertionError("Invalid keyboard reached the transport"))
    monkeypatch.setattr(api, "_request", request)
    with pytest.raises(ApiError):
        await api.send_text("chat", "hello", keyboard=keyboard)
    request.assert_not_awaited()


@pytest.mark.parametrize(
    ("url", "origin"),
    [
        ("https://faß.example/file?signature=opaque", "https://xn--fa-hia.example"),
        ("https://[0:0:0:0:0:0:0:1]:443/file", "https://[::1]"),
    ],
)
def test_download_origin_preserves_idna_identity_and_ip_equivalence(url, origin):
    assert _origin(url, False) == origin


async def test_idna_alias_cannot_reach_a_host_outside_the_download_allowlist(monkeypatch):
    api = TeamsAPI(Settings(base_url="https://fass.example/bot/v1", token="test-token"))
    client = Mock()
    client.get.side_effect = AssertionError("Untrusted hostname reached the HTTP client")
    monkeypatch.setattr(api, "_client", lambda: client)
    with pytest.raises(ApiError) as caught:
        await api.download("https://faß.example/file")
    assert caught.value.code == "untrusted_origin"
    client.get.assert_not_called()
