"""Check local-file boundaries and actual-byte voice/image classification."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hermes_vk_workspace.media import (
    Media,
    MediaError,
    content_type,
    local_file,
    native_voice_supported,
    normalize_voice,
    safe_filename,
    sniff_image,
    sniff_voice,
)


class MediaTests(unittest.IsolatedAsyncioTestCase):
    def test_safe_filename_handles_both_path_syntaxes_and_utf8_limit(self):
        self.assertEqual(safe_filename("../../report.txt"), "report.txt")
        self.assertEqual(safe_filename("C:\\secret\\report.txt"), "report.txt")
        self.assertEqual(safe_filename("../\x00.."), "attachment.bin")
        self.assertEqual(safe_filename("a\r\nb.txt"), "ab.txt")
        name = safe_filename("😀" * 500)
        self.assertLessEqual(len(name.encode("utf-8")), 180)
        self.assertNotIn("�", name)

    def test_image_signature_is_specific_and_overrides_misleading_filename(self):
        for data, mime in (
            (b"\x89PNG\r\n\x1a\n", "image/png"),
            (b"\xff\xd8\xff", "image/jpeg"),
            (b"GIF89a", "image/gif"),
            (b"RIFF0000WEBP", "image/webp"),
        ):
            self.assertEqual(sniff_image(data), mime)
        self.assertIsNone(sniff_image(b"RIFF0000WAVE"))
        self.assertIsNone(sniff_image(b"not an image"))

    def test_voice_containers_are_identified_from_bytes(self):
        examples = (
            (b"OggS", "audio/ogg", ".ogg"),
            (b"RIFF0000WAVE", "audio/wav", ".wav"),
            (b"ID3", "audio/mpeg", ".mp3"),
            (b"\xff\xfb\x90\x00", "audio/mpeg", ".mp3"),
            (b"\xff\xf1\x50\x80", "audio/aac", ".aac"),
            (b"\x00\x00\x00\x18ftypM4A ", "audio/mp4", ".m4a"),
            (b"\x1a\x45\xdf\xa3", "audio/webm", ".webm"),
        )
        for data, mime, suffix in examples:
            with self.subTest(mime=mime):
                self.assertEqual(sniff_voice(data), mime)
                media = normalize_voice(Media(data, "recording.wrong", "audio/wrong"))
                self.assertEqual(media.mime, mime)
                self.assertEqual(media.filename, "recording" + suffix)
                self.assertIs(media.data, data)
        with self.assertRaises(MediaError):
            normalize_voice(Media(b"not audio", "recording.ogg", "audio/ogg"))

    def test_native_voice_acceptance_uses_signature_not_extension(self):
        self.assertTrue(native_voice_supported(Media(b"OggS", "audio.mp3", "audio/mpeg")))
        self.assertFalse(native_voice_supported(Media(b"ID3", "audio.aac", "audio/aac")))
        self.assertFalse(native_voice_supported(Media(b"unknown", "audio.aac", "audio/aac")))
        value = normalize_voice(Media(b"OggS", "😀" * 100 + ".bin", "application/octet-stream"))
        self.assertTrue(value.filename.endswith(".ogg"))
        self.assertLessEqual(len(value.filename.encode("utf-8")), 180)

    def test_content_type_hints_are_sanitized(self):
        self.assertEqual(content_type("Audio/OGG; charset=binary", "voice"), "audio/ogg")
        self.assertEqual(content_type("application/octet-stream", "report.pdf"), "application/pdf")
        self.assertEqual(
            content_type("text/plain\r\nCookie: stolen", "blob.bin"), "application/octet-stream"
        )
        self.assertEqual(content_type(None, "photo.png"), "image/png")

    async def test_local_regular_file_is_read_and_byte_signature_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "misnamed.txt"
            data = b"\x89PNG\r\n\x1a\nexample"
            path.write_bytes(data)
            media = await local_file(path, len(data))
            self.assertEqual(media, Media(data, "misnamed.txt", "image/png"))
            self.assertNotIn(str(path), repr(media))

    async def test_local_file_size_is_bounded_even_if_fstat_underreports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.bin"
            path.write_bytes(b"x" * 33)
            with self.assertRaises(MediaError) as caught:
                await local_file(path, 32)
            self.assertEqual(caught.exception.code, "size_limit")
            real_stat = os.stat(path)
            values = list(real_stat)
            values[6] = 0
            with patch("hermes_vk_workspace.media.os.fstat", return_value=os.stat_result(values)):
                with self.assertRaises(MediaError) as caught:
                    await local_file(path, 32)
            self.assertEqual(caught.exception.code, "size_limit")

    async def test_missing_file_and_directory_fail_without_leaking_path(self):
        with tempfile.TemporaryDirectory() as directory:
            for path, code in (
                (Path(directory), "not_a_regular_file"),
                (Path(directory) / "private-secret.txt", "unreadable_file"),
            ):
                with self.assertRaises(MediaError) as caught:
                    await local_file(path, 10)
                self.assertEqual(caught.exception.code, code)
                self.assertNotIn(str(path), str(caught.exception))

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX FIFO contract")
    async def test_fifo_does_not_block_waiting_for_a_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fifo"
            os.mkfifo(path)
            with self.assertRaises(MediaError) as caught:
                await asyncio.wait_for(local_file(path, 100), timeout=1)
            self.assertEqual(caught.exception.code, "not_a_regular_file")

    async def test_invalid_media_bound_is_rejected_before_open(self):
        for limit in (True, 0, -1, 1.5):
            with self.assertRaises(MediaError) as caught:
                await local_file("unused", limit)
            self.assertEqual(caught.exception.code, "invalid_size_limit")
