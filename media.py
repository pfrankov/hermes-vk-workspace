"""Small, bounded media helpers shared by the transport and adapter."""

from __future__ import annotations

import asyncio
import mimetypes
import os
import re
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


class MediaError(Exception):
    """A media failure whose message never includes a path or remote URL."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(f"VK Teams media: {code.replace('_', ' ')}")


@dataclass(frozen=True, slots=True)
class Media:
    data: bytes = field(repr=False)
    filename: str
    mime: str


_MIME = re.compile(r"[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+\Z", re.ASCII)
_AUDIO_EXTENSIONS = {
    "audio/aac": ".aac",
    "audio/ogg": ".ogg",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/webm": ".webm",
    "audio/flac": ".flac",
}
_NATIVE_VOICE = frozenset({"audio/aac", "audio/ogg", "audio/mp4"})


def safe_filename(value: str | None) -> str:
    """Keep a short display filename; discard either platform's path separators."""
    if not isinstance(value, str):
        return "attachment.bin"
    name = value.replace("\\", "/").rsplit("/", 1)[-1]
    name = "".join(char for char in name if ord(char) >= 32 and ord(char) != 127)
    name = name.encode("utf-8", errors="replace")[:180].decode("utf-8", errors="ignore")
    name = name.strip().rstrip(". ")
    return name if name and name not in {".", ".."} else "attachment.bin"


def content_type(value: str | None, filename: str) -> str:
    """Normalize an HTTP MIME hint without allowing it to create headers."""
    candidate = value.partition(";")[0].strip().lower() if isinstance(value, str) else ""
    if _MIME.fullmatch(candidate) and candidate != "application/octet-stream":
        return candidate
    return mimetypes.guess_type(filename)[0] or "application/octet-stream"


def sniff_image(data: bytes) -> str | None:
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:6] in {b"GIF87a", b"GIF89a"}:
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def sniff_voice(data: bytes) -> str | None:
    """Recognize audio containers in voice context; this does not decode audio."""
    if data.startswith(b"fLaC"):
        return "audio/flac"
    if data.startswith(b"OggS"):
        return "audio/ogg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "audio/wav"
    if data.startswith(b"ID3"):
        return "audio/mpeg"
    if len(data) >= 2 and data[0] == 255:
        if data[1] & 0xF6 == 0xF0:
            return "audio/aac"
        if data[1] & 0xE0 == 0xE0 and data[1] & 0x06 and data[1] & 0x18 != 0x08:
            return "audio/mpeg"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        return "audio/mp4"
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return "audio/webm"
    return None


def normalize_voice(media: Media) -> Media:
    mime = sniff_voice(media.data)
    if mime is None:
        raise MediaError("unrecognized_voice")
    suffix = _AUDIO_EXTENSIONS[mime]
    name = safe_filename(media.filename)
    stem = Path(name).stem
    stem = stem.encode("utf-8")[: 180 - len(suffix)].decode("utf-8", errors="ignore")
    return Media(media.data, (stem or "voice") + suffix, mime)


def native_voice_supported(media: Media) -> bool:
    return sniff_voice(media.data) in _NATIVE_VOICE


def convert_voice(media: Media, limit: int) -> Media:
    """Convert normalized voice bytes in a worker that owns its temporary files."""
    from gateway.platforms.base import transcode_to_ogg_opus

    # Work on the already authorized, bounded bytes. The worker owns its temporary
    # files even if the caller is cancelled while ffmpeg is finishing.
    with tempfile.TemporaryDirectory(prefix="hermes-vk-voice-") as directory:
        source = Path(directory) / media.filename
        source.write_bytes(media.data)
        output = transcode_to_ogg_opus(
            str(source), output_path=str(Path(directory) / "converted.ogg")
        )
        if output is None:
            raise MediaError("voice_conversion_requires_ffmpeg")
        with open(output, "rb") as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise MediaError("media_size_limit")
        return normalize_voice(Media(data, "voice.ogg", "audio/ogg"))


def _read_local(path: str | os.PathLike[str], max_bytes: int) -> Media:
    try:
        # Hermes resolves and authorizes the path first. Reject a final-component
        # symlink substituted between that check and this open.
        extra_flags = (
            getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        path = os.fspath(path)
        with open(
            path, "rb", opener=lambda path, flags: os.open(path, flags | extra_flags)
        ) as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise MediaError("not_a_regular_file")
            if info.st_size > max_bytes:
                raise MediaError("size_limit")
            # The extra byte catches a file growing after fstat.
            data = handle.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise MediaError("size_limit")
        name = safe_filename(path)
        return Media(data, name, sniff_image(data) or content_type(None, name))
    except IsADirectoryError:
        raise MediaError("not_a_regular_file") from None
    except OSError, TypeError, ValueError:
        raise MediaError("unreadable_file") from None


async def local_file(path: str | os.PathLike[str], max_bytes: int) -> Media:
    """Read a host-authorized local regular file without blocking the event loop."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise MediaError("invalid_size_limit")
    return await asyncio.to_thread(_read_local, path, max_bytes)
