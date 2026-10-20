"""A private polling cursor, not a replacement for Hermes's execution journal."""

import asyncio
import json
import os
import stat
import tempfile
from pathlib import Path

from .events import event_id


class StateError(RuntimeError):
    pass


class CursorStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> int:
        extra_flags = getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            # open() owns the descriptor even if file-object construction fails.
            with open(
                self.path, "rb", opener=lambda path, flags: os.open(path, flags | extra_flags)
            ) as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_size > 1024:
                    raise ValueError("invalid state file")
                raw = stream.read(1025)
                if len(raw) > 1024:
                    raise ValueError("state file grew while reading")
                data = json.loads(raw)
            if (
                not isinstance(data, dict)
                or set(data) != {"version", "last_event_id"}
                or type(data["version"]) is not int
                or data["version"] != 1
            ):
                raise ValueError("invalid state schema")
            return event_id(data["last_event_id"])
        except FileNotFoundError:
            return 0
        except OSError:
            raise StateError("VK polling cursor cannot be read") from None
        except ValueError, TypeError:
            raise StateError("VK polling cursor is invalid; inspect it before restarting") from None

    def _save(self, value: int) -> None:
        temporary = None
        try:
            value = event_id(value)
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", prefix=".cursor-", dir=self.path.parent, delete=False
                ) as stream:
                    temporary = Path(stream.name)
                    json.dump({"version": 1, "last_event_id": value}, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        except OSError, ValueError, TypeError:
            raise StateError("VK polling cursor could not be saved; polling stopped") from None

    async def save(self, value: int) -> None:
        # Do not release the consumer lock while a cancelled write is still running.
        task = asyncio.create_task(asyncio.to_thread(self._save, value))
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        task.result()
        if cancelled:
            raise asyncio.CancelledError
