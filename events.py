"""Parse VK events without I/O; quoted content never becomes control input."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .config import ConfigError, identifier

MAX_PARTS = 100
MAX_ATTACHMENTS = 10
MAX_CONTEXT = 8000
MAX_INPUT_TEXT = 128 * 1024


def event_id(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("Invalid VK event ID")
    if isinstance(value, int) and 0 <= value < 10**30:
        return value
    if isinstance(value, str) and value.isascii() and value.isdigit() and len(value) <= 30:
        return int(value)
    raise ValueError("Invalid VK event ID")


def _id(value: Any) -> str | None:
    try:
        return identifier(value)
    except ConfigError, ValueError, TypeError:
        return None


def _text(value: Any, limit: int = MAX_INPUT_TEXT) -> str:
    if not isinstance(value, str):
        return ""
    # These bounds are plugin resource limits, not claims about the server.
    return value[:limit].encode("utf-8", "replace").decode("utf-8")


def _input_text(value: Any) -> str:
    """Bound primary instructions in UTF-8 bytes without cutting off their meaning."""
    if value is None:
        return ""
    if not isinstance(value, str) or len(value) > MAX_INPUT_TEXT:
        raise ValueError("invalid message text")
    encoded = value.encode("utf-8")
    if len(encoded) > MAX_INPUT_TEXT:
        raise ValueError("message text is too large")
    return value


def _name(person: dict) -> str:
    return (
        " ".join(
            filter(None, (_text(person.get("firstName"), 128), _text(person.get("lastName"), 128)))
        )
        or _id(person.get("userId"))
        or ""
    )


@dataclass(frozen=True)
class Attachment:
    file_id: str
    kind: str


@dataclass(frozen=True)
class Incoming:
    chat_id: str
    chat_type: str
    chat_name: str
    sender_id: str
    sender_name: str
    message_id: str
    text: str
    timestamp: datetime
    mentioned: bool
    attachments: tuple[Attachment, ...]
    parent_chat_id: str | None = None
    reply_id: str | None = None
    reply_text: str | None = None
    reply_author_id: str | None = None
    reply_author_name: str | None = None
    reply_is_own: bool = False
    context: str | None = None


def _parent_chat_id(payload: dict, chat_id: str, chat_type: str) -> str | None:
    """Accept a thread only with a distinct group parent and a valid root message."""
    if "parent_topic" not in payload:
        return None
    parent = payload["parent_topic"]
    if not isinstance(parent, dict) or chat_type == "dm":
        raise ValueError("invalid thread parent")
    parent_id = _id(parent.get("chatId"))
    if not parent_id or parent_id == chat_id or not _id(parent.get("messageId")):
        raise ValueError("invalid thread parent")
    return parent_id


def _direct_text(payload: dict, parts: list[dict]) -> str:
    """Use direct text or file captions as instructions, with one combined byte limit."""
    text = _input_text(payload.get("text"))
    if text.strip():
        return text
    captions = []
    size = 0
    for part in parts:
        if part.get("type") != "file":
            continue
        caption = _input_text(part["payload"].get("caption"))
        size += len(caption.encode("utf-8")) + bool(captions)
        if size > MAX_INPUT_TEXT:
            raise ValueError("message captions are too large")
        captions.append(caption)
    return "\n".join(captions).strip()


def _attachments(parts: list[dict]) -> tuple[Attachment, ...]:
    """Admit only direct files, keeping the first kind for each bounded file ID."""
    files: dict[str, Attachment] = {}
    for part in parts:
        file_id = _id(part["payload"].get("fileId"))
        kind = part.get("type")
        if kind not in ("file", "voice", "sticker") or not file_id or file_id in files:
            continue
        if kind == "file":
            declared = part["payload"].get("type")
            kind = declared if declared in ("image", "audio", "video") else "file"
        files[file_id] = Attachment(file_id, kind)
        if len(files) > MAX_ATTACHMENTS:
            raise ValueError("too many message attachments")
    return tuple(files.values())


def _forward_context(parts: list[dict]) -> str | None:
    """Quote a bounded preview without promoting nested mentions, commands, or files."""
    forwards = []
    for part in parts:
        nested = part["payload"].get("message")
        if part.get("type") != "forward" or not isinstance(nested, dict):
            continue
        author = nested.get("from") if isinstance(nested.get("from"), dict) else {}
        body = _text(nested.get("text"), 3000)
        if body:
            quoted = "\n".join("> " + line for line in body.splitlines())
            forwards.append(
                f"Forwarded message from {_name(author) or 'unknown sender'}:\n{quoted}"
            )
        if len(forwards) == 5:
            break
    return "\n\n".join(forwards)[:MAX_CONTEXT] or None


def _timestamp(value: Any) -> datetime:
    """Use receipt time when the provider timestamp cannot be represented."""
    try:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return datetime.fromtimestamp(value, UTC)
    except OverflowError, ValueError, OSError:
        pass
    return datetime.now(UTC)


def parse_message(event: dict, self_id: str) -> Incoming | None:
    if not isinstance(event, dict) or event.get("type") != "newMessage":
        return None
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return None
    chat, sender = payload.get("chat"), payload.get("from")
    if not isinstance(chat, dict) or not isinstance(sender, dict):
        return None
    chat_id, sender_id, msg_id = (
        _id(chat.get("chatId")),
        _id(sender.get("userId")),
        _id(payload.get("msgId")),
    )
    types = {"private": "dm", "group": "group", "channel": "channel"}
    wire_type = chat.get("type")
    chat_type = types.get(wire_type) if isinstance(wire_type, str) else None
    if not chat_id or not sender_id or not msg_id or not chat_type:
        return None
    if sender_id == self_id or sender.get("isBot", False) is not False:
        return None
    raw_parts = payload.get("parts", [])
    if not isinstance(raw_parts, list) or len(raw_parts) > MAX_PARTS:
        return None
    parts = [p for p in raw_parts if isinstance(p, dict) and isinstance(p.get("payload"), dict)]
    try:
        parent_id = _parent_chat_id(payload, chat_id, chat_type)
        text = _direct_text(payload, parts)
        attachments = _attachments(parts)
    except ValueError:
        return None
    reply = next(
        (
            p["payload"].get("message")
            for p in parts
            if p.get("type") == "reply" and isinstance(p["payload"].get("message"), dict)
        ),
        {},
    )
    reply_from = reply.get("from") if isinstance(reply.get("from"), dict) else {}
    reply_id = _id(reply.get("msgId"))
    reply_author = _id(reply_from.get("userId")) if reply_id else None
    reply_is_own = bool(reply_id and reply_author == self_id)
    mentioned = (
        reply_is_own
        or f"@[{self_id}]" in text
        or any(
            p.get("type") == "mention" and _id(p["payload"].get("userId")) == self_id for p in parts
        )
    )
    text = text.replace(f"@[{self_id}]", "").strip()
    context = _forward_context(parts)
    if not text and not attachments and not context and not reply_id:
        return None
    if not text:
        text = "[Attachment]" if attachments else "[Shared message]"
    return Incoming(
        chat_id=chat_id,
        chat_type=chat_type,
        chat_name=_text(chat.get("title"), 256),
        sender_id=sender_id,
        sender_name=_name(sender),
        message_id=msg_id,
        text=text,
        timestamp=_timestamp(payload.get("timestamp")),
        mentioned=mentioned,
        attachments=attachments,
        parent_chat_id=parent_id,
        reply_id=reply_id,
        reply_text=_text(reply.get("text"), MAX_CONTEXT) or None,
        reply_author_id=reply_author,
        reply_author_name=_name(reply_from) or None,
        reply_is_own=reply_is_own,
        context=context,
    )


def _same_supplied_ids(update: Any, original: Any, fields: tuple[str, ...]) -> bool:
    if not isinstance(update, dict) or not isinstance(original, dict):
        return False
    for field in fields:
        if field in update:
            supplied = _id(update[field])
            if supplied is None or supplied != _id(original.get(field)):
                return False
    return True


def _apply_message_change(original: dict | None, event: dict) -> dict | None:
    """Apply an edit/delete only to content; admission and thread identity stay fixed."""
    if original is None:
        return None
    before, payload = original["payload"], event["payload"]
    if (
        ("type" in payload["chat"] and payload["chat"]["type"] != before["chat"].get("type"))
        or (
            "from" in payload
            and not _same_supplied_ids(payload["from"], before.get("from"), ("userId",))
        )
        or (
            "parent_topic" in payload
            and not _same_supplied_ids(
                payload["parent_topic"], before.get("parent_topic"), ("chatId", "messageId")
            )
        )
    ):
        return original
    if event["type"] == "deletedMessage":
        return None
    updated = dict(before)
    if "text" in payload:
        updated["text"] = payload["text"]
    # An edited snapshot without parts must not replay removed attachments.
    updated["parts"] = payload.get("parts", [])
    return {**original, "payload": updated}


def coalesce_batch(events: list[dict]) -> list[dict]:
    """Apply an edit/delete in the same fetched batch before admitting its new message.

    Already running turns are never silently re-executed because a message changed.
    An edit cannot replace the sender, chat, or thread identity of the original.
    """
    pending: dict[tuple[str, str], int] = {}
    result: list[dict | None] = []
    for event in events:
        payload = event.get("payload")
        if not isinstance(payload, dict) or not isinstance(payload.get("chat"), dict):
            result.append(event)
            continue
        key = (_id(payload["chat"].get("chatId")), _id(payload.get("msgId")))
        kind = event.get("type")
        if None in key:
            result.append(event)
            continue
        if kind == "newMessage":
            if key in pending:
                continue
            pending[key] = len(result)
        elif kind in ("editedMessage", "deletedMessage") and key in pending:
            index = pending[key]
            result[index] = _apply_message_change(result[index], event)
            continue
        result.append(event)
    return [event for event in result if event is not None]
