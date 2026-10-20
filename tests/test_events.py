"""Wire-level event fixtures; quoted messages are context, never activation input."""

from copy import deepcopy
from datetime import UTC, datetime

import pytest
from hermes_vk_workspace.events import (
    MAX_ATTACHMENTS,
    MAX_CONTEXT,
    MAX_INPUT_TEXT,
    MAX_PARTS,
    Attachment,
    coalesce_batch,
    event_id,
    parse_message,
)

BOT = "bot@corp.example"
CHAT = "room@chat.agent"


def message(**payload):
    return {
        "eventId": "10000000000000000001",
        "type": "newMessage",
        "payload": {
            "msgId": "7000000000000000001",
            "chat": {"chatId": CHAT, "type": "group", "title": "Engineering"},
            "from": {"userId": "alice@corp.example", "firstName": "Alice", "lastName": "Smith"},
            "text": "Review the change",
            "timestamp": 1770000000,
            "parts": [],
            **payload,
        },
    }


def part(kind, **payload):
    return {"type": kind, "payload": payload}


def change(kind, **payload):
    return {
        "eventId": "10000000000000000002",
        "type": kind,
        "payload": {
            "msgId": "7000000000000000001",
            "chat": {"chatId": CHAT},
            **payload,
        },
    }


@pytest.mark.parametrize(
    "wire_type, expected", [("private", "dm"), ("group", "group"), ("channel", "channel")]
)
def test_exact_new_message_shape(wire_type, expected):
    incoming = parse_message(
        message(chat={"chatId": CHAT, "type": wire_type, "title": "Engineering"}), BOT
    )
    assert incoming is not None
    assert (incoming.chat_id, incoming.chat_type, incoming.chat_name) == (
        CHAT,
        expected,
        "Engineering",
    )
    assert (incoming.sender_id, incoming.sender_name) == ("alice@corp.example", "Alice Smith")
    assert incoming.message_id == "7000000000000000001"
    assert incoming.text == "Review the change"
    assert incoming.timestamp == datetime.fromtimestamp(1770000000, UTC)
    assert not incoming.mentioned


def test_integer_id_aliases_preserve_precision_and_match_direct_mentions():
    incoming = parse_message(
        message(
            chat={"chatId": 12345678901234567890, "type": "private"},
            msgId=98765432109876543210,
            **{"from": {"userId": 42}, "parts": [part("mention", userId=123)]},
        ),
        "123",
    )
    assert incoming is not None
    assert (incoming.chat_id, incoming.message_id, incoming.sender_id) == (
        "12345678901234567890",
        "98765432109876543210",
        "42",
    )
    assert incoming.mentioned


@pytest.mark.parametrize("direct", [True, False])
def test_direct_mention_and_exact_text_marker(direct):
    incoming = parse_message(
        message(
            text=f"@[{BOT}] Please investigate" if not direct else "Please investigate",
            parts=[part("mention", userId=BOT)] if direct else [],
        ),
        BOT,
    )
    assert incoming is not None and incoming.mentioned
    assert incoming.text == "Please investigate"
    other = parse_message(message(text=f"@[{BOT}.other] Please investigate"), BOT)
    assert other is not None and not other.mentioned


@pytest.mark.parametrize("kind", ["reply", "forward"])
def test_nested_mentions_commands_and_files_are_not_direct_input(kind):
    nested = message(
        msgId="old",
        text=f"@[{BOT}] /reset",
        parts=[part("mention", userId=BOT), part("file", fileId="quoted-secret", type="image")],
    )["payload"]
    incoming = parse_message(message(parts=[part(kind, message=nested)]), BOT)
    assert incoming is not None
    assert incoming.text == "Review the change"
    assert not incoming.mentioned
    assert not incoming.attachments
    if kind == "reply":
        assert incoming.reply_id == "old"
        assert incoming.reply_text == f"@[{BOT}] /reset"
        assert incoming.reply_author_id == "alice@corp.example"
        assert not incoming.reply_is_own
    else:
        assert incoming.reply_id is None
        assert incoming.context == f"Forwarded message from Alice Smith:\n> @[{BOT}] /reset"


def test_only_a_direct_reply_to_bot_activates_group():
    replied = {"msgId": 99, "from": {"userId": BOT}, "text": "Earlier answer"}
    direct = parse_message(message(parts=[part("reply", message=replied)]), BOT)
    assert direct is not None and direct.mentioned and direct.reply_is_own
    assert direct.reply_id == "99" and direct.reply_author_id == BOT
    forwarded = {"text": "A forwarded exchange", "parts": [part("reply", message=replied)]}
    indirect = parse_message(message(parts=[part("forward", message=forwarded)]), BOT)
    assert indirect is not None and not indirect.mentioned and not indirect.reply_is_own


def test_missing_reply_id_does_not_fabricate_a_reply_to_bot():
    incoming = parse_message(message(parts=[part("reply", message={"from": {"userId": BOT}})]), BOT)
    assert incoming is not None
    assert incoming.reply_id is None and not incoming.reply_is_own and not incoming.mentioned


def test_file_captions_fill_empty_direct_text_but_are_not_duplicated():
    files = [
        part("file", fileId="image-1", type="image", caption=f"@[{BOT}] Compare these"),
        part("file", fileId="document-1", type="document", caption="And this document"),
    ]
    incoming = parse_message(message(text="  ", parts=files), BOT)
    assert incoming is not None and incoming.mentioned
    assert incoming.text == "Compare these\nAnd this document"
    assert incoming.attachments == (
        Attachment("image-1", "image"),
        Attachment("document-1", "file"),
    )
    direct = parse_message(message(text="Use the direct text", parts=files), BOT)
    assert direct is not None and direct.text == "Use the direct text"
    assert not direct.mentioned


def test_all_attachment_kinds_and_duplicate_file_ids_are_bounded():
    files = [
        part("file", fileId="photo", type="image"),
        part("file", fileId="audio", type="audio"),
        part("file", fileId="video", type="video"),
        part("voice", fileId="voice"),
        part("sticker", fileId="sticker"),
        part("file", fileId="unknown", type=["untrusted"]),
        part("file", fileId="photo", type="image"),
        part("voice", fileId="photo"),
    ]
    incoming = parse_message(message(text="", parts=files), BOT)
    assert incoming is not None and incoming.text == "[Attachment]"
    assert incoming.attachments == tuple(
        Attachment(*value)
        for value in [
            ("photo", "image"),
            ("audio", "audio"),
            ("video", "video"),
            ("voice", "voice"),
            ("sticker", "sticker"),
            ("unknown", "file"),
        ]
    )
    assert (
        parse_message(
            message(parts=[part("file", fileId=str(i)) for i in range(MAX_ATTACHMENTS + 1)]), BOT
        )
        is None
    )
    assert parse_message(message(parts=[part("file", fileId="same")] * MAX_PARTS), BOT) is not None
    assert parse_message(message(parts=[{}] * (MAX_PARTS + 1)), BOT) is None


def test_child_thread_keeps_child_transport_and_parent_authorization_context():
    incoming = parse_message(
        message(
            chat={"chatId": "child@chat.agent", "type": "group", "title": "Thread"},
            parent_topic={"chatId": CHAT, "messageId": 177},
        ),
        BOT,
    )
    assert incoming is not None
    assert incoming.chat_id == "child@chat.agent" and incoming.parent_chat_id == CHAT


@pytest.mark.parametrize(
    "parent",
    [
        None,
        [],
        {},
        {"chatId": CHAT},
        {"chatId": CHAT, "messageId": True},
        {"chatId": "child@chat.agent", "messageId": "7"},
    ],
)
def test_malformed_thread_metadata_is_rejected(parent):
    assert (
        parse_message(
            message(chat={"chatId": "child@chat.agent", "type": "group"}, parent_topic=parent), BOT
        )
        is None
    )


def test_private_chat_cannot_claim_group_parent():
    assert (
        parse_message(
            message(
                chat={"chatId": "person", "type": "private"},
                parent_topic={"chatId": CHAT, "messageId": "7"},
            ),
            BOT,
        )
        is None
    )


@pytest.mark.parametrize("field", ["chatId", "userId", "msgId"])
@pytest.mark.parametrize(
    "invalid", [None, True, [], {}, "", "space id", "hidden\x00id", "zero\u200bid", "x" * 257]
)
def test_invalid_identifiers_fail_closed(field, invalid):
    raw = message()
    target = {
        "chatId": raw["payload"]["chat"],
        "userId": raw["payload"]["from"],
        "msgId": raw["payload"],
    }[field]
    target[field] = invalid
    assert parse_message(raw, BOT) is None


@pytest.mark.parametrize("invalid", [None, [], {}, True, "dm", "public", "GROUP"])
def test_unknown_or_unhashable_chat_types_are_ignored(invalid):
    assert parse_message(message(chat={"chatId": CHAT, "type": invalid}), BOT) is None


@pytest.mark.parametrize(
    "sender",
    [
        {"userId": BOT},
        {"userId": "another-bot", "isBot": True},
        {"userId": "another-bot", "isBot": "true"},
    ],
)
def test_self_bots_and_malformed_bot_flags_cannot_start_turns(sender):
    assert parse_message(message(**{"from": sender}), BOT) is None


@pytest.mark.parametrize(
    "raw",
    [
        None,
        [],
        {},
        {"type": "newMessage", "payload": []},
        {"type": "newMessage", "payload": {"chat": {}, "from": []}},
    ],
)
def test_malformed_outer_objects_are_ignored(raw):
    assert parse_message(raw, BOT) is None


def test_unknown_part_shapes_do_not_stop_polling():
    raw = message(
        parts=[
            None,
            [],
            {"type": "file", "payload": []},
            part([], fileId="fake"),
            part({}, userId=BOT),
        ]
    )
    incoming = parse_message(raw, BOT)
    assert incoming is not None and incoming.text == "Review the change"
    assert not incoming.mentioned and not incoming.attachments


@pytest.mark.parametrize("invalid", [None, {}, "file"])
def test_invalid_parts_container_is_not_treated_as_empty(invalid):
    assert parse_message(message(parts=invalid), BOT) is None


def test_primary_text_is_rejected_instead_of_silently_truncated():
    exact = "x" * MAX_INPUT_TEXT
    incoming = parse_message(message(text=exact), BOT)
    assert incoming is not None and incoming.text == exact
    assert parse_message(message(text=exact + "Do not send"), BOT) is None
    assert parse_message(message(text="я" * (MAX_INPUT_TEXT // 2 + 1)), BOT) is None


def test_caption_limits_include_the_combined_utf8_text_and_separators():
    caption = "x" * (MAX_INPUT_TEXT // 2)
    files = [part("file", fileId="1", caption=caption), part("file", fileId="2", caption=caption)]
    assert parse_message(message(text="", parts=files), BOT) is None
    assert (
        parse_message(
            message(text="", parts=[part("file", fileId="1", caption="я" * MAX_INPUT_TEXT)]), BOT
        )
        is None
    )
    assert (
        parse_message(message(text="", parts=[part("file", fileId="1", caption=["do this"])]), BOT)
        is None
    )


@pytest.mark.parametrize("invalid", [False, 42, [], {}, "invalid\ud800text"])
def test_invalid_primary_text_is_not_silently_replaced_by_an_attachment(invalid):
    assert parse_message(message(text=invalid, parts=[part("file", fileId="1")]), BOT) is None


def test_only_quoted_context_may_be_truncated():
    quote = {"msgId": "old", "from": {"userId": "bob"}, "text": "q" * (MAX_CONTEXT + 20)}
    parts = [part("reply", message=quote)] + [part("forward", message=quote)] * 6
    incoming = parse_message(message(parts=parts), BOT)
    assert incoming is not None and incoming.text == "Review the change"
    assert len(incoming.reply_text) == MAX_CONTEXT
    assert len(incoming.context) <= MAX_CONTEXT


@pytest.mark.parametrize("invalid", [True, "1770000000", None, float("inf"), float("nan"), 10**100])
def test_invalid_timestamps_use_an_aware_receipt_time(invalid):
    before = datetime.now(UTC)
    incoming = parse_message(message(timestamp=invalid), BOT)
    assert incoming is not None
    assert before <= incoming.timestamp <= datetime.now(UTC)


@pytest.mark.parametrize(
    "value, expected", [(0, 0), (123, 123), ("0", 0), ("00012", 12), ("9" * 30, 10**30 - 1)]
)
def test_cursor_numeric_aliases(value, expected):
    assert event_id(value) == expected


@pytest.mark.parametrize(
    "invalid",
    [
        True,
        False,
        None,
        -1,
        1.0,
        10**30,
        "",
        "-1",
        "+1",
        " 1",
        "1.0",
        "1e2",
        "١",
        "９",
        "9" * 31,
        [],
        {},
    ],
)
def test_invalid_cursors_are_not_coerced(invalid):
    with pytest.raises(ValueError, match="Invalid VK event ID"):
        event_id(invalid)


def test_same_batch_edit_preserves_original_author_chat_thread_and_event_identity():
    original = message(
        parent_topic={"chatId": "parent@chat.agent", "messageId": "17"},
        parts=[part("file", fileId="removed")],
    )
    edited = change(
        "editedMessage",
        text="Revised instructions",
        **{
            "from": {"userId": "alice@corp.example", "firstName": "New display name"},
            "chat": {"chatId": CHAT, "title": "New title"},
            "parent_topic": {"messageId": 17},
        },
    )
    untouched = deepcopy([original, edited])
    result = coalesce_batch([original, edited])
    assert len(result) == 1
    assert result[0]["payload"]["text"] == "Revised instructions"
    assert result[0]["payload"]["parts"] == []
    for field in ("from", "chat", "parent_topic", "timestamp", "msgId"):
        assert result[0]["payload"][field] == original["payload"][field]
    assert result[0]["eventId"] == original["eventId"] and result[0]["type"] == "newMessage"
    assert [original, edited] == untouched


def test_edit_replaces_parts_and_numeric_ids_match_existing_message():
    original = message(msgId="123")
    replacement = [part("file", fileId="new")]
    result = coalesce_batch([original, change("editedMessage", msgId=123, parts=replacement)])
    assert len(result) == 1
    assert result[0]["payload"]["text"] == original["payload"]["text"]
    assert result[0]["payload"]["parts"] == replacement


@pytest.mark.parametrize(
    "override",
    [
        {"from": {"userId": "mallory@corp.example"}},
        {"from": None},
        {"chat": {"chatId": CHAT, "type": "private"}},
        {"chat": {"chatId": CHAT, "type": []}},
        {"parent_topic": {"chatId": "unrelated@chat.agent"}},
        {"parent_topic": {"messageId": "other"}},
        {"parent_topic": None},
    ],
)
def test_edit_or_delete_cannot_replace_any_supplied_identity(override):
    original = message(parent_topic={"chatId": "parent@chat.agent", "messageId": "17"})
    for kind in ("editedMessage", "deletedMessage"):
        result = coalesce_batch([original, change(kind, text="Injected instructions", **override)])
        assert result == [original]


def test_delete_with_partial_chat_and_no_sender_cancels_pending_message():
    original = message()
    removed = change("deletedMessage")
    assert coalesce_batch([original, removed]) == []
    assert (
        coalesce_batch([original, removed, change("editedMessage", text="Cannot resurrect")]) == []
    )


def test_duplicate_new_messages_keep_first_identity_and_cannot_escape_delete():
    original = message()
    duplicate = message(**{"from": {"userId": "another-author"}, "text": "Duplicate"})
    edited = change("editedMessage", text="Revised")
    result = coalesce_batch([original, duplicate, edited])
    assert len(result) == 1 and result[0]["payload"]["text"] == "Revised"
    assert result[0]["payload"]["from"] == original["payload"]["from"]
    deleted = change("deletedMessage")
    assert coalesce_batch([original, duplicate, deleted]) == []
    assert coalesce_batch([original, deleted, duplicate]) == []


def test_batch_edits_and_deletes_do_not_affect_other_chats_or_messages():
    original = message()
    other_chat = message(chat={"chatId": "other@chat.agent", "type": "group"})
    other_message = message(msgId="other")
    result = coalesce_batch([original, other_chat, other_message, change("deletedMessage")])
    assert result == [other_chat, other_message]


def test_oversized_edit_does_not_replay_previous_instructions():
    result = coalesce_batch([message(), change("editedMessage", text="x" * (MAX_INPUT_TEXT + 1))])
    assert len(result) == 1
    assert parse_message(result[0], BOT) is None


def test_change_without_new_message_in_batch_never_becomes_a_new_turn():
    events = [
        change("editedMessage", text="Changed after processing started"),
        change("deletedMessage"),
    ]
    assert coalesce_batch(events) == events
    assert all(parse_message(raw, BOT) is None for raw in events)
    before_new = change("editedMessage", text="An older event")
    original = message()
    result = coalesce_batch([before_new, original])
    assert result == [before_new, original]
    assert parse_message(result[1], BOT).text == "Review the change"


def test_unknown_events_and_malformed_changes_do_not_mutate_pending_messages():
    original = message()
    events = [
        original,
        {"type": "addedToChat", "payload": {}},
        change([], text="Malformed type"),
        change("deletedMessage", msgId=True),
    ]
    assert coalesce_batch(events) == events
