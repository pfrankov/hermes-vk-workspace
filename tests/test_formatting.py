from html.parser import HTMLParser

import pytest
from hermes_vk_workspace.formatting import render_chunks, utf16_length


class Parsed(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.stack = []
        self.text = ""
        self.links = []
        self.feed(html)
        assert not self.stack

    def handle_starttag(self, tag, attrs):
        assert tag in {"b", "i", "u", "s", "code", "pre", "a"}
        self.stack.append(tag)
        if tag == "a":
            self.links.append(dict(attrs)["href"])

    def handle_endtag(self, tag):
        assert self.stack.pop() == tag

    def handle_data(self, data):
        self.text += data


def test_formats_markdown_and_preserves_code_and_links():
    chunks = render_chunks(
        "# Заголовок\n\n**bold** _italic_ `a < b`\n\n"
        '```python\nx = "<&"\n```\n\n[ссылка](https://example.org/?a=1&b=2)'
    )
    assert len(chunks) == 1
    parsed = Parsed(chunks[0])
    assert "Заголовок" in parsed.text
    assert 'x = "<&"' in parsed.text
    assert "<pre>" in chunks[0] and "<code>" in chunks[0]
    assert parsed.links == ["https://example.org/?a=1&b=2"]


@pytest.mark.parametrize(
    "source", ["&😀<" * 3000, "**bold " + "я😀" * 4000 + "**", "```\n" + "&x\n" * 3000 + "```"]
)
def test_chunking_is_balanced_and_preserves_all_displayed_text(source):
    whole = "".join(Parsed(c).text for c in render_chunks(source, limit=100_000))
    chunks = render_chunks(source, limit=256)
    assert len(chunks) > 1
    assert all(utf16_length(c) <= 256 for c in chunks)
    assert "".join(Parsed(c).text for c in chunks) == whole


def test_html_is_sanitized_and_unsafe_links_cannot_become_clickable():
    raw_html = "<script>alert(1)</script><img src=x onerror=alert(2)>"
    chunks = render_chunks(
        raw_html + "\n\n[run](javascript:alert) [secret](https://user:pass@example.org)"
    )
    parsed = Parsed("".join(chunks))
    assert parsed.links == []
    assert raw_html in parsed.text
    assert "<script" not in "".join(chunks)
    assert "<img" not in "".join(chunks)
    assert "user:pass" not in "".join(chunks)


@pytest.mark.parametrize(
    "source",
    [
        "<PLACEHOLDER>",
        "Set <TOKEN> as the value.",
        "Paste <client_id> and <client_secret>.",
        '<div data-x="1">visible <b>literal</b></div>',
        "<!-- keep this configuration note -->",
        "<!DOCTYPE html>",
    ],
)
def test_literal_html_and_placeholders_keep_their_visible_text(source):
    chunks = render_chunks(source)
    assert "".join(Parsed(chunk).text for chunk in chunks) == source


def test_angle_bracket_autolinks_and_code_keep_their_markdown_meaning():
    chunks = render_chunks(
        "<https://example.org/?x=1&y=2> and `<TOKEN>`\n\n```html\n<div><TOKEN></div>\n```"
    )
    parsed = Parsed("".join(chunks))
    assert parsed.links == ["https://example.org/?x=1&y=2"]
    assert "<code>&lt;TOKEN&gt;</code>" in "".join(chunks)
    assert "<div><TOKEN></div>\n" in parsed.text


def test_lists_and_tables_remain_readable():
    html = "".join(render_chunks("- one\n- two\n\nA | B\n--- | ---\nx | y"))
    parsed = Parsed(html)
    assert "• one" in parsed.text
    assert "• two" in parsed.text
    assert "x | y" in parsed.text


def test_limits_and_empty_text():
    assert render_chunks(" \n") == []
    with pytest.raises(ValueError):
        render_chunks("x" * (128 * 1024 + 1))
    with pytest.raises(UnicodeError):
        render_chunks("\ud800")
    with pytest.raises(ValueError):
        render_chunks("text", 1)
