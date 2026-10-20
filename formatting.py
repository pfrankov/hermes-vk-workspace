"""Render Hermes Markdown as a small, escaped VK HTML subset.

Markdown is already a core Hermes dependency. VK-specific work is limited to
selecting supported tags and splitting complete, independently balanced messages.
"""

from dataclasses import dataclass
from html import escape
from html.parser import HTMLParser
from urllib.parse import urlsplit

import markdown

MAX_TEXT = 128 * 1024
TEXT_LIMIT = 4096
CAPTION_LIMIT = 1024


def utf16_length(text: str) -> int:
    """A conservative wire budget; astral Unicode characters cost two units."""
    return len(text.encode("utf-16-le")) // 2


@dataclass(frozen=True)
class Run:
    text: str
    tags: tuple[tuple[str, str], ...] = ()


class _VKHTML(HTMLParser):
    _TAGS = {
        "strong": "b",
        "b": "b",
        "em": "i",
        "i": "i",
        "del": "s",
        "s": "s",
        "u": "u",
        "code": "code",
        "pre": "pre",
        "h1": "b",
        "h2": "b",
        "h3": "b",
        "h4": "b",
        "h5": "b",
        "h6": "b",
    }
    _BLOCKS = frozenset(
        {
            "p",
            "div",
            "pre",
            "blockquote",
            "ul",
            "ol",
            "table",
            "tr",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }
    )

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.runs: list[Run] = []
        self.stack: list[tuple[str, tuple[str, str] | None]] = []
        self._after_cell = False

    def _text(self, text: str) -> None:
        if text:
            tags = tuple(pair for _, pair in self.stack if pair)
            self.runs.append(Run(text, tags))

    def _line(self) -> None:
        if self.runs and not self.runs[-1].text.endswith("\n"):
            self._text("\n")

    def _markup(self, tag, attrs) -> tuple[str, str] | None:
        """Allow only VK formatting tags and bounded, credential-free web links."""
        if tag in self._TAGS:
            # VK pre already means code; nested <code> adds no useful information.
            if tag == "code" and any(t == "pre" for t, _ in self.stack):
                return None
            rendered = self._TAGS[tag]
            return f"<{rendered}>", f"</{rendered}>"
        if tag != "a":
            return None
        href = attrs.get("href", "")
        try:
            url = urlsplit(href)
            safe = (
                url.scheme in {"https", "http"}
                and url.hostname
                and not url.username
                and not url.password
                and len(href) <= 512
                and all(ord(c) > 32 and ord(c) != 127 for c in href)
            )
        except ValueError:
            return None
        if safe:
            return f'<a href="{escape(href, quote=True)}">', "</a>"
        return None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in self._BLOCKS:
            self._line()
        if tag == "br":
            self._text("\n")
            return
        if tag == "hr":
            self._line()
            self._text("—\n")
            return
        if tag == "img":
            self._text(attrs.get("alt") or "[Image]")
            return
        if tag == "li":
            self._line()
            self._text("• ")
        if tag in {"td", "th"}:
            if self._after_cell:
                self._text(" | ")
            self._after_cell = True
        if tag == "tr":
            self._after_cell = False
        # Bound rich-text nesting without growing tag overhead forever.
        if len(self.stack) < 32:
            self.stack.append((tag, self._markup(tag, attrs)))

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break
        if tag in self._BLOCKS or tag == "li":
            self._line()

    def handle_data(self, data):
        self._text(data)


def _markdown_runs(content: str) -> list[Run]:
    """Interpret Markdown while keeping literal HTML as visible message content."""
    renderer = markdown.Markdown(extensions=["fenced_code", "tables", "sane_lists"])
    # Raw tags are literal message content (including <TOKEN> placeholders).
    # Only Markdown-generated tags may reach the VK HTML renderer.
    renderer.preprocessors.deregister("html_block")
    renderer.inlinePatterns.deregister("html")
    # Pretty-print whitespace is not message content; it breaks table rows and
    # duplicates the block separators supplied by _VKHTML.
    renderer.treeprocessors.deregister("prettify")
    parser = _VKHTML()
    parser.feed(renderer.convert(content))
    parser.close()
    # Block separators are not content and must not become a whitespace-only
    # continuation at an exact wire boundary. Preserve whitespace inside code.
    while parser.runs and not any(tag[0] in {"<pre>", "<code>"} for tag in parser.runs[-1].tags):
        last = parser.runs.pop()
        if text := last.text.rstrip():
            parser.runs.append(Run(text, last.tags))
            break
    return parser.runs


def _chunk_runs(runs: list[Run], limit: int) -> list[str]:
    """Encode runs within the wire budget, closing and reopening tags at boundaries."""
    chunks: list[str] = []
    body: list[str] = []
    body_size = 0
    active: tuple[tuple[str, str], ...] = ()

    def finish():
        nonlocal body_size, active
        if body:
            chunks.append("".join(body) + "".join(t[1] for t in reversed(active)))
        body.clear()
        body_size, active = 0, ()

    for run in runs:
        tags = run.tags
        if sum(utf16_length(a + b) for a, b in tags) + 6 > limit:
            tags = ()  # The text remains visible even when a link cannot fit a small caption.
        closing = "".join(t[1] for t in reversed(tags))
        shared = 0
        while shared < min(len(active), len(tags)) and active[shared] == tags[shared]:
            shared += 1
        transition = "".join(t[1] for t in reversed(active[shared:])) + "".join(
            t[0] for t in tags[shared:]
        )
        for char in run.text:
            encoded = escape(char, quote=False)
            needed = utf16_length(transition + encoded + closing)
            if body_size + needed > limit:
                finish()
                transition = "".join(t[0] for t in tags)
            body.append(transition + encoded)
            body_size += utf16_length(transition + encoded)
            active = tags
            transition = ""
    finish()
    return chunks


def render_chunks(content: str, limit: int = TEXT_LIMIT) -> list[str]:
    """Balanced HTML chunks bounded *after* escaping, with no network or file I/O."""
    if not isinstance(content, str) or len(content) > MAX_TEXT:
        raise ValueError("Message exceeds the plugin text limit")
    if limit < 64:
        raise ValueError("Text limit must be at least 64")
    # Reject unpaired surrogates before the transport's encoder sees them.
    utf16_length(content)
    if not content.strip():
        return []
    return _chunk_runs(_markdown_runs(content), limit)
