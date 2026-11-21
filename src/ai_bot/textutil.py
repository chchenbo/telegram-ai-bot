"""Text helpers: split long messages the way Telegram measures them, and trim history."""

from __future__ import annotations

import re
from collections.abc import Iterable

# Telegram allows 4096 UTF-16 units per message; keep some headroom.
CHUNK_LIMIT = 3800

# Preferred break points, most natural first.
_BREAKS: tuple[tuple[str, int], ...] = (
    ("\n\n", 2),
    ("\n", 1),
    ("。", 1),
    ("！", 1),
    ("？", 1),
    ("；", 1),
    ("，", 1),
    (". ", 2),
    (" ", 1),
)


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_for_telegram(text: str) -> str:
    """Drop ASCII control characters Telegram rejects, keeping newlines and tabs."""
    return _CONTROL_CHARS.sub("", text).strip()


def utf16_len(text: str) -> int:
    """Telegram counts UTF-16 units, so emoji cost two."""
    return len(text.encode("utf-16-le")) // 2


def _prefix_len(text: str, limit: int) -> int:
    """Largest i such that text[:i] fits within limit UTF-16 units."""
    total = 0
    for i, ch in enumerate(text):
        total += 2 if ord(ch) > 0xFFFF else 1
        if total > limit:
            return max(i, 1)
    return len(text)


def take_prefix(text: str, limit: int = CHUNK_LIMIT) -> tuple[str, str]:
    """Cut off a piece no longer than limit, preferring a natural break.

    Returns (head, rest) with head + rest == text, so it can be accumulated while streaming.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    if utf16_len(text) <= limit:
        return text, ""

    idx = _prefix_len(text, limit)
    if idx >= len(text):
        return text, ""

    window = text[:idx]
    best_at, best_len = -1, 0
    for sep, sep_len in _BREAKS:
        pos = window.rfind(sep)
        if pos > best_at:
            best_at, best_len = pos, sep_len

    # A break too close to the start is not worth it; fall back to a hard cut.
    if best_at != -1 and best_at + best_len >= idx // 2:
        cut = best_at + best_len
        return text[:cut], text[cut:]
    return text[:idx], text[idx:]


def split_text(text: str, limit: int = CHUNK_LIMIT) -> list[str]:
    """Split a block of text into messages, trimming edges and never yielding empty chunks."""
    chunks: list[str] = []
    rest = text.strip()
    while rest:
        head, rest = take_prefix(rest, limit)
        head = head.strip()
        if head:
            chunks.append(head)
        rest = rest.lstrip()
    return chunks


def trim_history(
    history: Iterable[dict],
    max_messages: int = 40,
    max_chars: int = 48000,
) -> list[dict]:
    """Keep the most recent turns, and never start the window with an assistant message."""
    items = [dict(item) for item in history]
    total = sum(len(item.get("content") or "") for item in items)

    while items and (len(items) > max_messages or total > max_chars):
        dropped = items.pop(0)
        total -= len(dropped.get("content") or "")

    while items and items[0].get("role") != "user":
        items.pop(0)

    return items
