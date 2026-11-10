"""Split raw contract text into clause-sized segments with character offsets.

Real contracts (especially EDGAR exhibits like the CUAD corpus) are messy: section numbers
appear inline, headings sit on their own line, and pages carry stray numbers. The segmenter
is deliberately simple and deterministic:

1. split on blank lines;
2. split long blocks again where an inline section number starts ("   4. EXCLUSIVITY");
3. merge heading-only fragments into the block that follows them;
4. drop page numbers and other fragments that are too short to be a clause.

Every segment keeps ``start``/``end`` offsets into the original text, so any citation the
pipeline produces can be traced back to the exact source span.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_BLANK = re.compile(r"\n\s*\n")
# An inline section start: 2+ spaces, then "4." / "4.1" / "12.3.1" / "(a)" and a capital.
_INLINE_SECTION = re.compile(r"\s{2,}(?=(?:\d{1,2}(?:\.\d{1,2}){0,2}\.?|\([a-z]\))\s+[A-Z])")
_HEADING_MAX = 80
MIN_CLAUSE_CHARS = 40


@dataclass(frozen=True)
class Segment:
    index: int
    start: int
    end: int
    text: str


def _spans(text: str, pattern: re.Pattern[str], base: int = 0) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for m in pattern.finditer(text):
        spans.append((base + pos, base + m.start()))
        pos = m.end()
    spans.append((base + pos, base + len(text)))
    return spans


def _strip(text: str, start: int, end: int) -> tuple[int, int]:
    chunk = text[start:end]
    lead = len(chunk) - len(chunk.lstrip())
    trail = len(chunk) - len(chunk.rstrip())
    return start + lead, end - trail


def _is_heading(chunk: str) -> bool:
    return len(chunk) <= _HEADING_MAX and not chunk.rstrip().endswith((".", ";", ":"))


def segment(text: str, min_chars: int = MIN_CLAUSE_CHARS) -> list[Segment]:
    """Return clause segments in document order."""
    raw: list[tuple[int, int]] = []
    for b_start, b_end in _spans(text, _BLANK):
        block = text[b_start:b_end]
        for s, e in _spans(block, _INLINE_SECTION, base=b_start):
            s, e = _strip(text, s, e)
            if e > s:
                raw.append((s, e))

    merged: list[tuple[int, int]] = []
    pending: int | None = None  # start offset of a heading waiting for its body
    for s, e in raw:
        chunk = text[s:e]
        if _is_heading(chunk):
            if pending is None:
                pending = s
            continue
        merged.append((pending if pending is not None else s, e))
        pending = None

    out: list[Segment] = []
    for s, e in merged:
        body = text[s:e]
        if len(body) >= min_chars:
            out.append(Segment(index=len(out), start=s, end=e, text=body))
    return out
