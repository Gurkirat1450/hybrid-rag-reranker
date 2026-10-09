"""Markdown chunking for the FastAPI tutorial corpus.

Rules:
* doc_id is the file name (e.g. "query-params.md").
* mkdocs noise lines (starting with "{*", "///" or "{!") are dropped.
* Documents are split on "##" and "###" headings (never inside fenced code blocks).
  Each chunk's heading is "Doc title > Section title"; text before the first section
  uses just "Doc title".
* Sections longer than ``max_chars`` are split on paragraph boundaries with roughly
  ``overlap`` characters carried over from the previous piece.
* Sections shorter than ``min_chars`` are merged into the previous chunk of the same doc.
* chunk_ids are 0..N-1 in file-name-sorted order, so the same corpus always yields
  the same ids.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MAX_CHARS = 1800
OVERLAP_CHARS = 200
MIN_CHARS = 200

NOISE_PREFIXES = ("{*", "///", "{!")

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*?)\s*$")
_ANCHOR_RE = re.compile(r"\s*\{\s*#[^}]*\}\s*$")
_FENCE_PREFIXES = ("```", "~~~")


@dataclass
class Chunk:
    chunk_id: int
    doc_id: str
    heading: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def embedding_text(chunk: Chunk) -> str:
    """Text that gets embedded (and BM25-tokenized): heading, newline, body."""
    return f"{chunk.heading}\n{chunk.text}"


def strip_noise(markdown: str) -> str:
    """Drop mkdocs-specific lines (code includes, admonition fences, snippets)."""
    return "\n".join(
        line for line in markdown.splitlines() if not line.lstrip().startswith(NOISE_PREFIXES)
    )


def clean_heading(raw: str) -> str:
    """Remove the trailing "{ #anchor }" that the FastAPI docs put on headings."""
    return _ANCHOR_RE.sub("", raw).strip()


def _normalize_body(lines: list[str]) -> str:
    body = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", body)


def split_sections(markdown: str, doc_id: str) -> tuple[str, list[tuple[str | None, str]]]:
    """Return (doc_title, [(section_title_or_None, body), ...]).

    The first "# " heading is the doc title. "##" and "###" start new sections.
    Deeper headings ("####" and below) stay inside their parent section's body.
    Lines inside fenced code blocks are never treated as headings.
    """
    title: str | None = None
    sections: list[tuple[str | None, str]] = []
    current_title: str | None = None
    current_lines: list[str] = []
    in_fence = False

    for line in strip_noise(markdown).splitlines():
        if line.lstrip().startswith(_FENCE_PREFIXES):
            in_fence = not in_fence
            current_lines.append(line)
            continue
        if not in_fence:
            match = _HEADING_RE.match(line)
            if match:
                level = len(match.group(1))
                text = clean_heading(match.group(2))
                if level == 1 and title is None:
                    title = text
                    continue
                sections.append((current_title, _normalize_body(current_lines)))
                current_title = text
                current_lines = []
                continue
        current_lines.append(line)

    sections.append((current_title, _normalize_body(current_lines)))
    return (title or Path(doc_id).stem), sections


def overlap_tail(text: str, overlap: int) -> str:
    """Last ~``overlap`` characters of ``text``, starting on a word boundary.

    If the cut lands mid-word, the partial word is dropped. Returns "" when
    ``overlap`` <= 0 or no whole word fits.
    """
    if overlap <= 0 or not text:
        return ""
    if len(text) <= overlap:
        return text.strip()
    tail = text[-overlap:]
    if not text[-overlap - 1].isspace():
        ws = re.search(r"\s", tail)
        tail = tail[ws.start():] if ws else ""
    return tail.strip()


def _hard_split(paragraph: str, limit: int) -> list[str]:
    """Split one oversized paragraph at whitespace so each piece is <= limit chars."""
    pieces: list[str] = []
    rest = paragraph
    while len(rest) > limit:
        cut = max(rest.rfind(" ", 0, limit + 1), rest.rfind("\n", 0, limit + 1))
        if cut <= 0:
            cut = limit
        pieces.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        pieces.append(rest)
    return pieces


def split_long(text: str, max_chars: int = MAX_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split ``text`` on paragraph boundaries into pieces of about ``max_chars``.

    Each piece after the first starts with ``overlap_tail(previous_piece, overlap)``.
    Paragraphs are joined with a blank line ("\\n\\n").
    """
    if len(text) <= max_chars:
        return [text]

    limit = max(1, max_chars - overlap - 2)
    units: list[str] = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        units.extend(_hard_split(para, limit) if len(para) > limit else [para])

    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        add = len(unit) + (2 if current else 0)
        if current and current_len + add > max_chars:
            pieces.append("\n\n".join(current))
            tail = overlap_tail(pieces[-1], overlap)
            current = [tail] if tail else []
            current_len = len(tail)
            add = len(unit) + (2 if current else 0)
        current.append(unit)
        current_len += add
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def chunk_markdown(
    markdown: str,
    doc_id: str,
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP_CHARS,
    min_chars: int = MIN_CHARS,
) -> list[tuple[str, str]]:
    """Chunk one document. Returns [(heading, text), ...] without ids."""
    title, sections = split_sections(markdown, doc_id)
    chunks: list[tuple[str, str]] = []
    for section_title, body in sections:
        if not body:
            continue
        heading = title if section_title is None else f"{title} > {section_title}"
        if len(body) < min_chars and chunks:
            prev_heading, prev_text = chunks[-1]
            addition = body if section_title is None else f"{section_title}\n{body}"
            chunks[-1] = (prev_heading, f"{prev_text}\n\n{addition}")
            continue
        for piece in split_long(body, max_chars, overlap):
            chunks.append((heading, piece))
    return chunks


def chunk_corpus(
    corpus_dir: Path | str,
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP_CHARS,
    min_chars: int = MIN_CHARS,
) -> list[Chunk]:
    """Chunk every ``*.md`` file in ``corpus_dir`` (sorted by file name)."""
    out: list[Chunk] = []
    for path in sorted(Path(corpus_dir).glob("*.md"), key=lambda p: p.name):
        markdown = path.read_text(encoding="utf-8")
        for heading, text in chunk_markdown(markdown, path.name, max_chars, overlap, min_chars):
            out.append(Chunk(chunk_id=len(out), doc_id=path.name, heading=heading, text=text))
    return out
