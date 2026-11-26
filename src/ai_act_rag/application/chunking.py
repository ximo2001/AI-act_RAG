"""Structure-aware chunking of legal units.

Rules (chosen so that a chunk never mixes two articles and citations stay exact):

1. A chunk belongs to exactly one unit (article / recital / annex).
2. Consecutive paragraphs are packed greedily up to ``max_words``.
3. A single paragraph longer than ``max_words`` is split on its lines
   (list points such as ``(a)``, ``(b)``), and only as a last resort on words.
4. Every chunk gets a header (unit id, title, chapter) so that a chunk taken
   out of context still says where it comes from - this helps BM25, the
   embedding model and the LLM's citations.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_act_rag.domain.models import Chunk, LegalUnit, UnitType


@dataclass(frozen=True)
class ChunkingConfig:
    """Chunking parameters.

    Attributes:
        max_words: Soft upper bound of words per chunk body.
        include_recitals: Whether recitals are indexed (they explain intent
            but are not binding; useful for "why" questions).
    """

    max_words: int = 250
    include_recitals: bool = True


def _n_words(text: str) -> int:
    """Count whitespace-separated words.

    Args:
        text: Any text.

    Returns:
        Number of words.
    """
    return len(text.split())


def _split_long(paragraph: str, max_words: int) -> list[str]:
    """Split one oversized paragraph on line boundaries, then on words.

    Args:
        paragraph: Paragraph text (lines separated by newlines).
        max_words: Word budget per piece.

    Returns:
        Pieces of at most ``max_words`` words (except single huge lines that
        are cut on word boundaries).
    """
    pieces: list[str] = []
    current: list[str] = []
    count = 0
    for line in paragraph.split("\n"):
        n = _n_words(line)
        if n > max_words:
            if current:
                pieces.append("\n".join(current))
                current, count = [], 0
            words = line.split()
            pieces.extend(" ".join(words[i : i + max_words]) for i in range(0, n, max_words))
            continue
        if count + n > max_words and current:
            pieces.append("\n".join(current))
            current, count = [], 0
        current.append(line)
        count += n
    if current:
        pieces.append("\n".join(current))
    return pieces


def make_header(unit: LegalUnit) -> str:
    """Build the context header prepended to every chunk of a unit.

    Args:
        unit: The legal unit.

    Returns:
        e.g. ``"Article 5 - Prohibited AI practices | Chapter II - PROHIBITED AI PRACTICES"``.
    """
    parts = [unit.unit_id + (f" - {unit.title}" if unit.title else "")]
    if unit.chapter:
        parts.append(unit.chapter)
    if unit.section:
        parts.append(unit.section)
    return " | ".join(parts)


def chunk_unit(unit: LegalUnit, config: ChunkingConfig) -> list[Chunk]:
    """Split one legal unit into chunks.

    Args:
        unit: Article, recital or annex.
        config: Chunking parameters.

    Returns:
        Chunks with ids ``"<unit_id>#<i>"`` (i starting at 1).
    """
    pieces: list[str] = []
    for paragraph in unit.paragraphs:
        if _n_words(paragraph) > config.max_words:
            pieces.extend(_split_long(paragraph, config.max_words))
        else:
            pieces.append(paragraph)

    bodies: list[str] = []
    current: list[str] = []
    count = 0
    for piece in pieces:
        n = _n_words(piece)
        if count + n > config.max_words and current:
            bodies.append("\n\n".join(current))
            current, count = [], 0
        current.append(piece)
        count += n
    if current:
        bodies.append("\n\n".join(current))

    header = make_header(unit)
    return [
        Chunk(
            chunk_id=f"{unit.unit_id}#{i}",
            unit_id=unit.unit_id,
            unit_type=unit.unit_type,
            header=header,
            body=body,
        )
        for i, body in enumerate(bodies, start=1)
    ]


def chunk_units(units: list[LegalUnit], config: ChunkingConfig | None = None) -> list[Chunk]:
    """Chunk a whole corpus.

    Args:
        units: Parsed legal units.
        config: Chunking parameters (defaults to :class:`ChunkingConfig`).

    Returns:
        All chunks in document order.
    """
    config = config or ChunkingConfig()
    chunks: list[Chunk] = []
    for unit in units:
        if unit.unit_type == UnitType.RECITAL and not config.include_recitals:
            continue
        chunks.extend(chunk_unit(unit, config))
    return chunks
