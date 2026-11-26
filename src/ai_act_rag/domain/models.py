"""Domain models: the legal text, its chunks, retrieval hits and answers.

Pure data structures (pydantic) with no infrastructure dependencies.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

MAX_CITATIONS = 5


class UnitType(StrEnum):
    """Kind of structural unit of the Regulation."""

    ARTICLE = "article"
    RECITAL = "recital"
    ANNEX = "annex"


class LegalUnit(BaseModel):
    """One citable unit of the Regulation (an article, a recital or an annex).

    Attributes:
        unit_id: Human-readable, citable id, e.g. ``"Article 5"``, ``"Recital 27"``,
            ``"Annex III"``. This is what answers cite and what eval labels use.
        unit_type: Article, recital or annex.
        number: Number as printed (``"5"``, ``"27"``, ``"III"``).
        title: Article heading or annex title (recitals have none).
        chapter: e.g. ``"Chapter II - Prohibited AI practices"`` (articles only).
        section: e.g. ``"Section 1 - Classification of AI systems as high-risk"``.
        paragraphs: Text blocks in document order (numbered paragraphs for
            articles, a single block for recitals, top-level points for annexes).
    """

    unit_id: str
    unit_type: UnitType
    number: str
    title: str | None = None
    chapter: str | None = None
    section: str | None = None
    paragraphs: list[str]

    @property
    def text(self) -> str:
        """Return the full text of the unit.

        Returns:
            Paragraphs joined by blank lines.
        """
        return "\n\n".join(self.paragraphs)


class Chunk(BaseModel):
    """A retrievable piece of a :class:`LegalUnit`.

    Attributes:
        chunk_id: Unique id, e.g. ``"Article 5#2"`` (2nd chunk of Article 5).
        unit_id: Id of the unit this chunk belongs to (the citation target).
        unit_type: Type of the parent unit.
        header: Context line prepended for retrieval/generation
            (unit id, title, chapter).
        body: The chunk's own text.
    """

    chunk_id: str
    unit_id: str
    unit_type: UnitType
    header: str
    body: str

    @property
    def text(self) -> str:
        """Return header + body, the string that is embedded and indexed.

        Returns:
            The full chunk text.
        """
        return f"{self.header}\n{self.body}"


class RetrievedChunk(BaseModel):
    """A chunk returned by a retriever, with its score and 1-based rank."""

    chunk: Chunk
    score: float
    rank: int


class Confidence(StrEnum):
    """Self-reported confidence of the generated answer."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LLMAnswer(BaseModel):
    """Structured output the LLM must produce (validated JSON).

    Attributes:
        answer: Answer text, or the "not in the text" sentinel.
        cited_articles: Unit ids supporting the answer (e.g. ``"Article 5"``), at most
            :data:`MAX_CITATIONS`. The cap is also in the JSON schema (``maxItems``), so
            the constrained-decoding grammar cannot loop emitting citations forever.
        confidence: high / medium / low.
    """

    answer: str
    cited_articles: list[str] = Field(default_factory=list, max_length=MAX_CITATIONS)
    confidence: Confidence = Confidence.LOW

    @field_validator("cited_articles", mode="before")
    @classmethod
    def _clean(cls, value: Any) -> Any:
        """Strip, drop empty and duplicate citations, then keep the first few.

        Runs before validation so that an unconstrained provider returning more
        than :data:`MAX_CITATIONS` citations is truncated instead of rejected.

        Args:
            value: Raw citation list (anything else is left for pydantic to reject).

        Returns:
            Cleaned list, order preserved, at most :data:`MAX_CITATIONS` items.
        """
        if not isinstance(value, list):
            return value
        cleaned: list[Any] = []
        for item in value:
            if isinstance(item, str):
                item = item.strip()
            if item and item not in cleaned:
                cleaned.append(item)
        return cleaned[:MAX_CITATIONS]


class Answer(BaseModel):
    """Final answer returned to callers (API, MCP, demo, eval).

    Attributes:
        question: The user question.
        answer: Answer text (or the abstention sentinel).
        cited_articles: Citations, normalised and filtered to retrieved units.
        confidence: high / medium / low.
        abstained: True when the system answered "not stated in the text".
        sources: The chunks given to the LLM as context.
        latency_ms: Wall-clock latency split into retrieval / generation / total.
        raw_output: Raw LLM text, kept for debugging and eval.
        parse_error: True when the LLM output could not be parsed; the answer is then
            the abstention text, but eval must not count it as a deliberate abstention.
    """

    question: str
    answer: str
    cited_articles: list[str]
    confidence: Confidence
    abstained: bool
    sources: list[RetrievedChunk]
    latency_ms: dict[str, float]
    raw_output: str = ""
    parse_error: bool = False
