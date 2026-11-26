"""Answer-a-question use case: retrieve -> prompt -> LLM -> validate -> post-process.

Post-processing is where most of the robustness of a small-model RAG lives:

* the raw output is parsed into :class:`LLMAnswer` (Pydantic), extracting the
  first JSON object if the model added text around it;
* citations are normalised (``"Art. 5(1)"`` -> ``"Article 5"``) and **filtered
  to units that were actually in the context**, so the system never shows a
  citation it did not retrieve;
* the abstention sentinel ``NOT_IN_TEXT`` (or an equivalent phrase in the first
  sentence) becomes a localised "not stated in the text" answer;
* unparseable output also returns that text (safest for a user) but is flagged
  with ``parse_error=True`` so the evaluation never mistakes it for a deliberate
  abstention.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from pydantic import ValidationError

from ai_act_rag.application.prompts import PromptTemplate, format_context
from ai_act_rag.domain.exceptions import LLMOutputError
from ai_act_rag.domain.models import Answer, Confidence, LLMAnswer, RetrievedChunk
from ai_act_rag.domain.ports import LLM, Retriever

ABSTAIN_MESSAGES = {
    "en": "Not stated in the text of the AI Act (Regulation (EU) 2024/1689).",
    "es": "No consta en el texto del Reglamento de IA (Reglamento (UE) 2024/1689).",
}
# Phrases that mean "the context does not say". The v2 prompt asks for the
# NOT_IN_TEXT sentinel; the other phrasings catch models/prompts that abstain in
# their own words, so that every prompt version is scored the same way.
# A bare "not specified/mentioned/..." only counts when it ends the clause (".!?;"
# or end of text) or is followed by "in the text/context...": "X is not specified
# by the deployer" is an answer about the deployer, and "X is not specified, but
# Article 73 requires 15 days" goes on to answer, so neither is an abstention.
_ABSTAIN_PATTERNS = re.compile(
    r"not_in_text|not stated in the text|no consta"
    r"|\bnot (specified|mentioned|stated|provided)\b"
    r"(?=\s*($|[.!?;]|in (the |this )?(given |provided )?"
    r"(text|context|passages?|regulation|document)))"
    r"|\bno (specific |relevant )?information\b"
    r"|(context|text|passages?) (does|do) not (contain|mention|specify|say|provide)"
    r"|no se (mencion|especific|proporcion|indic|establec)\w*|no hay información",
    re.IGNORECASE,
)
_SPANISH_HINTS = frozenset(
    """el la los las de del que qué cuál cuáles cuándo cómo para por según una un es son
    sistemas inteligencia artificial reglamento proveedores deben debe está""".split()
)
_ENGLISH_HINTS = frozenset(
    """the of what which when how for under is are does do must should systems
    artificial intelligence regulation providers""".split()
)
_ROMAN = r"[IVXLC]+"
LANGUAGE_NAMES = {"en": "English", "es": "Spanish"}


def detect_language(text: str) -> str:
    """Guess whether a question is Spanish or English (enough for two languages).

    Args:
        text: Question text.

    Returns:
        ``"es"`` or ``"en"``.
    """
    words = re.findall(r"[a-záéíóúñü]+", text.lower())
    es = sum(w in _SPANISH_HINTS for w in words) + 2 * any(c in text for c in "¿¡ñáéíóú")
    en = sum(w in _ENGLISH_HINTS for w in words)
    return "es" if es > en else "en"


def normalize_citation(raw: str) -> str | None:
    """Map free-form citations to unit ids (``Article N``, ``Annex X``, ``Recital N``).

    Args:
        raw: e.g. ``"Art. 5(1)(a)"``, ``"artículo 6"``, ``"Annex III, point 4"``.

    Returns:
        The canonical unit id, or None if it is not recognisable.
    """
    s = raw.strip()
    m = re.match(r"(?i)^(article|art\.?|artículo)\s*(\d+)", s)
    if m:
        return f"Article {int(m.group(2))}"
    m = re.match(rf"(?i)^(annex|anexo)\s*({_ROMAN})\b", s)
    if m:
        return f"Annex {m.group(2).upper()}"
    m = re.match(r"(?i)^(recital|considerando)\s*\(?(\d+)\)?", s)
    if m:
        return f"Recital {int(m.group(2))}"
    return None


def parse_llm_output(raw: str) -> LLMAnswer:
    """Parse and validate raw LLM text into :class:`LLMAnswer`.

    Args:
        raw: Model output, ideally a JSON object.

    Returns:
        Validated answer.

    Raises:
        LLMOutputError: If no valid JSON object matching the schema is found.
    """
    candidates = [raw]
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        candidates.append(match.group(0))
    last_error: Exception | None = None
    for text in candidates:
        try:
            data = json.loads(text)
            if isinstance(data, dict) and isinstance(data.get("confidence"), str):
                data["confidence"] = data["confidence"].strip().lower()
            return LLMAnswer.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
    raise LLMOutputError(f"Invalid LLM output: {last_error}; raw={raw[:200]!r}")


class AnswerService:
    """Implements the ``ask`` use case on top of the Retriever and LLM ports."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        prompt: PromptTemplate,
        top_k: int = 5,
        max_tokens: int = 384,
    ) -> None:
        """Wire the use case.

        Args:
            retriever: Any retriever (dense, BM25, hybrid...).
            llm: Any LLM provider.
            prompt: Prompt template version to use.
            top_k: Number of chunks put in the context.
            max_tokens: Generation budget.
        """
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt
        self.top_k = top_k
        self.max_tokens = max_tokens

    @staticmethod
    def json_schema() -> dict[str, Any]:
        """JSON schema of the structured output, used for constrained decoding.

        Returns:
            The schema of :class:`LLMAnswer`, with every field required so
            the grammar forces the model to emit citations and confidence.
        """
        schema = LLMAnswer.model_json_schema()
        schema["required"] = ["answer", "cited_articles", "confidence"]
        return schema

    def ask(self, question: str) -> Answer:
        """Answer a question with citations, or abstain.

        Args:
            question: User question (English or Spanish).

        Returns:
            The final :class:`Answer`.
        """
        lang = detect_language(question)
        t0 = time.perf_counter()
        hits = self.retriever.search(question, k=self.top_k)
        t1 = time.perf_counter()
        raw = self.llm.generate(
            system=self.prompt.system,
            user=self.prompt.render_user(question, format_context(hits), LANGUAGE_NAMES[lang]),
            json_schema=self.json_schema(),
            max_tokens=self.max_tokens,
        )
        t2 = time.perf_counter()
        answer = self._postprocess(question, lang, raw, hits)
        answer.latency_ms = {
            "retrieval": round((t1 - t0) * 1000, 1),
            "generation": round((t2 - t1) * 1000, 1),
            "total": round((t2 - t0) * 1000, 1),
        }
        return answer

    def _postprocess(
        self, question: str, lang: str, raw: str, hits: list[RetrievedChunk]
    ) -> Answer:
        """Validate the output, ground the citations and handle abstention.

        Args:
            question: User question.
            lang: Language of the question (``"en"`` or ``"es"``).
            raw: Raw LLM output.
            hits: Context chunks.

        Returns:
            Final answer (latency filled in by the caller).
        """
        try:
            parsed = parse_llm_output(raw)
        except LLMOutputError:
            # Unparseable output: show the abstention text (safest for the user),
            # but flag it so the evaluation reports it as a failure, not an abstention.
            return self._abstain(question, lang, hits, raw, parse_error=True)

        context_units = {h.chunk.unit_id for h in hits}
        grounded: list[str] = []
        for cit in parsed.cited_articles:
            unit = normalize_citation(cit)
            if unit and unit in context_units and unit not in grounded:
                grounded.append(unit)

        # Only the first sentence (or clause before ";") is checked, so an answer
        # that mentions a limitation later on ("... but the text does not specify X")
        # is kept.
        first_sentence = re.split(r"(?<=[.!?;])\s", parsed.answer.strip(), maxsplit=1)[0]
        if not parsed.answer.strip() or _ABSTAIN_PATTERNS.search(first_sentence):
            return self._abstain(question, lang, hits, raw)
        return Answer(
            question=question,
            answer=parsed.answer.strip(),
            cited_articles=grounded,
            confidence=parsed.confidence,
            abstained=False,
            sources=hits,
            latency_ms={},
            raw_output=raw,
        )

    @staticmethod
    def _abstain(
        question: str,
        lang: str,
        hits: list[RetrievedChunk],
        raw: str,
        parse_error: bool = False,
    ) -> Answer:
        """Build the "not stated in the text" answer.

        Args:
            question: User question.
            lang: ``"en"`` or ``"es"``.
            hits: Context chunks (still returned for transparency).
            raw: Raw LLM output.
            parse_error: Whether the abstention comes from unparseable output.

        Returns:
            Abstention answer.
        """
        return Answer(
            question=question,
            answer=ABSTAIN_MESSAGES[lang],
            cited_articles=[],
            confidence=Confidence.LOW,
            abstained=True,
            sources=hits,
            latency_ms={},
            raw_output=raw,
            parse_error=parse_error,
        )
