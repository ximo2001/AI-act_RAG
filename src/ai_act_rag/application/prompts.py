"""Versioned prompt templates stored as text files in ``prompts/``.

File format (``prompts/answer_<version>.txt``)::

    # free comment lines
    ### system
    ...system prompt...
    ### user
    ...user prompt with {context} and {question}...

Placeholders are replaced with ``str.replace`` (not ``str.format``) so prompts
can contain literal JSON braces.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ai_act_rag.domain.models import RetrievedChunk


@dataclass(frozen=True)
class PromptTemplate:
    """A system + user prompt pair.

    Attributes:
        version: Version tag (file suffix), recorded in eval results.
        system: System prompt.
        user: User prompt template with ``{context}`` and ``{question}``.
    """

    version: str
    system: str
    user: str

    def render_user(self, question: str, context: str, language: str = "English") -> str:
        """Fill the user template.

        Args:
            question: User question.
            context: Formatted context passages.
            language: Answer language name, for templates with ``{language}``.

        Returns:
            The user prompt.
        """
        return (
            self.user.replace("{context}", context)
            .replace("{question}", question)
            .replace("{language}", language)
        )


def load_prompt(
    version: str, prompts_dir: Path = Path("prompts"), kind: str = "answer"
) -> PromptTemplate:
    """Load ``prompts/<kind>_<version>.txt``.

    Args:
        version: e.g. ``"v2"``.
        prompts_dir: Folder with prompt files.
        kind: Prompt family (``"answer"`` or ``"translate"``).

    Returns:
        The parsed template.

    Raises:
        FileNotFoundError: If the version does not exist.
        ValueError: If a section is missing.
    """
    text = (prompts_dir / f"{kind}_{version}.txt").read_text(encoding="utf-8")
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("### "):
            current = line[4:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
        # lines before the first section are comments
    if "system" not in sections or "user" not in sections:
        raise ValueError(f"Prompt {version} needs '### system' and '### user' sections")
    return PromptTemplate(
        version=version,
        system="\n".join(sections["system"]).strip(),
        user="\n".join(sections["user"]).strip(),
    )


def format_context(hits: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as citable passages.

    Each passage starts with its citation id in square brackets so the model
    can copy it verbatim into ``cited_articles``.

    Args:
        hits: Retrieved chunks.

    Returns:
        Passages separated by blank lines.
    """
    return "\n\n".join(f"[{h.chunk.unit_id}] {h.chunk.text}" for h in hits)
