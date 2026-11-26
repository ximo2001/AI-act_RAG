"""Minimal Gradio demo: ``uv run python -m ai_act_rag.infrastructure.demo``."""

from __future__ import annotations

from functools import lru_cache

import gradio as gr

from ai_act_rag.application.answering import AnswerService
from ai_act_rag.infrastructure.config import Settings
from ai_act_rag.infrastructure.container import build_answer_service

EXAMPLES = [
    "Which AI practices are prohibited?",
    "What fines apply for non-compliance with the prohibited AI practices?",
    "¿Qué obligaciones de transparencia tienen los chatbots?",
    "What is the corporate income tax rate for AI companies?",
]


@lru_cache(maxsize=1)
def _service() -> AnswerService:
    """Build the pipeline once.

    Returns:
        The answer service.
    """
    return build_answer_service(Settings.from_env())


def answer(question: str) -> tuple[str, str]:
    """Gradio callback.

    Args:
        question: User question.

    Returns:
        (answer markdown, sources markdown).
    """
    ans = _service().ask(question)
    head = f"**{ans.answer}**\n\n"
    head += f"Cited: {', '.join(ans.cited_articles) or '-'} · confidence: {ans.confidence.value}"
    head += f" · {ans.latency_ms['total'] / 1000:.1f}s"
    sources = "\n\n---\n\n".join(f"**{h.chunk.chunk_id}**\n\n{h.chunk.body}" for h in ans.sources)
    return head, sources


def build() -> gr.Blocks:
    """Build the UI.

    Returns:
        Gradio app.
    """
    with gr.Blocks(title="AI Act RAG") as demo:
        gr.Markdown("# AI Act assistant\nAnswers about Regulation (EU) 2024/1689, with citations.")
        question = gr.Textbox(label="Question", lines=2)
        button = gr.Button("Ask", variant="primary")
        out = gr.Markdown()
        with gr.Accordion("Retrieved passages", open=False):
            sources = gr.Markdown()
        gr.Examples(EXAMPLES, inputs=question)
        button.click(answer, inputs=question, outputs=[out, sources])
        question.submit(answer, inputs=question, outputs=[out, sources])
    return demo  # type: ignore[no-any-return]  # gradio ships no type hints


if __name__ == "__main__":
    build().launch(server_name="0.0.0.0")
