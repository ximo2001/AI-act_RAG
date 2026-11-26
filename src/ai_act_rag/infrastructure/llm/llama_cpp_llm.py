"""Local GGUF model through llama-cpp-python (default provider, CPU-only)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_act_rag.domain.exceptions import ProviderUnavailableError


class LlamaCppLLM:
    """Runs a GGUF chat model in-process with llama.cpp.

    When a JSON schema is given, llama.cpp compiles it into a grammar, so the
    sampler can only emit tokens that keep the output valid JSON for the
    schema. That guarantees well-formed JSON only if generation is not cut by
    ``max_tokens``: a truncated object is still invalid. The caller validates
    with Pydantic and reports such output as ``parse_error`` (in the stored
    results, q17 of the 1.5B v1 run and q40 of the 1.5B v3 run).
    """

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> None:
        """Load the model.

        Args:
            model_path: Path to the ``.gguf`` file.
            n_ctx: Context window in tokens.
            n_threads: CPU threads (None = llama.cpp default).
            temperature: Sampling temperature (0 = greedy, reproducible).
            seed: Sampling seed.

        Raises:
            ProviderUnavailableError: If llama-cpp-python is not installed or the
                ``.gguf`` file does not exist.
        """
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ProviderUnavailableError("Install the 'llm' extra: uv sync --extra llm") from exc
        if not Path(model_path).is_file():
            raise ProviderUnavailableError(
                f"GGUF model not found at {model_path}. Run: uv run python "
                "scripts/download_models.py (or set AIACT_LLM_MODEL)"
            )
        self._path = model_path
        self._temperature = temperature
        self._llm: Any = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            seed=seed,
            verbose=False,
        )

    @property
    def name(self) -> str:
        """Provider/model id.

        Returns:
            ``"llama_cpp:<file name>"``.
        """
        return f"llama_cpp:{self._path.rsplit('/', 1)[-1]}"

    def generate(
        self,
        system: str,
        user: str,
        json_schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
    ) -> str:
        """Generate a chat completion.

        Args:
            system: System prompt.
            user: User prompt.
            json_schema: Optional JSON schema to constrain the output.
            max_tokens: Generation budget.

        Returns:
            Generated text.
        """
        kwargs: dict[str, Any] = {}
        if json_schema is not None:
            kwargs["response_format"] = {"type": "json_object", "schema": json_schema}
        out = self._llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=self._temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return str(out["choices"][0]["message"]["content"] or "")
