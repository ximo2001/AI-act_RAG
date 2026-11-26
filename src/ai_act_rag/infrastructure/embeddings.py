"""Sentence-transformers embedder and cross-encoder reranker adapters.

Imports of ``sentence_transformers`` are lazy so the rest of the package (and
the test-suite) works without torch installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ai_act_rag.domain.exceptions import ProviderUnavailableError
from ai_act_rag.domain.models import RetrievedChunk

# Query/passage prefixes recommended by each model family's authors. Getting
# these wrong silently costs several points of recall. Families are recognised
# by the start of the model's final path component (``models/bge-small-en-v1.5``
# -> ``bge-small-en-v1.5``), so a folder name such as ``models/e5_exp/minilm``
# or an org prefix cannot trigger the wrong family.
_PREFIXES: dict[tuple[str, ...], tuple[str, str]] = {
    ("e5", "multilingual-e5"): ("query: ", "passage: "),
    ("bge",): ("Represent this sentence for searching relevant passages: ", ""),
}


def _prefixes_for(model_name: str) -> tuple[str, str]:
    """Return the (query, passage) prefixes for a model name.

    Args:
        model_name: HF id (``BAAI/bge-small-en-v1.5``) or local path of the model.

    Returns:
        (query_prefix, passage_prefix); empty strings if the family is unknown.
    """
    base = Path(model_name).name.lower()
    for starts, prefixes in _PREFIXES.items():
        if base.startswith(starts):
            return prefixes
    return ("", "")


class SentenceTransformerEmbedder:
    """Embedder backed by a sentence-transformers bi-encoder."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32) -> None:
        """Load the model.

        Args:
            model_name: HF model id or local directory.
            device: Torch device.
            batch_size: Encoding batch size.

        Raises:
            ProviderUnavailableError: If sentence-transformers is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ProviderUnavailableError("Install the 'ml' extra: uv sync --extra ml") from exc
        self._name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._batch_size = batch_size
        self._query_prefix, self._doc_prefix = _prefixes_for(model_name)

    @property
    def name(self) -> str:
        """Model identifier.

        Returns:
            The model name/path given at construction.
        """
        return self._name

    def _encode(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Encode and L2-normalise.

        Args:
            texts: Texts (already prefixed).

        Returns:
            Float32 array (n, dim).
        """
        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> npt.NDArray[np.float32]:
        """Embed passages.

        Args:
            texts: Passages.

        Returns:
            Array (n, dim).
        """
        return self._encode([self._doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> npt.NDArray[np.float32]:
        """Embed a query.

        Args:
            text: Query.

        Returns:
            Array (dim,).
        """
        vec: npt.NDArray[np.float32] = self._encode([self._query_prefix + text])[0]
        return vec


class CrossEncoderReranker:
    """Reranker backed by a sentence-transformers cross-encoder."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        """Load the cross-encoder.

        Args:
            model_name: HF model id or local directory.
            device: Torch device.

        Raises:
            ProviderUnavailableError: If sentence-transformers is not installed.
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ProviderUnavailableError("Install the 'ml' extra: uv sync --extra ml") from exc
        self._model: Any = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, hits: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        """Score (query, chunk) pairs jointly and keep the top-k.

        Args:
            query: Query.
            hits: Candidates.
            k: Number to keep.

        Returns:
            Re-ranked hits with cross-encoder scores.
        """
        if not hits:
            return []
        scores = self._model.predict([(query, h.chunk.text) for h in hits], show_progress_bar=False)
        order = np.argsort(-np.asarray(scores))[:k]
        return [
            RetrievedChunk(chunk=hits[i].chunk, score=float(scores[i]), rank=r)
            for r, i in enumerate(order, start=1)
        ]
