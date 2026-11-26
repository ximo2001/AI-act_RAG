"""Persistence of the chunk corpus and its FAISS index.

Layout of an index directory::

    index_dir/
      chunks.jsonl      one Chunk per line (row i <-> vector i)
      vectors.faiss     FAISS IndexFlatIP over L2-normalised embeddings
      meta.json         embedder name, chunking config, corpus hash

``IndexFlatIP`` is exact search: with ~1k chunks there is no reason to trade
accuracy for speed with an approximate index (IVF/HNSW).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import numpy.typing as npt

from ai_act_rag.domain.exceptions import IndexNotBuiltError
from ai_act_rag.domain.models import Chunk


class FaissIndex:
    """Thin wrapper around ``faiss.IndexFlatIP`` matching the ``VectorIndex`` port."""

    def __init__(self, index: Any) -> None:
        """Wrap an existing FAISS index.

        Args:
            index: A FAISS index.
        """
        self.index = index

    @classmethod
    def build(cls, vectors: npt.NDArray[np.float32]) -> FaissIndex:
        """Build an exact inner-product index.

        Args:
            vectors: L2-normalised array (n, dim).

        Returns:
            The wrapped index.
        """
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        return cls(index)

    def search(self, query: npt.NDArray[np.float32], k: int) -> tuple[list[int], list[float]]:
        """Top-k inner-product search.

        Args:
            query: Vector (dim,).
            k: Number of neighbours.

        Returns:
            (ids, scores).
        """
        scores, ids = self.index.search(np.asarray(query, dtype=np.float32)[None, :], k)
        return [int(i) for i in ids[0]], [float(s) for s in scores[0]]


def save_index(
    index_dir: Path, chunks: list[Chunk], vectors: npt.NDArray[np.float32], meta: dict[str, Any]
) -> None:
    """Write chunks, vectors and metadata to ``index_dir``.

    Args:
        index_dir: Target directory (created if missing).
        chunks: Chunks, aligned with ``vectors`` rows.
        vectors: Embeddings (n, dim).
        meta: Free-form metadata (embedder name, chunking config...).
    """
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")
    faiss.write_index(FaissIndex.build(vectors).index, str(index_dir / "vectors.faiss"))
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_chunks(index_dir: Path) -> list[Chunk]:
    """Load the chunk corpus of an index.

    Args:
        index_dir: Index directory.

    Returns:
        Chunks in index order.

    Raises:
        IndexNotBuiltError: If the directory has no ``chunks.jsonl``.
    """
    path = index_dir / "chunks.jsonl"
    if not path.exists():
        raise IndexNotBuiltError(f"No index at {index_dir}. Run: ai-act-rag build-index")
    return [Chunk.model_validate_json(line) for line in path.read_text("utf-8").splitlines()]


def load_meta(index_dir: Path) -> dict[str, Any]:
    """Load the metadata of an index (embedder, chunk size, corpus hash).

    Args:
        index_dir: Index directory.

    Returns:
        The ``meta.json`` content.

    Raises:
        IndexNotBuiltError: If ``meta.json`` is missing.
    """
    path = index_dir / "meta.json"
    if not path.exists():
        raise IndexNotBuiltError(f"Missing {path}. Run: ai-act-rag build-index")
    meta: dict[str, Any] = json.loads(path.read_text("utf-8"))
    return meta


def load_index(index_dir: Path) -> tuple[list[Chunk], FaissIndex, dict[str, Any]]:
    """Load chunks, FAISS index and metadata.

    Args:
        index_dir: Index directory.

    Returns:
        (chunks, index, meta).

    Raises:
        IndexNotBuiltError: If files are missing.
    """
    chunks = load_chunks(index_dir)
    faiss_path = index_dir / "vectors.faiss"
    if not faiss_path.exists():
        raise IndexNotBuiltError(f"Missing {faiss_path}")
    return chunks, FaissIndex(faiss.read_index(str(faiss_path))), load_meta(index_dir)
