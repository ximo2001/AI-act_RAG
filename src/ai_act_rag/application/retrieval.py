"""Dense, hybrid (Reciprocal Rank Fusion) and reranked retrieval.

The dense retriever depends only on the ``Embedder`` port and a
``VectorIndex`` (implemented with FAISS in infrastructure), so it can be
tested with a fake embedder and a NumPy index.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt

from ai_act_rag.domain.models import Chunk, RetrievedChunk
from ai_act_rag.domain.ports import Embedder, Reranker, Retriever


class VectorIndex(Protocol):
    """Nearest-neighbour index over L2-normalised vectors (inner product)."""

    def search(self, query: npt.NDArray[np.float32], k: int) -> tuple[list[int], list[float]]:
        """Return the ids and similarities of the k nearest vectors.

        Args:
            query: Query vector of shape (dim,).
            k: Number of neighbours.

        Returns:
            (ids, scores) sorted by decreasing similarity.
        """
        ...


class NumpyIndex:
    """Exact inner-product search with NumPy (used in tests)."""

    def __init__(self, vectors: npt.NDArray[np.float32]) -> None:
        """Store the vectors.

        Args:
            vectors: Array (n, dim), L2-normalised.
        """
        self.vectors = vectors

    def search(self, query: npt.NDArray[np.float32], k: int) -> tuple[list[int], list[float]]:
        """Exact top-k by inner product.

        Args:
            query: Vector (dim,).
            k: Number of neighbours.

        Returns:
            (ids, scores) sorted by decreasing similarity.
        """
        sims = self.vectors @ query
        top = np.argsort(-sims)[:k]
        return [int(i) for i in top], [float(sims[i]) for i in top]


class DenseRetriever:
    """Embeds the query and looks up nearest chunks in a vector index."""

    def __init__(self, chunks: list[Chunk], embedder: Embedder, index: VectorIndex) -> None:
        """Create the retriever.

        Args:
            chunks: Corpus, aligned with the index rows.
            embedder: Query embedder (must be the model used to build the index).
            index: Vector index over the chunk embeddings.
        """
        self.chunks = chunks
        self.embedder = embedder
        self.index = index

    def search(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        """Return the top-k chunks by cosine similarity.

        Args:
            query: Natural-language query.
            k: Number of results.

        Returns:
            Ranked chunks.
        """
        ids, scores = self.index.search(self.embedder.embed_query(query), k)
        return [
            RetrievedChunk(chunk=self.chunks[i], score=s, rank=r)
            for r, (i, s) in enumerate(zip(ids, scores, strict=True), start=1)
            if i >= 0
        ]


def reciprocal_rank_fusion(
    rankings: list[list[RetrievedChunk]], k: int = 10, rrf_k: int = 60
) -> list[RetrievedChunk]:
    """Fuse several rankings with Reciprocal Rank Fusion.

    ``score(d) = sum_r 1 / (rrf_k + rank_r(d))``. RRF only uses ranks, so it
    needs no calibration between BM25 scores (unbounded) and cosine
    similarities ([-1, 1]); ``rrf_k=60`` is the value from Cormack et al. (2009).

    Args:
        rankings: One ranked list per retriever.
        k: Number of fused results to return.
        rrf_k: Smoothing constant that dampens the weight of top ranks.

    Returns:
        Fused ranking with RRF scores.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in rankings:
        for hit in ranking:
            cid = hit.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + hit.rank)
            chunks[cid] = hit.chunk
    # Ties broken by chunk id for determinism.
    order = sorted(scores, key=lambda c: (-scores[c], c))[:k]
    return [
        RetrievedChunk(chunk=chunks[c], score=scores[c], rank=r) for r, c in enumerate(order, 1)
    ]


class HybridRetriever:
    """Dense + BM25 fused with RRF, optionally followed by a cross-encoder reranker."""

    def __init__(
        self,
        retrievers: list[Retriever],
        candidates: int = 30,
        rrf_k: int = 60,
        reranker: Reranker | None = None,
    ) -> None:
        """Create the hybrid retriever.

        Args:
            retrievers: Retrievers to fuse (typically dense and BM25).
            candidates: How many results to take from each retriever before fusing.
            rrf_k: RRF smoothing constant.
            reranker: Optional reranker applied to the fused candidates.
        """
        self.retrievers = retrievers
        self.candidates = candidates
        self.rrf_k = rrf_k
        self.reranker = reranker

    def search(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        """Retrieve with every retriever, fuse with RRF and optionally rerank.

        Args:
            query: Natural-language query.
            k: Number of results.

        Returns:
            Ranked chunks.
        """
        rankings = [r.search(query, self.candidates) for r in self.retrievers]
        n_fused = self.candidates if self.reranker else k
        fused = reciprocal_rank_fusion(rankings, k=n_fused, rrf_k=self.rrf_k)
        if self.reranker:
            return self.reranker.rerank(query, fused, k)
        return fused


def unique_units(hits: list[RetrievedChunk]) -> list[str]:
    """Collapse a chunk ranking into a ranking of unit ids (first occurrence wins).

    Retrieval metrics are computed at unit level (articles/annexes/recitals),
    because that is what the answer cites and what the gold labels contain.

    Args:
        hits: Ranked chunks.

    Returns:
        Unit ids in rank order, without duplicates.
    """
    seen: dict[str, None] = {}
    for h in hits:
        seen.setdefault(h.chunk.unit_id, None)
    return list(seen)
