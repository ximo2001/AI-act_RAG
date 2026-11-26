"""BM25 (Okapi) sparse retriever, implemented from scratch (~60 lines).

Implemented here rather than imported so the scoring is explicit and testable:

    score(q, d) = sum_t idf(t) * tf(t,d) * (k1 + 1) / (tf(t,d) + k1 * (1 - b + b * |d| / avgdl))
    idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

Legal text is full of exact terms ("biometric categorisation", "Article 6(2)"),
which is exactly where lexical matching complements dense embeddings.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ai_act_rag.domain.models import Chunk, RetrievedChunk

_TOKEN = re.compile(r"[a-z0-9]+")

# Small English stopword list: enough to stop "the/of/and" dominating scores.
STOPWORDS = frozenset(
    """a an and are as at be been but by for from has have if in into is it its
    of on or such that the their there these this to was were which will with
    shall may not any other where what when who how does do can""".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics, drop stopwords and naive plurals.

    Args:
        text: Raw text.

    Returns:
        Normalised tokens.
    """
    tokens = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        # Very light stemming: "systems" -> "system", "practices" -> "practice".
        if len(tok) > 4 and tok.endswith("s") and not tok.endswith("ss"):
            tok = tok[:-1]
        tokens.append(tok)
    return tokens


class BM25Retriever:
    """In-memory Okapi BM25 over a list of chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        """Index the chunks.

        Args:
            chunks: Corpus to index.
            k1: Term-frequency saturation parameter.
            b: Length-normalisation parameter.
        """
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._tfs = [Counter(tokenize(c.text)) for c in chunks]
        self._lens = [sum(tf.values()) for tf in self._tfs]
        self._avgdl = sum(self._lens) / max(len(self._lens), 1)
        df: Counter[str] = Counter()
        for tf in self._tfs:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        """Score every chunk against the query.

        Args:
            query: Natural-language query.

        Returns:
            One BM25 score per chunk (same order as the corpus).
        """
        q_terms = [t for t in tokenize(query) if t in self._idf]
        out = []
        for tf, dl in zip(self._tfs, self._lens, strict=True):
            norm = self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            s = 0.0
            for t in q_terms:
                f = tf.get(t, 0)
                if f:
                    s += self._idf[t] * f * (self.k1 + 1) / (f + norm)
            out.append(s)
        return out

    def search(self, query: str, k: int = 10) -> list[RetrievedChunk]:
        """Return the top-k chunks by BM25 score.

        Args:
            query: Natural-language query.
            k: Number of results.

        Returns:
            Ranked chunks (chunks with score 0 are excluded).
        """
        scores = self.scores(query)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = [i for i in order[:k] if scores[i] > 0]
        return [
            RetrievedChunk(chunk=self.chunks[i], score=scores[i], rank=r)
            for r, i in enumerate(hits, start=1)
        ]
