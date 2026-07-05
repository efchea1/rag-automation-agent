"""
retriever.py
------------
Hybrid retrieval over indexed chunks: combines sparse keyword search (BM25)
with dense vector similarity (TF-IDF cosine by default; swap in a real
embedding model such as sentence-transformers or an API embedding endpoint
by implementing `EmbeddingBackend` and passing it to HybridRetriever).

Hybrid search matters in enterprise IT content because exact identifiers
(error codes, ticket IDs, hostnames, policy numbers) are common, and pure
dense retrieval frequently misses them. BM25 covers exact/lexical matches;
the vector component covers paraphrase and semantic similarity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Protocol

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    bm25_score: float
    vector_score: float


class EmbeddingBackend(Protocol):
    """Swap-in interface: implement this against OpenAI/Anthropic/Voyage/
    sentence-transformers embeddings for production-grade semantic recall."""

    def embed(self, texts: List[str]) -> np.ndarray:  # returns (N, D) array
        ...


def _tokenize(text: str) -> List[str]:
    return text.lower().split()


class HybridRetriever:
    """
    Indexes a list of Chunks and supports hybrid BM25 + vector search.

    Default vector backend is TF-IDF cosine similarity, which requires no
    external API calls and works fully offline/on-prem — a real
    consideration for enterprise IT deployments with data-residency rules.
    """

    def __init__(self, chunks: List[Chunk], embedding_backend: Optional[EmbeddingBackend] = None):
        self.chunks = chunks
        self._embedding_backend = embedding_backend

        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus) if chunks else None

        self._vectorizer = TfidfVectorizer(stop_words="english")
        if chunks:
            self._tfidf_matrix = self._vectorizer.fit_transform([c.text for c in chunks])
        else:
            self._tfidf_matrix = None

        if embedding_backend is not None and chunks:
            self._dense_matrix = embedding_backend.embed([c.text for c in chunks])
        else:
            self._dense_matrix = None

    def _vector_scores(self, query: str) -> np.ndarray:
        if self._embedding_backend is not None and self._dense_matrix is not None:
            query_vec = self._embedding_backend.embed([query])
            sims = cosine_similarity(query_vec, self._dense_matrix)[0]
            return sims

        if self._tfidf_matrix is None:
            return np.zeros(len(self.chunks))

        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._tfidf_matrix)[0]
        return sims

    def search(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ) -> List[RetrievalResult]:
        if not self.chunks:
            return []

        bm25_scores = np.array(self._bm25.get_scores(_tokenize(query)))
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        vector_scores = self._vector_scores(query)
        if vector_scores.max() > 0:
            vector_scores = vector_scores / vector_scores.max()

        combined = bm25_weight * bm25_scores + vector_weight * vector_scores

        ranked_idx = np.argsort(-combined)[:top_k]

        results = [
            RetrievalResult(
                chunk=self.chunks[i],
                score=float(combined[i]),
                bm25_score=float(bm25_scores[i]),
                vector_score=float(vector_scores[i]),
            )
            for i in ranked_idx
            if combined[i] > 0
        ]

        logger.info("Query '%s' -> %d results (top score %.3f)",
                     query, len(results), results[0].score if results else 0.0)
        return results
