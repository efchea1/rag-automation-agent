"""
vector_store.py
----------------
Real vector database integration (ChromaDB) — replaces the TF-IDF-only
fallback with actual vector storage, persistence, and similarity search
infrastructure, addressing the JD's "hands-on experience with knowledge
retrieval systems, vector databases" preferred skill.

Embedding function note: ChromaDB's default embedding function downloads
a small ONNX model from the internet on first use. To keep this runnable
in network-restricted environments (a real concern for on-prem/regulated
IT deployments), this module ships a deterministic, fully-offline
`TfidfEmbeddingFunction` that fits a TF-IDF vectorizer over the corpus and
projects it to a fixed-width dense vector Chroma can index.

In a production deployment with internet/model access, swap in
`chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction`
or an API-based embedding function (OpenAI, Voyage, Azure OpenAI
embeddings) — nothing else about the storage/query layer changes, because
both implement the same `EmbeddingFunction` protocol.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from .chunker import Chunk

logger = logging.getLogger(__name__)

from chromadb import EmbeddingFunction, Documents, Embeddings


class TfidfEmbeddingFunction(EmbeddingFunction):
    """
    Chroma-compatible embedding function requiring no network access.

    Fits TF-IDF + truncated SVD (a lightweight LSA) over the ingested
    corpus to produce fixed-width dense vectors. Not as strong as a real
    sentence-transformer model, but it is deterministic, fast, and fully
    on-prem — often the right trade-off for regulated IT/security content
    where sending document text to an external embeddings API is not
    acceptable.
    """

    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._svd: Optional[TruncatedSVD] = None
        self._fitted = False

    def fit(self, corpus: List[str]) -> None:
        tfidf = self._vectorizer.fit_transform(corpus)
        n_components = min(self.n_components, max(2, tfidf.shape[1] - 1), tfidf.shape[0])
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(tfidf)
        self._fitted = True

    def __call__(self, input: Documents) -> Embeddings:
        if not self._fitted:
            # cold start: fit on whatever we're given, then embed it
            self.fit(list(input))
        tfidf = self._vectorizer.transform(list(input))
        dense = self._svd.transform(tfidf)
        return dense.tolist()

    def embed_query(self, input: Documents) -> Embeddings:
        # queries reuse the already-fitted vectorizer/SVD; never refit on a query
        tfidf = self._vectorizer.transform(list(input))
        dense = self._svd.transform(tfidf)
        return dense.tolist()

    def name(self) -> str:
        return "tfidf_svd_offline"

    def get_config(self) -> dict:
        return {"n_components": self.n_components}

    @staticmethod
    def build_from_config(config: dict) -> "TfidfEmbeddingFunction":
        return TfidfEmbeddingFunction(n_components=config.get("n_components", 128))


class ChromaVectorStore:
    """
    Thin wrapper around a Chroma collection for indexing and querying
    document chunks. This is the real persistent vector database layer;
    `HybridRetriever` in retriever.py remains available as the
    zero-dependency default for quick local testing.
    """

    def __init__(
        self,
        collection_name: str = "rag_agent_chunks",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[TfidfEmbeddingFunction] = None,
    ):
        self.embedding_function = embedding_function or TfidfEmbeddingFunction()

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
            logger.info("ChromaVectorStore persisting to %s", persist_directory)
        else:
            self._client = chromadb.Client()
            logger.info("ChromaVectorStore running in-memory (no persistence)")

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
        )

    def index(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        # Fit the offline embedding function over the full corpus first so
        # SVD components capture corpus-wide structure rather than just the
        # first batch indexed.
        if isinstance(self.embedding_function, TfidfEmbeddingFunction):
            self.embedding_function.fit([c.text for c in chunks])

        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {"source_path": c.source_path, "source_type": c.source_type, "chunk_index": c.chunk_index}
                for c in chunks
            ],
        )
        logger.info("Indexed %d chunks into Chroma collection '%s'", len(chunks), self._collection.name)

    def query(self, query_text: str, top_k: int = 5) -> List[dict]:
        results = self._collection.query(query_texts=[query_text], n_results=top_k)
        out = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            similarity = 1.0 / (1.0 + dist)  # convert distance to a bounded similarity-like score
            out.append({"chunk_id": _id, "text": doc, "metadata": meta, "score": similarity})
        return out

    def count(self) -> int:
        return self._collection.count()
