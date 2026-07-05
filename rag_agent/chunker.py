"""
chunker.py
----------
Splits normalized document text into overlapping, retrieval-sized chunks.

Uses a paragraph-aware sliding window: it tries to break on paragraph/sentence
boundaries first, and only hard-splits very long paragraphs. Overlap preserves
context across chunk boundaries, which matters a lot for incident logs and
runbooks where the "why" and the "how" often sit in adjacent sentences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .document_loader import Document


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_path: str
    source_type: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_into_sentences(paragraph: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]


def chunk_text(
    doc: Document,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[Chunk]:
    """
    Chunk a Document into overlapping windows of ~chunk_size characters,
    preferring to break on sentence boundaries.
    """
    paragraphs = [p.strip() for p in doc.text.split("\n") if p.strip()]

    sentences: List[str] = []
    for para in paragraphs:
        sentences.extend(_split_into_sentences(para))

    chunks: List[Chunk] = []
    current = ""
    idx = 0

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = f"{current} {sentence}".strip()
            continue

        if current:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.source_path}::{idx}",
                    text=current,
                    source_path=doc.source_path,
                    source_type=doc.source_type,
                    chunk_index=idx,
                    metadata=doc.metadata,
                )
            )
            idx += 1
            # carry the overlap tail forward into the next chunk
            current = current[-overlap:] + " " + sentence
        else:
            # a single sentence longer than chunk_size: hard split
            for i in range(0, len(sentence), chunk_size):
                piece = sentence[i : i + chunk_size]
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.source_path}::{idx}",
                        text=piece,
                        source_path=doc.source_path,
                        source_type=doc.source_type,
                        chunk_index=idx,
                        metadata=doc.metadata,
                    )
                )
                idx += 1
            current = ""

    if current.strip():
        chunks.append(
            Chunk(
                chunk_id=f"{doc.source_path}::{idx}",
                text=current.strip(),
                source_path=doc.source_path,
                source_type=doc.source_type,
                chunk_index=idx,
                metadata=doc.metadata,
            )
        )

    return chunks
