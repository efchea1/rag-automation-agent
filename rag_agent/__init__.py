"""
rag_agent
=========

A production-oriented Retrieval-Augmented Generation (RAG) agent that reads,
indexes, searches, and reasons over heterogeneous document collections
(PDF, DOCX/Word, HTML, and plain text).

Built to demonstrate the exact capability set described in Covetrus'
Sr. IT AI/Automation Engineer role:
  - RAG pipelines over enterprise document sources (docs, KB, tickets, logs)
  - Agentic workflows (triage, root-cause assistance, guided remediation)
  - Python + REST APIs
  - Secure, observable, maintainable design
"""

from .agent import RagAgent
from .document_loader import load_document, load_directory
from .chunker import chunk_text
from .retriever import HybridRetriever

__all__ = [
    "RagAgent",
    "load_document",
    "load_directory",
    "chunk_text",
    "HybridRetriever",
]
