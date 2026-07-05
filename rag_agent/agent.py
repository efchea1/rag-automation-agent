"""
agent.py
--------
The agent loop that ties document loading, chunking, retrieval, and an LLM
together into the workflow described in the job posting:

    1. Rewrite the user's query to improve retrieval recall.
    2. Retrieve the most relevant chunks (hybrid BM25 + vector search).
    3. Read the retrieved chunks and produce a grounded, cited answer.
    4. Classify/triage the request (e.g. informational vs. incident vs.
       access-request) so it can be routed appropriately.
    5. Propose a concrete action plan / next steps (guided remediation,
       escalation, or self-service resolution).

Every step is logged and returned in the AgentResponse so the pipeline is
observable and auditable — not a black box producing an answer.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .chunker import Chunk, chunk_text
from .document_loader import load_directory, load_document
from .llm_client import LLMClient, get_default_client
from .retriever import HybridRetriever, RetrievalResult
from .governance.pii_redaction import redact
from .governance.audit_log import AuditLogger, build_audit_record

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    source_path: str
    source_type: str
    chunk_index: int
    excerpt: str
    score: float


@dataclass
class Triage:
    category: str
    urgency: str
    recommended_action: str
    rationale: str


@dataclass
class MatchedRunbook:
    name: str
    preview_steps: List[str]
    requires_approval: bool


@dataclass
class AgentResponse:
    original_query: str
    rewritten_query: str
    answer: str
    citations: List[Citation]
    triage: Triage
    action_plan: str
    pii_redactions: List[dict] = field(default_factory=list)
    matched_runbook: Optional[MatchedRunbook] = None
    trace: dict = field(default_factory=dict)  # full step-by-step trace for observability


_QUERY_REWRITE_SYSTEM = (
    "You rewrite the user's query to maximize retrieval recall against a "
    "hybrid keyword + vector search index of IT/enterprise documents. "
    "Expand abbreviations, add likely synonyms, keep it concise. "
    "Return ONLY the rewritten query, nothing else."
)

_ANSWER_SYSTEM = (
    "You are an IT operations assistant. Answer the user's question using ONLY "
    "the retrieved context provided. Cite sources inline using [source_path] "
    "notation. If the context does not contain the answer, say so explicitly "
    "rather than guessing."
)

_TRIAGE_SYSTEM = (
    "You perform triage classification for an IT operations request. "
    "Given the user's query and the retrieved context, classify it and "
    "return STRICT JSON with keys: category "
    "(one of: informational_query, incident, access_request, "
    "how_to_request), urgency (low, medium, high), recommended_action "
    "(short string), rationale (one sentence). Return ONLY JSON."
)

_ACTION_SYSTEM = (
    "You generate an action_plan: concrete next steps for resolving or "
    "responding to the request, given the triage classification and the "
    "grounded answer already produced. Be specific and operational. "
    "3-5 numbered steps, no preamble."
)


class RagAgent:
    """High-level agent: ingest documents, then answer queries end-to-end."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        chunk_size: int = 800,
        overlap: int = 150,
        top_k: int = 5,
        redact_pii: bool = True,
        audit_logger: Optional[AuditLogger] = None,
        runbook_executor=None,  # RunbookExecutor, kept untyped here to avoid a hard import cycle
    ):
        self.llm = llm_client or get_default_client()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.redact_pii = redact_pii
        self.audit_logger = audit_logger
        self.runbook_executor = runbook_executor
        self._chunks: List[Chunk] = []
        self._retriever: Optional[HybridRetriever] = None

    # ---------- Ingestion ----------

    def ingest_path(self, path: str) -> int:
        """Ingest a single file or every supported file in a directory. Returns chunk count."""
        p = Path(path)
        docs = [load_document(p)] if p.is_file() else list(load_directory(p))

        new_chunks: List[Chunk] = []
        for doc in docs:
            new_chunks.extend(chunk_text(doc, chunk_size=self.chunk_size, overlap=self.overlap))

        self._chunks.extend(new_chunks)
        self._retriever = HybridRetriever(self._chunks)

        logger.info("Ingested %d document(s) -> %d chunks (total index: %d chunks)",
                     len(docs), len(new_chunks), len(self._chunks))
        return len(new_chunks)

    # ---------- Agent steps ----------

    def _rewrite_query(self, query: str) -> str:
        rewritten = self.llm.complete(
            _QUERY_REWRITE_SYSTEM, query, max_tokens=100, step="rewrite"
        ).strip()
        return rewritten or query

    def _retrieve(self, query: str) -> List[RetrievalResult]:
        if self._retriever is None:
            return []
        return self._retriever.search(query, top_k=self.top_k)

    def _build_context_block(self, results: List[RetrievalResult]) -> str:
        blocks = []
        for r in results:
            blocks.append(
                f"[{r.chunk.source_path} | chunk {r.chunk.chunk_index} | "
                f"score={r.score:.2f}]\n{r.chunk.text}"
            )
        return "\n\n---\n\n".join(blocks)

    def _answer(self, query: str, context_block: str) -> str:
        prompt = f"Question: {query}\n\nRetrieved context:\n{context_block}"
        return self.llm.complete(_ANSWER_SYSTEM, prompt, max_tokens=600, step="answer").strip()

    def _triage(self, query: str, context_block: str) -> Triage:
        prompt = f"User query: {query}\n\nRetrieved context:\n{context_block}"
        raw = self.llm.complete(_TRIAGE_SYSTEM, prompt, max_tokens=300, step="triage").strip()
        cleaned = self._extract_json(raw)
        try:
            data = json.loads(cleaned)
            return Triage(
                category=data.get("category", "informational_query"),
                urgency=data.get("urgency", "low"),
                recommended_action=data.get("recommended_action", "answer_with_citations"),
                rationale=data.get("rationale", ""),
            )
        except (json.JSONDecodeError, TypeError):
            logger.warning("Triage output was not valid JSON even after cleanup, using fallback: %r", raw)
            return Triage(
                category="informational_query",
                urgency="low",
                recommended_action="answer_with_citations",
                rationale="Fallback classification (LLM output was not parseable JSON).",
            )

    @staticmethod
    def _extract_json(raw: str) -> str:
        """
        Real LLMs frequently wrap JSON in markdown code fences (```json ... ```)
        even when explicitly told to return only JSON. Strip fences first; if
        that still isn't valid JSON, fall back to extracting the substring
        between the first '{' and the last '}', which handles cases where the
        model adds a short preamble/postamble around the JSON object.
        """
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        try:
            json.loads(cleaned)
            return cleaned
        except json.JSONDecodeError:
            pass

        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return cleaned[start : end + 1]
        return cleaned

    def _action_plan(self, query: str, answer: str, triage: Triage) -> str:
        prompt = (
            f"User query: {query}\n\nGrounded answer:\n{answer}\n\n"
            f"Triage: category={triage.category}, urgency={triage.urgency}, "
            f"recommended_action={triage.recommended_action}"
        )
        return self.llm.complete(_ACTION_SYSTEM, prompt, max_tokens=300, step="action").strip()

    # ---------- Public entrypoint ----------

    def run(self, query: str) -> AgentResponse:
        trace = {"original_query": query}

        pii_findings: List[dict] = []
        effective_query = query
        if self.redact_pii:
            redaction = redact(query)
            if redaction.had_redactions:
                pii_findings.extend(redaction.findings)
                effective_query = redaction.redacted_text
                logger.info("Redacted PII from query before LLM calls: %s", redaction.findings)

        rewritten_query = self._rewrite_query(effective_query)
        trace["rewritten_query"] = rewritten_query

        results = self._retrieve(rewritten_query)
        trace["num_results"] = len(results)

        context_block = self._build_context_block(results)
        trace["context_block"] = context_block

        answer = self._answer(effective_query, context_block)
        trace["answer"] = answer

        triage = self._triage(effective_query, context_block)
        trace["triage"] = triage.__dict__

        action_plan = self._action_plan(effective_query, answer, triage)
        trace["action_plan"] = action_plan

        citations = [
            Citation(
                source_path=r.chunk.source_path,
                source_type=r.chunk.source_type,
                chunk_index=r.chunk.chunk_index,
                excerpt=r.chunk.text[:200],
                score=r.score,
            )
            for r in results
        ]

        matched_runbook = None
        if self.runbook_executor is not None:
            runbook = self.runbook_executor.find_matching_runbook(query)
            if runbook is not None:
                matched_runbook = MatchedRunbook(
                    name=runbook.name,
                    preview_steps=self.runbook_executor.preview(runbook),
                    requires_approval=any(step.requires_approval for step in runbook.steps),
                )
                trace["matched_runbook"] = runbook.name

        response = AgentResponse(
            original_query=query,
            rewritten_query=rewritten_query,
            answer=answer,
            citations=citations,
            triage=triage,
            action_plan=action_plan,
            pii_redactions=pii_findings,
            matched_runbook=matched_runbook,
            trace=trace,
        )

        if self.audit_logger is not None:
            record = build_audit_record(response, pii_findings=pii_findings)
            record.query = effective_query  # never persist raw PII to the audit log
            self.audit_logger.log(record)

        return response
