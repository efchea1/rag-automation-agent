"""
api.py
------
REST API for the RAG agent, built with FastAPI.

Endpoints:
    POST /ingest   - ingest a file or directory path into the index
    POST /query    - run the full agent loop (rewrite -> retrieve -> answer
                     -> triage -> action plan) against the current index
    GET  /health   - liveness/readiness probe for orchestration platforms

Run locally:
    uvicorn api:app --reload --port 8000

This is intentionally small — the point is to show the same agent used in
the CLI/tests is directly reusable behind a REST interface, which is how
this would actually plug into ITSM tools, chatops, or a self-service portal.
"""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from rag_agent.agent import RagAgent
from rag_agent.governance.api_security import enforce_rate_limit
from rag_agent.governance.audit_log import AuditLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_agent.api")

app = FastAPI(
    title="RAG Automation Agent API",
    description="Agentic RAG over PDF / Word / HTML / text document collections",
    version="0.2.0",
)

# Single shared agent/index for this demo process. In production this would
# be backed by a persistent vector store (e.g. pgvector, Chroma, or a managed
# service) keyed by tenant/workspace, not an in-memory singleton.
agent = RagAgent(audit_logger=AuditLogger(log_path="api_audit_log.jsonl"))


class IngestRequest(BaseModel):
    path: str


class IngestResponse(BaseModel):
    chunks_indexed: int
    total_chunks: int


class QueryRequest(BaseModel):
    query: str


class CitationOut(BaseModel):
    source_path: str
    source_type: str
    chunk_index: int
    excerpt: str
    score: float


class TriageOut(BaseModel):
    category: str
    urgency: str
    recommended_action: str
    rationale: str


class QueryResponse(BaseModel):
    original_query: str
    rewritten_query: str
    answer: str
    citations: list[CitationOut]
    triage: TriageOut
    action_plan: str
    pii_redactions: list[dict]
    matched_runbook: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": len(agent._chunks)}


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest, api_key: str = Depends(enforce_rate_limit)) -> IngestResponse:
    try:
        new_count = agent.ingest_path(req.path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IngestResponse(chunks_indexed=new_count, total_chunks=len(agent._chunks))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, api_key: str = Depends(enforce_rate_limit)) -> QueryResponse:
    if not agent._chunks:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed yet. POST /ingest first.",
        )

    result = agent.run(req.query)

    return QueryResponse(
        original_query=result.original_query,
        rewritten_query=result.rewritten_query,
        answer=result.answer,
        citations=[CitationOut(**c.__dict__) for c in result.citations],
        triage=TriageOut(**result.triage.__dict__),
        action_plan=result.action_plan,
        pii_redactions=result.pii_redactions,
        matched_runbook=result.matched_runbook.name if result.matched_runbook else None,
    )
