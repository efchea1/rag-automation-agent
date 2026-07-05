"""
End-to-end and unit tests for the RAG agent pipeline.

Run with: pytest -v
Uses MockLLMClient throughout, so this suite requires no API key and no
network access — it validates the mechanics (loading, chunking, retrieval,
orchestration, error handling), not model output quality.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.agent import RagAgent
from rag_agent.chunker import chunk_text
from rag_agent.document_loader import Document, load_directory, load_document
from rag_agent.llm_client import MockLLMClient
from rag_agent.retriever import HybridRetriever

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_docs"


class TestDocumentLoader:
    def test_load_pdf(self):
        doc = load_document(SAMPLE_DIR / "password_reset_policy.pdf")
        assert doc.source_type == "pdf"
        assert "lockout" in doc.text.lower()

    def test_load_docx(self):
        doc = load_document(SAMPLE_DIR / "new_hire_onboarding_runbook.docx")
        assert doc.source_type == "docx"
        assert "onboarding" in doc.text.lower()
        # table content should be captured too
        assert "Azure AD" in doc.text or "azure ad" in doc.text.lower()

    def test_load_html(self):
        doc = load_document(SAMPLE_DIR / "vpn_troubleshooting_kb.html")
        assert doc.source_type == "html"
        assert "vpn" in doc.text.lower()
        # script/nav/footer should be stripped or at least not dominate
        assert "<html>" not in doc.text

    def test_unsupported_extension_raises(self, tmp_path):
        bad_file = tmp_path / "file.exe"
        bad_file.write_bytes(b"\x00\x01")
        with pytest.raises(ValueError):
            load_document(bad_file)

    def test_load_directory_loads_all_types(self):
        docs = list(load_directory(SAMPLE_DIR))
        types = {d.source_type for d in docs}
        assert {"pdf", "docx", "html"}.issubset(types)


class TestChunker:
    def test_chunk_respects_size_roughly(self):
        doc = Document(source_path="x.txt", source_type="text", text="Sentence one. " * 200)
        chunks = chunk_text(doc, chunk_size=500, overlap=50)
        assert len(chunks) > 1
        assert all(len(c.text) <= 500 + 60 for c in chunks)  # small slack for overlap carry

    def test_chunk_empty_document(self):
        doc = Document(source_path="empty.txt", source_type="text", text="")
        chunks = chunk_text(doc)
        assert chunks == []


class TestRetriever:
    def setup_method(self):
        docs = list(load_directory(SAMPLE_DIR))
        self.chunks = []
        for d in docs:
            self.chunks.extend(chunk_text(d))
        self.retriever = HybridRetriever(self.chunks)

    def test_retrieves_relevant_chunk_for_vpn_query(self):
        results = self.retriever.search("VPN won't connect error code 3", top_k=3)
        assert len(results) > 0
        assert any("vpn" in r.chunk.source_path.lower() for r in results)

    def test_retrieves_relevant_chunk_for_password_query(self):
        results = self.retriever.search("how many failed logins before lockout", top_k=3)
        assert len(results) > 0
        top_sources = [r.chunk.source_path.lower() for r in results]
        assert any("password" in s for s in top_sources)

    def test_empty_index_returns_no_results(self):
        empty_retriever = HybridRetriever([])
        assert empty_retriever.search("anything") == []


class TestAgentEndToEnd:
    def setup_method(self):
        self.agent = RagAgent(llm_client=MockLLMClient())
        self.agent.ingest_path(str(SAMPLE_DIR))

    def test_ingest_indexes_all_documents(self):
        assert len(self.agent._chunks) > 0

    def test_run_produces_full_response_shape(self):
        response = self.agent.run("A user's VPN keeps failing with error code 3, what should I do?")
        assert response.original_query
        assert response.rewritten_query
        assert response.answer
        assert isinstance(response.citations, list)
        assert response.triage.category in {
            "informational_query", "incident", "access_request", "how_to_request",
        }
        assert response.action_plan
        assert "context_block" in response.trace

    def test_run_with_no_documents_returns_no_citations(self):
        empty_agent = RagAgent(llm_client=MockLLMClient())
        response = empty_agent.run("anything at all")
        assert response.citations == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
