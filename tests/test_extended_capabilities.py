"""
Tests for the expanded system: vector database, cloud/enterprise
integrations, event-driven architecture, governance, and guided
remediation runbooks. All use mock/offline backends — no external
credentials or network access required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.document_loader import load_directory
from rag_agent.chunker import chunk_text
from rag_agent.vector_store import ChromaVectorStore, TfidfEmbeddingFunction
from rag_agent.governance.pii_redaction import redact
from rag_agent.governance.audit_log import AuditLogger, build_audit_record
from rag_agent.events.bus import EventBus, Event, agent_event_handler
from rag_agent.identity.lifecycle import MockGraphIdentityClient
from rag_agent.itsm.servicenow_client import MockServiceNowClient
from rag_agent.power_automate import MockPowerAutomateClient, PowerAutomateWebhookPayload
from rag_agent.runbooks.executor import build_default_runbooks
from rag_agent.agent import RagAgent
from rag_agent.llm_client import MockLLMClient

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_docs"


class TestVectorStore:
    def test_index_and_query(self):
        docs = list(load_directory(SAMPLE_DIR))
        chunks = []
        for d in docs:
            chunks.extend(chunk_text(d))

        store = ChromaVectorStore(collection_name="pytest_collection")
        store.index(chunks)
        assert store.count() == len(chunks)

        results = store.query("VPN certificate expired error code 3", top_k=3)
        assert len(results) > 0
        assert any("vpn" in r["metadata"]["source_path"].lower() for r in results)

    def test_embedding_function_dims_consistent(self):
        ef = TfidfEmbeddingFunction(n_components=8)
        ef.fit(["password reset policy", "vpn troubleshooting guide", "onboarding runbook"])
        vecs = ef(["a new query about passwords"])
        assert len(vecs) == 1
        assert len(vecs[0]) <= 8


class TestPiiRedaction:
    def test_redacts_email_and_phone(self):
        result = redact("Contact jane.doe@example.com or 555-123-4567 please")
        assert "jane.doe@example.com" not in result.redacted_text
        assert "555-123-4567" not in result.redacted_text
        assert result.had_redactions
        categories = {f["category"] for f in result.findings}
        assert "EMAIL" in categories
        assert "PHONE" in categories

    def test_no_redaction_needed(self):
        result = redact("The VPN gateway is down for everyone")
        assert not result.had_redactions
        assert result.redacted_text == "The VPN gateway is down for everyone"

    def test_findings_never_contain_raw_value(self):
        result = redact("SSN is 123-45-6789")
        for finding in result.findings:
            assert "123-45-6789" not in str(finding)


class TestAuditLog:
    def test_log_and_read_roundtrip(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path=str(log_path))

        agent = RagAgent(llm_client=MockLLMClient())
        agent.ingest_path(str(SAMPLE_DIR))
        response = agent.run("VPN keeps failing")

        record = build_audit_record(response)
        logger.log(record)

        records = logger.read_all()
        assert len(records) == 1
        assert records[0]["category"] == response.triage.category

    def test_agent_never_logs_raw_pii(self, tmp_path):
        log_path = tmp_path / "audit.jsonl"
        logger = AuditLogger(log_path=str(log_path))

        agent = RagAgent(llm_client=MockLLMClient(), audit_logger=logger)
        agent.ingest_path(str(SAMPLE_DIR))
        agent.run("Email me at test.user@example.com about the VPN issue")

        records = logger.read_all()
        assert "test.user@example.com" not in records[0]["query"]
        assert "[REDACTED_EMAIL]" in records[0]["query"]


class TestEventBus:
    def test_publish_calls_subscribed_handler(self):
        bus = EventBus()
        received = []
        bus.subscribe("ticket_created", lambda e: received.append(e))

        bus.publish(Event(event_type="ticket_created", payload={"id": 1}, source="test"))
        assert len(received) == 1
        assert received[0].payload == {"id": 1}

    def test_wildcard_handler_receives_all_events(self):
        bus = EventBus()
        received = []
        bus.subscribe("*", lambda e: received.append(e.event_type))

        bus.publish(Event(event_type="a", payload={}, source="test"))
        bus.publish(Event(event_type="b", payload={}, source="test"))
        assert received == ["a", "b"]

    def test_agent_event_handler_auto_triages_and_escalates(self):
        agent = RagAgent(llm_client=MockLLMClient())
        agent.ingest_path(str(SAMPLE_DIR))
        itsm = MockServiceNowClient()
        pa = MockPowerAutomateClient()

        handler = agent_event_handler(agent, itsm_client=itsm, power_automate_client=pa)
        handler(Event(
            event_type="log_alert",
            payload={"text": "VPN gateway down, everyone locked out, urgent outage"},
            source="log_watcher",
        ))

        assert len(itsm.created_tickets) == 1
        assert len(pa.sent_payloads) == 1

    def test_agent_event_handler_low_urgency_does_not_escalate(self):
        agent = RagAgent(llm_client=MockLLMClient())
        agent.ingest_path(str(SAMPLE_DIR))
        itsm = MockServiceNowClient()
        pa = MockPowerAutomateClient()

        handler = agent_event_handler(agent, itsm_client=itsm, power_automate_client=pa)
        handler(Event(
            event_type="log_alert",
            payload={"text": "How do I reset my password?"},
            source="log_watcher",
        ))

        assert len(itsm.created_tickets) == 0
        assert len(pa.sent_payloads) == 0


class TestITSMIntegration:
    def test_create_incident_returns_ticket_number(self):
        client = MockServiceNowClient()
        result = client.create_incident("VPN down", "Details here", urgency="1")
        assert result.success
        assert result.ticket_number.startswith("INC")

    def test_multiple_tickets_get_unique_numbers(self):
        client = MockServiceNowClient()
        t1 = client.create_incident("Issue 1", "desc")
        t2 = client.create_incident("Issue 2", "desc")
        assert t1.ticket_number != t2.ticket_number


class TestIdentityLifecycle:
    def test_onboarding_sequence(self):
        client = MockGraphIdentityClient()
        result = client.create_user("New Hire", "new.hire@example.com", "TempPass1!")
        assert result.success
        assert "create_user:new.hire@example.com" in client.calls

    def test_offboarding_sequence(self):
        client = MockGraphIdentityClient()
        disable_result = client.disable_account("leaver@example.com")
        revoke_result = client.revoke_sessions("leaver@example.com")
        assert disable_result.success
        assert revoke_result.success
        assert disable_result.dry_run is True  # mock always dry-runs — never mutates a real tenant


class TestPowerAutomate:
    def test_mock_client_records_payload_without_network_call(self):
        client = MockPowerAutomateClient()
        payload = PowerAutomateWebhookPayload(
            event_type="incident_triaged", query="VPN down", category="incident",
            urgency="high", summary="VPN gateway outage",
        )
        result = client.trigger_flow("notify_oncall", payload)
        assert result["triggered"]
        assert len(client.sent_payloads) == 1
        assert client.sent_payloads[0][0] == "notify_oncall"

    def test_real_client_raises_on_unconfigured_flow(self):
        from rag_agent.power_automate import PowerAutomateClient

        client = PowerAutomateClient(flow_webhook_urls={})
        payload = PowerAutomateWebhookPayload(
            event_type="x", query="x", category="x", urgency="low", summary="x"
        )
        try:
            client.trigger_flow("nonexistent_flow", payload)
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "nonexistent_flow" in str(exc)


class TestRunbookExecutor:
    def setup_method(self):
        self.graph = MockGraphIdentityClient()
        self.itsm = MockServiceNowClient()
        self.executor = build_default_runbooks(self.graph, self.itsm)

    def test_matches_lockout_query(self):
        runbook = self.executor.find_matching_runbook("user keeps getting locked out")
        assert runbook is not None
        assert runbook.name == "clear_account_lockout"

    def test_matches_vpn_query(self):
        runbook = self.executor.find_matching_runbook("vpn certificate error code 3")
        assert runbook is not None
        assert runbook.name == "regenerate_vpn_certificate"

    def test_no_match_for_unrelated_query(self):
        runbook = self.executor.find_matching_runbook("what is the capital of France")
        assert runbook is None

    def test_execution_blocked_without_approval(self):
        runbook = self.executor.find_matching_runbook("locked out repeatedly")
        result = self.executor.execute(runbook, user_principal_name="a@example.com")
        assert result.executed is False
        assert result.approved is False

    def test_execution_proceeds_with_approval(self):
        runbook = self.executor.find_matching_runbook("locked out repeatedly")
        result = self.executor.execute(runbook, approved_by="manager@example.com", user_principal_name="a@example.com")
        assert result.executed is True
        assert result.approved is True
        assert len(self.itsm.created_tickets) == 1


class TestAgentWithRunbookMatching:
    def test_agent_response_includes_matched_runbook(self):
        graph = MockGraphIdentityClient()
        itsm = MockServiceNowClient()
        executor = build_default_runbooks(graph, itsm)

        agent = RagAgent(llm_client=MockLLMClient(), runbook_executor=executor)
        agent.ingest_path(str(SAMPLE_DIR))

        response = agent.run("I keep getting locked out of my account")
        assert response.matched_runbook is not None
        assert response.matched_runbook.name == "clear_account_lockout"
        assert response.matched_runbook.requires_approval is True

    def test_agent_response_no_runbook_for_generic_query(self):
        graph = MockGraphIdentityClient()
        itsm = MockServiceNowClient()
        executor = build_default_runbooks(graph, itsm)

        agent = RagAgent(llm_client=MockLLMClient(), runbook_executor=executor)
        agent.ingest_path(str(SAMPLE_DIR))

        response = agent.run("What is the onboarding process for new hires?")
        assert response.matched_runbook is None


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
