"""
app.py
-------
Standalone, portable demo UI for the RAG automation agent — no claude.ai
dependency. Wraps the *real* RagAgent, HybridRetriever, runbook executor,
governance modules, and mock ITSM/Graph/Power Automate clients directly,
so this is a faithful representation of the actual codebase, not a
simplified JS reimplementation of it.

Run locally:
    pip install streamlit
    streamlit run app.py

Deploy for a shareable public link (free tiers exist for both):
    - Streamlit Community Cloud: push this repo to GitHub, connect it at
      share.streamlit.io, point it at app.py. No server management.
    - Hugging Face Spaces: create a Space with the Streamlit SDK, push
      this repo's contents to it.

By default this runs entirely on MockLLMClient (no API key required) so
it's demoable out of the box. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or
the AZURE_OPENAI_* variables in the environment (or a local `.env` file
loaded before `streamlit run`) to get real LLM-generated answers instead
of the mock placeholder text.
"""

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads a local .env file, if present, into os.environ
except ImportError:
    pass  # python-dotenv is optional; env vars set another way still work fine

from rag_agent.agent import RagAgent
from rag_agent.llm_client import get_default_client, MockLLMClient
from rag_agent.governance.audit_log import AuditLogger, build_audit_record
from rag_agent.runbooks.executor import build_default_runbooks
from rag_agent.identity.lifecycle import MockGraphIdentityClient
from rag_agent.itsm.servicenow_client import MockServiceNowClient
from rag_agent.power_automate import MockPowerAutomateClient, PowerAutomateWebhookPayload

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_docs"

st.set_page_config(page_title="RAG Automation Agent — live demo", layout="centered")


@st.cache_resource
def build_agent():
    llm = get_default_client()
    graph = MockGraphIdentityClient()
    itsm = MockServiceNowClient()
    power_automate = MockPowerAutomateClient()
    executor = build_default_runbooks(graph, itsm)
    audit_logger = AuditLogger(log_path="streamlit_demo_audit.jsonl")

    agent = RagAgent(
        llm_client=llm,
        runbook_executor=executor,
        audit_logger=audit_logger,
    )
    agent.ingest_path(str(SAMPLE_DIR))
    return agent, graph, itsm, power_automate, audit_logger


agent, graph_client, itsm_client, power_automate_client, audit_logger = build_agent()

if "history" not in st.session_state:
    st.session_state.history = []
if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None
if "query_text" not in st.session_state:
    st.session_state.query_text = ""

st.title("RAG automation agent")
st.caption(
    "Real RagAgent from the delivered codebase — hybrid retrieval, PII redaction, "
    "triage, auto-escalation, guided remediation, and audit logging. "
    f"LLM backend: `{type(agent.llm).__name__}`"
    + (" (no API key set — using deterministic mock output)" if isinstance(agent.llm, MockLLMClient) else "")
)

with st.expander("Indexed documents"):
    for path in sorted(SAMPLE_DIR.glob("*")):
        st.write(f"- `{path.name}`")

examples = {
    "VPN issue with PII": "Contact me at jane.doe@example.com or 555-123-4567, my VPN keeps failing with error code 3",
    "Urgent VPN outage": "The VPN gateway is completely down and everyone is locked out, urgent",
    "Account lockout": "User keeps getting locked out after repeated failed logins",
}

cols = st.columns(len(examples))
for col, (label, text) in zip(cols, examples.items()):
    if col.button(label, use_container_width=True):
        st.session_state.query_text = text

query = st.text_input(
    "Ask a question about the indexed documents",
    key="query_text",
    placeholder="e.g. VPN keeps failing with error code 3",
)
run_clicked = st.button("Run agent", type="primary")

if run_clicked and query.strip():
    with st.spinner("Running the agent pipeline..."):
        response = agent.run(query.strip())
    st.session_state.history.append(response)
    st.session_state.pending_approval = None

if st.session_state.history:
    response = st.session_state.history[-1]

    st.subheader("1. PII redaction")
    if response.pii_redactions:
        st.warning(
            "Redacted before any LLM call: "
            + ", ".join(f"{f['category']} x{f['count']}" for f in response.pii_redactions)
        )
    else:
        st.write("No PII patterns detected.")

    st.subheader("2. Query rewrite")
    st.code(response.rewritten_query, language=None)

    st.subheader("3. Hybrid retrieval")
    for c in response.citations:
        st.markdown(f"**{c.source_path}** &nbsp; score {c.score:.2f}")
        st.caption(c.excerpt)

    st.subheader("4. Grounded answer")
    st.write(response.answer)

    st.subheader("5. Triage classification")
    t = response.triage
    badge_col1, badge_col2, badge_col3 = st.columns(3)
    badge_col1.metric("Category", t.category)
    badge_col2.metric("Urgency", t.urgency)
    badge_col3.metric("Action", t.recommended_action)
    st.caption(t.rationale)

    st.subheader("6. Auto-escalation")
    if t.urgency == "high":
        ticket = itsm_client.create_incident(
            short_description=f"[Auto-triaged] {response.original_query[:100]}",
            description=f"{response.answer}\n\nAction plan:\n{response.action_plan}",
            urgency="1",
        )
        payload = PowerAutomateWebhookPayload(
            event_type="incident_triaged", query=response.original_query,
            category=t.category, urgency=t.urgency, summary=response.answer[:280],
            source_documents=[c.source_path for c in response.citations],
        )
        power_automate_client.trigger_flow("notify_oncall", payload)
        st.success(f"ServiceNow incident `{ticket.ticket_number}` auto-created, Power Automate `notify_oncall` triggered.")
    else:
        st.write("Urgency is not high — no ticket opened, no flow triggered.")

    st.subheader("7. Guided remediation runbook")
    if response.matched_runbook:
        rb = response.matched_runbook
        st.write(f"Matched runbook: `{rb.name}`")
        for step in rb.preview_steps:
            st.write(f"- {step}")
        if rb.requires_approval:
            if st.button("Approve and execute (dry run)", key=f"approve-{len(st.session_state.history)}"):
                runbook = agent.runbook_executor.find_matching_runbook(response.original_query)
                result = agent.runbook_executor.execute(
                    runbook, approved_by="demo-user@example.com",
                    user_principal_name="demo-user@example.com",
                )
                st.success(f"Approved and executed: {result.detail}")
        else:
            st.caption("No approval required for this runbook's steps.")
    else:
        st.write("No predefined remediation runbook matches this query.")

    st.subheader("8. Audit log (this session)")
    all_records = audit_logger.read_all()
    st.json(all_records[-3:] if len(all_records) > 3 else all_records)
    st.caption(f"{len(all_records)} record(s) written this session. Raw PII is never persisted.")
