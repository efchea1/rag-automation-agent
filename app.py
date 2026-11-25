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
import tempfile
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
if "ingested_uploads" not in st.session_state:
    st.session_state.ingested_uploads = set()
if "upload_dir" not in st.session_state:
    st.session_state.upload_dir = tempfile.mkdtemp(prefix="rag_agent_uploads_")

st.title("RAG automation agent")
st.markdown(
    "**A working AI assistant for IT operations** — it reads your documents, "
    "answers questions about them with cited sources, and automatically "
    "classifies and escalates issues that need human attention."
)
st.caption(
    "Real RagAgent from [this project's codebase](https://github.com/efchea1/rag-automation-agent) — hybrid retrieval, PII redaction, "
    "triage, auto-escalation, guided remediation, and audit logging. "
    f"LLM backend: `{type(agent.llm).__name__}`"
    + (" (no API key set — using deterministic mock output)" if isinstance(agent.llm, MockLLMClient) else "")
)

with st.expander("How this works — what happens when you click \"Run agent\"", expanded=False):
    st.markdown(
        "Every query runs through the same 8-step pipeline as the underlying "
        "`RagAgent.run()` in the actual codebase:\n\n"
        "1. **PII redaction** — emails and phone numbers are stripped before anything reaches the LLM\n"
        "2. **Query rewrite** — expands your question for better retrieval recall\n"
        "3. **Hybrid retrieval** — searches the indexed documents (keyword + vector search)\n"
        "4. **Grounded answer** — the LLM answers using only the retrieved passages, with citations\n"
        "5. **Triage classification** — categorizes and rates the urgency of the request\n"
        "6. **Auto-escalation** — high-urgency incidents automatically open a ServiceNow ticket "
        "and trigger a Power Automate notification (simulated in this demo)\n"
        "7. **Guided remediation** — if a known pattern matches, a specific fix is proposed "
        "(and requires your approval before it would ever execute)\n"
        "8. **Audit log** — every step of every run is recorded, with PII already redacted"
    )

st.subheader("Add your own documents")
uploaded_files = st.file_uploader(
    "Upload PDF, Word (.docx), HTML, or text files to add them to the index",
    type=["pdf", "docx", "html", "htm", "txt"],
    accept_multiple_files=True,
)
st.caption(
    "Demo only. Don't upload personal or confidential files. Uploads are visible "
    "to other visitors and cleared when the app restarts."
)

if uploaded_files:
    newly_ingested = []
    for uf in uploaded_files:
        # Security: uf.name is attacker-controlled (whatever the uploading
        # browser reports). Without sanitizing it, a crafted filename like
        # "../../../etc/something" or an absolute path could write outside
        # the intended upload directory (path traversal / arbitrary write).
        # Normalize backslashes to forward slashes first, then take the
        # basename — Path(...).name alone only treats "/" as a separator on
        # POSIX, so a Windows-style "..\\..\\evil.txt" would slip through
        # unmodified there; this app runs locally on Windows as well as on
        # Linux (Streamlit Cloud), so sanitization must not assume either OS.
        safe_name = os.path.basename(uf.name.replace("\\", "/"))
        if not safe_name:
            st.error("Skipped a file with an invalid name.")
            continue

        file_key = f"{safe_name}:{uf.size}"
        if file_key in st.session_state.ingested_uploads:
            continue  # already indexed this exact file this session; don't re-ingest on every rerun

        dest_path = Path(st.session_state.upload_dir) / safe_name
        with open(dest_path, "wb") as f:
            f.write(uf.getbuffer())

        try:
            n = agent.ingest_path(str(dest_path))
            st.session_state.ingested_uploads.add(file_key)
            newly_ingested.append((safe_name, n))
        except ValueError as exc:
            st.error(f"Couldn't index `{safe_name}`: {exc}")

    for name, n in newly_ingested:
        st.success(f"Indexed `{name}` — {n} chunk(s) added to the index.")

with st.expander("Indexed documents"):
    st.write(f"Total indexed chunks: {len(agent._chunks)}")
    st.caption("Sample documents (always indexed):")
    for path in sorted(SAMPLE_DIR.glob("*")):
        st.write(f"- `{path.name}`")
    if st.session_state.ingested_uploads:
        st.caption("Your uploaded documents:")
        for key in sorted(st.session_state.ingested_uploads):
            st.write(f"- `{key.rsplit(':', 1)[0]}`")

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
        st.success(f"ServiceNow incident `{ticket.ticket_number}` auto-created, Power Automate `notify_oncall` triggered (simulated in this demo).")
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
    st.caption(
        f"{len(all_records)} record(s) written this session. "
        "Raw PII is never persisted — expand below to verify: the `query` "
        "field shows a tag like `[REDACTED_EMAIL]` instead of your original text."
    )
    with st.expander("View raw audit records"):
        st.json(all_records[-3:] if len(all_records) > 3 else all_records)

st.divider()
st.caption(
    "Built by Emmanuel Fle Chea · "
    "[GitHub](https://github.com/efchea1/rag-automation-agent) · "
    "[LinkedIn](https://www.linkedin.com/in/emmanuel-fle-chea/)"
)
