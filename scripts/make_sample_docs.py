"""
Generates three generic sample documents used to demonstrate that the
loader/chunker/retriever pipeline works uniformly across PDF, Word, and HTML.
Run once: python scripts/make_sample_docs.py
"""

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "sample_docs"
OUT.mkdir(exist_ok=True)


def make_pdf():
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    path = OUT / "password_reset_policy.pdf"
    c = canvas.Canvas(str(path), pagesize=letter)
    lines = [
        "Password Reset and Account Lockout Policy",
        "",
        "1. Scope",
        "This policy applies to all corporate identity accounts, including",
        "Active Directory, SSO-federated SaaS applications, and VPN access.",
        "",
        "2. Lockout Threshold",
        "Accounts are locked after 5 consecutive failed login attempts within",
        "a 15-minute window. Lockout duration is 30 minutes unless manually",
        "cleared by IT Operations.",
        "",
        "3. Self-Service Reset",
        "Employees may reset their own password via the self-service portal",
        "after verifying identity with MFA. This resolves roughly 70% of",
        "password-related tickets without opening a helpdesk ticket.",
        "",
        "4. Manual Reset Procedure (IT Operations)",
        "If self-service is unavailable, verify the employee's identity via",
        "manager confirmation or video call, then reset the password in the",
        "identity provider console and force a change at next login.",
        "",
        "5. Escalation",
        "Repeated lockouts (3+ in 24 hours) for the same account should be",
        "escalated to Security as a potential credential-stuffing indicator.",
    ]
    y = 740
    for line in lines:
        c.drawString(50, y, line)
        y -= 18
    c.save()
    return path


def make_docx():
    import docx

    path = OUT / "new_hire_onboarding_runbook.docx"
    doc = docx.Document()
    doc.add_heading("New Hire IT Onboarding Runbook", level=1)

    doc.add_heading("Overview", level=2)
    doc.add_paragraph(
        "This runbook describes the standard IT onboarding sequence for new "
        "hires, covering identity provisioning, hardware, and access "
        "requests. Target SLA: all accounts active by 9am on start date."
    )

    doc.add_heading("Steps", level=2)
    steps = [
        "HR system triggers a new-hire event 5 business days before start date.",
        "Identity account is provisioned in Azure AD with department-based group membership.",
        "Standard SaaS licenses (email, chat, ticketing) are auto-assigned via group policy.",
        "Laptop is imaged and shipped (remote) or staged at desk (on-site).",
        "Manager receives a checklist to confirm role-specific access (e.g. VPN, specific ITSM projects, shared drives) on day 1.",
        "IT Operations confirms all accounts are active in the onboarding dashboard.",
    ]
    for i, step in enumerate(steps, start=1):
        doc.add_paragraph(f"{i}. {step}")

    doc.add_heading("Common Issues", level=2)
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Issue"
    table.rows[0].cells[1].text = "Resolution"
    rows = [
        ("Account not provisioned by start date", "Check HR event triggered on time; manually provision via Azure AD if delayed."),
        ("Manager access checklist incomplete", "Escalate to manager's director after 48 hours; access requests without manager sign-off require Security exception."),
        ("Laptop DOA on arrival", "Advance-replace from spare pool; do not wait for RMA before reissuing."),
    ]
    for issue, resolution in rows:
        row = table.add_row()
        row.cells[0].text = issue
        row.cells[1].text = resolution

    doc.save(str(path))
    return path


def make_html():
    path = OUT / "vpn_troubleshooting_kb.html"
    html = """<!DOCTYPE html>
<html>
<head><title>KB0042 - VPN Connection Troubleshooting</title></head>
<body>
<nav>Knowledge Base &gt; Networking</nav>
<h1>KB0042: VPN Connection Troubleshooting</h1>
<p>Applies to: GlobalProtect client, all supported OS versions.</p>

<h2>Symptom</h2>
<p>User reports VPN client fails to connect, hangs at "Connecting..." or
returns error code 3 ("Unable to connect to the service").</p>

<h2>Common Root Causes</h2>
<ul>
<li>Expired or revoked client certificate (most common - roughly 40% of tickets)</li>
<li>Local firewall or endpoint security software blocking the VPN adapter</li>
<li>Stale DNS cache pointing to a decommissioned gateway</li>
<li>Gateway-side capacity saturation during peak hours (8-9am local time)</li>
</ul>

<h2>Triage Steps</h2>
<ol>
<li>Check the identity provider for certificate expiration on the user's account.</li>
<li>Have the user run the built-in diagnostic tool and capture the log bundle.</li>
<li>If certificate is valid, check gateway health dashboard for regional saturation.</li>
<li>If none of the above resolve it within 15 minutes, escalate to Network Engineering with the log bundle attached.</li>
</ol>

<h2>Self-Service Fix</h2>
<p>For expired certificates, users can regenerate their own client certificate
from the self-service identity portal without opening a ticket. This is the
single highest-impact self-service deflection opportunity in the VPN
category.</p>

<footer>Last reviewed: internal IT knowledge base</footer>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path


if __name__ == "__main__":
    p1 = make_pdf()
    p2 = make_docx()
    p3 = make_html()
    print(f"Created:\n  {p1}\n  {p2}\n  {p3}")
