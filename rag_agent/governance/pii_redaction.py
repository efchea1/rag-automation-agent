"""
governance/pii_redaction.py
------------------------------
Security and data-protection control: scrubs common PII patterns from text
before it's sent to any external LLM provider. Addresses "Establish
standards for: Security and data protection" and "Responsible AI usage"
in the job posting's Leadership & Enablement section.

This is intentionally regex-based rather than ML-based: it's deterministic,
auditable, has zero external dependencies, and — critically for a
security control — its behavior can be explained precisely rather than
"the model decided." It is a first line of defense, not a guarantee; real
deployments should pair this with provider-side data handling agreements
and DLP tooling appropriate to the sensitivity of the content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

_PATTERNS = {
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "PHONE": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    "IPV4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}


@dataclass
class RedactionResult:
    redacted_text: str
    findings: List[dict] = field(default_factory=list)

    @property
    def had_redactions(self) -> bool:
        return len(self.findings) > 0


def redact(text: str, categories: tuple = tuple(_PATTERNS.keys())) -> RedactionResult:
    """
    Replace matches of the given PII categories with `[REDACTED_<CATEGORY>]`.

    Returns both the redacted text and a findings list (category + count),
    never the raw matched values — findings are for audit logging, and
    logging the PII itself would defeat the purpose of redacting it.
    """
    redacted = text
    findings = []

    for category in categories:
        pattern = _PATTERNS.get(category)
        if pattern is None:
            continue
        matches = pattern.findall(redacted)
        if matches:
            findings.append({"category": category, "count": len(matches)})
            redacted = pattern.sub(f"[REDACTED_{category}]", redacted)

    return RedactionResult(redacted_text=redacted, findings=findings)
