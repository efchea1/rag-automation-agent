"""
governance/audit_log.py
--------------------------
Structured, append-only audit logging for every agent run — addresses
"Establish standards for: ... Operational resilience" and the general
"secure, scalable, observable, and maintainable" requirement, plus
supports the "familiarity with governance, risk, and compliance
considerations for enterprise AI" preferred skill.

Writes one JSON line per agent run (JSONL — easy to ship to a SIEM or
log-analytics pipeline). Never logs raw PII — callers should pass
already-redacted text (see pii_redaction.py) or explicitly acknowledge
they're logging sensitive content.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AuditRecord:
    timestamp: str
    query: str
    rewritten_query: str
    category: str
    urgency: str
    recommended_action: str
    sources_used: list
    pii_redactions: list = field(default_factory=list)
    downstream_actions: list = field(default_factory=list)  # e.g. ["itsm:INC1001", "power_automate:notify_oncall"]
    approved_by: Optional[str] = None  # set when a human approves a guided-remediation action


class AuditLogger:
    """Appends one JSON line per record to a log file."""

    def __init__(self, log_path: str = "audit_log.jsonl"):
        self.log_path = Path(log_path)

    def log(self, record: AuditRecord) -> None:
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")
        logger.info(
            "Audit record written: category=%s urgency=%s sources=%d",
            record.category, record.urgency, len(record.sources_used),
        )

    def read_all(self) -> list:
        if not self.log_path.exists():
            return []
        records = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


def build_audit_record(agent_response, pii_findings: Optional[list] = None) -> AuditRecord:
    """Convenience constructor from an agent.AgentResponse."""
    return AuditRecord(
        timestamp=datetime.now(timezone.utc).isoformat(),
        query=agent_response.original_query,
        rewritten_query=agent_response.rewritten_query,
        category=agent_response.triage.category,
        urgency=agent_response.triage.urgency,
        recommended_action=agent_response.triage.recommended_action,
        sources_used=[c.source_path for c in agent_response.citations],
        pii_redactions=pii_findings or [],
    )
