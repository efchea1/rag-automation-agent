"""
power_automate.py
------------------
Integration with Microsoft Power Automate — the "hands-on experience with
Microsoft Power Automate and related Power Platform components" required
skill in the job posting.

The standard integration pattern is: a Power Automate flow starts with an
"When an HTTP request is received" trigger, which hands you a webhook URL.
External systems (like this agent) POST a JSON payload to that URL to kick
off the flow. This client wraps that pattern, plus a mirror-image
`PowerAutomateWebhookPayload` builder so the JSON shape sent is predictable
and matches what a flow's request-body JSON schema would expect.

Typical flows this agent would trigger in a real deployment:
  - "Create/update ServiceNow incident from AI Ops triage"
  - "Notify on-call via Teams when urgency=high"
  - "Kick off SharePoint approval flow for an access request"
  - "Log agent decision to a Dataverse table for audit/compliance"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class PowerAutomateWebhookPayload:
    """Structured payload matching a typical Power Automate HTTP-trigger schema."""

    event_type: str  # e.g. "incident_triaged", "access_request", "onboarding_task"
    query: str
    category: str
    urgency: str
    summary: str
    source_documents: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "eventType": self.event_type,
            "query": self.query,
            "category": self.category,
            "urgency": self.urgency,
            "summary": self.summary,
            "sourceDocuments": self.source_documents,
            "metadata": self.metadata,
        }


class PowerAutomateClient:
    """
    Triggers Power Automate flows via their HTTP-request trigger webhooks.

    Flow URLs are typically per-environment secrets (they embed a SAS-style
    signature), so this client takes them from environment variables by
    default rather than hardcoding them anywhere.
    """

    def __init__(self, flow_webhook_urls: Optional[dict] = None, timeout_seconds: int = 10):
        # flow_webhook_urls maps a logical flow name -> webhook URL, e.g.
        # {"notify_oncall": "https://prod-00.westus.logic.azure.com/workflows/.../triggers/manual/paths/invoke?..."}
        self.flow_webhook_urls = flow_webhook_urls or self._load_from_env()
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _load_from_env() -> dict:
        urls = {}
        prefix = "POWER_AUTOMATE_FLOW_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                flow_name = key[len(prefix):].lower()
                urls[flow_name] = value
        return urls

    def trigger_flow(self, flow_name: str, payload: PowerAutomateWebhookPayload) -> dict:
        """
        POST the payload to the named flow's HTTP trigger.

        Returns a dict with status info. Raises RuntimeError if the flow
        name isn't configured, so misconfiguration fails loudly rather than
        silently no-op'ing in production.
        """
        webhook_url = self.flow_webhook_urls.get(flow_name)
        if not webhook_url:
            raise RuntimeError(
                f"No webhook URL configured for flow '{flow_name}'. "
                f"Set POWER_AUTOMATE_FLOW_{flow_name.upper()} or pass flow_webhook_urls= explicitly. "
                f"Configured flows: {list(self.flow_webhook_urls.keys())}"
            )

        body = payload.to_dict()
        logger.info("Triggering Power Automate flow '%s' with event_type=%s", flow_name, payload.event_type)

        response = requests.post(webhook_url, json=body, timeout=self.timeout_seconds)
        response.raise_for_status()

        return {
            "flow_name": flow_name,
            "status_code": response.status_code,
            "triggered": True,
        }


class MockPowerAutomateClient(PowerAutomateClient):
    """
    Dependency-free stand-in for demos/tests: logs what would have been
    sent instead of making a real HTTP call. Same interface as
    PowerAutomateClient, so agent code doesn't change between the two.
    """

    def __init__(self):
        super().__init__(flow_webhook_urls={})
        self.sent_payloads: list = []

    def trigger_flow(self, flow_name: str, payload: PowerAutomateWebhookPayload) -> dict:
        logger.info("[MOCK] Would trigger Power Automate flow '%s': %s", flow_name, payload.to_dict())
        self.sent_payloads.append((flow_name, payload.to_dict()))
        return {"flow_name": flow_name, "status_code": 200, "triggered": True, "mock": True}
