"""
itsm/servicenow_client.py
----------------------------
ITSM integration — addresses "background in ITSM automation, incident
management" and lets the agent actually create/update tickets rather than
just recommend that someone else does.

ServiceNow's Table API is REST over HTTPS with basic or OAuth auth; this
client covers the incident table specifically since that's the natural
target for the agent's triage output, but `create_record`/`update_record`
are generic and work against any table (change_request, sc_request, etc.)
for the access-request and how-to-request categories too.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)


@dataclass
class TicketResult:
    ticket_number: str
    sys_id: str
    table: str
    url: str
    success: bool
    detail: str = ""


class ServiceNowClient:
    """Thin wrapper over the ServiceNow Table REST API."""

    def __init__(
        self,
        instance_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout_seconds: int = 15,
    ):
        # instance_url looks like https://yourcompany.service-now.com
        self.instance_url = (instance_url or os.environ.get("SERVICENOW_INSTANCE_URL", "")).rstrip("/")
        self.username = username or os.environ.get("SERVICENOW_USERNAME")
        self.password = password or os.environ.get("SERVICENOW_PASSWORD")
        self.timeout_seconds = timeout_seconds

        if not all([self.instance_url, self.username, self.password]):
            raise RuntimeError(
                "ServiceNow credentials incomplete. Set SERVICENOW_INSTANCE_URL, "
                "SERVICENOW_USERNAME, SERVICENOW_PASSWORD, or pass explicitly."
            )

    def _auth(self) -> HTTPBasicAuth:
        return HTTPBasicAuth(self.username, self.password)

    def create_incident(
        self,
        short_description: str,
        description: str,
        urgency: str = "3",       # ServiceNow: 1=high, 2=medium, 3=low
        category: str = "software",
        caller_id: Optional[str] = None,
    ) -> TicketResult:
        body = {
            "short_description": short_description,
            "description": description,
            "urgency": urgency,
            "category": category,
        }
        if caller_id:
            body["caller_id"] = caller_id

        url = f"{self.instance_url}/api/now/table/incident"
        response = requests.post(url, json=body, auth=self._auth(), timeout=self.timeout_seconds)

        if response.status_code >= 300:
            return TicketResult(
                ticket_number="", sys_id="", table="incident", url="",
                success=False, detail=f"ServiceNow API error {response.status_code}: {response.text}",
            )

        result = response.json().get("result", {})
        return TicketResult(
            ticket_number=result.get("number", ""),
            sys_id=result.get("sys_id", ""),
            table="incident",
            url=f"{self.instance_url}/nav_to.do?uri=incident.do?sys_id={result.get('sys_id', '')}",
            success=True,
            detail="Incident created.",
        )

    def add_work_note(self, sys_id: str, note: str, table: str = "incident") -> bool:
        url = f"{self.instance_url}/api/now/table/{table}/{sys_id}"
        response = requests.patch(
            url, json={"work_notes": note}, auth=self._auth(), timeout=self.timeout_seconds
        )
        return response.status_code < 300


def urgency_to_servicenow_code(urgency: str) -> str:
    """Map the agent's triage urgency vocabulary onto ServiceNow's numeric codes."""
    return {"high": "1", "medium": "2", "low": "3"}.get(urgency.lower(), "3")


class MockServiceNowClient(ServiceNowClient):
    """Dependency-free stand-in for demos/tests: no network calls, no credentials required."""

    def __init__(self):
        self.instance_url = "https://mock.service-now.com"
        self.username = "mock"
        self.password = "mock"
        self.timeout_seconds = 15
        self.created_tickets: list = []

    def create_incident(self, short_description, description, urgency="3", category="software", caller_id=None):
        ticket_number = f"INC{1000 + len(self.created_tickets)}"
        self.created_tickets.append(
            {"short_description": short_description, "description": description, "urgency": urgency}
        )
        logger.info("[MOCK] Created ServiceNow incident %s: %s", ticket_number, short_description)
        return TicketResult(
            ticket_number=ticket_number, sys_id=f"mock-{ticket_number}", table="incident",
            url=f"{self.instance_url}/nav_to.do?uri=incident.do?sys_id=mock-{ticket_number}",
            success=True, detail="[MOCK] Incident created.",
        )
