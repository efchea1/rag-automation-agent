"""
events/bus.py
---------------
Event-driven operations layer — addresses "evolving from traditional
infrastructure and scripting models to AI-driven, agentic, and
event-based operations" and "improve IT service delivery speed,
consistency, and quality through intelligent automation and event-driven
design."

A minimal, dependency-free in-process pub/sub bus. In a real deployment
this would be backed by something durable (Azure Service Bus, AWS
EventBridge/SQS, Kafka) — the point here is the *shape* of the
architecture: the agent doesn't only respond to a human typing a question,
it also reacts to events emitted by other systems (a new ticket landing,
a log line matching an error pattern, a monitoring alert firing).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class Event:
    event_type: str
    payload: dict
    source: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


Handler = Callable[[Event], None]


class EventBus:
    """Simple synchronous pub/sub. Handlers run in registration order."""

    def __init__(self):
        self._handlers: Dict[str, List[Handler]] = defaultdict(list)
        self._history: List[Event] = []

    def subscribe(self, event_type: str, handler: Handler) -> None:
        self._handlers[event_type].append(handler)
        logger.info("Handler '%s' subscribed to event_type='%s'", getattr(handler, "__name__", handler), event_type)

    def publish(self, event: Event) -> None:
        self._history.append(event)
        logger.info("Event published: type=%s source=%s", event.event_type, event.source)

        for handler in self._handlers.get(event.event_type, []):
            try:
                handler(event)
            except Exception:  # noqa: BLE001
                logger.exception("Handler failed for event_type='%s'", event.event_type)

        # wildcard subscribers (event_type="*") receive every event
        for handler in self._handlers.get("*", []):
            try:
                handler(event)
            except Exception:  # noqa: BLE001
                logger.exception("Wildcard handler failed for event_type='%s'", event.event_type)

    def history(self) -> List[Event]:
        return list(self._history)


def agent_event_handler(agent, itsm_client=None, power_automate_client=None):
    """
    Factory: builds an event handler that runs the RAG agent against an
    incoming event's text, and — if the resulting triage is a high-urgency
    incident — automatically opens an ITSM ticket and/or triggers a Power
    Automate flow. This is the concrete "agentic AI workflow for
    automated incident triage and response" wired end-to-end: event in,
    triage + ticket + notification out, no human required to kick it off.
    """
    from ..power_automate import PowerAutomateWebhookPayload

    def handle(event: Event) -> None:
        query_text = event.payload.get("text", "")
        if not query_text:
            logger.warning("Event %s had no 'text' payload field; skipping agent run.", event.event_type)
            return

        response = agent.run(query_text)
        logger.info(
            "Event-triggered agent run: category=%s urgency=%s",
            response.triage.category, response.triage.urgency,
        )

        if response.triage.urgency == "high" and itsm_client is not None:
            ticket = itsm_client.create_incident(
                short_description=f"[Auto-triaged] {query_text[:100]}",
                description=f"{response.answer}\n\nAction plan:\n{response.action_plan}",
                urgency="1",
            )
            logger.info("Auto-created ticket %s from event %s", ticket.ticket_number, event.event_type)

        if response.triage.urgency == "high" and power_automate_client is not None:
            payload = PowerAutomateWebhookPayload(
                event_type="incident_triaged",
                query=query_text,
                category=response.triage.category,
                urgency=response.triage.urgency,
                summary=response.answer[:280],
                source_documents=[c.source_path for c in response.citations],
            )
            power_automate_client.trigger_flow("notify_oncall", payload)

    return handle
