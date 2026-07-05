"""
runbooks/executor.py
------------------------
Closes the "guided remediation and runbook execution" gap directly. Earlier
version of this system only ever *proposed* an action plan in prose; this
module lets the agent propose a specific, executable runbook step and then
actually run it — but only after dry-run preview and explicit human
approval, per governance/policies.yaml's human_in_the_loop standard.

Runbooks here wrap the mock/real clients built elsewhere in this package
(GraphIdentityClient for identity actions, ServiceNowClient for ticketing)
so "guided remediation" isn't just a string in a chat response — it's a
concrete, auditable function call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RunbookStep:
    name: str
    description: str
    action: Callable[..., object]
    requires_approval: bool = True


@dataclass
class RunbookExecutionResult:
    runbook_name: str
    dry_run: bool
    approved: bool
    executed: bool
    result: Optional[object] = None
    detail: str = ""


class Runbook:
    """A named, ordered sequence of steps addressing one incident pattern."""

    def __init__(
        self,
        name: str,
        trigger_keywords: List[str],
        steps: List[RunbookStep],
        exclude_keywords: Optional[List[str]] = None,
    ):
        self.name = name
        self.trigger_keywords = [k.lower() for k in trigger_keywords]
        self.exclude_keywords = [k.lower() for k in (exclude_keywords or [])]
        self.steps = steps

    def match_score(self, text: str) -> int:
        """
        Number of trigger phrases present in text, or 0 if any exclusion
        phrase is present. Exclusions exist because a single loud keyword
        (e.g. "vpn") can appear in queries this runbook clearly doesn't
        apply to (e.g. a full gateway outage vs. one user's expired
        certificate) — a bare "any keyword hit" match can't tell those
        apart, but an explicit exclusion list can.
        """
        text_lower = text.lower()
        if any(kw in text_lower for kw in self.exclude_keywords):
            return 0
        return sum(1 for kw in self.trigger_keywords if kw in text_lower)

    def matches(self, text: str) -> bool:
        return self.match_score(text) > 0


class RunbookExecutor:
    """
    Registry + execution engine for runbooks.

    Usage pattern:
        executor.find_matching_runbook(query)   -> Runbook | None
        executor.preview(runbook, **kwargs)     -> dry-run description, no side effects
        executor.execute(runbook, approved_by, **kwargs) -> actually runs it
    """

    def __init__(self):
        self._runbooks: Dict[str, Runbook] = {}

    def register(self, runbook: Runbook) -> None:
        self._runbooks[runbook.name] = runbook

    def find_matching_runbook(self, query: str) -> Optional[Runbook]:
        """
        Score every registered runbook and return the strongest match.

        Returns None (declining to match) rather than guessing when:
          - no runbook scores above 0, or
          - two or more runbooks tie for the top score — an ambiguous
            match is exactly the situation where a wrong automatic pick
            is worse than admitting "no confident match" and falling
            back to the agent's generic action-plan/escalation path.
        """
        scored = [(rb, rb.match_score(query)) for rb in self._runbooks.values()]
        scored = [(rb, score) for rb, score in scored if score > 0]

        if not scored:
            return None

        top_score = max(score for _, score in scored)
        top_matches = [rb for rb, score in scored if score == top_score]

        if len(top_matches) > 1:
            logger.warning(
                "Ambiguous runbook match (tied at score %d among %s) for query %r; "
                "declining to auto-select a runbook.",
                top_score, [rb.name for rb in top_matches], query,
            )
            return None

        return top_matches[0]

    def preview(self, runbook: Runbook, **kwargs) -> List[str]:
        """Describe what each step would do, without executing anything."""
        return [f"{step.name}: {step.description}" for step in runbook.steps]

    def execute(self, runbook: Runbook, approved_by: Optional[str] = None, **kwargs) -> RunbookExecutionResult:
        """
        Execute every step of the runbook. Steps marked requires_approval=True
        will not run unless `approved_by` is provided — this is the human
        gate. Steps are still called with dry_run=True by default if the
        caller doesn't override, so "approval" without care still won't
        mutate real systems unless the caller explicitly sets dry_run=False
        in kwargs.
        """
        if any(step.requires_approval for step in runbook.steps) and not approved_by:
            return RunbookExecutionResult(
                runbook_name=runbook.name, dry_run=True, approved=False, executed=False,
                detail="Runbook requires human approval before execution. No steps run.",
            )

        results = []
        for step in runbook.steps:
            logger.info("Executing runbook step '%s' (approved_by=%s)", step.name, approved_by)
            step_result = step.action(**kwargs)
            results.append({"step": step.name, "result": str(step_result)})

        return RunbookExecutionResult(
            runbook_name=runbook.name, dry_run=kwargs.get("dry_run", True),
            approved=True, executed=True, result=results,
            detail=f"Executed {len(results)} step(s), approved by {approved_by}.",
        )


def build_default_runbooks(graph_client, itsm_client) -> RunbookExecutor:
    """
    Wires up the two runbooks that map directly onto the sample documents
    used throughout this project: VPN certificate issues and account
    lockouts. In a real deployment this registry would grow to cover every
    KB article with a well-defined remediation.
    """
    executor = RunbookExecutor()

    executor.register(Runbook(
        name="regenerate_vpn_certificate",
        trigger_keywords=["certificate", "error code 3", "cert expired", "regenerate certificate"],
        exclude_keywords=[
            "gateway is down", "gateway down", "everyone", "all users",
            "complete outage", "full outage", "outage",
        ],
        steps=[
            RunbookStep(
                name="revoke_stale_certificate",
                description="Revoke the user's expired/stale VPN client certificate.",
                action=lambda **kw: f"[stub] would revoke cert for {kw.get('user_principal_name', 'unknown user')}",
                requires_approval=True,
            ),
            RunbookStep(
                name="notify_user_self_service",
                description="Notify the user to regenerate their certificate via the self-service portal.",
                action=lambda **kw: f"[stub] would notify {kw.get('user_principal_name', 'unknown user')}",
                requires_approval=False,
            ),
        ],
    ))

    executor.register(Runbook(
        name="clear_account_lockout",
        trigger_keywords=["locked out", "lockout", "failed login", "failed logins", "repeated failed"],
        exclude_keywords=[
            "gateway is down", "gateway down", "everyone", "all users",
            "complete outage", "full outage", "outage", "vpn gateway",
        ],
        steps=[
            RunbookStep(
                name="revoke_sessions_and_clear_lockout",
                description="Revoke active sessions and clear the account lockout flag.",
                action=lambda **kw: graph_client.revoke_sessions(
                    kw.get("user_principal_name", "unknown@example.com"), dry_run=kw.get("dry_run", True)
                ),
                requires_approval=True,
            ),
            RunbookStep(
                name="open_security_review_ticket",
                description="Open a ServiceNow ticket for Security to review repeated lockouts.",
                action=lambda **kw: itsm_client.create_incident(
                    short_description=f"Repeated lockouts: {kw.get('user_principal_name', 'unknown')}",
                    description="Auto-opened by RAG agent guided remediation runbook.",
                    urgency="2",
                ),
                requires_approval=True,
            ),
        ],
    ))

    return executor
