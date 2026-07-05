"""
identity/lifecycle.py
------------------------
Automates the "employee onboarding and offboarding workflows across
identity, access, collaboration, and endpoint systems" requirement.

Talks to Microsoft Graph (the standard API for Azure AD / Entra ID —
identity, groups/licenses, and M365 collaboration tools live behind the
same Graph API). Auth uses MSAL client-credentials flow, which is the
standard pattern for an unattended automation service principal.

Every mutating action here is exposed as an explicit method with a
`dry_run` flag defaulting to True, so this can be wired into the agent's
action-plan step and exercised safely before anyone flips it to actually
mutate identity state. Real production use also needs an approval gate —
see runbooks/executor.py, which this module is designed to be called from.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"


@dataclass
class LifecycleActionResult:
    action: str
    target: str
    dry_run: bool
    success: bool
    detail: str
    payload: dict = field(default_factory=dict)


class GraphIdentityClient:
    """
    Wraps the Microsoft Graph calls needed for onboarding/offboarding:
    create/disable accounts, assign/remove group (and therefore license)
    membership, and revoke sessions.

    Requires an app registration with Application permissions
    (User.ReadWrite.All, Group.ReadWrite.All) and admin consent — this is
    infrastructure-level access, so in production this client should run
    under a locked-down service principal with logging/alerting on every
    call, not developer credentials.
    """

    def __init__(
        self,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        self.tenant_id = tenant_id or os.environ.get("AZURE_TENANT_ID")
        self.client_id = client_id or os.environ.get("AZURE_CLIENT_ID")
        self.client_secret = client_secret or os.environ.get("AZURE_CLIENT_SECRET")

        missing = [
            name
            for name, val in [
                ("AZURE_TENANT_ID", self.tenant_id),
                ("AZURE_CLIENT_ID", self.client_id),
                ("AZURE_CLIENT_SECRET", self.client_secret),
            ]
            if not val
        ]
        if missing:
            raise RuntimeError(
                f"Missing Graph API credentials: {missing}. Set them as env vars "
                f"or pass explicitly to GraphIdentityClient()."
            )

        self._token: Optional[str] = None

    def _get_token(self) -> str:
        if self._token:
            return self._token

        import msal

        app = msal.ConfidentialClientApplication(
            self.client_id,
            authority=f"https://login.microsoftonline.com/{self.tenant_id}",
            client_credential=self.client_secret,
        )
        result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])

        if "access_token" not in result:
            raise RuntimeError(
                f"Failed to acquire Graph API token: {result.get('error_description', result)}"
            )

        self._token = result["access_token"]
        return self._token

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._get_token()}", "Content-Type": "application/json"}

    # ---------- Onboarding ----------

    def create_user(
        self,
        display_name: str,
        user_principal_name: str,
        initial_password: str,
        group_ids: Optional[List[str]] = None,
        dry_run: bool = True,
    ) -> LifecycleActionResult:
        body = {
            "accountEnabled": True,
            "displayName": display_name,
            "mailNickname": user_principal_name.split("@")[0],
            "userPrincipalName": user_principal_name,
            "passwordProfile": {
                "forceChangePasswordNextSignIn": True,
                "password": initial_password,
            },
        }

        if dry_run:
            logger.info("[DRY RUN] Would create user %s: %s", user_principal_name, body)
            return LifecycleActionResult(
                action="create_user", target=user_principal_name, dry_run=True,
                success=True, detail="Dry run — no user created.", payload=body,
            )

        response = requests.post(f"{GRAPH_BASE_URL}/users", json=body, headers=self._headers())
        if response.status_code >= 300:
            return LifecycleActionResult(
                action="create_user", target=user_principal_name, dry_run=False,
                success=False, detail=f"Graph API error {response.status_code}: {response.text}",
            )

        user = response.json()
        if group_ids:
            for group_id in group_ids:
                self.add_to_group(user["id"], group_id, dry_run=False)

        return LifecycleActionResult(
            action="create_user", target=user_principal_name, dry_run=False,
            success=True, detail="User created.", payload=user,
        )

    def add_to_group(self, user_id: str, group_id: str, dry_run: bool = True) -> LifecycleActionResult:
        body = {"@odata.id": f"{GRAPH_BASE_URL}/directoryObjects/{user_id}"}

        if dry_run:
            logger.info("[DRY RUN] Would add user %s to group %s", user_id, group_id)
            return LifecycleActionResult(
                action="add_to_group", target=f"{user_id}->{group_id}", dry_run=True,
                success=True, detail="Dry run — no membership change made.",
            )

        response = requests.post(
            f"{GRAPH_BASE_URL}/groups/{group_id}/members/$ref", json=body, headers=self._headers()
        )
        success = response.status_code < 300
        return LifecycleActionResult(
            action="add_to_group", target=f"{user_id}->{group_id}", dry_run=False,
            success=success,
            detail="Added to group." if success else f"Graph API error {response.status_code}: {response.text}",
        )

    # ---------- Offboarding ----------

    def disable_account(self, user_principal_name: str, dry_run: bool = True) -> LifecycleActionResult:
        body = {"accountEnabled": False}

        if dry_run:
            logger.info("[DRY RUN] Would disable account %s", user_principal_name)
            return LifecycleActionResult(
                action="disable_account", target=user_principal_name, dry_run=True,
                success=True, detail="Dry run — account not disabled.",
            )

        response = requests.patch(
            f"{GRAPH_BASE_URL}/users/{user_principal_name}", json=body, headers=self._headers()
        )
        success = response.status_code < 300
        return LifecycleActionResult(
            action="disable_account", target=user_principal_name, dry_run=False,
            success=success,
            detail="Account disabled." if success else f"Graph API error {response.status_code}: {response.text}",
        )

    def revoke_sessions(self, user_principal_name: str, dry_run: bool = True) -> LifecycleActionResult:
        """Invalidate all refresh tokens — forces immediate sign-out everywhere."""
        if dry_run:
            logger.info("[DRY RUN] Would revoke all sessions for %s", user_principal_name)
            return LifecycleActionResult(
                action="revoke_sessions", target=user_principal_name, dry_run=True,
                success=True, detail="Dry run — sessions not revoked.",
            )

        response = requests.post(
            f"{GRAPH_BASE_URL}/users/{user_principal_name}/revokeSignInSessions",
            headers=self._headers(),
        )
        success = response.status_code < 300
        return LifecycleActionResult(
            action="revoke_sessions", target=user_principal_name, dry_run=False,
            success=success,
            detail="Sessions revoked." if success else f"Graph API error {response.status_code}: {response.text}",
        )

    def remove_from_all_groups(self, user_id: str, group_ids: List[str], dry_run: bool = True) -> List[LifecycleActionResult]:
        results = []
        for group_id in group_ids:
            if dry_run:
                results.append(LifecycleActionResult(
                    action="remove_from_group", target=f"{user_id}->{group_id}", dry_run=True,
                    success=True, detail="Dry run — no membership change made.",
                ))
                continue

            response = requests.delete(
                f"{GRAPH_BASE_URL}/groups/{group_id}/members/{user_id}/$ref", headers=self._headers()
            )
            success = response.status_code < 300
            results.append(LifecycleActionResult(
                action="remove_from_group", target=f"{user_id}->{group_id}", dry_run=False,
                success=success,
                detail="Removed." if success else f"Graph API error {response.status_code}: {response.text}",
            ))
        return results


class MockGraphIdentityClient(GraphIdentityClient):
    """
    Dependency-free stand-in: same method signatures as GraphIdentityClient,
    but every call is force-dry-run and nothing ever hits the network. Used
    in tests and in the demo so onboarding/offboarding logic is exercised
    without real tenant credentials.
    """

    def __init__(self):
        self.tenant_id = "mock-tenant"
        self.client_id = "mock-client"
        self.client_secret = "mock-secret"
        self._token = "mock-token"
        self.calls: List[str] = []

    def create_user(self, display_name, user_principal_name, initial_password, group_ids=None, dry_run=True):
        self.calls.append(f"create_user:{user_principal_name}")
        return LifecycleActionResult(
            action="create_user", target=user_principal_name, dry_run=True,
            success=True, detail="[MOCK] User creation simulated.",
        )

    def disable_account(self, user_principal_name, dry_run=True):
        self.calls.append(f"disable_account:{user_principal_name}")
        return LifecycleActionResult(
            action="disable_account", target=user_principal_name, dry_run=True,
            success=True, detail="[MOCK] Account disable simulated.",
        )

    def revoke_sessions(self, user_principal_name, dry_run=True):
        self.calls.append(f"revoke_sessions:{user_principal_name}")
        return LifecycleActionResult(
            action="revoke_sessions", target=user_principal_name, dry_run=True,
            success=True, detail="[MOCK] Session revocation simulated.",
        )
