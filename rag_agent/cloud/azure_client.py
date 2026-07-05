"""
cloud/azure_client.py
-----------------------
Azure integration for the RAG agent — the Azure half of "experience
working in cloud environments (AWS and/or Azure)".

Two integration points mirroring the AWS module:
  - Blob Storage as a document source
  - Key Vault for secrets

Uses `azure-identity` (DefaultAzureCredential, so it works with managed
identity in production and `az login` locally without code changes) and
`azure-storage-blob`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from azure.core.exceptions import ResourceNotFoundError, ClientAuthenticationError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

logger = logging.getLogger(__name__)


class BlobDocumentSource:
    """Lists and downloads documents from an Azure Blob Storage container."""

    def __init__(self, account_url: str, container_name: str, prefix: str = ""):
        # account_url looks like https://<account>.blob.core.windows.net
        self.container_name = container_name
        self.prefix = prefix
        credential = DefaultAzureCredential()
        self._service_client = BlobServiceClient(account_url=account_url, credential=credential)
        self._container_client = self._service_client.get_container_client(container_name)

    def list_documents(self, suffixes: tuple = (".pdf", ".docx", ".html", ".htm", ".txt")) -> List[str]:
        try:
            blobs = self._container_client.list_blobs(name_starts_with=self.prefix)
            names = [b.name for b in blobs if b.name.lower().endswith(suffixes)]
        except ClientAuthenticationError as exc:
            raise RuntimeError(
                "Azure authentication failed. Ensure a managed identity, "
                "`az login` session, or service principal env vars are available."
            ) from exc

        logger.info("Found %d ingestible blobs in %s/%s", len(names), self.container_name, self.prefix)
        return names

    def download_to(self, blob_name: str, local_dir: str) -> str:
        local_path = Path(local_dir) / Path(blob_name).name
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            blob_client = self._container_client.get_blob_client(blob_name)
            with open(local_path, "wb") as f:
                f.write(blob_client.download_blob().readall())
        except ResourceNotFoundError as exc:
            raise RuntimeError(f"Blob '{blob_name}' not found in container '{self.container_name}'.") from exc

        logger.info("Downloaded blob %s -> %s", blob_name, local_path)
        return str(local_path)

    def sync_all(self, local_dir: str) -> List[str]:
        return [self.download_to(name, local_dir) for name in self.list_documents()]


class KeyVaultClient:
    """Fetches secrets from Azure Key Vault via DefaultAzureCredential."""

    def __init__(self, vault_url: str):
        from azure.keyvault.secrets import SecretClient  # imported lazily; optional extra

        credential = DefaultAzureCredential()
        self._client = SecretClient(vault_url=vault_url, credential=credential)

    def get_secret(self, secret_name: str) -> str:
        try:
            secret = self._client.get_secret(secret_name)
        except ResourceNotFoundError as exc:
            raise RuntimeError(f"Secret '{secret_name}' not found in Key Vault.") from exc
        except ClientAuthenticationError as exc:
            raise RuntimeError("Azure authentication failed when accessing Key Vault.") from exc

        return secret.value
