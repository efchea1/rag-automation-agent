"""
cloud/aws_client.py
--------------------
AWS integration for the RAG agent — addresses the "experience working in
cloud environments (AWS and/or Azure)" and "integrate with cloud
infrastructure" requirements.

Two concrete integration points, chosen because they're the ones an actual
IT-ops RAG deployment on AWS would need on day one:
  - S3 as a document source (ingest runbooks/policies stored in a bucket
    instead of only local disk)
  - Secrets Manager for LLM API keys / webhook URLs, instead of relying on
    plaintext environment variables in production

Uses boto3. All methods degrade to clear, actionable errors if credentials
or the target resource aren't available, rather than silently returning
empty data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class S3DocumentSource:
    """Lists and downloads documents from an S3 bucket/prefix for ingestion."""

    def __init__(self, bucket: str, prefix: str = "", region_name: Optional[str] = None):
        self.bucket = bucket
        self.prefix = prefix
        self._s3 = boto3.client("s3", region_name=region_name)

    def list_documents(self, suffixes: tuple = (".pdf", ".docx", ".html", ".htm", ".txt")) -> List[str]:
        """Return S3 keys under the configured prefix matching supported extensions."""
        keys: List[str] = []
        paginator = self._s3.get_paginator("list_objects_v2")

        try:
            for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if key.lower().endswith(suffixes):
                        keys.append(key)
        except NoCredentialsError as exc:
            raise RuntimeError(
                "No AWS credentials found. Configure via environment variables, "
                "an IAM role, or ~/.aws/credentials."
            ) from exc
        except ClientError as exc:
            raise RuntimeError(f"Failed to list objects in s3://{self.bucket}/{self.prefix}: {exc}") from exc

        logger.info("Found %d ingestible objects in s3://%s/%s", len(keys), self.bucket, self.prefix)
        return keys

    def download_to(self, key: str, local_dir: str) -> str:
        """Download a single S3 object to local_dir, returning the local path."""
        local_path = Path(local_dir) / Path(key).name
        local_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._s3.download_file(self.bucket, key, str(local_path))
        except ClientError as exc:
            raise RuntimeError(f"Failed to download s3://{self.bucket}/{key}: {exc}") from exc

        logger.info("Downloaded s3://%s/%s -> %s", self.bucket, key, local_path)
        return str(local_path)

    def sync_all(self, local_dir: str) -> List[str]:
        """Download every ingestible document under the prefix. Returns local paths."""
        local_paths = []
        for key in self.list_documents():
            local_paths.append(self.download_to(key, local_dir))
        return local_paths


class SecretsManagerClient:
    """
    Fetches secrets (LLM API keys, Power Automate webhook URLs, ITSM
    credentials) from AWS Secrets Manager instead of plaintext env vars —
    the standard for the "secure, scalable, observable, and maintainable"
    requirement in a real AWS-hosted deployment.
    """

    def __init__(self, region_name: Optional[str] = None):
        self._client = boto3.client("secretsmanager", region_name=region_name)

    def get_secret(self, secret_id: str) -> str:
        try:
            response = self._client.get_secret_value(SecretId=secret_id)
        except NoCredentialsError as exc:
            raise RuntimeError("No AWS credentials found for Secrets Manager access.") from exc
        except ClientError as exc:
            raise RuntimeError(f"Failed to retrieve secret '{secret_id}': {exc}") from exc

        return response.get("SecretString", "")
