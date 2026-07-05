"""
llm_client.py
-------------
Thin, provider-agnostic LLM client interface. The agent never talks to a
specific vendor SDK directly — it talks to `LLMClient`. This is what the
job description calls "integrating LLMs (OpenAI, Anthropic, Microsoft) into
secure, production-grade automation solutions using industry-standard
frameworks": the automation logic shouldn't be rewritten every time the
organization swaps or adds a model provider.

Includes a MockLLMClient with zero external dependencies so the pipeline,
tests, and demo are fully runnable without an API key.
"""

from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    @abstractmethod
    def complete(self, system: str, prompt: str, max_tokens: int = 800, step: str = "generic") -> str:
        """
        Return the model's text completion for a system+user prompt.

        `step` is an explicit tag identifying which stage of the agent
        pipeline is calling (e.g. "rewrite", "answer", "triage", "action").
        Real LLM backends can ignore it; MockLLMClient uses it to return
        stage-appropriate canned output instead of sniffing prompt text,
        which is what a production implementation should also prefer over
        string-matching on prompt content.
        """
        raise NotImplementedError


class AnthropicClient(LLMClient):
    """Calls Claude via the Anthropic Messages API."""

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Export it or pass api_key= explicitly."
            )
        import anthropic  # imported lazily so the package is optional

        self._client = anthropic.Anthropic(api_key=self.api_key)

    def complete(self, system: str, prompt: str, max_tokens: int = 800, step: str = "generic") -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAIClient(LLMClient):
    """Calls an OpenAI chat completion endpoint."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Export it or pass api_key= explicitly.")
        import openai  # imported lazily so the package is optional

        self._client = openai.OpenAI(api_key=self.api_key)

    def complete(self, system: str, prompt: str, max_tokens: int = 800, step: str = "generic") -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""


class AzureOpenAIClient(LLMClient):
    """
    Calls an Azure OpenAI Service deployment. This is the "Microsoft" LLM
    provider named explicitly in the job posting ("Integrate LLMs (OpenAI,
    Anthropic, Microsoft)"). Azure OpenAI uses deployment names rather than
    raw model names, and auth via either an API key or Azure AD
    (DefaultAzureCredential) — both are supported here since enterprise IT
    environments frequently mandate AD-based auth over static keys.
    """

    def __init__(
        self,
        deployment_name: str,
        azure_endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-10-21",
        use_azure_ad: bool = False,
    ):
        self.deployment_name = deployment_name
        self.azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        if not self.azure_endpoint:
            raise RuntimeError(
                "AZURE_OPENAI_ENDPOINT not set. Export it or pass azure_endpoint= explicitly."
            )

        import openai  # imported lazily so the package is optional

        if use_azure_ad:
            from azure.identity import DefaultAzureCredential, get_bearer_token_provider

            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(
                credential, "https://cognitiveservices.azure.com/.default"
            )
            self._client = openai.AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                azure_ad_token_provider=token_provider,
                api_version=api_version,
            )
        else:
            key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
            if not key:
                raise RuntimeError(
                    "AZURE_OPENAI_API_KEY not set (or pass use_azure_ad=True for AD auth)."
                )
            self._client = openai.AzureOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=key,
                api_version=api_version,
            )

    def complete(self, system: str, prompt: str, max_tokens: int = 800, step: str = "generic") -> str:
        response = self._client.chat.completions.create(
            model=self.deployment_name,  # Azure OpenAI addresses deployments, not model names
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""


class MockLLMClient(LLMClient):
    """
    Deterministic, dependency-free stand-in for a real LLM.

    Lets the retrieval/chunking/agent-orchestration logic be fully unit
    tested and demoed without network access or API keys. Swap in
    AnthropicClient / OpenAIClient for production use — nothing else in
    the agent changes.
    """

    def complete(self, system: str, prompt: str, max_tokens: int = 800, step: str = "generic") -> str:
        logger.debug("MockLLMClient.complete called (step=%s)", step)

        if step == "rewrite":
            # naive query expansion: just return the original query, trimmed
            return prompt.strip().splitlines()[-1].strip()

        if step == "triage":
            urgent = any(w in prompt.lower() for w in ("down", "outage", "can't access", "locked out", "urgent"))
            return json.dumps(
                {
                    "category": "incident" if urgent else "informational_query",
                    "urgency": "high" if urgent else "low",
                    "recommended_action": "escalate_to_oncall" if urgent else "answer_with_citations",
                    "rationale": (
                        "Mock classifier: urgency keyword detected in query."
                        if urgent
                        else "Mock classifier: no urgent-incident keywords detected."
                    ),
                }
            )

        if step == "action":
            return (
                "Suggested next steps:\n"
                "1. Share the retrieved passages with the requester.\n"
                "2. If this recurs, consider adding it to the self-service KB.\n"
                "3. No escalation required based on retrieved context."
            )

        if step == "answer":
            return (
                "Based on the retrieved passages, here is a synthesized answer. "
                "[MockLLMClient placeholder response — connect a real LLMClient for live output]"
            )

        return "[MockLLMClient placeholder response for unrecognized step]"


def get_default_client() -> LLMClient:
    """Best-effort factory: prefer Anthropic, then Azure OpenAI, then OpenAI, else mock."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicClient()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Anthropic client unavailable: %s", exc)
    if os.environ.get("AZURE_OPENAI_ENDPOINT") and os.environ.get("AZURE_OPENAI_DEPLOYMENT"):
        try:
            return AzureOpenAIClient(deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Azure OpenAI client unavailable: %s", exc)
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIClient()
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAI client unavailable: %s", exc)
    return MockLLMClient()
