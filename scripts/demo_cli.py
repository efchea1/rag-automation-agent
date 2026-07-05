"""
Interactive CLI demo: ingests the generic sample documents, then lets you
ask questions against them and see the full agent trace.

Uses a real LLM (Anthropic or OpenAI) if an API key is present in the
environment; otherwise falls back to MockLLMClient so the demo still runs.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.agent import RagAgent  # noqa: E402
from rag_agent.llm_client import get_default_client  # noqa: E402

logging.basicConfig(level=logging.WARNING)

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_docs"


def main():
    client = get_default_client()
    print(f"Using LLM backend: {type(client).__name__}")

    agent = RagAgent(llm_client=client)
    n = agent.ingest_path(str(SAMPLE_DIR))
    print(f"Indexed {n} chunks from {SAMPLE_DIR}\n")

    print("Ask a question about the sample docs (password resets, VPN issues, ")
    print("new-hire onboarding). Type 'quit' to exit.\n")

    while True:
        query = input("> ").strip()
        if not query or query.lower() in {"quit", "exit"}:
            break

        response = agent.run(query)

        print(f"\n--- Rewritten query ---\n{response.rewritten_query}")
        print(f"\n--- Answer ---\n{response.answer}")
        print(f"\n--- Triage ---\n{response.triage}")
        print(f"\n--- Action plan ---\n{response.action_plan}")
        print("\n--- Citations ---")
        for c in response.citations:
            print(f"  [{c.score:.2f}] {c.source_path} (chunk {c.chunk_index}): {c.excerpt[:100]}...")
        print()


if __name__ == "__main__":
    main()
