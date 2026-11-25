"""
run_answer_eval.py
------------------
Manual answer-quality check. Uses the real LLM, so it needs ANTHROPIC_API_KEY
(or another key RagAgent supports). Not part of the pytest run.

Cost: 18 questions x 4 LLM calls (rewrite, answer, triage, action plan) on
the default model, typically well under US$0.50 per run.

Runs every question in eval/questions.jsonl through the real RagAgent.run()
over sample_docs/ (plus eval/corpus/ with --with-distractors) and checks:

  Answerable questions
    cites_something      the answer contains at least one inline [source] citation
    cites_only_retrieved every inline citation names a document that was
                         actually retrieved for this run (no invented sources)
    cites_expected       at least one inline citation is an expected document
  Unanswerable questions
    declined             the answer says the documents don't cover it
                         (phrase heuristic; read the saved answers to confirm)
  All questions
    rewrite_hit@5        retrieval after the LLM query rewrite, scored the same
                         way as run_retrieval_eval.py, to compare with the
                         offline (no rewrite) numbers

Every answer is saved to eval/results/answer_eval_<date>.jsonl for review.
The decline check is a heuristic: skim the saved answers before quoting it.

Usage:
  python eval/run_answer_eval.py
  python eval/run_answer_eval.py --with-distractors
  python eval/run_answer_eval.py --mock   # wiring check only, scores are meaningless
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import List, Set

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:  # load the key from .env when python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from rag_agent.agent import RagAgent  # noqa: E402
from rag_agent.llm_client import MockLLMClient, get_default_client  # noqa: E402

_spec = importlib.util.spec_from_file_location("run_retrieval_eval", ROOT / "eval" / "run_retrieval_eval.py")
retrieval_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retrieval_eval)

_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")
_FILE_RE = re.compile(r"[\w./\\-]+\.(?:pdf|docx|html?|txt|md)", re.IGNORECASE)

DECLINE_PATTERNS = [
    r"does(?: not|n't) (?:contain|include|cover|mention|address|provide|specify|say)",
    r"do(?: not|n't) (?:contain|include|cover|mention|address|provide|specify|say)",
    r"(?:is|are)(?: not|n't) (?:covered|mentioned|addressed|included|specified|available)",
    r"no (?:information|details|mention|guidance)",
    r"not (?:able|possible) to (?:answer|find|determine)",
    r"(?:unable|cannot|can't|could not|couldn't) (?:to )?(?:answer|find|determine|locate)",
    r"outside (?:the )?(?:scope|retrieved)",
]
_DECLINE_RE = re.compile("|".join(DECLINE_PATTERNS), re.IGNORECASE)


def cited_docs(answer: str) -> Set[str]:
    names = set()
    for inside in _CITATION_RE.findall(answer):
        for path in _FILE_RE.findall(inside):
            names.add(Path(path.replace("\\", "/")).name)
    return names


def declined(answer: str) -> bool:
    return bool(_DECLINE_RE.search(answer))


def _rate(rows: List[dict], key: str) -> str:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return f"{sum(vals)}/{len(vals)}" if vals else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--with-distractors", action="store_true", help="also index eval/corpus/")
    parser.add_argument("--mock", action="store_true", help="use MockLLMClient (wiring check only)")
    args = parser.parse_args()

    client = MockLLMClient() if args.mock else get_default_client()
    if isinstance(client, MockLLMClient) and not args.mock:
        raise SystemExit("No API key found. Set ANTHROPIC_API_KEY (or add it to .env), or pass --mock.")

    agent = RagAgent(llm_client=client)
    agent.ingest_path(str(retrieval_eval.SAMPLE_DIR))
    if args.with_distractors:
        agent.ingest_path(str(retrieval_eval.DISTRACTOR_DIR))

    questions = retrieval_eval.load_questions()
    rows = []
    for q in questions:
        resp = agent.run(q["question"])
        retrieved = {Path(c.source_path).name for c in resp.citations}
        cited = cited_docs(resp.answer)

        row = {
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "rewritten_query": resp.rewritten_query,
            "retrieved_docs": sorted(retrieved),
            "cited_docs": sorted(cited),
            "answer": resp.answer,
            "cites_something": None,
            "cites_only_retrieved": None,
            "cites_expected": None,
            "declined": None,
            "rewrite_hit@5": None,
        }
        if q["type"] == "unanswerable":
            row["declined"] = declined(resp.answer)
        else:
            row["cites_something"] = bool(cited)
            row["cites_only_retrieved"] = cited <= retrieved if cited else None
            row["cites_expected"] = bool(cited & set(q["expected_docs"]))
            ranked = [r.chunk for r in agent._retriever.search(resp.rewritten_query, top_k=agent.top_k)]
            rank = retrieval_eval._first_hit_rank(q, ranked)
            row["rewrite_hit@5"] = rank is not None
        rows.append(row)
        print(f"{q['id']:4} {q['type']:13} cited={sorted(cited)} declined={row['declined']}")

    suffix = "_distractors" if args.with_distractors else ""
    suffix += "_MOCK" if args.mock else ""
    out = retrieval_eval.EVAL_DIR / "results" / f"answer_eval_{date.today().isoformat()}{suffix}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    print()
    print(f"Model client: {type(client).__name__}" + ("  (MOCK: scores are meaningless)" if args.mock else ""))
    print(f"Answerable ({len(answerable)}):")
    print(f"  cites at least one source      {_rate(answerable, 'cites_something')}")
    print(f"  cites only retrieved documents {_rate(answerable, 'cites_only_retrieved')}")
    print(f"  cites an expected document     {_rate(answerable, 'cites_expected')}")
    print(f"  retrieval hit@5 after rewrite  {_rate(answerable, 'rewrite_hit@5')}")
    print(f"Unanswerable ({len(unanswerable)}):")
    print(f"  declined (heuristic)           {_rate(unanswerable, 'declined')}")
    print(f"\nSaved every answer to {out.relative_to(ROOT)}; skim them before quoting the decline rate.")


if __name__ == "__main__":
    main()
