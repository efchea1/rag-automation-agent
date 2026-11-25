"""
run_retrieval_eval.py
---------------------
Offline retrieval evaluation. No API key, no network, no cost.

Calls HybridRetriever.search() directly with each raw question from
eval/questions.jsonl, so it measures retrieval WITHOUT the LLM query rewrite
that RagAgent.run() does first.

Scoring
  single  A hit at k means a chunk in the top k comes from the expected
          document AND contains the question's answer_phrase (whitespace- and
          case-insensitive). Labels are phrases, not chunk IDs, so they
          survive changes to chunk size.
  multi   A hit at k means every expected document appears among the top-k
          chunks. Rank = position at which the last expected document first
          appears, so MRR rewards getting all of them early.
  unanswerable questions are skipped here; they are tested by
          run_answer_eval.py, because declining is done by the LLM prompt.

Two corpora are scored:
  demo        sample_docs/ only (what the live demo indexes)
  distractors sample_docs/ + eval/corpus/ (similar-topic documents that make
              retrieval choose between near neighbours)

Usage:
  python eval/run_retrieval_eval.py            # print table, write results JSON
  python eval/run_retrieval_eval.py --verbose  # also print each question's top 3
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag_agent.agent import RagAgent  # noqa: E402
from rag_agent.chunker import Chunk, chunk_text  # noqa: E402
from rag_agent.document_loader import load_directory  # noqa: E402
from rag_agent.retriever import HybridRetriever  # noqa: E402

EVAL_DIR = ROOT / "eval"
QUESTIONS_PATH = EVAL_DIR / "questions.jsonl"
SAMPLE_DIR = ROOT / "sample_docs"
DISTRACTOR_DIR = EVAL_DIR / "corpus"
RESULTS_PATH = EVAL_DIR / "results" / "retrieval_results.json"

KS = (1, 3, 5)

# Use the same settings the agent uses, read from RagAgent's defaults so the
# eval cannot silently drift from the app.
_AGENT_DEFAULTS = {
    name: p.default
    for name, p in inspect.signature(RagAgent.__init__).parameters.items()
    if p.default is not inspect.Parameter.empty
}
_SEARCH_DEFAULTS = {
    name: p.default
    for name, p in inspect.signature(HybridRetriever.search).parameters.items()
    if p.default is not inspect.Parameter.empty
}
SETTINGS = {
    "chunk_size": _AGENT_DEFAULTS["chunk_size"],
    "overlap": _AGENT_DEFAULTS["overlap"],
    "agent_top_k": _AGENT_DEFAULTS["top_k"],
    "bm25_weight": _SEARCH_DEFAULTS["bm25_weight"],
    "vector_weight": _SEARCH_DEFAULTS["vector_weight"],
    "vector_method": "TF-IDF cosine (scikit-learn, English stop words)",
    "query_rewrite": "off (raw question)",
}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def load_questions(path: Path = QUESTIONS_PATH) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_chunks(dirs: List[Path]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for d in dirs:
        for doc in load_directory(d):
            chunks.extend(
                chunk_text(doc, chunk_size=SETTINGS["chunk_size"], overlap=SETTINGS["overlap"])
            )
    return chunks


def _doc_name(chunk: Chunk) -> str:
    return Path(chunk.source_path).name


def _first_hit_rank(q: dict, ranked: List[Chunk]) -> Optional[int]:
    """1-based rank at which the question counts as answered, or None."""
    if q["type"] == "single":
        expected = q["expected_docs"][0]
        phrase = _norm(q["answer_phrase"])
        for i, c in enumerate(ranked, start=1):
            if _doc_name(c) == expected and phrase in _norm(c.text):
                return i
        return None
    if q["type"] == "multi":
        remaining = set(q["expected_docs"])
        for i, c in enumerate(ranked, start=1):
            remaining.discard(_doc_name(c))
            if not remaining:
                return i
        return None
    raise ValueError(f"not scored offline: {q['type']}")


def check_labels(questions: List[dict], demo_chunks: List[Chunk], distractor_chunks: List[Chunk]) -> List[str]:
    """Problems that would make the labels unsound. Empty list means OK."""
    problems = []
    for q in questions:
        if q["type"] != "single":
            continue
        phrase = _norm(q["answer_phrase"])
        if not any(_doc_name(c) == q["expected_docs"][0] and phrase in _norm(c.text) for c in demo_chunks):
            problems.append(f"{q['id']}: answer_phrase not found in any chunk of {q['expected_docs'][0]}")
        for c in distractor_chunks:
            if phrase in _norm(c.text):
                problems.append(f"{q['id']}: answer_phrase also appears in distractor {_doc_name(c)}")
    return problems


def evaluate(questions: List[dict], chunks: List[Chunk]) -> Dict:
    retriever = HybridRetriever(chunks)
    depth = max(KS)
    per_question = []
    for q in questions:
        if q["type"] not in ("single", "multi"):
            continue
        results = retriever.search(
            q["question"],
            top_k=depth,
            bm25_weight=SETTINGS["bm25_weight"],
            vector_weight=SETTINGS["vector_weight"],
        )
        ranked = [r.chunk for r in results]
        rank = _first_hit_rank(q, ranked)
        per_question.append(
            {
                "id": q["id"],
                "type": q["type"],
                "question": q["question"],
                "rank": rank,
                "top3": [f"{_doc_name(c)}#{c.chunk_index}" for c in ranked[:3]],
            }
        )

    summary = {}
    for qtype in ("single", "multi"):
        rows = [r for r in per_question if r["type"] == qtype]
        if not rows:
            continue
        n = len(rows)
        s = {"n": n}
        for k in KS:
            s[f"recall@{k}"] = round(sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n, 3)
        s["mrr"] = round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 3)
        summary[qtype] = s
    return {"summary": summary, "per_question": per_question}


def run_all() -> Dict:
    questions = load_questions()
    demo_chunks = build_chunks([SAMPLE_DIR])
    distractor_chunks = build_chunks([DISTRACTOR_DIR])
    problems = check_labels(questions, demo_chunks, distractor_chunks)
    if problems:
        raise SystemExit("Label problems:\n  " + "\n  ".join(problems))

    demo_docs = len({_doc_name(c) for c in demo_chunks})
    distractor_docs = len({_doc_name(c) for c in distractor_chunks})
    return {
        "date": date.today().isoformat(),
        "settings": SETTINGS,
        "question_counts": {
            t: sum(1 for q in questions if q["type"] == t) for t in ("single", "multi", "unanswerable")
        },
        "corpora": {
            "demo": {
                "documents": demo_docs,
                "chunks": len(demo_chunks),
                **evaluate(questions, demo_chunks),
            },
            "distractors": {
                "documents": demo_docs + distractor_docs,
                "chunks": len(demo_chunks) + len(distractor_chunks),
                **evaluate(questions, demo_chunks + distractor_chunks),
            },
        },
    }


def _table(report: Dict) -> str:
    lines = [
        "| Corpus | Question type | n | recall@1 | recall@3 | recall@5 | MRR |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, corpus in report["corpora"].items():
        label = f"{name} ({corpus['documents']} docs, {corpus['chunks']} chunks)"
        for qtype, s in corpus["summary"].items():
            # a multi-document question needs 2+ chunks, so recall@1 is 0 by definition
            r1 = "n/a" if qtype == "multi" else f"{s['recall@1']:.2f}"
            lines.append(
                f"| {label} | {qtype} | {s['n']} | {r1} | "
                f"{s['recall@3']:.2f} | {s['recall@5']:.2f} | {s['mrr']:.2f} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true", help="print each question's rank and top 3 chunks")
    args = parser.parse_args()

    report = run_all()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Retrieval eval, {report['date']}")
    print("Settings: " + ", ".join(f"{k}={v}" for k, v in report["settings"].items()))
    print()
    print(_table(report))
    print()
    for name, corpus in report["corpora"].items():
        misses = [r for r in corpus["per_question"] if not r["rank"] or r["rank"] > 3]
        if misses or args.verbose:
            print(f"[{name}]")
        rows = corpus["per_question"] if args.verbose else misses
        for r in rows:
            rank = r["rank"] if r["rank"] else "miss"
            print(f"  {r['id']} rank={rank}  top3={r['top3']}  \"{r['question']}\"")
    print(f"\nWrote {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
