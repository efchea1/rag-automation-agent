"""
Retrieval regression test, built on eval/run_retrieval_eval.py.

Offline and deterministic: no API key, no network. Fails if retrieval quality
drops below floors set slightly under the first measured scores (2026-11-24),
so small tuning changes don't break the build but real regressions do.

First measured recall@3:
  demo corpus         single 0.89 (8/9)   multi 0.75 (3/4)
  with distractors    single 0.78 (7/9)   multi 0.75 (3/4)
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("run_retrieval_eval", ROOT / "eval" / "run_retrieval_eval.py")
retrieval_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(retrieval_eval)

# (corpus, question type) -> minimum recall@3
FLOORS = {
    ("demo", "single"): 0.80,
    ("demo", "multi"): 0.50,
    ("distractors", "single"): 0.70,
    ("distractors", "multi"): 0.50,
}


@pytest.fixture(scope="module")
def report():
    return retrieval_eval.run_all()


def test_labels_are_sound():
    questions = retrieval_eval.load_questions()
    demo = retrieval_eval.build_chunks([retrieval_eval.SAMPLE_DIR])
    distractors = retrieval_eval.build_chunks([retrieval_eval.DISTRACTOR_DIR])
    assert retrieval_eval.check_labels(questions, demo, distractors) == []


def test_question_set_shape():
    questions = retrieval_eval.load_questions()
    ids = [q["id"] for q in questions]
    assert len(ids) == len(set(ids)), "duplicate question ids"
    for q in questions:
        assert q["type"] in ("single", "multi", "unanswerable")
        if q["type"] == "single":
            assert len(q["expected_docs"]) == 1 and q["answer_phrase"]
        if q["type"] == "multi":
            assert len(q["expected_docs"]) >= 2
        if q["type"] == "unanswerable":
            assert q["expected_docs"] == []


@pytest.mark.parametrize("corpus,qtype", sorted(FLOORS))
def test_recall_at_3_floor(report, corpus, qtype):
    score = report["corpora"][corpus]["summary"][qtype]["recall@3"]
    floor = FLOORS[(corpus, qtype)]
    assert score >= floor, f"{corpus}/{qtype} recall@3 {score:.2f} fell below floor {floor:.2f}"
