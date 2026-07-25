"""
app/metrics/accuracy.py

Diagnostic Accuracy scorer.

Scores a system response against the ground-truth manifest on a graded 0-2 scale
(per the approved evaluation design):
    2 = Fully correct  : correct root cause AND correct file identified.
    1 = Partially right : correct root cause OR correct file (not both).
    0 = Incorrect       : neither.

The score is computed from two independent checks against the ground truth:
  - file_correct : the correct file path appears in the response text.
  - cause_correct: at least one root-cause keyword appears in the response text.

This keyword/path matching is deterministic and reproducible. Its limitations
(e.g. it rewards mentioning the right file even in a wrong explanation) are
documented so they can be discussed in the evaluation chapter.
"""

from typing import Optional


def score_response(response_text: str, ground_truth: dict) -> int:
    """Score one response against its ground-truth record. Returns 0, 1, or 2."""
    text = (response_text or "").lower()

    # File check: does the correct file path appear in the response?
    correct_file = (ground_truth.get("file_path") or "").lower()
    file_correct = bool(correct_file) and correct_file in text
    # Also accept the bare filename (e.g. "uploads.py") as a partial path match
    if not file_correct and correct_file:
        filename = correct_file.rsplit("/", 1)[-1]
        file_correct = filename in text

    # Cause check: does any root-cause keyword appear?
    keywords = [k.lower() for k in ground_truth.get("root_cause_keywords", [])]
    cause_correct = any(kw in text for kw in keywords)

    if file_correct and cause_correct:
        return 2
    if file_correct or cause_correct:
        return 1
    return 0


def summarise_scores(scores: list) -> dict:
    """Aggregate a list of 0/1/2 scores into summary statistics."""
    n = len(scores)
    if n == 0:
        return {"n": 0, "mean_score": 0.0, "percentage": 0.0,
                "count_2": 0, "count_1": 0, "count_0": 0}
    total = sum(scores)
    return {
        "n": n,
        "mean_score": round(total / n, 3),
        "max_possible": n * 2,
        "percentage": round((total / (n * 2)) * 100, 1),
        "count_2": scores.count(2),
        "count_1": scores.count(1),
        "count_0": scores.count(0),
    }
