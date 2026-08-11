"""
benchmark/run_judge_symmetric.py

Phase 10c (part 2) — the SYMMETRIC hallucination judge pass.

This complements run_judge_pass.py. Where that script judged each system
against the REAL evidence it retrieved (which is asymmetric, because the
baseline retrieves nothing), THIS script judges BOTH systems against the
SAME standard: ground truth only. Every response, regardless of system, is
evaluated by whether its specific claims contradict the known correct answer.

Purpose: a fair, like-for-like hallucination comparison to report ALONGSIDE
the evidence-grounded rate. Reporting both is more rigorous than either alone:
  - evidence-grounded rate  -> reveals the multi-agent evidence-overstatement
                               failure mode (from run_judge_pass.py)
  - symmetric rate (this)   -> fair comparison under an identical standard

Output:
  results/hallucination_verdicts_symmetric.json

Usage (from project root, venv active):
    python -m benchmark.run_judge_symmetric
    python -m benchmark.run_judge_symmetric --limit 10
"""

import argparse
import asyncio
import json

from app.metrics.hallucination import judge_response, compute_rate

RESULTS_FILE = "results/benchmark_results.json"
OUTPUT_FILE = "results/hallucination_verdicts_symmetric.json"

# The evidence given to the judge is identical for every response: none.
# This forces the judge to assess each response purely on whether its specific
# claims are consistent with the ground truth — the same bar for both systems.
SYMMETRIC_EVIDENCE = (
    "No retrieved evidence is provided. Judge the response solely on whether "
    "its specific technical claims are consistent with the GROUND_TRUTH below. "
    "Treat any specific claim that contradicts the ground truth as a hallucination."
)


def _load_results() -> list:
    with open(RESULTS_FILE) as f:
        return json.load(f)


async def main(limit: int | None) -> None:
    results = _load_results()
    if limit:
        results = results[:limit]

    verdicts = []
    total = len(results)
    print(f"Symmetric judging of {total} responses (ground-truth-only, both systems)...")
    print("-" * 60)

    for i, record in enumerate(results, 1):
        if record.get("response") is None:
            continue

        response = record["response"]
        response_text = response.get("response_text", "")
        ground_truth = record.get("ground_truth", {})

        try:
            verdict = await judge_response(response_text, SYMMETRIC_EVIDENCE, ground_truth)
        except Exception as e:  # noqa: BLE001
            verdict = {"hallucination_detected": False, "confidence": 0.0,
                       "reason": f"judge error: {e}"}

        vrec = {
            "query_id": record["query_id"],
            "system": record["system"],
            "replication": record["replication"],
            "category": record.get("category"),
            "difficulty": record.get("difficulty"),
            "hallucination_detected": verdict.get("hallucination_detected"),
            "judge_confidence": verdict.get("confidence"),
            "judge_reason": verdict.get("reason"),
        }
        verdicts.append(vrec)

        flag = "HALLUC" if verdict.get("hallucination_detected") else "clean "
        print(f"  [{i:3}/{total}] {record['system']:11} {record['query_id']:8} "
              f"rep{record['replication']} {flag} conf={verdict.get('confidence')}")

        with open(OUTPUT_FILE, "w") as f:
            json.dump(verdicts, f, indent=2, default=str)

    print("-" * 60)
    print(f"Judged {len(verdicts)} responses. Saved to {OUTPUT_FILE}")

    print("\n=== SYMMETRIC HALLUCINATION RATE (ground-truth-only, both systems) ===")
    for system in ["multi_agent", "baseline"]:
        sub = [v for v in verdicts if v["system"] == system]
        rate = compute_rate([
            {"hallucination_detected": v["hallucination_detected"],
             "confidence": float(v["judge_confidence"] or 0.0)}
            for v in sub
        ])
        print(f"{system:12} {rate}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(main(args.limit))
