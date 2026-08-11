"""
benchmark/run_judge_pass.py

Phase 10c — the hallucination judge pass.

For every response collected in results/benchmark_results.json, this script
determines the REAL evidence the system had access to and asks the judge
(app.metrics.hallucination.judge_response) whether the response contains
hallucinated (ungrounded, incorrect-specific) claims.

Evidence reconstruction (Option C — the rigorous approach):
  - multi_agent: the ACTUAL Researcher similar_tickets and Coder diagnosis /
    inspected files, read from the matching raw log in results/raw_logs/.
  - baseline: no retrieval by design, so evidence = "none". This asymmetry is
    intentional and reflects the architectural difference under study.

Matching strategy:
  Benchmark records are keyed by (query_id, system, replication). Raw logs are
  named TICKETID_TIMESTAMP_system.json. For each (query_id, system) we sort the
  matching logs by timestamp and align them to replications 1..N in order. This
  is robust because the benchmark ran sequentially, so the k-th log for a ticket
  corresponds to the k-th replication that completed.

  Note: dry-run logs for the first few bug queries may add extra logs. We handle
  this by taking the LAST N logs per (query_id, system) where N = number of
  replications actually present in the results file, because the benchmark run
  happened after any dry run. (If counts still mismatch, we fall back to
  ground-truth-only judging for that record and flag it.)

Output:
  results/hallucination_verdicts.json  — one verdict per response, plus the
  reconstructed evidence used, so every judgement is auditable.

Usage (from project root, venv active, API key in .env):
    python -m benchmark.run_judge_pass
    python -m benchmark.run_judge_pass --limit 10     # test on first 10
"""

import argparse
import asyncio
import glob
import json
import os
from collections import defaultdict

from app.metrics.hallucination import judge_response, compute_rate

RESULTS_FILE = "results/benchmark_results.json"
RAW_LOGS_DIR = "results/raw_logs"
OUTPUT_FILE = "results/hallucination_verdicts.json"


def _load_results() -> list:
    with open(RESULTS_FILE) as f:
        return json.load(f)


def _index_raw_logs() -> dict:
    """Return {(query_id, system): [sorted log dicts by timestamp]}."""
    index = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(RAW_LOGS_DIR, "*.json"))):
        fn = os.path.basename(path)
        if fn.endswith("_multiagent.json"):
            system = "multi_agent"
        elif fn.endswith("_baseline.json"):
            system = "baseline"
        else:
            continue
        query_id = fn.split("_")[0]
        # timestamp is embedded; sorted() on filename sorts by it since format is fixed
        try:
            with open(path) as f:
                log = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        index[(query_id, system)].append((fn, log))
    # Sort each list by filename (chronological due to fixed timestamp format)
    for key in index:
        index[key].sort(key=lambda t: t[0])
    return index


def _reconstruct_evidence(system: str, log: dict | None) -> str:
    """Build the evidence string the judge will see, from the REAL run log."""
    if system == "baseline":
        return "none (the baseline system performs no retrieval by design)"
    if log is None:
        return "none available"

    parts = []
    researcher = log.get("researcher")
    if researcher and researcher.get("similar_tickets"):
        lines = ["RETRIEVED PAST TICKETS (from the ticket database):"]
        for t in researcher["similar_tickets"]:
            lines.append(
                f"  - Past ticket {t.get('past_ticket_id')} "
                f"(similarity {t.get('similarity_score')}): "
                f"root cause: {t.get('root_cause','')}"
            )
        parts.append("\n".join(lines))
    else:
        parts.append("RETRIEVED PAST TICKETS: none found.")

    coder = log.get("coder")
    if coder:
        files = ", ".join(f.get("path", "") for f in coder.get("relevant_files", [])) or "none"
        parts.append(
            "CODE INSPECTION (from the GitHub repository):\n"
            f"  Files actually inspected: {files}\n"
            f"  Coder diagnosis: {coder.get('diagnostic_hypothesis','')}"
        )
    else:
        parts.append("CODE INSPECTION: not performed (access ticket or skipped).")

    return "\n\n".join(parts)


async def main(limit: int | None) -> None:
    results = _load_results()
    if limit:
        results = results[:limit]

    raw_index = _index_raw_logs()

    # Figure out replications present per (query_id, system) in the results
    reps_present = defaultdict(list)
    for r in results:
        reps_present[(r["query_id"], r["system"])].append(r["replication"])
    for key in reps_present:
        reps_present[key] = sorted(reps_present[key])

    verdicts = []
    total = len(results)
    print(f"Judging {total} responses...")
    print("-" * 60)

    for i, record in enumerate(results, 1):
        query_id = record["query_id"]
        system = record["system"]
        replication = record["replication"]

        # Skip error records (no response to judge)
        if record.get("response") is None:
            continue

        response = record["response"]
        response_text = response.get("response_text", "")
        ground_truth = record.get("ground_truth", {})

        # --- Match this record to its raw log ---
        logs = raw_index.get((query_id, system), [])
        matched_log = None
        reps = reps_present[(query_id, system)]
        n_reps = len(reps)
        # Align: take the LAST n_reps logs (benchmark ran after any dry run),
        # then pick the one at this replication's position.
        if logs:
            aligned = [lg for (_fn, lg) in logs[-n_reps:]] if len(logs) >= n_reps else [lg for (_fn, lg) in logs]
            try:
                pos = reps.index(replication)
                if pos < len(aligned):
                    matched_log = aligned[pos]
            except ValueError:
                matched_log = None

        evidence = _reconstruct_evidence(system, matched_log)

        # --- Run the judge ---
        try:
            verdict = await judge_response(response_text, evidence, ground_truth)
        except Exception as e:  # noqa: BLE001
            verdict = {
                "hallucination_detected": False,
                "confidence": 0.0,
                "reason": f"judge error: {e}",
            }

        vrec = {
            "query_id": query_id,
            "system": system,
            "replication": replication,
            "category": record.get("category"),
            "difficulty": record.get("difficulty"),
            "hallucination_detected": verdict.get("hallucination_detected"),
            "judge_confidence": verdict.get("confidence"),
            "judge_reason": verdict.get("reason"),
            "evidence_used": evidence,
            "matched_log": matched_log is not None,
        }
        verdicts.append(vrec)

        flag = "HALLUC" if verdict.get("hallucination_detected") else "clean "
        conf = verdict.get("confidence", 0.0)
        print(f"  [{i:3}/{total}] {system:11} {query_id:8} rep{replication} "
              f"{flag} conf={conf}")

        # Incremental save
        with open(OUTPUT_FILE, "w") as f:
            json.dump(verdicts, f, indent=2, default=str)

    print("-" * 60)
    print(f"Judged {len(verdicts)} responses. Saved to {OUTPUT_FILE}")

    # Quick summary
    print("\n=== HALLUCINATION RATE BY SYSTEM ===")
    for system in ["multi_agent", "baseline"]:
        sub = [v for v in verdicts if v["system"] == system]
        rate = compute_rate([
            {"hallucination_detected": v["hallucination_detected"],
             "confidence": float(v["judge_confidence"] or 0.0)}
            for v in sub
        ])
        print(f"{system:12} {rate}")
    unmatched = sum(1 for v in verdicts if not v["matched_log"] and v["system"] == "multi_agent")
    if unmatched:
        print(f"\nNote: {unmatched} multi_agent records could not be matched to a raw log "
              f"and were judged with limited evidence.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(main(args.limit))
