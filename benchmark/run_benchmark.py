"""
benchmark/run_benchmark.py

Drives the full experiment. For each query in test_suite.json, sends the ticket
to BOTH systems (multi-agent /resolve and baseline /baseline/resolve) via the
running API, for N replications. Scores diagnostic accuracy against the ground
truth and records latency. Results are saved incrementally so the run is
resumable if interrupted.

Design (per the approved Technical Implementation Document): the benchmark calls
the API endpoints over HTTP using httpx, exactly as an external client would.

Usage (server must be running in another terminal):
    python benchmark/run_benchmark.py --replications 3
    python benchmark/run_benchmark.py --replications 1        # quick validation
    python benchmark/run_benchmark.py --limit 5               # dry run, 5 queries
"""

import argparse
import asyncio
import json
import os
import random
from datetime import datetime, timezone

import httpx

from app.metrics.accuracy import score_response

BASE_URL = "http://localhost:8000/api/v1"
SUITE_FILE = "benchmark/test_suite.json"
RESULTS_DIR = "results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "benchmark_results.json")

# One record per (query, system, replication). Response timeout is generous
# because the multi-agent pipeline can take 15-20 seconds per ticket.
REQUEST_TIMEOUT = 180.0
MAX_RETRIES = 2


def _load_existing() -> list:
    """Load any previously saved results so the run can resume."""
    if os.path.exists(RESULTS_FILE):
        try:
            with open(RESULTS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


def _save(results: list) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)


def _done_keys(results: list) -> set:
    """Set of (query_id, system, replication) already completed."""
    return {(r["query_id"], r["system"], r["replication"]) for r in results}


async def _call(client: httpx.AsyncClient, endpoint: str, ticket: dict) -> dict:
    """Call one endpoint with retries. Returns the parsed JSON response."""
    url = f"{BASE_URL}/{endpoint}"
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await client.post(url, json=ticket, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:  # noqa: BLE001 - we want to retry on any error
            last_error = e
            await asyncio.sleep(2 * attempt)
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {last_error}")


async def run(replications: int, limit: int | None) -> None:
    with open(SUITE_FILE) as f:
        queries = json.load(f)
    if limit:
        queries = queries[:limit]

    results = _load_existing()
    done = _done_keys(results)
    total_planned = len(queries) * 2 * replications
    print(f"Queries: {len(queries)} | Systems: 2 | Replications: {replications}")
    print(f"Planned runs: {total_planned} | Already done: {len(results)}")
    print("-" * 60)

    systems = [("resolve", "multi_agent"), ("baseline/resolve", "baseline")]

    async with httpx.AsyncClient() as client:
        for rep in range(1, replications + 1):
            # Randomise order each replication to avoid order effects
            order = queries[:]
            random.shuffle(order)
            for query in order:
                for endpoint, system in systems:
                    key = (query["query_id"], system, rep)
                    if key in done:
                        continue  # already completed (resume)

                    ticket = query["ticket_input"]
                    try:
                        response = await _call(client, endpoint, ticket)
                    except Exception as e:  # noqa: BLE001
                        print(f"  [rep {rep}] {system:11} {query['query_id']:8} ERROR: {e}")
                        record = {
                            "query_id": query["query_id"],
                            "system": system,
                            "replication": rep,
                            "category": query["category"],
                            "difficulty": query["difficulty"],
                            "error": str(e),
                            "accuracy_score": None,
                            "latency_ms": None,
                            "response": None,
                            "ground_truth": query["ground_truth"],
                        }
                        results.append(record)
                        _save(results)
                        continue

                    # Score accuracy against ground truth
                    gt = query["ground_truth"]
                    score = score_response(response.get("response_text", ""), gt)
                    # Also check category correctness (from router, multi-agent only)
                    record = {
                        "query_id": query["query_id"],
                        "system": system,
                        "replication": rep,
                        "category": query["category"],
                        "difficulty": query["difficulty"],
                        "accuracy_score": score,
                        "latency_ms": response.get("latency_ms"),
                        "confidence": response.get("confidence"),
                        "response": response,
                        "ground_truth": gt,
                    }
                    results.append(record)
                    done.add(key)
                    _save(results)  # incremental save after every call

                    print(f"  [rep {rep}] {system:11} {query['query_id']:8} "
                          f"score={score} latency={response.get('latency_ms')}ms")

    print("-" * 60)
    print(f"Complete. {len(results)} records saved to {RESULTS_FILE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replications", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    asyncio.run(run(args.replications, args.limit))
