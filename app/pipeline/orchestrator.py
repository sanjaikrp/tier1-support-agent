"""
app/pipeline/orchestrator.py

Coordinates the full multi-agent pipeline:
    Router -> Researcher -> Coder -> Communicator

Responsibilities:
  - Run each agent in order, threading a PipelinePayload through them.
  - Skip the Coder for access tickets (handled inside run_coder).
  - Record per-agent and total latency.
  - Write a complete structured JSON log of every run for auditability and
    later evaluation (satisfies the logging requirement, FR-08).
"""

import os
import json
import time
from datetime import datetime, timezone

from app.schemas.models import TicketInput, PipelinePayload, ResolutionResponse
from app.agents.router_agent import run_router
from app.agents.researcher_agent import run_researcher
from app.agents.coder_agent import run_coder
from app.agents.communicator_agent import run_communicator
from app.config import LOG_DIR


async def run_multi_agent_pipeline(ticket: TicketInput) -> ResolutionResponse:
    """Run a ticket through all four agents and return the final resolution."""
    total_start = time.perf_counter()
    latencies: dict = {}

    payload = PipelinePayload(original_ticket=ticket)

    # Stage 1: Router (classify)
    payload.router_output = await run_router(ticket, latencies)

    # Stage 2: Researcher (always runs)
    payload.researcher_output = await run_researcher(
        ticket, payload.router_output, latencies
    )

    # Stage 3: Coder (skipped for access tickets, handled inside run_coder)
    payload.coder_output = await run_coder(
        ticket, payload.router_output, latencies
    )

    # Stage 4: Communicator (synthesise)
    response = await run_communicator(payload, latencies, total_start)

    # Log the complete run
    _log_run(ticket, payload, response)

    return response


def _log_run(
    ticket: TicketInput,
    payload: PipelinePayload,
    response: ResolutionResponse,
) -> None:
    """Write a complete structured JSON log of one pipeline run."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    log_path = os.path.join(LOG_DIR, f"{ticket.ticket_id}_{timestamp}_multiagent.json")

    record = {
        "ticket_id": ticket.ticket_id,
        "system": "multi_agent",
        "timestamp_utc": timestamp,
        "input": ticket.model_dump(),
        "router": payload.router_output.model_dump() if payload.router_output else None,
        "researcher": payload.researcher_output.model_dump() if payload.researcher_output else None,
        "coder": payload.coder_output.model_dump() if payload.coder_output else None,
        "response": response.model_dump(),
    }

    try:
        with open(log_path, "w") as f:
            json.dump(record, f, indent=2, default=str)
    except OSError:
        # Logging must never break the pipeline
        pass
