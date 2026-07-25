"""
app/baseline/single_prompt.py

The single-prompt baseline — the experimental CONTROL condition.

A single Claude call receives the ticket and produces a ResolutionResponse in
the identical schema as the multi-agent pipeline. It has NO agents, NO database
access, and NO GitHub access: it can rely only on the model's parametric
knowledge.

Experimental control: this uses the SAME model, temperature, and max_tokens as
every agent in the multi-agent system (from app.config). The ONLY difference
between the two systems is architecture — which is what the study measures.

The system prompt asks the model to be honest about uncertainty, so that any
hallucination observed reflects the architecture's limitations rather than a
prompt that actively encourages fabrication. This keeps the comparison fair.
"""

import os
import json
import time
from datetime import datetime, timezone

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.schemas.models import TicketInput, ResolutionResponse
from app.config import (
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL, ANTHROPIC_TEMP, ANTHROPIC_MAX_TOKENS,
    LOG_DIR,
)

llm = ChatAnthropic(
    model=ANTHROPIC_MODEL,
    temperature=ANTHROPIC_TEMP,
    max_tokens=ANTHROPIC_MAX_TOKENS,
    api_key=ANTHROPIC_API_KEY,
)

parser = PydanticOutputParser(pydantic_object=ResolutionResponse)

SYSTEM_PROMPT = """You are a senior technical support analyst.

A software engineer has submitted a support ticket. Diagnose the problem and
provide a resolution based on your own knowledge.

Be honest about your confidence:
  - If you are not certain of the exact root cause, set confidence to "low".
  - Do not fabricate specific file paths, function names, or past resolution
    precedents that you cannot actually verify.

Return a JSON object matching this schema exactly:
{format_instructions}"""

USER_PROMPT = """Ticket ID: {ticket_id}
Title: {title}
Description: {description}
Severity: {severity}

Provide a complete diagnostic resolution."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT),
])

baseline_chain = prompt | llm | parser


async def run_baseline(ticket: TicketInput) -> ResolutionResponse:
    """Produce a resolution using a single Claude call (no agents/tools)."""
    total_start = time.perf_counter()

    result = await baseline_chain.ainvoke({
        "ticket_id":           ticket.ticket_id,
        "title":               ticket.title,
        "description":         ticket.description,
        "severity":            ticket.severity,
        "format_instructions": parser.get_format_instructions(),
    })

    latency_ms = round((time.perf_counter() - total_start) * 1000, 2)

    # Fill in metadata in code (not the model's responsibility)
    result.ticket_id = ticket.ticket_id
    result.latency_ms = latency_ms
    result.agent_latencies = {"single_prompt": latency_ms}
    result.system = "baseline"

    _log_run(ticket, result)
    return result


def _log_run(ticket: TicketInput, response: ResolutionResponse) -> None:
    """Write a structured JSON log of one baseline run (mirrors the pipeline log)."""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    log_path = os.path.join(LOG_DIR, f"{ticket.ticket_id}_{timestamp}_baseline.json")
    record = {
        "ticket_id": ticket.ticket_id,
        "system": "baseline",
        "timestamp_utc": timestamp,
        "input": ticket.model_dump(),
        "response": response.model_dump(),
    }
    try:
        with open(log_path, "w") as f:
            json.dump(record, f, indent=2, default=str)
    except OSError:
        pass
