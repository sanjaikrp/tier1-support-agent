"""
app/agents/communicator_agent.py

The Communicator Agent is the final stage of the pipeline. It synthesises the
outputs of the upstream agents (Router, Researcher, Coder) into a single,
human-readable resolution response.

Design (per the approved Technical Implementation Document):
  - Pattern: a single LLMChain (prompt -> llm -> parser). No tools: its job is
    synthesis over structured inputs, not data retrieval.
  - Output: a validated ResolutionResponse.

Grounding (primary hallucination-suppression mechanism):
  The system prompt instructs the model to base every claim strictly on the
  evidence provided by the upstream agents, to acknowledge explicitly when the
  Researcher found nothing or the Coder was not run (access tickets), and never
  to invent file paths, function names, commit hashes, or past resolutions.
"""

import time

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.schemas.models import PipelinePayload, ResolutionResponse
from app.config import (
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL, ANTHROPIC_TEMP, ANTHROPIC_MAX_TOKENS,
)

llm = ChatAnthropic(
    model=ANTHROPIC_MODEL,
    temperature=ANTHROPIC_TEMP,
    max_tokens=ANTHROPIC_MAX_TOKENS,
    api_key=ANTHROPIC_API_KEY,
)

parser = PydanticOutputParser(pydantic_object=ResolutionResponse)

SYSTEM_PROMPT = """You are a senior technical support analyst writing the final
resolution for a support ticket.

You are given evidence gathered by upstream systems:
  1. The original ticket.
  2. Similar past tickets and their resolutions (from the Researcher).
  3. A code-level diagnosis of the relevant source file (from the Coder), which
     may be absent for access-related tickets.

STRICT GROUNDING RULES:
  - Base every statement ONLY on the evidence provided below.
  - If the Researcher found no similar tickets, say so explicitly.
  - If no code diagnosis is provided (e.g. an access ticket), do not reference
    any source code.
  - NEVER invent file paths, function names, commit hashes, or past resolutions
    that are not present in the evidence.
  - If the evidence is weak or incomplete, set confidence to "low" and say so.

Return a JSON object matching this schema exactly:
{format_instructions}"""

USER_PROMPT = """ORIGINAL TICKET
Ticket ID: {ticket_id}
Title: {title}
Description: {description}
Category (from Router): {category}

RESEARCHER EVIDENCE (similar past tickets)
{researcher_evidence}

CODER EVIDENCE (code-level diagnosis)
{coder_evidence}

Write the final resolution response now."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT),
])

comm_chain = prompt | llm | parser


def _format_researcher(payload: PipelinePayload) -> str:
    r = payload.researcher_output
    if r is None or not r.similar_tickets:
        return "No similar past tickets were found in the database."
    blocks = []
    for t in r.similar_tickets:
        blocks.append(
            f"- Past ticket {t.past_ticket_id} (similarity {t.similarity_score}):\n"
            f"    Root cause: {t.root_cause}\n"
            f"    Resolution: {t.resolution}"
        )
    return "\n".join(blocks)


def _format_coder(payload: PipelinePayload) -> str:
    c = payload.coder_output
    if c is None:
        return "No code diagnosis (not applicable for this ticket category)."
    files = ", ".join(f.path for f in c.relevant_files) or "none"
    return (
        f"Diagnostic hypothesis: {c.diagnostic_hypothesis}\n"
        f"Confidence: {c.confidence}\n"
        f"Files inspected: {files}"
    )


async def run_communicator(
    payload: PipelinePayload,
    latencies: dict,
    total_start: float,
) -> ResolutionResponse:
    """Synthesise the upstream evidence into a final ResolutionResponse."""
    start = time.perf_counter()

    ticket = payload.original_ticket
    category = payload.router_output.category if payload.router_output else "unknown"

    result = await comm_chain.ainvoke({
        "ticket_id":           ticket.ticket_id,
        "title":               ticket.title,
        "description":         ticket.description,
        "category":            category,
        "researcher_evidence": _format_researcher(payload),
        "coder_evidence":      _format_coder(payload),
        "format_instructions": parser.get_format_instructions(),
    })

    latencies["communicator"] = round((time.perf_counter() - start) * 1000, 2)

    # Fill in metadata the model should not be responsible for
    result.ticket_id = ticket.ticket_id
    result.latency_ms = round((time.perf_counter() - total_start) * 1000, 2)
    result.agent_latencies = dict(latencies)
    result.system = "multi_agent"
    return result
