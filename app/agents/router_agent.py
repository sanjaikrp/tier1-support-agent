"""
app/agents/router_agent.py

The Router Agent is the entry point of the multi-agent pipeline.
It classifies an incoming support ticket into one of three categories:
bug, config, or access. It uses a single Claude call with a structured
output parser that enforces the RouterOutput schema.
"""

import time
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from app.schemas.models import TicketInput, RouterOutput
from app.config import (
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL,
    ANTHROPIC_TEMP, ANTHROPIC_MAX_TOKENS,
)

# The LLM client — same settings every agent will use
llm = ChatAnthropic(
    model=ANTHROPIC_MODEL,
    temperature=ANTHROPIC_TEMP,
    max_tokens=ANTHROPIC_MAX_TOKENS,
    api_key=ANTHROPIC_API_KEY,
)

# The parser forces Claude's output into a valid RouterOutput object
parser = PydanticOutputParser(pydantic_object=RouterOutput)

SYSTEM_PROMPT = """You are a technical support ticket classifier.
Classify the given ticket into exactly one category:
  - bug: A software defect causing incorrect or unexpected behaviour.
  - config: A configuration, environment, or dependency problem.
  - access: A permissions, authentication, or authorisation problem.

Provide a confidence score between 0.0 and 1.0 and brief reasoning.
Return a JSON object matching this schema exactly:
{format_instructions}"""

USER_PROMPT = """Ticket ID: {ticket_id}
Title: {title}
Description: {description}
Severity: {severity}

Classify this ticket."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT),
])

# The chain: prompt -> llm -> parser (LangChain pipe syntax)
router_chain = prompt | llm | parser


async def run_router(ticket: TicketInput, latencies: dict) -> RouterOutput:
    """Classify a ticket and record how long it took (in milliseconds)."""
    start = time.perf_counter()
    result = await router_chain.ainvoke({
        "ticket_id":           ticket.ticket_id,
        "title":               ticket.title,
        "description":         ticket.description,
        "severity":            ticket.severity,
        "format_instructions": parser.get_format_instructions(),
    })
    latencies["router"] = round((time.perf_counter() - start) * 1000, 2)
    # Ensure the ticket_id is always correct (the model can occasionally alter it)
    result.ticket_id = ticket.ticket_id
    return result
