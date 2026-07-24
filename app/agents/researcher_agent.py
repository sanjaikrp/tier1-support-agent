"""
app/agents/researcher_agent.py

The Researcher Agent queries the historical ticket database to find past
incidents similar to the incoming ticket.

Design (per the approved Technical Implementation Document):
  - Pattern: ReAct / tool-calling agent with a single registered tool
    (sql_executor_tool), capped at 3 iterations to prevent runaway loops.
  - The agent reasons about WHAT to search for and invokes the tool.

Grounding decision:
  The returned similar_tickets are assembled IN CODE from the rows the tool
  actually retrieved (via the tool's capture store), not re-typed by the model.
  This guarantees that every reported past ticket corresponds to a real database
  row. In a project whose central metric is hallucination rate, allowing the
  model to restate retrieved records would introduce an avoidable source of
  fabrication.

similarity_score:
  Computed deterministically as the Jaccard coefficient between the token sets
  of the incoming ticket text and each retrieved ticket's title+description:
      J(A, B) = |A intersect B| / |A union B|
  This yields a reproducible, mathematically defined value in [0, 1] rather than
  a model-invented number. The approved document defines the field but does not
  specify its derivation; this fills that gap.
"""

import re
import time

from langchain_anthropic import ChatAnthropic
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from app.tools.sql_tool import sql_executor_tool, reset_capture, get_capture
from app.schemas.models import (
    TicketInput, RouterOutput, ResearcherOutput, SimilarTicket,
)
from app.config import (
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL, ANTHROPIC_TEMP, ANTHROPIC_MAX_TOKENS,
)

llm = ChatAnthropic(
    model=ANTHROPIC_MODEL,
    temperature=ANTHROPIC_TEMP,
    max_tokens=ANTHROPIC_MAX_TOKENS,
    api_key=ANTHROPIC_API_KEY,
)

tools = [sql_executor_tool]

SYSTEM_PROMPT = """You are a technical support researcher.

Your job is to search a database of past RESOLVED support tickets to find
incidents similar to the current ticket.

Use the sql_executor_tool. Pass it:
  - query: a short natural-language description of the technical problem,
           focusing on distinctive technical nouns (e.g. "upload timeout",
           "session cookie", "database connection").
  - category: the category given to you below.

If your first search returns no results, try ONE more search using different,
broader keywords. Do not search more than twice.

When you have your results, reply with a one-sentence summary of what you found.
Do not list the tickets in your reply; they are recorded automatically."""

USER_PROMPT = """Current ticket:
Title: {title}
Description: {description}

Category: {category}

Search the database for similar past tickets."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=3,
    verbose=False,
    handle_parsing_errors=True,
)

# Common English and domain words that carry no discriminating signal
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "to", "of", "in", "on", "at", "for", "with", "by", "from", "as",
    "it", "its", "this", "that", "these", "those", "when", "if", "not", "no",
    "i", "we", "you", "they", "he", "she", "my", "our", "their",
    "error", "issue", "problem", "fail", "fails", "failed", "failing",
    "ticket", "user", "users", "does", "do", "did", "can", "cannot",
}


def _tokenise(text: str) -> set:
    """Lowercase, split into word tokens, and drop stopwords and short words."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {w for w in words if len(w) > 2 and w not in _STOPWORDS}


def _jaccard(a: set, b: set) -> float:
    """Jaccard coefficient: |A n B| / |A u B|. Returns 0.0 for empty union."""
    if not a or not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return round(len(a & b) / len(union), 4)


async def run_researcher(
    ticket: TicketInput,
    router: RouterOutput,
    latencies: dict,
) -> ResearcherOutput:
    """Search the ticket database and return grounded, scored results."""
    start = time.perf_counter()

    # Clear any rows captured by a previous run
    reset_capture()

    try:
        await executor.ainvoke({
            "title":       ticket.title,
            "description": ticket.description,
            "category":    router.category,
        })
    except Exception as e:
        # The agent failed; return an empty but valid result rather than crashing
        latencies["researcher"] = round((time.perf_counter() - start) * 1000, 2)
        return ResearcherOutput(
            ticket_id=ticket.ticket_id,
            similar_tickets=[],
            query_executed=f"Researcher agent error: {e}",
        )

    # Assemble results from the rows the tool ACTUALLY retrieved
    capture = get_capture()
    rows = capture["rows"]
    sql = capture["sql"] or "No query was executed."

    ticket_tokens = _tokenise(f"{ticket.title} {ticket.description}")

    similar = []
    for row in rows:
        row_tokens = _tokenise(f"{row.get('title', '')} {row.get('description', '')}")
        similar.append(
            SimilarTicket(
                past_ticket_id=str(row.get("id")),
                description=row.get("description") or "",
                root_cause=row.get("root_cause") or "",
                resolution=row.get("resolution") or "",
                similarity_score=_jaccard(ticket_tokens, row_tokens),
            )
        )

    # Most relevant first; keep the top 3 per the approved design
    similar.sort(key=lambda s: s.similarity_score, reverse=True)
    similar = similar[:3]

    latencies["researcher"] = round((time.perf_counter() - start) * 1000, 2)

    return ResearcherOutput(
        ticket_id=ticket.ticket_id,
        similar_tickets=similar,
        query_executed=sql,
    )
