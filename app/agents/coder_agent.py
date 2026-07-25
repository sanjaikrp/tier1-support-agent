"""
app/agents/coder_agent.py

The Coder Agent inspects the target GitHub repository to produce a
code-grounded diagnosis for bug and config tickets.

Design (per the approved Technical Implementation Document):
  - Pattern: ReAct / tool-calling agent with two registered tools
    (github_file_fetcher, github_commit_searcher), capped at 5 iterations.
  - Activation: runs ONLY for 'bug' and 'config' category tickets. For
    'access' tickets it is skipped and returns None (access problems are
    permission issues, not code defects).

File discovery:
  The agent is given the list of available support_demo module paths in its
  system prompt. This mirrors how a support engineer knows the rough layout of
  their own codebase, and keeps the agent focused without an extra repo-listing
  API call. This is a deliberate, documented design choice.

Grounding:
  The agent is instructed to base its diagnosis only on file contents and
  commit history it has actually retrieved, and not to invent file paths,
  function names, or commit hashes.
"""

import json
import time

from langchain_anthropic import ChatAnthropic
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from app.tools.github_tool import github_file_fetcher, github_commit_searcher
from app.schemas.models import TicketInput, RouterOutput, CoderOutput, RelevantFile
from app.config import (
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL, ANTHROPIC_TEMP, ANTHROPIC_MAX_TOKENS,
)

llm = ChatAnthropic(
    model=ANTHROPIC_MODEL,
    temperature=ANTHROPIC_TEMP,
    max_tokens=ANTHROPIC_MAX_TOKENS,
    api_key=ANTHROPIC_API_KEY,
)

tools = [github_file_fetcher, github_commit_searcher]

# The known layout of the target application (documented design choice).
AVAILABLE_FILES = """
support_demo/auth/views.py        - authentication (login, logout)
support_demo/pagination.py        - pagination helpers
support_demo/helpers.py           - response helpers (JSON responses)
support_demo/uploads.py           - file upload handling
support_demo/models/product.py    - product model (ratings)
support_demo/export.py            - data export (CSV, dates)
support_demo/cache.py             - in-memory caching
support_demo/views/contact.py     - contact form handling (redirects)
support_demo/models/stats.py      - statistics counters
support_demo/mailer.py            - email sending (retries)
support_demo/billing.py           - billing / invoice calculations
support_demo/validators.py        - validation and feature flags
support_demo/reports.py           - report building
support_demo/retry.py             - retry / backoff logic
support_demo/security.py          - security (redirects, headers)
"""

SYSTEM_PROMPT = f"""You are a software debugging specialist.

Given a support ticket, identify which source file and function most likely
contains the defect, using the GitHub tools available to you:
  - github_file_fetcher: read a file's contents.
  - github_commit_searcher: see recent commits for a file (commit messages
    often describe what changed).

The target application has these files:
{AVAILABLE_FILES}

Strategy:
1. From the ticket, decide which 1-3 files are most likely relevant.
2. Use github_file_fetcher to read them, and github_commit_searcher to check
   their history.
3. Identify the specific file and function responsible.

Base your diagnosis ONLY on file contents and commit history you have actually
retrieved. Do NOT invent file paths, function names, or commit hashes.

When done, provide your final answer as a JSON object with exactly these keys:
  "diagnostic_hypothesis": a concise explanation of the root cause,
  "primary_file": the single most likely file path,
  "confidence": a number between 0.0 and 1.0.
Return ONLY that JSON object as your final answer."""

USER_PROMPT = """Ticket:
Title: {title}
Description: {description}

Diagnose which file and function contains the defect."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,
    verbose=False,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
)


def _normalise_output(output) -> str:
    """Normalise an AgentExecutor output into a plain string.

    Claude tool-calling can return the final output either as a string or as a
    list of content blocks (dicts with a "text" key). This handles both.
    """
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts = []
        for block in output:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(output)


def _parse_final(output) -> dict:
    """Extract the JSON diagnosis from the agent's final answer text."""
    text = _normalise_output(output).strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```")[1] if "```" in text[3:] else text
        text = text.replace("json", "", 1).strip("`").strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Fallback: find the first {...} block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                pass
    return {}


async def run_coder(
    ticket: TicketInput,
    router: RouterOutput,
    latencies: dict,
) -> CoderOutput | None:
    """Diagnose a bug/config ticket against the codebase. None for access tickets."""
    # Skip for access tickets — not a code problem
    if router.category == "access":
        latencies["coder"] = 0.0
        return None

    start = time.perf_counter()

    try:
        result = await executor.ainvoke({
            "title": ticket.title,
            "description": ticket.description,
        })
    except Exception as e:
        latencies["coder"] = round((time.perf_counter() - start) * 1000, 2)
        return CoderOutput(
            ticket_id=ticket.ticket_id,
            relevant_files=[],
            diagnostic_hypothesis=f"Coder agent error: {e}",
            confidence=0.0,
        )

    latencies["coder"] = round((time.perf_counter() - start) * 1000, 2)

    parsed = _parse_final(result.get("output", ""))

    # Collect the files the agent actually fetched, from its intermediate steps
    fetched_files = []
    for action, observation in result.get("intermediate_steps", []):
        tool_name = getattr(action, "tool", "")
        tool_input = getattr(action, "tool_input", {})
        if tool_name == "github_file_fetcher":
            path = tool_input.get("file_path", "") if isinstance(tool_input, dict) else ""
            if path:
                # Keep a short excerpt of what was retrieved
                excerpt = str(observation)[:400]
                fetched_files.append(
                    RelevantFile(
                        path=path,
                        content_excerpt=excerpt,
                        last_commit_message="",
                        last_commit_date="",
                    )
                )

    return CoderOutput(
        ticket_id=ticket.ticket_id,
        relevant_files=fetched_files,
        diagnostic_hypothesis=parsed.get("diagnostic_hypothesis", "No diagnosis produced."),
        confidence=float(parsed.get("confidence", 0.0)),
    )
