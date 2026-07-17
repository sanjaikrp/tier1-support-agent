"""
app/tools/sql_tool.py

The SQL Executor Tool — the Researcher Agent's only route to the ticket
database.

Design (per the approved Technical Implementation Document):
  1. The agent supplies a natural-language description of what to search for.
  2. A dedicated LLM call translates that into a single SQL SELECT statement.
  3. The statement is validated (SELECT-only, no stacked statements) and run.
  4. Matching rows are returned to the agent as readable text.

Safety (NFR-03): only SELECT statements are permitted. The generated SQL must
begin with SELECT and must contain no semicolons, which prevents stacked
statements such as "SELECT 1; DROP TABLE tickets".

Auditability: the generated SQL and the raw rows retrieved are recorded in a
module-level capture store so the Researcher Agent can report the exact query
executed (ResearcherOutput.query_executed) and assemble results from real
database rows rather than model-reconstructed text.

Note: the capture store is not safe for concurrent requests. This is acceptable
for the sequential benchmark protocol used in this project and is documented as
a known limitation.
"""

import re
import psycopg2
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

from app.config import (
    DATABASE_URL, ANTHROPIC_API_KEY, ANTHROPIC_MODEL, ANTHROPIC_TEMP,
)

# A dedicated small LLM call for NL -> SQL translation.
# Same model and temperature as every other call in the system (experimental control).
_llm = ChatAnthropic(
    model=ANTHROPIC_MODEL,
    temperature=ANTHROPIC_TEMP,
    max_tokens=256,
    api_key=ANTHROPIC_API_KEY,
)

SCHEMA_DESCRIPTION = """
Table: tickets
Columns:
  id                      INT     - unique ticket number
  category                VARCHAR - one of: bug, config, access
  severity                VARCHAR - one of: low, medium, high, critical
  title                   TEXT    - short headline
  description             TEXT    - full problem description
  root_cause              TEXT    - the diagnosed root cause
  resolution              TEXT    - how it was fixed
  file_path               TEXT    - relevant source file (NULL for access tickets)
  created_at              TIMESTAMP
  resolved_at             TIMESTAMP
  resolution_time_minutes INT
"""

# ── Capture store: records what the last tool invocation actually did ─────
_capture = {"sql": None, "rows": []}


def reset_capture() -> None:
    """Clear the capture store. Called before each Researcher Agent run."""
    _capture["sql"] = None
    _capture["rows"] = []


def get_capture() -> dict:
    """Return the SQL executed and the rows retrieved by the last invocation."""
    return {"sql": _capture["sql"], "rows": list(_capture["rows"])}


def _clean_sql(raw: str) -> str:
    """Strip markdown fences and trailing semicolons from the model's output."""
    sql = raw.strip()
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip().rstrip(";").strip()


def _validate_sql(sql: str) -> None:
    """Enforce the read-only safety contract. Raises ValueError if unsafe."""
    if not sql.upper().startswith("SELECT"):
        raise ValueError(f"Unsafe SQL rejected (not a SELECT): {sql[:80]}")
    if ";" in sql:
        raise ValueError(f"Unsafe SQL rejected (stacked statement): {sql[:80]}")


def _translate_to_sql(natural_language_query: str, category: str) -> str:
    """Use the LLM to turn a natural-language request into a safe SELECT.

    The prompt instructs the model to decompose the request into individual
    keywords rather than matching multi-word phrases. Multi-word ILIKE patterns
    such as '%file upload%error%' require those terms to appear adjacently in a
    single column and therefore almost never match real ticket text.
    """
    prompt = f"""Given this PostgreSQL schema:
{SCHEMA_DESCRIPTION}

Write a single PostgreSQL SELECT query to find past tickets related to:
"{natural_language_query}"

CRITICAL RULES for building the WHERE clause:
- Extract 2 to 4 individual DISTINCTIVE SINGLE WORDS from the request.
  Choose specific technical nouns (e.g. upload, timeout, session, permission).
  Ignore generic words such as: error, issue, problem, fail, failing, the, a, on, with.
- Each keyword must be a SINGLE WORD. Never put two words inside one ILIKE pattern.
- Build one ILIKE condition per keyword against title, and one against description.
- Join ALL keyword conditions with OR (never AND).
- Always filter by category = '{category}' and combine it with AND.

Correct example for the request "file upload failing with an error":
SELECT id, category, severity, title, description, root_cause, resolution, file_path
FROM tickets
WHERE category = 'bug'
AND (title ILIKE '%upload%' OR description ILIKE '%upload%'
     OR title ILIKE '%file%' OR description ILIKE '%file%')
LIMIT 5

WRONG (never do this): title ILIKE '%file upload%error%'

Select these columns: id, category, severity, title, description, root_cause, resolution, file_path
LIMIT the result to 5 rows.
Return ONLY the SQL query. No explanation, no markdown fences, no semicolon.
"""
    response = _llm.invoke(prompt)
    sql = _clean_sql(response.content)
    _validate_sql(sql)
    return sql


@tool
def sql_executor_tool(query: str, category: str) -> str:
    """Search the historical ticket database for past resolved incidents.

    Args:
        query: Natural-language description of what to search for, e.g.
               "file upload failing with 500 error on large files".
        category: The ticket category to filter by. Must be one of:
                  "bug", "config", or "access".

    Returns:
        Readable text describing matching past tickets and their resolutions,
        or a message stating that no matches were found.
    """
    try:
        sql = _translate_to_sql(query, category)
    except ValueError as e:
        return f"Query rejected by safety check: {e}"

    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute(sql)
        column_names = [desc[0] for desc in cur.description]
        raw_rows = cur.fetchall()
        cur.close()
    except Exception as e:
        return f"Database error while executing query: {e}"
    finally:
        if conn is not None:
            conn.close()

    # Convert rows to dictionaries and record them for the agent to assemble
    rows = [dict(zip(column_names, r)) for r in raw_rows]
    _capture["sql"] = sql
    _capture["rows"] = rows

    if not rows:
        return "No similar past tickets found for that search."

    # Format for the agent to read
    blocks = []
    for r in rows:
        blocks.append(
            f"PAST TICKET {r.get('id')} | category: {r.get('category')} | "
            f"severity: {r.get('severity')}\n"
            f"Title:       {r.get('title')}\n"
            f"Description: {r.get('description')}\n"
            f"Root Cause:  {r.get('root_cause')}\n"
            f"Resolution:  {r.get('resolution')}\n"
            f"File:        {r.get('file_path')}"
        )
    return f"SQL executed: {sql}\n\nFound {len(rows)} matching ticket(s):\n\n" + \
           "\n---\n".join(blocks)
