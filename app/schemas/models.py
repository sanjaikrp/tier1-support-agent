"""
app/schemas/models.py

Defines ALL data structures used across the system. Every agent, tool,
and API route imports its data shapes from here. This is the single
source of truth for what data looks like as it flows through the pipeline.

Pydantic BaseModel classes validate types at runtime: if an agent produces
data that doesn't match the schema, an error is raised immediately rather
than silently passing bad data downstream.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime


# - INPUT: what arrives at the API
class TicketInput(BaseModel):
    """A support ticket submitted by an engineer."""
    ticket_id: str
    title: str
    description: str
    severity: Literal["low", "medium", "high", "critical"]


# - ROUTER OUTPUT 
class RouterOutput(BaseModel):
    """The Router Agent's classification decision."""
    ticket_id: str
    category: Literal["bug", "config", "access"]
    confidence: float
    routing_notes: str


# - RESEARCHER OUTPUT 
class SimilarTicket(BaseModel):
    """One past resolved ticket retrieved from the database."""
    past_ticket_id: str
    description: str
    root_cause: str
    resolution: str
    similarity_score: float


class ResearcherOutput(BaseModel):
    """The Researcher Agent's findings from the ticket database."""
    ticket_id: str
    similar_tickets: List[SimilarTicket]
    query_executed: str


# - CODER OUTPUT (None for access tickets)
class RelevantFile(BaseModel):
    """One source file retrieved from GitHub."""
    path: str
    content_excerpt: str
    last_commit_message: str
    last_commit_date: str


class CoderOutput(BaseModel):
    """The Coder Agent's code-grounded diagnosis."""
    ticket_id: str
    relevant_files: List[RelevantFile]
    diagnostic_hypothesis: str
    confidence: float


# - PIPELINE PAYLOAD: passed between all agents 
class PipelinePayload(BaseModel):
    """
    The central object that flows through the whole pipeline.
    Starts with only the ticket; each agent fills in its output field.
    """
    original_ticket: TicketInput
    router_output: Optional[RouterOutput] = None
    researcher_output: Optional[ResearcherOutput] = None
    coder_output: Optional[CoderOutput] = None
    start_time: datetime = Field(default_factory=datetime.utcnow)


# - FINAL RESPONSE: returned by either system
class ResolutionResponse(BaseModel):
    """The final resolution returned by the multi-agent or baseline system."""
    ticket_id: str
    root_cause_summary: str
    recommended_fix: str
    supporting_evidence: List[str]
    confidence: Literal["high", "medium", "low"]
    response_text: str
    latency_ms: float = 0.0
    agent_latencies: dict = Field(default_factory=dict)
    system: Literal["multi_agent", "baseline"] = "multi_agent"
