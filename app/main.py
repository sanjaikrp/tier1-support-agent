"""
app/main.py

The FastAPI application entry point. Exposes:
  - GET  /api/v1/health           : server health check
  - POST /api/v1/classify         : Router Agent only (single-stage demo)
  - POST /api/v1/resolve          : full multi-agent pipeline (experimental)
  - POST /api/v1/baseline/resolve : single-prompt baseline (control)
"""

from fastapi import FastAPI, HTTPException

from app.config import ANTHROPIC_MODEL
from app.schemas.models import TicketInput, RouterOutput, ResolutionResponse
from app.agents.router_agent import run_router
from app.pipeline.orchestrator import run_multi_agent_pipeline
from app.baseline.single_prompt import run_baseline

app = FastAPI(
    title="Tier 1 Support Automation API",
    description="Multi-agent vs single-prompt LLM benchmark system",
    version="1.0.0",
)


@app.get("/api/v1/health")
async def health_check():
    """Confirms the server is running and reports the configured model."""
    return {"status": "ok", "model": ANTHROPIC_MODEL}


@app.post("/api/v1/classify", response_model=RouterOutput)
async def classify_ticket(ticket: TicketInput) -> RouterOutput:
    """Run only the Router Agent on a ticket (single-stage demonstration)."""
    try:
        latencies = {}
        return await run_router(ticket, latencies)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/resolve", response_model=ResolutionResponse)
async def resolve_ticket(ticket: TicketInput) -> ResolutionResponse:
    """Run a ticket through the full multi-agent pipeline (experimental condition)."""
    try:
        return await run_multi_agent_pipeline(ticket)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/baseline/resolve", response_model=ResolutionResponse)
async def resolve_ticket_baseline(ticket: TicketInput) -> ResolutionResponse:
    """Run a ticket through the single-prompt baseline (control condition)."""
    try:
        return await run_baseline(ticket)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
