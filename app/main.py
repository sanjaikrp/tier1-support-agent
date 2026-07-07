"""
app/main.py

The FastAPI application entry point. Exposes:
  - GET  /api/v1/health         : server health check
  - POST /api/v1/classify       : run the Router Agent on a ticket

More endpoints (full pipeline, baseline) are added in later phases.
"""

from fastapi import FastAPI, HTTPException
from app.config import ANTHROPIC_MODEL
from app.schemas.models import TicketInput, RouterOutput
from app.agents.router_agent import run_router

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
    """
    Run the Router Agent on a single ticket and return its classification.
    This demonstrates the first stage of the multi-agent pipeline.
    """
    try:
        latencies = {}
        result = await run_router(ticket, latencies)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
