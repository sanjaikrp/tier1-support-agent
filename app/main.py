"""
app/main.py

The FastAPI application entry point. For now it exposes a single health-check
endpoint so we can confirm the server runs. Agent endpoints are added later.
"""

from fastapi import FastAPI
from app.config import ANTHROPIC_MODEL

app = FastAPI(
    title="Tier 1 Support Automation API",
    description="Multi-agent vs single-prompt LLM benchmark system",
    version="1.0.0",
)


@app.get("/api/v1/health")
async def health_check():
    """Confirms the server is running and reports the configured model."""
    return {"status": "ok", "model": ANTHROPIC_MODEL}
