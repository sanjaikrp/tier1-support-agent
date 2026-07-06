"""
app/config.py

Central configuration. Reads all settings from the .env file and exposes
them as typed Python constants. Every other module imports from here rather
than reading environment variables directly, so settings live in one place.
"""

import os
from dotenv import load_dotenv

# Load the .env file into the environment
load_dotenv()

# ── Anthropic settings — shared by ALL agents and the baseline ───────────
ANTHROPIC_API_KEY    = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL      = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-3-5-20241022")
ANTHROPIC_TEMP       = float(os.environ.get("ANTHROPIC_TEMPERATURE", "0"))
ANTHROPIC_MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "1024"))

# ── Database settings ────────────────────────────────────────────────────
DB_HOST = os.environ.get("POSTGRES_HOST", "localhost")
DB_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
DB_NAME = os.environ.get("POSTGRES_DB", "tier1_tickets")
DB_USER = os.environ.get("POSTGRES_USER", "tier1_user")
DB_PASS = os.environ.get("POSTGRES_PASSWORD", "")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# ── GitHub settings (used later, in Phase 6) ─────────────────────────────
GITHUB_TOKEN      = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO_OWNER = os.environ.get("GITHUB_REPO_OWNER", "")
GITHUB_REPO_NAME  = os.environ.get("GITHUB_REPO_NAME", "")

# ── Experiment logging ───────────────────────────────────────────────────
LOG_DIR = os.environ.get("LOG_DIR", "results/raw_logs")
