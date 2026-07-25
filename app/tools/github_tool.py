"""
app/tools/github_tool.py

Two tools that let the Coder Agent read the target GitHub repository via the
GitHub REST API:

  - github_file_fetcher:    returns the contents of a source file.
  - github_commit_searcher: returns the recent commit history for a file path.

Authentication uses the fine-grained Personal Access Token from .env, which
raises the rate limit from 60 to 5,000 requests/hour (Risk 2 mitigation).

Cost control (NFR-04): file contents are truncated to 500 lines before being
returned to the agent, bounding token usage.
"""

import base64
import requests

from langchain_core.tools import tool
from app.config import GITHUB_TOKEN, GITHUB_REPO_OWNER, GITHUB_REPO_NAME

_BASE_URL = f"https://api.github.com/repos/{GITHUB_REPO_OWNER}/{GITHUB_REPO_NAME}"
_HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
_MAX_LINES = 500
_TIMEOUT = 15  # seconds


@tool
def github_file_fetcher(file_path: str) -> str:
    """Fetch the contents of a source file from the target GitHub repository.

    Args:
        file_path: Path to the file within the repository, for example
                   "support_demo/uploads.py".

    Returns:
        The file's contents (first 500 lines), or a clear error message if the
        file could not be found or retrieved.
    """
    url = f"{_BASE_URL}/contents/{file_path}"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
    except requests.RequestException as e:
        return f"Network error fetching {file_path}: {e}"

    if resp.status_code == 404:
        return f"File not found: {file_path}"
    if resp.status_code == 403:
        return "GitHub API rate limit or permission error (403)."
    if resp.status_code != 200:
        return f"GitHub API error {resp.status_code} for {file_path}."

    payload = resp.json()
    # The contents endpoint returns base64-encoded file content
    if payload.get("encoding") != "base64" or "content" not in payload:
        return f"Unexpected response format for {file_path}."

    try:
        content = base64.b64decode(payload["content"]).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as e:
        return f"Could not decode {file_path}: {e}"

    lines = content.splitlines()
    truncated = "\n".join(lines[:_MAX_LINES])
    note = "" if len(lines) <= _MAX_LINES else \
        f"\n\n[truncated: showing first {_MAX_LINES} of {len(lines)} lines]"
    return f"FILE: {file_path}\n\n{truncated}{note}"


@tool
def github_commit_searcher(file_path: str) -> str:
    """Retrieve the most recent commits that modified a specific file.

    Args:
        file_path: Path to the file within the repository, for example
                   "support_demo/uploads.py".

    Returns:
        A formatted list of up to 5 recent commits (short hash, date, message),
        or a clear message if no commits or an error occurred.
    """
    url = f"{_BASE_URL}/commits"
    params = {"path": file_path, "per_page": 5}
    try:
        resp = requests.get(url, headers=_HEADERS, params=params, timeout=_TIMEOUT)
    except requests.RequestException as e:
        return f"Network error fetching commits for {file_path}: {e}"

    if resp.status_code == 403:
        return "GitHub API rate limit or permission error (403)."
    if resp.status_code != 200:
        return f"GitHub API error {resp.status_code} for commits of {file_path}."

    commits = resp.json()
    if not commits:
        return f"No commits found for {file_path}."

    lines = []
    for c in commits:
        sha = c.get("sha", "")[:7]
        commit = c.get("commit", {})
        message = commit.get("message", "").split("\n")[0]  # first line only
        date = commit.get("author", {}).get("date", "")[:10]
        lines.append(f"[{sha}] {date}: {message}")

    return f"COMMITS for {file_path}:\n" + "\n".join(lines)
