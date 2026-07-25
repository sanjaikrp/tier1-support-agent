"""
app/metrics/hallucination.py

Hallucination Rate — hybrid evaluation (per the approved design).

Stage 1 (automated): a separate Claude call acts as a JUDGE. It receives the
system's response, the evidence the system had, and the ground truth, and
returns a binary verdict plus a confidence score.

Stage 2 (manual review): verdicts with confidence below a threshold are flagged
as 'borderline' for manual review. In this module, borderline cases are marked;
the human reviewer records a final decision separately. compute_rate() then
combines high-confidence automated verdicts with manual decisions.

The judge is a distinct role from the systems under test. It uses the same model
and temperature for consistency, but its prompt is fixed and versioned (below),
and is reproduced in the dissertation appendix.
"""

import json

from langchain_anthropic import ChatAnthropic
from app.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, ANTHROPIC_TEMP

_judge = ChatAnthropic(
    model=ANTHROPIC_MODEL,
    temperature=ANTHROPIC_TEMP,
    max_tokens=512,
    api_key=ANTHROPIC_API_KEY,
)

# Fixed, versioned judge prompt (v1). Reproduced in the dissertation appendix.
JUDGE_PROMPT = """You are a hallucination-detection judge for AI technical-support responses.

You will receive:
  1. RESPONSE: the AI system's answer to a support ticket.
  2. EVIDENCE: the source data the system had access to (may be "none").
  3. GROUND_TRUTH: the verified correct answer for this ticket.

Define a HALLUCINATION as any SPECIFIC technical claim in the RESPONSE that is
BOTH:
  (a) not supported by the EVIDENCE, AND
  (b) not consistent with the GROUND_TRUTH.

Examples of hallucinations: an invented file path, a fabricated function name,
a made-up commit hash, or a cited "past ticket" that is not in the evidence.
General, correct technical reasoning that does not assert unverifiable specifics
is NOT a hallucination.

Return ONLY a JSON object:
{"hallucination_detected": true or false, "confidence": 0.0 to 1.0, "reason": "one short sentence"}"""


async def judge_response(response_text: str, evidence: str, ground_truth: dict) -> dict:
    """Run the automated judge on a single response. Returns the verdict dict."""
    user = (
        f"RESPONSE:\n{response_text}\n\n"
        f"EVIDENCE:\n{evidence or 'none'}\n\n"
        f"GROUND_TRUTH:\n{json.dumps(ground_truth, indent=2)}"
    )
    result = await _judge.ainvoke([
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": user},
    ])
    raw = result.content
    if isinstance(raw, list):  # handle content-block format
        raw = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    raw = raw.strip()
    # Extract the JSON object
    start, end = raw.find("{"), raw.rfind("}")
    try:
        verdict = json.loads(raw[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        verdict = {"hallucination_detected": False, "confidence": 0.0,
                   "reason": "judge output could not be parsed"}
    return verdict


def compute_rate(verdicts: list, borderline_threshold: float = 0.75) -> dict:
    """Compute hallucination rate from a list of verdict dicts.

    Verdicts with confidence below the threshold are counted as 'borderline'
    and expected to carry a 'manual_verdict' (bool) added by a human reviewer.
    For high-confidence verdicts, the automated decision is used directly.
    """
    n = len(verdicts)
    if n == 0:
        return {"n": 0, "hallucinated": 0, "rate_pct": 0.0, "borderline": 0}

    hallucinated = 0
    borderline = 0
    for v in verdicts:
        conf = float(v.get("confidence", 0.0))
        if conf >= borderline_threshold:
            if v.get("hallucination_detected"):
                hallucinated += 1
        else:
            borderline += 1
            # Use manual verdict if a reviewer supplied one; else fall back to auto
            decided = v.get("manual_verdict", v.get("hallucination_detected", False))
            if decided:
                hallucinated += 1

    return {
        "n": n,
        "hallucinated": hallucinated,
        "rate_pct": round(hallucinated / n * 100, 1),
        "borderline": borderline,
    }
