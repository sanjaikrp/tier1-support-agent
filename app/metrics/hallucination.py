"""
app/metrics/hallucination.py

Hallucination Rate — hybrid evaluation (per the approved design).

Stage 1 (automated): a separate Claude call acts as a JUDGE. It receives the
system's response, the evidence the system had, and the ground truth, and
returns a binary verdict plus a confidence score.

Stage 2 (manual review): verdicts with confidence below a threshold are flagged
as 'borderline' for manual review. compute_rate() combines high-confidence
automated verdicts with any manual decisions.

Robust parsing: the judge is instructed to return strict JSON, but models
occasionally wrap it or include awkward characters in the free-text 'reason'.
_extract_verdict recovers the boolean and confidence even from imperfect JSON,
so valid judgements are not lost as parse errors.
"""

import json
import re

from langchain_anthropic import ChatAnthropic
from app.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, ANTHROPIC_TEMP

_judge = ChatAnthropic(
    model=ANTHROPIC_MODEL,
    temperature=ANTHROPIC_TEMP,
    max_tokens=512,
    api_key=ANTHROPIC_API_KEY,
)

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

Return ONLY a JSON object on a single line, with no code fences, in exactly this
form:
{"hallucination_detected": true, "confidence": 0.0, "reason": "..."}
Keep the reason under 200 characters and avoid using double quotes inside it."""


def _extract_verdict(raw) -> dict:
    """Robustly extract the verdict from the judge's raw output.

    Handles: content-block lists, markdown fences, and reason fields containing
    awkward characters. Falls back to regex extraction of the boolean and
    confidence if full JSON parsing fails, so a genuine judgement is never lost.
    """
    # Normalise to a string
    if isinstance(raw, list):
        raw = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    text = str(raw).strip()
    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()

    # First attempt: clean JSON object
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end + 1]
        try:
            obj = json.loads(candidate)
            return {
                "hallucination_detected": bool(obj.get("hallucination_detected", False)),
                "confidence": float(obj.get("confidence", 0.0)),
                "reason": str(obj.get("reason", ""))[:300],
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass  # fall through to regex recovery

    # Recovery: pull the boolean and confidence out with regex
    detected = None
    m = re.search(r'"?hallucination_detected"?\s*:\s*(true|false)', text, re.IGNORECASE)
    if m:
        detected = m.group(1).lower() == "true"
    conf = 0.0
    mc = re.search(r'"?confidence"?\s*:\s*([0-9]*\.?[0-9]+)', text)
    if mc:
        try:
            conf = float(mc.group(1))
        except ValueError:
            conf = 0.0
    # The reason is whatever the model wrote; keep the text as the reason
    mr = re.search(r'"?reason"?\s*:\s*"?(.+?)"?\s*}?\s*$', text, re.DOTALL)
    reason = (mr.group(1)[:300] if mr else text[:300])

    if detected is not None:
        return {"hallucination_detected": detected, "confidence": conf, "reason": reason}

    # Genuinely unparseable — surface as an explicit error verdict
    return {"hallucination_detected": False, "confidence": 0.0,
            "reason": f"unparseable judge output: {text[:150]}"}


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
    return _extract_verdict(result.content)


def compute_rate(verdicts: list, borderline_threshold: float = 0.75) -> dict:
    """Compute hallucination rate from a list of verdict dicts."""
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
            decided = v.get("manual_verdict", v.get("hallucination_detected", False))
            if decided:
                hallucinated += 1

    return {
        "n": n,
        "hallucinated": hallucinated,
        "rate_pct": round(hallucinated / n * 100, 1),
        "borderline": borderline,
    }
