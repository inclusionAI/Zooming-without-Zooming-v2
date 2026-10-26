# Selected definitions from the project-adapted reference source.
# Not a relicense of the original. See MANIFEST.json and THIRD_PARTY_NOTICES.md.
from __future__ import annotations
import re
import json
from decimal import Decimal, ROUND_HALF_UP
JUDGE_SYSTEM = """You are an exact-answer audit judge. Do NOT solve the visual
question yourself and do NOT infer missing information. Compare the assistant's
written response with the provided reference answer. Mark correct only when the
assistant unambiguously states an answer equivalent to the reference answer.
An unfinished, ambiguous, contradictory, or differently formatted movement
answer is incorrect unless it is clearly semantically equivalent.

Return exactly one JSON object with these fields:
{"verdict":"correct"|"incorrect","recovered_answer":"short string or empty","reason":"brief"}
"""

def parse_json(content: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.S | re.I)
    candidate = fenced.group(1) if fenced else content.strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            raise
        value = json.loads(match.group(0))
    verdict = str(value.get("verdict", "")).lower().strip()
    if verdict not in {"correct", "incorrect"}:
        raise ValueError(f"Invalid verdict: {verdict!r}")
    return {
        "verdict": verdict,
        "recovered_answer": str(value.get("recovered_answer", "")).strip(),
        "reason": str(value.get("reason", "")).strip(),
    }
