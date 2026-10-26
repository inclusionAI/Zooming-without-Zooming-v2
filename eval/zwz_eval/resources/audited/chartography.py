# Selected definitions from the project-adapted reference source.
# Not a relicense of the original. See MANIFEST.json and THIRD_PARTY_NOTICES.md.
from __future__ import annotations
import re
import json
from decimal import Decimal, ROUND_HALF_UP
JUDGE_PROMPT_TEMPLATE = """Read the following prompt and response pair and evaluate it against the golden answer.

Prompt: {{ rubric.response_generation_prompt }}
Response: {{ rubric.response_to_grade }}
Golden answer: {{ rubric.all_criteria }}

Return a JSON object according to the specified format. The score must be either 0 or 1. Use "1" if the response matches all parts of the golden answer (see below for more detail). Use "0" if any part of the response does not match the golden answer.  Do not use any other values or decimals.

STEP 1: EXTRACT THE MODEL'S FINAL ANSWER
- Identify the answer the model is actually committing to. The answer may appear at the start or end of the response.
- If the model states one answer and later revises to a different one, use its LAST/final stated answer.
- Ignore all reasoning, explanation, hedging, and formatting. Grade only the substantive answer.

STEP 2: COMPARE TO THE GOLDEN ANSWER
Apply the rule that matches the answer type:

Numerical answer:
- If an acceptable range is given, the model is CORRECT if its value falls within that range (inclusive), INCORRECT otherwise.
- If no range is given (the value is exact/printed), the model is CORRECT only if it matches the golden value (allow trivial rounding and equivalent notation, e.g., "$4.2M" = "4.2 million" = "4,200,000"; "31 percentage points" = "31 pp").
- Do not give credit for being "close" if the value is outside the range. Do not penalize a more precise value that still falls inside the range (e.g., 3.850 is correct for an acceptable range of "3.8 to 3.9").
- Do not penalize models for different formatting, as long as the actual answer is the same

Categorical answer (one or more categories, e.g., a single item or a top-N list):
- CORRECT only if the model identifies exactly the right set: every required category is present and no extra/incorrect ones are added. Order does not matter unless the prompt asked for a ranking (e.g., "Give me the top 3 in order"). If the prompt does ask for an answer in a specific order / ranking, then the order should be an exact match. Accept minor wording/label variants that clearly refer to the same item.
- Do not penalize models for different formatting, as long as the actual answer is the same

Multi-part answer:
- Every part must be correct (all-or-nothing). If any part is wrong, missing, or outside its range, the whole answer is INCORRECT.
- Do not penalize models for different formatting, as long as the actual answer is the same

"Cannot be determined" answers:
- If the golden answer is a value/category but the model says the prompt is unanswerable or refuses, that is INCORRECT.
- If the golden answer is itself "cannot be determined from the chart," the model is CORRECT only if it reaches that same conclusion.

STEP 3: OUTPUT
Return a JSON object with a 'scores' array. Each entry has a 'criterion_id', an 'extracted_answer', a 'score', and a 'rationale'.
- 'criterion_id': use "criterion_n" where n is an integer in order of appearance ("criterion_1", "criterion_2", ...).
- 'extracted_answer': the model's final answer, verbatim or lightly normalized.
- 'score': must be either 1 or 0. Use 1 if the response matches all parts of the golden answer (CORRECT per the rules above). Use 0 if any part of the response does not match the golden answer (INCORRECT per the rules above). Do not use any other values or decimals.
- 'rationale': one or two sentences explaining the comparison and which rule applied.

Be strict and literal about the substantive value, but lenient about formatting, units, phrasing, extra explanation, and equivalent notation. When the golden answer and the model's answer refer to the same value/category within the allowed tolerance, mark it correct."""

JUDGE_GUIDED_JSON = {
    "type": "object",
    "required": ["scores"],
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["criterion_id", "extracted_answer", "score", "rationale"],
                "properties": {
                    "criterion_id": {"type": "string"},
                    "extracted_answer": {"type": "string"},
                    "score": {"type": "integer", "enum": [0, 1]},
                    "rationale": {"type": "string"},
                },
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}

def _judge_prompt(question: str, golden_answer: str, response_text: str) -> str:
    return (
        JUDGE_PROMPT_TEMPLATE
        .replace("{{ rubric.response_generation_prompt }}", question)
        .replace("{{ rubric.response_to_grade }}", response_text)
        .replace("{{ rubric.all_criteria }}", golden_answer)
    )

def _parse_verdict(raw_text: str) -> list[dict] | None:
    '\u5bb9\u9519 parse: \u4f18\u5148 json.loads; \u5931\u8d25\u5c31 find \u7b2c\u4e00\u6bb5 ```json``` \u6216\u6700\u5916\u5c42 { ... }'
    cleaned = raw_text.strip()
    # Preserved upstream logic; see the corresponding protocol documentation.
    cleaned = re.sub(r"\A```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```\Z", "", cleaned)
    # Preserved upstream logic; see the corresponding protocol documentation.
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        # Preserved upstream logic; see the corresponding protocol documentation.
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            return None
        try:
            payload = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    entries = payload.get("scores")
    if not isinstance(entries, list) or not entries:
        return None
    out = []
    for e in entries:
        if not isinstance(e, dict):
            return None
        try:
            # Preserved upstream logic; see the corresponding protocol documentation.
            raw_s = e.get("score", "0")
            if isinstance(raw_s, bool):
                s = 1 if raw_s else 0
            elif isinstance(raw_s, (int, float)):
                s = int(raw_s)
            else:
                s = int(Decimal(str(raw_s).strip()))
        except Exception:
            return None
        out.append({
            "criterion_id": str(e.get("criterion_id", "criterion_1")),
            "extracted_answer": str(e.get("extracted_answer", "")),
            "score": s,
            "rationale": str(e.get("rationale", "")),
        })
    return out

def _verdict_passes(score: int) -> bool:
    return Decimal(score).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP) >= 1
