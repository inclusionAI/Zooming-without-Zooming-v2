import re
import time

from .answer_norm import _norm, norm_match
from .call_qwen_vl import call_qwen_vl
from .llm import extract_json


AUDIT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail", "uncertain"]},
        "answer_complete": {"type": "boolean"},
        "question_unambiguous": {"type": "boolean"},
        "evidence_sufficient": {"type": "boolean"},
        "independent_answer": {"type": "string", "maxLength": 200},
        "issues": {
            "type": "array",
            "items": {"type": "string", "maxLength": 160},
            "maxItems": 5,
        },
    },
    "required": [
        "verdict", "answer_complete", "question_unambiguous",
        "evidence_sufficient", "independent_answer", "issues",
    ],
    "additionalProperties": False,
}


# These tasks have a larger semantic or reasoning error surface; they are
# flagged in the audit result so downstream consumers can review them, though
# the Qwen-VL screen is the single auditor.
HIGH_RISK_QUESTION_TYPES = {
    "ocr_visual_verification",
    "spatial_chain",
    "dependent_chain",
    "viewpoint_transform",
    "agent_relative_direction",
    "geometric_line_relation",
    "topological_relation",
    "depth_ordering",
    "spatial_transitivity",
    "cross_image_composition",
    "arithmetic_composition",
    "conditional_composition",
    "boolean_composition",
    "relational_composition",
    "multi_hop_reasoning",
}


_MULTI_OUTPUT_QUESTION = re.compile(
    r"\b(?:and\s+(?:what|which|who|where|report|state|give|provide|identify)|"
    r"both|respectively)\b",
    re.IGNORECASE,
)
_NEGATION_OR_ALTERNATIVE = re.compile(
    r"\b(?:not|no|isn't|isnt|aren't|arent|neither|unknown|unclear|or)\b",
    re.IGNORECASE,
)
_GENERIC_WRAPPER_WORDS = {
    "a", "an", "the", "is", "it", "appears", "to", "be",
    "answer", "region", "box", "outline", "area", "sign", "brand",
    "label", "text", "word", "number", "code", "shape", "color",
    "coloured", "colored", "object", "item", "window", "vehicle",
    "person", "man", "woman", "building", "creamery",
}


def _set_atoms(text):
    raw = str(text or "").strip().lower()
    parts = re.split(r"\s*(?:\+|,|;|、|，|\band\b|\以及\s*|和)\s*", raw)
    atoms = [_norm(part) for part in parts]
    return sorted(atom for atom in atoms if atom)


def _reference_match(independent_answer, voted_answer, candidate):
    """Return (matched, mode) using conservative, answer-type-aware rules."""
    if norm_match(independent_answer, voted_answer):
        return True, "normalized_exact"

    answer = _norm(independent_answer)
    reference = _norm(voted_answer)
    if not answer or not reference:
        return False, "none"

    # OCR identifiers commonly vary only in separators: TRF.718 / TRF 718.
    # Require both letters and digits so decimal numbers remain strict.
    def identifier_key(value):
        return re.sub(r"[._\-/]", "", value)

    if (re.search(r"[a-z]", answer) and re.search(r"\d", answer)
            and re.search(r"[a-z]", reference) and re.search(r"\d", reference)
            and identifier_key(answer) == identifier_key(reference)):
        return True, "ocr_identifier"

    semantics = str(candidate.get("answer_semantics") or "").lower()
    answer_kind = str(candidate.get("final_answer_kind") or "").lower()
    if semantics in {"set", "set_complete"} or answer_kind in {
        "semantic_set", "anchor_set", "set"
    }:
        answer_atoms = _set_atoms(independent_answer)
        reference_atoms = _set_atoms(voted_answer)
        if answer_atoms and answer_atoms == reference_atoms:
            return True, "unordered_set"
        return False, "none"

    # Permit a short scalar answer wrapped by harmless type words, e.g.
    # "SoCo creamery" for "SoCo". Never apply this to multi-output prompts,
    # negations, alternatives, or numeric references.
    question = str(candidate.get("question") or "")
    if (semantics == "scalar"
            and answer_kind in {"semantic", "text", "anchor"}
            and not _MULTI_OUTPUT_QUESTION.search(question)
            and not _NEGATION_OR_ALTERNATIVE.search(str(independent_answer))
            and not re.fullmatch(r"-?\d+(?:\.\d+)?", reference)):
        answer_words = re.findall(r"[a-z0-9]+", str(independent_answer).lower())
        reference_words = re.findall(r"[a-z0-9]+", str(voted_answer).lower())
        remaining = list(answer_words)
        for word in reference_words:
            if word not in remaining:
                break
            remaining.remove(word)
        else:
            if remaining and all(word in _GENERIC_WRAPPER_WORDS for word in remaining):
                return True, "scalar_wrapper"

    return False, "none"


def _high_risk_reasons(candidate):
    reasons = []
    question_type = str(candidate.get("question_type") or "").strip().lower()
    if question_type in HIGH_RISK_QUESTION_TYPES:
        reasons.append(f"question_type:{question_type}")
    answer_kind = str(candidate.get("final_answer_kind") or "").strip().lower()
    answer_space = str(candidate.get("answer_space") or "").strip().lower()
    if answer_kind in {"semantic", "text"} or answer_space == "open":
        reasons.append("open_answer")
    if (len(candidate.get("evidence_regions") or []) >= 6
            and str(candidate.get("necessity_mode") or "") == "chain_dependency"):
        reasons.append("large_dependency_chain")
    return reasons


def _audit_prompt(candidate):
    region_count = len(candidate.get("evidence_regions") or [])
    vocabulary = [
        str(value).strip() for value in (candidate.get("answer_vocabulary") or [])
        if str(value).strip()
    ]
    vocabulary_text = (
        "\nAllowed neutral answer labels: " + " | ".join(vocabulary)
        if vocabulary else ""
    )
    return f"""You are the final accuracy auditor for a visual-question dataset.

The image is a privileged layout containing {region_count} evidence crops placed in
their original relative positions. Colored outlines are reference markers, not object
attributes. Independently inspect the pixels and audit this proposed sample.

Question: {candidate.get('question', '')}
Answer semantics: {candidate.get('answer_semantics', 'scalar')}
{vocabulary_text}

No proposed reference answer is provided. Solve the question independently from the
pixels. Do not guess what another model may have answered.

Check all of the following:
1. The question is objective, unambiguous, and uniquely answerable from the image.
2. Your independent answer completely covers every output explicitly requested.
3. Your answer has the requested scalar/set/pair structure.
4. The question does not rely on invisible, speculative, or unreadable details.

Do not judge whether every crop is necessary; a separate drop-one test handles that.
Use verdict=fail only when there is a definite objective defect. Use uncertain when
the pixels, wording, or your answer cannot be assessed confidently. Return JSON only:
{{
  "verdict": "pass", "fail", or "uncertain",
  "answer_complete": true or false,
  "question_unambiguous": true or false,
  "evidence_sufficient": true or false,
  "independent_answer": "complete short answer",
  "issues": ["short objective issue"]
}}"""


def _parse_audit_response(raw):
    parsed = extract_json(raw, dict)
    required_keys = set(AUDIT_RESPONSE_SCHEMA["required"])
    if not isinstance(parsed, dict) or not required_keys.issubset(parsed):
        return None, "missing_required_fields"
    if set(parsed) != required_keys:
        return None, "unexpected_fields"
    if parsed.get("verdict") not in {"pass", "fail", "uncertain"}:
        return None, "invalid_verdict"
    for key in ("answer_complete", "question_unambiguous", "evidence_sufficient"):
        if type(parsed.get(key)) is not bool:
            return None, f"invalid_boolean:{key}"
    answer = parsed.get("independent_answer")
    if not isinstance(answer, str) or len(answer) > 200:
        return None, "invalid_independent_answer"
    if parsed["verdict"] == "pass" and not answer.strip():
        return None, "empty_passing_answer"
    issues = parsed.get("issues")
    if (not isinstance(issues, list) or len(issues) > 5
            or any(not isinstance(issue, str) or len(issue) > 160 for issue in issues)):
        return None, "invalid_issues"
    return parsed, None


def _normalize_result(model_name, raw, voted_answer, candidate, latency_seconds):
    parsed, parse_error = _parse_audit_response(raw)
    if parse_error:
        return {
            "model": model_name,
            "decision": "uncertain",
            "pass": False,
            "parse_error": parse_error,
            "latency_seconds": round(latency_seconds, 3),
            "raw": raw,
        }
    independent_answer = parsed.get("independent_answer", "")
    reference_match, reference_match_mode = _reference_match(
        independent_answer, voted_answer, candidate
    )
    quality_flags_pass = all(
        parsed.get(key) is True for key in (
            "answer_complete", "question_unambiguous", "evidence_sufficient"
        )
    )
    if parsed["verdict"] == "fail":
        decision = "fail"
    elif parsed["verdict"] == "pass" and quality_flags_pass and reference_match:
        decision = "pass"
    else:
        # A plausible but different answer is treated as uncertain rather than
        # discarded outright; this protects valid open-answer paraphrases that
        # the single Qwen-VL auditor could not match exactly.
        decision = "uncertain"
    return {
        "model": model_name,
        "decision": decision,
        "pass": decision == "pass",
        "verdict": parsed.get("verdict"),
        "answer_complete": parsed.get("answer_complete"),
        "question_unambiguous": parsed.get("question_unambiguous"),
        "evidence_sufficient": parsed.get("evidence_sufficient"),
        "independent_answer": independent_answer,
        "reference_match": reference_match,
        "reference_match_mode": reference_match_mode,
        "issues": parsed.get("issues"),
        "latency_seconds": round(latency_seconds, 3),
        "raw": raw,
    }


def audit_candidate(candidate, voted_answer, privileged_layout_path, api_key,
                    qwen_vl_model="Qwen3.5-VL-72B",
                    timeout=180, retries=2):
    """Single-tier audit: Qwen-VL independently screens the sample."""
    prompt = _audit_prompt(candidate)
    high_risk_reasons = _high_risk_reasons(candidate)
    generation_source = str(candidate.get("generation_source") or "qwen").lower()

    def run_qwen_vl():
        started = time.monotonic()
        try:
            raw = call_qwen_vl(
                prompt, privileged_layout_path, api_key=api_key,
                model=qwen_vl_model, timeout=timeout, retries=retries,
                # The audit JSON has 6 required fields plus an issues array;
                # qwen_vl tends to wrap output in ```json fences with generous
                # whitespace. 512 was truncating mid-object (raw len ~124) and
                # failing the required-fields check. 2048 leaves comfortable
                # headroom for a complete, parseable response.
                max_tokens=2048,
            )
            result = _normalize_result(
                qwen_vl_model, raw, voted_answer, candidate,
                time.monotonic() - started
            )
            result["audit_mode"] = "screen"
            return result
        except Exception as exc:
            return {
                "model": qwen_vl_model,
                "decision": "uncertain",
                "pass": False,
                "error": str(exc),
                "latency_seconds": round(time.monotonic() - started, 3),
                "audit_mode": "screen",
            }

    started = time.monotonic()
    qwen_vl_result = run_qwen_vl()
    qwen_vl_decision = qwen_vl_result.get("decision")

    if qwen_vl_decision == "pass":
        passed = True
        route = "qwen_vl_pass"
    elif qwen_vl_decision == "fail":
        passed = False
        route = "qwen_vl_reject"
    else:
        # Uncertain: no auditor could commit, so conservatively reject.
        passed = False
        route = "qwen_vl_uncertain"

    return {
        "pass": passed,
        "reason": "ok" if passed else "premium_audit_failed",
        "route": route,
        "generation_source": generation_source,
        "high_risk_reasons": high_risk_reasons,
        "qwen_vl": qwen_vl_result,
        "wall_latency_seconds": round(time.monotonic() - started, 3),
    }
