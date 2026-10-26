"""Perception/MCQ rule-based reward function for verl.

Adapted from easyR1's examples/reward_function/perception_rule.py to match
verl's reward-manager calling convention:

    compute_score(data_source, solution_str, ground_truth, extra_info) -> float | dict

The original logic (MCQ option extraction, normalized string match, mathruler
`grade_answer` fallback) is preserved. `mathruler` is optional: when unavailable
we fall back to a normalized equality check, so this module remains importable
in environments where mathruler is not installed.

This file is self-contained and does not modify any existing verl code.
"""

from __future__ import annotations

import re
from typing import Any, Optional

try:
    from mathruler.grader import grade_answer as _mathruler_grade  # type: ignore
    _HAS_MATHRULER = True
except Exception:  # pragma: no cover - depends on optional dep
    _HAS_MATHRULER = False
    _mathruler_grade = None


REWARD_NAME = "perceptual"
REWARD_TYPE = "batch"

SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]


def extract_mcq_option(answer: str) -> str:
    """Return the leading MCQ option letter (A-F) if `answer` looks like an MCQ answer, else ''."""
    if not isinstance(answer, str) or not answer:
        return ""
    text = answer.strip()
    pattern = r'^[ (\[]*([A-F])(?:(?=$)|[\.\)\]]|(?:[\:\-]\s+))'
    match = re.match(pattern, text)
    if match:
        return match.group(1)
    return ""


def check_question_is_mcq(question: str) -> bool:
    newlines = re.findall(r'\n', question or "")
    if "A" in (question or "") and len(newlines) >= 2:
        return True
    return False


def extract_first_option(text: str) -> str:
    if not text:
        return ""
    match = re.search(r'\(([A-Z])\)', text)
    if match:
        return match.group(1)
    match = re.search(r'([A-Z])[\.\)\s]', text)
    if match:
        return match.group(1)
    match = re.search(r'([A-Z])', text)
    if match:
        return match.group(1)
    return ""


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final-answer string (ported verbatim from easyR1)."""
    if not isinstance(final_answer, str):
        return ""
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer.strip()


def check_format_match(response: str) -> bool:
    pattern = re.compile(r"<answer>.*?</answer>", re.DOTALL)
    return bool(re.search(pattern, response or ""))


def string_match(response: str, ground_truth: str) -> bool:
    match = re.findall(r"(?i)Answer\s*:\s*([^\n]+)", response or "")
    answer = match[-1] if match else "[INVALID]"
    return normalize_final_answer(answer) == normalize_final_answer(ground_truth)


def _fallback_grade(answer_text: str, ground_truth: str) -> bool:
    """Used when mathruler is unavailable."""
    return normalize_final_answer(answer_text) == normalize_final_answer(ground_truth)


def grade_answer(answer_text: str, ground_truth: str) -> bool:
    if _HAS_MATHRULER:
        try:
            return bool(_mathruler_grade(answer_text, ground_truth))
        except Exception:
            return _fallback_grade(answer_text, ground_truth)
    return _fallback_grade(answer_text, ground_truth)


def _score_one(response: str, ground_truth: str, format_weight: float = 0.0) -> dict[str, float]:
    format_match = check_format_match(response)
    content_match = re.search(r"<answer>(.*?)</answer>", response or "")
    answer_text = content_match.group(1).strip() if content_match else (response or "").strip()
    format_score = 1.0 if format_match else 0.0

    if not answer_text:
        return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}

    if len(answer_text) >= 300:
        accuracy_score = 0.0
        return {
            "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
            "format": format_score,
            "accuracy": accuracy_score,
        }

    if string_match(answer_text, ground_truth):
        accuracy_score = 1.0
        return {
            "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
            "format": format_score,
            "accuracy": accuracy_score,
        }

    mcq_option = extract_mcq_option(ground_truth)
    if mcq_option:
        answer = extract_first_option(answer_text)
        if mcq_option == answer:
            accuracy_score = 1.0
            return {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }

    if grade_answer(answer_text, ground_truth):
        accuracy_score = 1.0
        return {
            "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
            "format": format_score,
            "accuracy": accuracy_score,
        }

    accuracy_score = 0.0
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    format_weight: float = 0.0,
    **kwargs: Any,
) -> dict[str, float]:
    """verl-compatible reward function (single-item).

    Args:
        data_source: dataset source tag (unused but kept for verl signature).
        solution_str: the model's decoded response.
        ground_truth: the reference answer (string).
        extra_info: optional verl extra_info dict (unused).
        format_weight: weight on format score; default 0.0 (accuracy only).

    Returns:
        dict with keys ``score`` (verl reads this), ``overall``, ``format``,
        ``accuracy``, ``acc`` for convenience.
    """
    gt = ground_truth if isinstance(ground_truth, str) else str(ground_truth)
    res = _score_one(solution_str or "", gt, format_weight=format_weight)
    res["score"] = res["overall"]
    res["acc"] = res["accuracy"]
    return res


# Backward-compatible batch entry point matching easyR1's signature, so the same
# module can be used from code that still passes a list of reward-input dicts.
def compute_score_batch(reward_inputs: list[dict[str, Any]], format_weight: float = 0.0) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for item in reward_inputs:
        res = _score_one(item.get("response", ""), item.get("ground_truth", ""), format_weight=format_weight)
        out.append(res)
    return out