# Selected definitions from the project-adapted reference source.
# Not a relicense of the original. See MANIFEST.json and THIRD_PARTY_NOTICES.md.
from __future__ import annotations
import re
import json
from decimal import Decimal, ROUND_HALF_UP
PROMPT_TEMPLATE = (
    "Your task is to judge whether the response expresses the same meaning "
    "as the answer of a question.\n"
    "The question is: {question}\n"
    "The answer is: {gt}\n"
    "The response is: {response}\n"
    "Please check and compare them and then judge. "
    "If the response is correct, your output should be Yes. "
    "Otherwise, your output should be No. Directly give me your output."
)

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
    "square", "ways", "integers", "dollars", "mph", "inches", "hours", "km",
    "units", "\\ldots", "sue", "points", "feet", "minutes", "digits", "cents",
    "degrees", "cm", "gm", "pounds", "meters", "meals", "edges", "students",
    "childrentickets", "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
    "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}", r"\mathrm{th}",
    r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"', "\\dots",
]

def normalize_final_answer(final_answer: str) -> str:
    'Normalize a final answer (\u5bf9\u9f50 perception_rule.normalize_final_answer)\u3002'
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
        # Preserved upstream logic; see the corresponding protocol documentation.
        # Preserved upstream logic; see the corresponding protocol documentation.
        # Preserved upstream logic; see the corresponding protocol documentation.
        final_answer = final_answer.replace(",", "").lstrip("0") or "0"
    # Preserved upstream logic; see the corresponding protocol documentation.
    return final_answer.strip().lower()

def string_match(predicted: str, ground_truth: str) -> bool:
    '\u5f52\u4e00\u5316\u540e\u505a\u7cbe\u786e\u5b57\u7b26\u4e32\u6bd4\u8f83\uff08\u6570\u5b57 / \u77ed\u6587\u672c\u7b54\u6848\u76f4\u63a5\u547d\u4e2d\uff09\u3002'
    if not isinstance(predicted, str) or not isinstance(ground_truth, str):
        return False
    return normalize_final_answer(predicted) == normalize_final_answer(ground_truth)

def parse_llm_judge_response(text):
    '\u4ece judge \u8f93\u51fa\u4e2d\u63d0\u53d6 Yes/No\uff08\u517c\u5bb9 thinking / \u4ee3\u7801\u5757\uff09\u3002'
    if not text:
        return "No"
    cleaned = text.strip()
    low = cleaned.lower()
    if low in ("yes", "no"):
        return cleaned.capitalize()
    parts = cleaned.split("```")
    after_last_fence = parts[-1].strip() if len(parts) > 1 else ""
    for word in reversed(after_last_fence.split()):
        if word.lower() in ("yes", "no"):
            return word.capitalize()
    tokens = cleaned.split()
    for token in reversed(tokens):
        token_stripped = token.rstrip(".,;:!?")
        if token_stripped.lower() in ("yes", "no"):
            return token_stripped.capitalize()
    return "No"

def extract_answer(model_answer_raw):
    answer = None
    if "<answer>" in model_answer_raw:
        start = model_answer_raw.find("<answer>")
        end = model_answer_raw.find("</answer>")
        if start != -1 and end != -1:
            answer = model_answer_raw[start + len("<answer>"):end].strip()
    if answer is None and "Answer:" in model_answer_raw:
        answer = model_answer_raw[model_answer_raw.find("Answer:"):].strip()
    if answer is None:
        lines = [line.strip() for line in model_answer_raw.strip().split("\n\n") if line.strip()]
        answer = "\n\n".join(lines[-3:]) if lines else model_answer_raw.strip()
    return answer
