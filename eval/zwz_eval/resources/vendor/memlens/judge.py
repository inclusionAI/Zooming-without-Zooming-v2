#!/usr/bin/env python3
'\nMemLens \u5b98\u65b9 LLM-as-Judge \u8bc4\u6d4b\u811a\u672c (\u81ea\u5305\u542b, \u4e0d\u4f9d\u8d56 MEMLENS repo)\n\n\u8f93\u5165: \u7528\u6237 infer jsonl  (\u5b57\u6bb5: sample_id, category, meta.question_subtype, response, model_answer, query)\n\u5224\u5b98: \u8fdc\u7a0b OpenAI-\u517c\u5bb9 vLLM (\u5982 Qwen3.6-35B-A3B, TP=4, 16 GPU \u8282\u70b9\u4e0a\u542f\u7684)\n\u8f93\u51fa: judge_metrics.json + judge_details.json + .jsonl (\u65ad\u70b9\u7eed\u8dd1)\n\n\u5b8c\u5168\u5bf9\u9f50 MEMLENS \u5b98\u65b9 main \u5206\u652f (2026-08 \u65f6\u70b9):\n  - parse_model_output / normalize_for_judge / detect_degenerate   (parse_utils.py)\n  - 5 \u7c7b \xd7 10 task_keys \u7684 CRITERIA / EXAMPLES                     (judge_prompts/*.py)\n  - get_task_key / build_judge_prompt                              (judge_prompts/__init__.py)\n  - _parse_judge_response / _judge_one / evaluate                  (llm_judge.py)\n  - \u8f93\u51fa metrics \u7ed3\u6784 (overall / by_question_type / by_question_subtype / by_task_key)\n  - JSONL resume + ThreadPoolExecutor \u5e76\u53d1\n\n\u5b57\u6bb5\u6620\u5c04 (\u7528\u6237 jsonl -> \u5b98\u65b9 llm_judge \u671f\u671b\u5b57\u6bb5):\n  sample_id            -> question_id\n  query                -> question\n  response             -> reference_answer\n  model_answer         -> prediction\n  category             -> question_type\n  meta.question_subtype-> question_subtype\n  (\u65e0)                 -> old_answer  (bench \u6570\u636e\u96c6\u672a\u66b4\u9732)\n  (\u65e0)                 -> output_len  (\u9ed8\u8ba4 0)\n\n\u53c2\u8003: https://github.com/xrenaf/MEMLENS  (MIT)\n'

import argparse
import io
import json
import os
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

# =============================================================================
# parse_utils.py — official parsing / degenerate detection / tail truncate
# =============================================================================

QWEN_THINK_END = chr(60) + "/think" + chr(62)  # qwen think-end token, built dynamically
def parse_model_output(text: str) -> str:
    """Strip Qwen thinking / special tokens / GLM box tokens / Kimi filter."""
    if not text:
        return ""
    # 1. Qwen thinking trace: keep only text after </think>
    if QWEN_THINK_END in text:
        text = text.split(QWEN_THINK_END, 1)[1]
    # 2. <｜...｜> style tokens
    text = re.sub(r"<\|[^>]*\|>", "", text)
    # 3. box tokens (uncommon; regex above already strips most)
    text = re.sub(r"<\|begin_of_box\|>(.*?)<\|end_of_box\|>", r"\1", text, flags=re.DOTALL)
    # 4. Kimi content filter
    if "[CONTENT_FILTER_REJECTED]" in text:
        text = text.replace("[CONTENT_FILTER_REJECTED]", "")
    return text.strip()


_HEDGING_MARKERS = [
    "wait", "let me re-read", "let me re read", "let me double check",
    "let me double-check", "let me reconsider", "let me look again",
    "let's reconsider", "let s reconsider", "re-evaluating", "re evaluating",
    "actually looking", "actually let me", "hold on",
    "let me look at", "let me check", "let's re-read", "let s re-read",
    "i need to reconsider", "wait let me",
]

TAIL_CHAR_LIMIT = 6000
TAIL_FRACTION = 0.3
MIN_LENGTH_FOR_TRUNCATION = 3000


def detect_degenerate(text: str) -> Tuple[bool, Dict[str, Any]]:
    '\u5b98\u65b9 detect_degenerate: >=6 hedging markers OR <40% unique 5-grams.'
    if not text or len(text) < 500:
        return False, {"hedging_count": 0, "unique_5gram_ratio": 1.0}
    text_lower = text.lower()
    hedging_count = sum(text_lower.count(m) for m in _HEDGING_MARKERS)
    words = text_lower.split()
    n = 5
    if len(words) >= n:
        ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        unique_ratio = len(set(ngrams)) / len(ngrams) if ngrams else 1.0
    else:
        unique_ratio = 1.0
    is_degenerate = (hedging_count >= 6) or (unique_ratio < 0.40)
    return is_degenerate, {"hedging_count": hedging_count, "unique_5gram_ratio": round(unique_ratio, 3)}


def normalize_for_judge(text: str) -> Dict[str, Any]:
    """parse -> detect degenerate -> tail truncate. Returns dict for judge."""
    if not text:
        return {"text": "", "original_len": 0, "was_truncated": False,
                "is_degenerate": False, "diagnostics": {}}
    original_len = len(text)
    text = parse_model_output(text)
    is_degenerate, diagnostics = detect_degenerate(text)
    was_truncated = False
    if len(text) > MIN_LENGTH_FOR_TRUNCATION and len(text) > TAIL_CHAR_LIMIT:
        tail_start = int(len(text) * (1 - TAIL_FRACTION))
        text = "[...truncated earlier reasoning...]\n" + text[tail_start:]
        was_truncated = True
    return {"text": text, "original_len": original_len, "was_truncated": was_truncated,
            "is_degenerate": is_degenerate, "diagnostics": diagnostics}


# =============================================================================
# Preserved upstream logic; see the corresponding protocol documentation.
# =============================================================================

def get_task_key(question_type: str, question_subtype: str, reference: str) -> str:
    '\u5b98\u65b9 get_task_key \u2014 exactly mirrors judge_prompts/__init__.py.'
    if question_type == "multi_session_reasoning":
        if question_subtype == "arithmetic":
            return "MSR_Arithmetic"
        if question_subtype == "counting":
            return "MSR_Counting"
        if question_subtype == "entity_resolution":
            return "MSR_YesNo" if reference.strip() in ("Yes", "No") else "MSR_Counting"
        return "MSR_Counting"
    if question_type == "temporal_reasoning":
        if question_subtype == "duration_comparison":
            return "TR_DurationComparison"
        if question_subtype == "order_ranking":
            return "TR_OrderRanking"
        if question_subtype == "temporal_info_extraction":
            return "TR_DateExtraction"
        return "TR_DateExtraction"
    if question_type == "knowledge_update":
        return "KU_KnowledgeUpdate"
    if question_type == "answer_refusal":
        return "AR_AnswerRefusal"
    if question_type == "information_extraction":
        if question_subtype == "previnfo":
            return "IE_PreviousInfo"
        return "IE_Entity"
    return "IE_Entity"


# ---- judge_prompts/base.py ----

QA_BENCHMARK_JUDGE_PROMPT = """Now your role is a grading teacher. Your task is to review and score student answers based on reference standard answers for a question-answering benchmark. You need to notice the following key points:
- First, extract the final answer from the student's solution, then analyze and judge whether the answer is correct.
- Scoring should only refer to the final answer obtained by the student; there is no need to examine whether the intermediate problem-solving steps are correct.
- If the response contains both hesitation and a clear answer, judge the answer itself.
- If the response contains both a refusal and a guessed answer, judge the final committed answer.
- If the response gives multiple inconsistent answers, assign 0 points.
- If the student's response shows circular reasoning — repeatedly revisiting the same evidence, flip-flopping between answers (e.g., "yes... wait no... actually yes... let me reconsider... no"), with no clear final commitment — assign 0 points. A correct answer mentioned during reasoning does NOT count unless it is the clearly stated final conclusion.
- Only the student's LAST clearly stated answer counts. If the student initially says "yes" but later concludes "no" (or vice versa), score based on the LAST position only. Intermediate answers during reasoning are not final answers.
- If the student's output is entirely a reasoning trace (e.g., "let me check... scanning evidence... re-reading session 5...") with no clear concluding answer statement, assign 0 points — even if the correct answer appears somewhere within the reasoning.
- If the student's output begins with "[...truncated earlier reasoning...]", this means only the final portion of a longer response is shown. Focus on extracting the answer from this final portion.

Below are examples of problematic student outputs that apply to ALL task types. Study these before grading.

[Universal Example A — Circular Reasoning -> 0 points]
<Question>: Is the cat in Mark and Jenny's 5-year anniversary post named Mittens?
<Standard Answer>: Yes
<Student Answer>: let me check session 5 mark has cat named mittens session 7 mark and jenny anniversary post there is no link wait let me re-read session 5 says mittens session 7 says anniversary actually looking at this again wait is it possible let me reconsider mark has mittens in session 5 the post is about couple therefore answer is likely no wait let me double check session 5 mark has cat mittens actually maybe yes but text doesnt say
[Scoring Rationale]: The output flip-flops between "yes" and "no" multiple times with no clear final commitment. The last position is ambiguous ("maybe yes but text doesnt say"). This is circular reasoning.
In summary, the student's answer deserves 0 points.
[JSON]: {{"answer_score": 0}}

[Universal Example B — Redundant But Committed -> Score Normally]
<Question>: Is the trailing plant in the cafe safe for my kitten?
<Standard Answer>: No
<Student Answer>: the plant looks like pothos pothos is toxic to cats therefore answer is no let me double check is pothos safe for cats no is answer no yes wait let me verify image 3 is pothos image 9 is pothos pothos is toxic therefore answer must be no
[Scoring Rationale]: Despite excessive self-verification, the student consistently commits to "no" throughout and never wavers to a different answer. The final answer is clearly "No", which matches the standard answer.
In summary, the student's answer deserves 1 point.
[JSON]: {{"answer_score": 1}}

[Universal Example C — Reasoning Trace With No Answer -> 0 points]
<Question>: How many sessions mentioned the coffee shop?
<Standard Answer>: 3
<Student Answer>: scanning session 1 yes coffee mentioned session 2 no session 3 yes coffee again session 4 no session 5 maybe let me re-read session 5 it mentions cafe is that same as coffee shop need to check session 6 no session 7 unclear let me look at session 5 again
[Scoring Rationale]: The student walks through sessions but never states a final count. The output is entirely a reasoning trace with no concluding answer.
In summary, the student's answer deserves 0 points.
[JSON]: {{"answer_score": 0}}

Now proceed with grading. Remember: the universal examples above apply regardless of task type.

- When analyzing and judging whether the answer is correct, you need to write down the scoring rationale, organize it into clear statements that follow the logical flow. The summary of the scoring rationale should be placed at the end, using the following format: "In summary, the student's answer deserves x points" (where x represents the student's specific score).
- Keep the whole process concise, within 150 words.
- Provide the score based on your analysis and display it in a code block in "JSON" format.
- An item is covered if it is strictly mentioned or unambiguously implied by a semantic equivalence. This includes numerical equivalence (e.g., 10% and 0.1), synonyms (e.g., UK and United Kingdom), plural/singular forms (e.g., "apple" and "apples"), and equivalent date formats (e.g., 2024-01-15 and January 15, 2024). However, do not accept loosely related concepts.
- Ignore minor formatting differences, capitalization, punctuation, and equivalent wording when meaning is unchanged.
Your output format is:
[Scoring Rationale]:
[Score]: x points
[JSON]:
{{"answer_score": <integer_value>}}

Below is the grading rubric:
[Scores]:
The scoring scale consists of 2 levels in total, from highest to lowest: 1 point, 0 points (the minimum is 0 points).
[Tier Details]:
1 point: Assign 1 point if the student's final answer matches the standard answer under the task-specific criteria below.
0 points: Assign 0 points if the student's final answer does not match the standard answer, the student refuses to answer, claims insufficient information (except for AR tasks), or does not clearly answer.

[Task-Specific Criteria]:
{task_criteria}

{examples}
"""


# ---- judge_prompts/information_extraction.py ----

IE_CRITERIA = {
    "IE_Entity": """[IE — Entity / Attribute Extraction]
This is an information extraction question about a concrete entity, count, attribute, text, object, location, or visual detail from the provided memory context.
- Assign 1 point if the student's response contains the core information from the standard answer. Minor wording differences are acceptable, but the essential information must be present and correct.
- Assign 0 points if the core information is missing, contradicted, too vague, refused, or incorrect.
- For numeric or count answers, require the exact value unless the standard answer itself is approximate.""",

    "IE_PreviousInfo": """[IE — Previous Information Extraction]
This is an information extraction question about previously mentioned information, spatial relations, counts, or attributes from earlier conversation/image context.
- Assign 1 point if the student's response recovers the same previous information as the standard answer, including the correct relation, location, count, or attribute.
- Minor wording differences are acceptable if they preserve the same meaning.
- Assign 0 points if the answer refers to the wrong previous item, gives the wrong relation/count/location, is too vague, refuses, or is unsupported.""",
}

IE_EXAMPLES = {
    "IE_Entity": """
[Example 1]
<Question>: What specific word is painted in large white letters on the pavement in the middle lanes?
<Standard Answer>: SCHOOL
<Student Answer>: SCHOOL

[Scoring Rationale]: The student's answer is "SCHOOL", which exactly matches the standard answer.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: Where on the sign is the photo of the mountains located?
<Standard Answer>: In the bottom right corner
<Student Answer>: The photo of the mountains is located in the upper right corner of the sign.

[Scoring Rationale]: The student says "upper right corner" but the standard answer is "bottom right corner". The location is incorrect — "upper" vs "bottom" is a substantive difference.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",

    "IE_PreviousInfo": """
[Example 1]
<Question>: Where on the sign is the photo of the mountains located?
<Standard Answer>: In the bottom right corner
<Student Answer>: The photo is in the bottom-right corner of the sign.

[Scoring Rationale]: The student's answer preserves the same location relation as the standard answer.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: Relative to the baking sheet, where is the salmon fillet placed?
<Standard Answer>: The right side
<Student Answer>: It is on the left side of the baking sheet.

[Scoring Rationale]: The student's answer gives the opposite spatial relation from the standard answer.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",
}


# ---- judge_prompts/knowledge_update.py ----

KU_CRITERIA = {
    "KU_KnowledgeUpdate": """[KU — Knowledge Update]
This is a knowledge update question testing whether the model tracks the latest state.
The correct current answer and the outdated old answer are both provided.
- Assign 1 point if the student gives the current (correct) answer.
- Assign 0 points if the student gives the outdated old answer, any other incorrect answer, refuses, or does not clearly answer.""",
}

KU_EXAMPLES = {
    "KU_KnowledgeUpdate": """
[Example 1]
<Question>: What do I treat as my must-have now that I plan my meals around it?
<Standard Answer>: brioche
<Old (Outdated) Answer>: sourdough
<Student Answer>: You now consider brioche your must-have bread that you plan meals around.

[Scoring Rationale]: The student's answer is "brioche", which matches the current standard answer, not the outdated answer "sourdough".
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: What do I treat as my must-have now that I plan my meals around it?
<Standard Answer>: brioche
<Old (Outdated) Answer>: sourdough
<Student Answer>: A versatile, high-quality protein source like chicken, beans, or tofu is a great starting point for building a week's worth of meals.

[Scoring Rationale]: The student's answer mentions generic protein sources, which is neither the current answer "brioche" nor the outdated answer "sourdough". The answer is simply incorrect.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",
}


# ---- judge_prompts/temporal_reasoning.py ----

TR_CRITERIA = {
    "TR_DurationComparison": """[TR — Duration Comparison (A/B)]
This is a binary choice question. The student must select one of two options.
- Assign 1 point if the student clearly selects the same option as the standard answer, regardless of phrasing.
- Assign 0 points if the student selects the other option, refuses, or does not make a clear selection.""",

    "TR_OrderRanking": """[TR — Order Ranking]
This is an ordering question. The standard answer is an exact sequence of items.
- Assign 1 point ONLY if the student gives the exact same sequence in the same order.
- Assign 0 points if any item is missing, added, duplicated, misplaced, or if the order is unclear.""",

    "TR_DateExtraction": """[TR — Date Extraction]
This is a date extraction question. The standard answer is a specific date.
- Assign 1 point if the student gives the same date in any unambiguous equivalent format (e.g., "Jan 15, 2024" and "2024-01-15" and "15/01/2024").
- Assign 0 points if the date is different, incomplete when a full date is required, ambiguous, or missing.""",
}

TR_EXAMPLES = {
    "TR_DurationComparison": """
[Example 1]
<Question>: Which of the following two durations is longer? Duration 1: The time I spent on Miller Hall. Duration 2: The time I spent on Music Festival Job. A. Duration 1 is longer. B. Duration 2 is longer.
<Standard Answer>: B
<Student Answer>: B. Duration 2 is longer.

[Scoring Rationale]: The student clearly selects "B", which matches the standard answer.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: Which of the following two durations is longer? Duration 1: The time I spent with the NutriPeak Nutrition. Duration 2: The time I spent with the BlendIt Pro Blender. A. Duration 1 is longer. B. Duration 2 is longer.
<Standard Answer>: A
<Student Answer>: B. Duration 2 is longer.

[Scoring Rationale]: The student selects "B", but the standard answer is "A". The selection is opposite.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",

    "TR_OrderRanking": """
[Example 1]
<Question>: Rank the following cities by population from largest to smallest: Tokyo, Delhi, Shanghai.
<Standard Answer>: Tokyo, Delhi, Shanghai
<Student Answer>: 1. Tokyo 2. Delhi 3. Shanghai

[Scoring Rationale]: The student's sequence is Tokyo, Delhi, Shanghai, which exactly matches the standard answer.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: Rank events A, B, C in chronological order.
<Standard Answer>: B, A, C
<Student Answer>: A, B, C

[Scoring Rationale]: The student's sequence is A, B, C, but the standard answer is B, A, C. The order does not match.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",

    "TR_DateExtraction": """
[Example 1]
<Question>: When did I complete the Scandinavian design course at Copenhagen Design Academy?
<Standard Answer>: 2024/08/31
<Student Answer>: You completed the Scandinavian design course on August 31st, 2024.

[Scoring Rationale]: The student's answer "August 31st, 2024" is an unambiguous equivalent of the standard answer "2024/08/31".
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: When did I visit Marutama Ra-men to taste their chicken broth?
<Standard Answer>: 2023/01/20
<Student Answer>: January 20, 2021

[Scoring Rationale]: The student's answer is "January 20, 2021", but the standard answer is "2023/01/20" — the year is wrong (2021 vs 2023).
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",
}


# ---- judge_prompts/multi_session_reasoning.py ----

MSR_CRITERIA = {
    "MSR_YesNo": """[MSR — Yes/No (Entity Resolution)]
This is a yes/no question.
- Assign 1 point if the student's final answer is semantically equivalent to the standard answer (e.g., "Yes" matches "Yes", "No" matches "No"), even if phrased differently (e.g., "That's correct" for "Yes").
- Assign 0 points if the answer means the opposite, refuses, or does not clearly answer.""",

    "MSR_Counting": """[MSR — Counting]
This is a counting question. The standard answer is a specific number.
- Assign 1 point if the student clearly states the exact number in any equivalent form (e.g., "3", "three", "3.0").
- Assign 0 points if the student gives a different number, a range, an approximation, refuses, or does not clearly answer.""",

    "MSR_Arithmetic": """[MSR — Arithmetic]
This is an arithmetic question. The standard answer is a specific value.
- Assign 1 point if the student gives the same value in an equivalent format (e.g., "0.5" and "1/2" and "50%").
- Assign 0 points if the student gives a different value, an unsupported approximation, refuses, or does not clearly answer.""",
}

MSR_EXAMPLES = {
    "MSR_YesNo": """
[Example 1]
<Question>: Is the cat in Mark and Jenny's 5-year anniversary post named Mittens?
<Standard Answer>: Yes
<Student Answer>: Yes, the cat in Mark and Jenny's 5-year anniversary post is named Mittens.

[Scoring Rationale]: The student's final answer is "Yes", which matches the standard answer "Yes". The additional explanation does not contradict the answer.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: Did the cat named after a crypto creator knock over the phone displaying the Binance alert?
<Standard Answer>: Yes
<Student Answer>: The text does not explicitly state whether the cat knocked over the phone. It only mentions the cat was sitting next to the phone when it crashed.

[Scoring Rationale]: The student does not give a clear "Yes". Instead, the student hedges and effectively refuses to commit to an answer. The standard answer is "Yes".
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",

    "MSR_Counting": "",

    "MSR_Arithmetic": """
[Example 1]
<Question>: How much total have I spent on coffee makers?
<Standard Answer>: $260.00
<Student Answer>: Based on the conversation, you spent $110 on the first coffee maker and $150 on the second one, totaling $260.

[Scoring Rationale]: The student's final answer is $260, which is numerically equivalent to the standard answer $260.00.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: How much total have I spent on coffee makers?
<Standard Answer>: $260.00
<Student Answer>: I do not have access to your personal financial data, including your coffee maker purchases. I cannot provide you with a total amount spent on coffee makers.

[Scoring Rationale]: The student refuses to answer and claims insufficient information, but this is not an AR task. The information was available in the conversation context.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",
}


# ---- judge_prompts/answer_refusal.py ----

AR_CRITERIA = {
    "AR_AnswerRefusal": """[AR — Answer Refusal]
- Assign 1 point if the student explicitly refuses, says there is not enough information, or indicates the question is unanswerable from the provided context.
- Assign 0 points if the student gives a substantive answer, even if hedged.""",
}

AR_EXAMPLES = {
    "AR_AnswerRefusal": """
[Example 1]
<Question>: What is the model number printed on the power bank?
<Standard Answer>: Insufficient information
<Student Answer>: I am sorry, but I cannot fulfill your request. The provided conversation history does not contain any information about a power bank or its model number.

[Scoring Rationale]: The student correctly refuses to answer, stating the information is not available in the conversation. This is the expected behavior for an unanswerable question.
In summary, the student's answer deserves 1 point.
[Score]: 1 point
[JSON]:
{
  "answer_score": 1
}

[Example 2]
<Question>: How many vertical cables are visible on one side of the bridge?
<Standard Answer>: Insufficient information
<Student Answer>: Three

[Scoring Rationale]: The student provides a substantive answer ("Three") rather than refusing. For an unanswerable question, the correct response is to refuse or indicate insufficient information.
In summary, the student's answer deserves 0 points.
[Score]: 0 points
[JSON]:
{
  "answer_score": 0
}
""",
}


TASK_CRITERIA: Dict[str, str] = {**IE_CRITERIA, **KU_CRITERIA, **TR_CRITERIA, **MSR_CRITERIA, **AR_CRITERIA}
TASK_EXAMPLES: Dict[str, str] = {**IE_EXAMPLES, **KU_EXAMPLES, **TR_EXAMPLES, **MSR_EXAMPLES, **AR_EXAMPLES}


def build_judge_prompt(task_type: str, question: str, reference: str,
                       prediction: str, old_answer: str = "") -> str:
    '\u5b98\u65b9 build_judge_prompt.'
    criteria = TASK_CRITERIA.get(task_type, "")
    examples = TASK_EXAMPLES.get(task_type, "")
    system_prompt = QA_BENCHMARK_JUDGE_PROMPT.format(task_criteria=criteria, examples=examples)
    if task_type == "KU_KnowledgeUpdate" and old_answer:
        current_case = (
            "[Current Case]\n"
            f"<Question>: {question}\n"
            f"<Standard Answer>: {reference}\n"
            f"<Old (Outdated) Answer>: {old_answer}\n"
            f"<Student Answer>: {prediction}\n\n"
        )
    elif task_type == "AR_AnswerRefusal":
        current_case = (
            "[Current Case]\n"
            f"<Question>: {question}\n"
            "<Standard Answer>: Insufficient information\n"
            f"<Student Answer>: {prediction}\n\n"
        )
    else:
        current_case = (
            "[Current Case]\n"
            f"<Question>: {question}\n"
            f"<Standard Answer>: {reference}\n"
            f"<Student Answer>: {prediction}\n\n"
        )
    return system_prompt + current_case + "[Scoring Rationale]:"


# =============================================================================
# Preserved upstream logic; see the corresponding protocol documentation.
# =============================================================================

_clients: List[OpenAI] = []
_rr_idx: int = 0
_rr_lock = threading.Lock()
_model_name: Optional[str] = None
_is_api_model: bool = False


def init_judge(base_url: str, model_name: str, api_key: str = "EMPTY", is_api: bool = False):
    '\u521d\u59cb\u5316 OpenAI \u517c\u5bb9\u5ba2\u6237\u7aef (vLLM \u9ed8\u8ba4). \u652f\u6301\u9017\u53f7\u5206\u9694\u7684\u591a endpoint \u8f6e\u8be2.'
    global _clients, _model_name, _is_api_model
    urls = [u.strip() for u in base_url.split(",") if u.strip()]
    if not urls:
        raise ValueError(f"no valid base_url parsed from: {base_url}")
    _clients = [OpenAI(base_url=u, api_key=api_key) for u in urls]
    _model_name = model_name
    _is_api_model = is_api
    print(f"[init_judge] registered {len(_clients)} endpoint(s): {urls}", flush=True)


def _next_client() -> OpenAI:
    """Thread-safe round-robin pick across _clients."""
    global _rr_idx
    with _rr_lock:
        c = _clients[_rr_idx % len(_clients)]
        _rr_idx += 1
        return c


def parse_score_from_judge(text: str) -> int:
    '\u5b98\u65b9 _parse_judge_response \u5b8c\u5168\u4e00\u81f4, \u8fd4\u56de 0 \u6216 1.'
    if not text:
        return 0
    m = re.search(r'"answer_score"\s*:\s*(\d+)', text)
    if m:
        return min(int(m.group(1)), 1)
    m = re.search(r"deserves\s+(\d+)\s+point", text, re.IGNORECASE)
    if m:
        return min(int(m.group(1)), 1)
    m = re.search(r"\[Score\]\s*:\s*(\d+)", text)
    if m:
        return min(int(m.group(1)), 1)
    return 0


def judge_one(question: str, reference: str, prediction_text: str,
              question_type: str, question_subtype: str = "",
              old_answer: str = "", max_tokens_judge: int = 2048) -> Tuple[int, Dict[str, Any]]:
    '\u6253\u5206\u5355\u6761 \u2014 \u542b 3 \u6b21\u91cd\u8bd5. \u4e0e\u5b98\u65b9 _judge_one \u7b49\u4ef7.'
    task_key = get_task_key(question_type, question_subtype, reference)
    prompt = build_judge_prompt(task_key, question, reference, prediction_text, old_answer)
    for attempt in range(3):
        try:
            api_kwargs = dict(
                model=_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            if _is_api_model:
                api_kwargs["max_completion_tokens"] = 256
            else:
                api_kwargs["max_tokens"] = max_tokens_judge
                api_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            resp = _next_client().chat.completions.create(**api_kwargs)
            raw = (resp.choices[0].message.content or "").strip()
            return parse_score_from_judge(raw), {}
        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (2 ** attempt))
                continue
            print(f"[Judge] API error after 3 retries: {e}", flush=True)
            raise RuntimeError(f"Judge API failed after 3 retries: {e}")


_PARSED_WORDS_THRESHOLD = 500  # Preserved upstream logic; see the corresponding protocol documentation.


def evaluate(items: List[Dict[str, Any]], jsonl_path: str,
             num_workers: int = 32, max_tokens_judge: int = 2048,
             max_samples: Optional[int] = None) -> Tuple[Dict, List[Dict]]:
    '\u5bf9 list of dict \u8dd1\u5224\u5206. \u6bcf\u4e2a item \u5b57\u6bb5\u540d\u5bf9\u9f50\u5b98\u65b9:\n       question_id, question, reference_answer, prediction, question_type, question_subtype,\n       old_answer, output_len.\n    '
    eval_data = items[:max_samples] if max_samples else items
    total = len(eval_data)

    # ---- JSONL resume ----
    completed: Dict[int, Dict[str, Any]] = {}
    if Path(jsonl_path).exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                idx = rec["idx"]
                if idx >= total:
                    continue
                item = eval_data[idx]
                if rec.get("question_id") != item.get("question_id"):
                    continue
                expected_task_key = get_task_key(
                    item.get("question_type", ""),
                    item.get("question_subtype", ""),
                    item["reference_answer"],
                )
                if rec.get("task_key") == expected_task_key:
                    completed[idx] = rec
        print(f"[Judge] Resuming: {len(completed)}/{total} already done", flush=True)

    remaining = [i for i in range(total) if i not in completed]
    if not remaining:
        print(f"[Judge] All {total} samples already judged", flush=True)
    else:
        print(f"[Judge] Processing {len(remaining)} samples with {num_workers} workers...", flush=True)

    write_lock = threading.Lock()
    fout = open(jsonl_path, "a")
    done_count = [len(completed)]

    def process(idx: int) -> Dict[str, Any]:
        item = eval_data[idx]
        task_key = get_task_key(
            item.get("question_type", ""),
            item.get("question_subtype", ""),
            item["reference_answer"],
        )
        raw_pred = item.get("prediction") or ""
        prediction_info = normalize_for_judge(raw_pred)
        parsed_words = len(prediction_info["text"].split())
        auto_zero = parsed_words > _PARSED_WORDS_THRESHOLD

        diagnostics = {
            "original_len": prediction_info["original_len"],
            "was_truncated": prediction_info["was_truncated"],
            "is_degenerate": prediction_info["is_degenerate"],
            **prediction_info.get("diagnostics", {}),
        }
        if auto_zero:
            score = 0
        else:
            score, _ = judge_one(
                question=item.get("question", ""),
                reference=item["reference_answer"],
                prediction_text=prediction_info["text"],
                question_type=item.get("question_type", "unknown"),
                question_subtype=item.get("question_subtype", ""),
                old_answer=item.get("old_answer", ""),
                max_tokens_judge=max_tokens_judge,
            )
        rec = {
            "idx": idx,
            "question_id": item.get("question_id"),
            "question_type": item.get("question_type", ""),
            "question_subtype": item.get("question_subtype", ""),
            "task_key": task_key,
            "judge_score": score,
            "original_len": diagnostics["original_len"],
            "was_truncated": diagnostics["was_truncated"],
            "is_degenerate": diagnostics["is_degenerate"],
            "hedging_count": diagnostics.get("hedging_count", 0),
            "unique_5gram_ratio": diagnostics.get("unique_5gram_ratio", 1.0),
            "auto_zero": auto_zero,
            "parsed_words": parsed_words,
            "output_len": item.get("output_len", 0),
        }
        with write_lock:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            done_count[0] += 1
            if done_count[0] % 100 == 0:
                print(f"  {done_count[0]}/{total}", flush=True)
        return rec

    t0 = time.time()
    if remaining:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(process, i): i for i in remaining}
            for fut in as_completed(futures):
                rec = fut.result()
                completed[rec["idx"]] = rec
    fout.close()
    if remaining:
        elapsed = time.time() - t0
        rate = len(remaining) / elapsed if elapsed else 0
        print(f"[Judge] Done {len(remaining)} samples in {elapsed:.1f}s ({rate:.1f} samples/s)", flush=True)

    # Preserved upstream logic; see the corresponding protocol documentation.
    details: List[Dict[str, Any]] = []
    by_type: Dict[str, List[int]] = defaultdict(list)
    by_subtype: Dict[str, List[int]] = defaultdict(list)
    by_task_key: Dict[str, List[int]] = defaultdict(list)
    degen_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "degenerate": 0, "truncated": 0, "auto_zero": 0}
    )

    for idx in range(total):
        rec = completed[idx]
        details.append({
            "question_id": rec["question_id"],
            "question_type": rec["question_type"],
            "question_subtype": rec.get("question_subtype", ""),
            "task_key": rec["task_key"],
            "judge_score": rec["judge_score"],
            "original_len": rec.get("original_len", 0),
            "was_truncated": rec.get("was_truncated", False),
            "is_degenerate": rec.get("is_degenerate", False),
            "hedging_count": rec.get("hedging_count", 0),
            "unique_5gram_ratio": rec.get("unique_5gram_ratio", 1.0),
            "auto_zero": rec.get("auto_zero", False),
            "parsed_words": rec.get("parsed_words", 0),
            "output_len": rec.get("output_len", 0),
        })
        qtype = rec["question_type"]
        subtype = rec.get("question_subtype", "") or "unknown"
        task_key = rec["task_key"]
        by_type[qtype].append(rec["judge_score"])
        by_subtype[f"{qtype}/{subtype}"].append(rec["judge_score"])
        by_task_key[task_key].append(rec["judge_score"])
        degen_counts[qtype]["total"] += 1
        if rec.get("is_degenerate"):
            degen_counts[qtype]["degenerate"] += 1
        if rec.get("was_truncated"):
            degen_counts[qtype]["truncated"] += 1
        if rec.get("auto_zero"):
            degen_counts[qtype]["auto_zero"] += 1

    all_scores = [d["judge_score"] for d in details]
    yes = sum(all_scores)
    total_degenerate = sum(dc["degenerate"] for dc in degen_counts.values())
    total_truncated = sum(dc["truncated"] for dc in degen_counts.values())
    total_auto_zero = sum(dc["auto_zero"] for dc in degen_counts.values())

    def summarize_groups(groups: Dict[str, List[int]]) -> Dict[str, Dict[str, Any]]:
        out = {}
        for name, scores in sorted(groups.items()):
            n_yes = sum(scores)
            n = len(scores)
            out[name] = {"accuracy": n_yes / n if n else 0,
                         "yes": n_yes, "no": n - n_yes, "count": n}
        return out

    by_type_metrics = {}
    for qtype, scores in sorted(by_type.items()):
        n_yes = sum(scores)
        n = len(scores)
        dc = degen_counts[qtype]
        by_type_metrics[qtype] = {
            "accuracy": n_yes / n if n else 0,
            "yes": n_yes, "no": n - n_yes, "count": n,
            "degenerate_count": dc["degenerate"],
            "truncated_count": dc["truncated"],
            "auto_zero_count": dc["auto_zero"],
            "degenerate_pct": round(dc["degenerate"] / dc["total"] * 100, 1) if dc["total"] else 0,
        }

    metrics = {
        "overall": {
            "accuracy": yes / total if total else 0,
            "yes": yes, "no": total - yes, "total": total,
            "degenerate_count": total_degenerate,
            "degenerate_pct": round(total_degenerate / total * 100, 1) if total else 0,
            "truncated_count": total_truncated,
            "truncated_pct": round(total_truncated / total * 100, 1) if total else 0,
            "auto_zero_count": total_auto_zero,
            "auto_zero_pct": round(total_auto_zero / total * 100, 1) if total else 0,
        },
        "by_question_type": by_type_metrics,
        "by_question_subtype": summarize_groups(by_subtype),
        "by_task_key": summarize_groups(by_task_key),
    }
    return metrics, details


# =============================================================================
# Preserved upstream logic; see the corresponding protocol documentation.
# =============================================================================

def load_infer_as_official(infer_file: str) -> List[Dict[str, Any]]:
    '\u5b57\u6bb5\u6620\u5c04:\n       sample_id            -> question_id\n       query                -> question\n       response             -> reference_answer\n       model_answer         -> prediction\n       category             -> question_type\n       meta.question_subtype-> question_subtype\n       (\u65e0 old_answer; bench \u6570\u636e\u96c6\u672a\u66b4\u9732)\n       (\u65e0 output_len; \u9ed8\u8ba4 0)\n    '
    items = []
    with io.open(infer_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            meta = row.get("meta") or {}
            items.append({
                "question_id": row.get("sample_id", ""),
                "question": row.get("query", "") or "",
                "reference_answer": row.get("response", "") or "",
                "prediction": row.get("model_answer", "") or "",
                "question_type": row.get("category", "") or "",
                "question_subtype": meta.get("question_subtype", "") or "",
                "old_answer": "",
                "output_len": 0,
            })
    return items


def print_metrics(metrics: Dict[str, Any], label: str = ""):
    '\u4e0e\u5b98\u65b9 print_metrics \u4e00\u81f4\u7684\u6253\u5370\u683c\u5f0f.'
    o = metrics["overall"]
    print("")
    print("=" * 70)
    if label:
        print(f"  {label}")
        print("=" * 70)
    print(f"  Overall: {o['accuracy'] * 100:.1f}%  ({o['yes']}/{o['total']})")
    print(f"  Degenerate: {o.get('degenerate_count', 0)} ({o.get('degenerate_pct', 0)}%)  "
          f"Truncated: {o.get('truncated_count', 0)} ({o.get('truncated_pct', 0)}%)  "
          f"AutoZero: {o.get('auto_zero_count', 0)} ({o.get('auto_zero_pct', 0)}%)")
    print("=" * 70)
    print(f"  {'Type':<25} {'Acc':>8} {'Yes':>5} {'No':>5} {'N':>5} {'Degen':>6}")
    print(f"  {'-' * 62}")
    for qtype, s in sorted(metrics["by_question_type"].items()):
        print(f"  {qtype:<25} {s['accuracy'] * 100:>7.1f}% {s['yes']:>5} {s['no']:>5} "
              f"{s['count']:>5} {s.get('degenerate_count', 0):>5}")
    if "by_task_key" in metrics:
        print(f"  {'-' * 62}")
        print(f"  {'Task key':<28} {'Acc':>8} {'Yes':>5} {'No':>5} {'N':>5}")
        print(f"  {'-' * 62}")
        for tk, s in sorted(metrics["by_task_key"].items()):
            print(f"  {tk:<28} {s['accuracy'] * 100:>7.1f}% {s['yes']:>5} {s['no']:>5} {s['count']:>5}")
    print("=" * 70)
    print("")


# =============================================================================
# Preserved upstream logic; see the corresponding protocol documentation.
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description='MemLens \u5b98\u65b9 LLM-as-Judge \u8bc4\u6d4b (\u81ea\u5305\u542b, \u76f4\u63a5\u8bfb\u7528\u6237 infer jsonl)'
    )
    ap.add_argument("--infer_file", required=True,
                    help='\u7528\u6237\u683c\u5f0f infer jsonl: \u5b57\u6bb5 sample_id / category / meta / query / response / model_answer')
    ap.add_argument("--output_dir", required=True,
                    help='\u7ed3\u679c\u8f93\u51fa\u76ee\u5f55 (\u5199 judge_metrics.json / judge_details.json / .jsonl)')
    ap.add_argument("--judge_base_url", default="http://127.0.0.1:8000/v1",
                    help='\u5224\u5b98 vLLM \u7684 OpenAI \u517c\u5bb9\u7aef\u70b9')
    ap.add_argument("--judge_model", default="memlens-judge",
                    help='\u5224\u5b98 served-model-name (\u987b\u4e0e vLLM \u542f\u52a8\u65f6 --served-model-name \u5bf9\u9f50)')
    # Portable wrapper: credentials stay in the environment, not process arguments.
    ap.add_argument("--judge_api_key", default=os.environ.get("JUDGE_API_KEY", "EMPTY"))
    ap.add_argument("--num_workers", type=int, default=32,
                    help='\u5e76\u53d1\u7ebf\u7a0b\u6570 (\u5355 split 789 \u6761\u7528 32-128 \u90fd\u884c)')
    ap.add_argument("--max_tokens_judge", type=int, default=2048,
                    help='\u5224\u5b98\u6700\u5927\u8f93\u51fa token (vLLM \u6a21\u5f0f)')
    ap.add_argument("--max_samples", type=int, default=None,
                    help='\u8c03\u8bd5\u7528: \u53ea\u5224\u524d N \u6761')
    ap.add_argument("--label", default="",
                    help="\u8f93\u51fa\u7684\u6807\u9898 (\u5982 'Baseline_qwen35_4B / memlens-32k')")
    args = ap.parse_args()

    if not os.path.exists(args.infer_file):
        print(f"ERROR: infer_file not found: {args.infer_file}")
        sys.exit(1)
    items = load_infer_as_official(args.infer_file)
    print(f"[Load] {args.infer_file}: {len(items)} samples", flush=True)

    init_judge(base_url=args.judge_base_url, model_name=args.judge_model,
               api_key=args.judge_api_key, is_api=False)

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "judge_metrics.json")
    details_path = os.path.join(args.output_dir, "judge_details.json")
    jsonl_path = metrics_path + ".jsonl"

    metrics, details = evaluate(
        items=items,
        jsonl_path=jsonl_path,
        num_workers=args.num_workers,
        max_tokens_judge=args.max_tokens_judge,
        max_samples=args.max_samples,
    )

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(details_path, "w") as f:
        json.dump(details, f, indent=2, ensure_ascii=False)

    print_metrics(metrics, args.label)
    print(f"[Judge] Saved: {metrics_path}")
    print(f"[Judge] Saved: {details_path}")
    print(f"[Judge] Resume JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
