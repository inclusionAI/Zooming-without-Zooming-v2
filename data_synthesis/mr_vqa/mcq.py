import random
import re
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor

from .answer_norm import _norm, norm_match
from .image_utils import file_to_b64
from .llm import call_chat, extract_json, image_message


_UNMARKED_FORBIDDEN_PHRASES = (
    "marked box", "marked boxes", "marked region", "marked regions",
    "outlined box", "outlined boxes", "outlined region", "outlined regions",
    "colored box", "colored boxes", "colored region", "colored regions",
)


def _assemble_mcq(question, answer, distractors, answer_vocabulary=None):
    vocabulary = [str(x).strip() for x in (answer_vocabulary or []) if str(x).strip()]
    closed_scalar_vocabulary = (
        "+" not in str(answer)
        and not re.fullmatch(r"-?\d+(?:\.\d+)?", str(answer).strip())
        and len(vocabulary) >= 2
        and any(norm_match(answer, value) for value in vocabulary)
    )
    set_atoms = []
    if "+" in str(answer):
        for value in [str(answer)] + vocabulary:
            for atom in re.split(r"\s*\+\s*", value):
                atom = atom.strip()
                if atom and _norm(atom) not in {_norm(x) for x in set_atoms}:
                    set_atoms.append(atom)

    def valid_closed_set(candidate):
        if set_atoms:
            parts = [x.strip() for x in re.split(r"\s*\+\s*", candidate) if x.strip()]
            normalized = [_norm(x) for x in parts]
            allowed = {_norm(x) for x in set_atoms}
            return bool(parts) and len(normalized) == len(set(normalized)) and all(
                x in allowed for x in normalized
            )
        if closed_scalar_vocabulary:
            return any(norm_match(candidate, value) for value in vocabulary)
        return True

    unique = []
    for candidate in distractors:
        candidate = str(candidate).strip()
        if (not candidate or norm_match(candidate, answer)
                or not valid_closed_set(candidate)):
            continue
        if any(norm_match(candidate, existing) or
               _norm(candidate) == _norm(existing) for existing in unique):
            continue
        unique.append(candidate)
    # Selection questions already carry a verified closed answer vocabulary.
    # Use it as a deterministic fallback when the model returns duplicates or
    # an order-equivalent version of the correct set.
    if "+" in str(answer):
        answer_size = len([x for x in re.split(r"\s*\+\s*", str(answer)) if x.strip()])
        sizes = sorted(range(1, len(set_atoms) + 1), key=lambda n: (n != answer_size, abs(n - answer_size)))
        for size in sizes:
            for subset in combinations(set_atoms, size):
                candidate = " + ".join(subset)
                if norm_match(candidate, answer):
                    continue
                if any(norm_match(candidate, existing) or
                       _norm(candidate) == _norm(existing) for existing in unique):
                    continue
                unique.append(candidate)
                if len(unique) >= 3:
                    break
            if len(unique) >= 3:
                break
    for candidate in vocabulary:
        if norm_match(candidate, answer):
            continue
        if any(norm_match(candidate, existing) or
               _norm(candidate) == _norm(existing) for existing in unique):
            continue
        unique.append(candidate)
    # Count questions should not fail solely because the distractor model
    # returned duplicates. Fill missing options deterministically around the
    # verified integer answer; visual correctness still comes from the voted GT.
    if re.fullmatch(r"-?\d+", str(answer).strip()):
        value = int(str(answer).strip())
        deltas = (-1, 1, -2, 2, -3, 3, 4, 5)
        for delta in deltas:
            candidate = str(value + delta)
            if int(candidate) < 0 or norm_match(candidate, answer):
                continue
            if any(norm_match(candidate, existing) for existing in unique):
                continue
            unique.append(candidate)
            if len(unique) >= 3:
                break
    # A closed set with only three candidates provides just two candidate
    # distractors. Add a standard mutually exclusive fallback rather than
    # failing MCQ construction after the answer has already been verified.
    generic_fallbacks = (
        ("all of the above", "cannot be determined")
        if "none" in _norm(answer)
        else ("none of the above", "all of the above", "cannot be determined")
    )
    for candidate in generic_fallbacks:
        if norm_match(candidate, answer):
            continue
        if any(norm_match(candidate, existing) or
               _norm(candidate) == _norm(existing) for existing in unique):
            continue
        unique.append(candidate)
        if len(unique) >= 3:
            break
    if len(unique) < 3:
        return None
    options = [answer] + unique[:3]
    random.shuffle(options)
    correct = chr(65 + options.index(answer))
    return {
        "mcq_question": question + "\n\n" + "\n".join(
            f"{chr(65 + i)}. {x}" for i, x in enumerate(options)
        ) + "\n\nAnswer with the option letter.",
        "neutral_question": question,
        "all_options": options,
        "correct_option": correct,
        "distractors": unique[:3],
        "correct_answer_text": answer,
    }


def _audit_grounding_matches(audit, expected_regions, image_width, image_height):
    """Require a one-to-one match between audit boxes and expected evidence areas."""
    boxes = audit.get("evidence_boxes") or []
    if (not expected_regions or len(boxes) != len(expected_regions)
            or not image_width or not image_height):
        return False

    predicted = []
    for item in boxes:
        box = item.get("bbox_2d") if isinstance(item, dict) else item
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            return False
        try:
            x1, y1, x2, y2 = [float(v) for v in box]
        except (TypeError, ValueError):
            return False
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            return False
        predicted.append((x1, y1, x2, y2))

    expected = []
    for region in expected_regions:
        x1, y1, x2, y2 = region["bbox"]
        expected.append((1000 * x1 / image_width, 1000 * y1 / image_height,
                         1000 * x2 / image_width, 1000 * y2 / image_height))

    def compatible(pred, exp):
        px = (pred[0] + pred[2]) / 2
        py = (pred[1] + pred[3]) / 2
        ew = exp[2] - exp[0]
        eh = exp[3] - exp[1]
        # Audit boxes may tightly cover an object while proposals include context.
        return (exp[0] - .15 * ew <= px <= exp[2] + .15 * ew
                and exp[1] - .15 * eh <= py <= exp[3] + .15 * eh)

    edges = [[j for j, exp in enumerate(expected) if compatible(pred, exp)]
             for pred in predicted]
    if any(not choices for choices in edges):
        return False

    matched = [-1] * len(expected)

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if matched[j] < 0 or augment(matched[j], seen):
                matched[j] = i
                return True
        return False

    return all(augment(i, set()) for i in range(len(predicted)))


def generate_mcq(image_path, question, answer, api_url, api_key, model,
                 answer_vocabulary=None):
    image_b64 = file_to_b64(image_path)
    prompt = f"""You are given a visual question and its correct answer.

Original question: {question}
Correct answer: {answer}

The provided image is the marked original image with colored evidence boxes. Your task is to generate three plausible but incorrect multiple-choice distractors.

Rules:
- DO NOT rewrite, paraphrase, simplify, or neutralize the original question.
- Keep the final MCQ stem exactly equal to the original question.
- Generate three plausible but incorrect distractors.
- Distractors should be visually relevant and clearly distinct from the correct answer.
- Distractors should have the same answer format and similar length as the correct answer.
- For an open-vocabulary short answer, create three concrete alternatives with the
  same grammatical form, specificity, and roughly the same word length. The answer may
  be any objective expression under 10 words, not only an object or attribute. Do not use
  "none of the above", "all of the above", or "cannot be determined".
- Do not include explanations.

Return JSON only:
[
  {{"distractors": ["d1", "d2", "d3"]}}
]"""
    text = call_chat(
        api_url, api_key, model,
        [{"role": "user", "content": [image_message(image_b64), {"type": "text", "text": prompt}]}],
        temperature=0.8,
        max_tokens=1024,
        retries=4,
    )
    parsed = extract_json(text, list)
    neutral_question = question
    distractors = []
    if parsed and isinstance(parsed[0], dict):
        distractors = [str(x).strip() for x in (parsed[0].get("distractors") or [])]
    else:
        # fallback: also accept bare distractor list
        distractors = [str(x).strip() for x in parsed] if parsed else []
    return _assemble_mcq(
        neutral_question, answer, distractors,
        answer_vocabulary=answer_vocabulary,
    )


def generate_unmarked_mcq(image_path, anchored_question, answer,
                          expected_regions, image_width, image_height,
                          primary_api_url, primary_api_key, primary_model,
                          audit_api_url, audit_api_key, audit_model,
                          audit_samples=2):
    """Try to remove visual anchors without changing an anchor-independent answer.

    The primary model minimally de-anchors the stem and creates distractors. A
    second model, which is not shown the expected answer, independently checks
    referential uniqueness, multi-region dependence, and the answer itself.
    """
    image_b64 = file_to_b64(image_path)
    rewrite_prompt = f"""You are given an unmarked original image, an anchored visual
question, and its verified correct answer.

Anchored question: {anchored_question}
Verified answer: {answer}

Create a minimally changed version of the question that works on the UNMARKED image.
Replace references to boxes/outlines with natural, visually unique descriptions or a
clearly exhaustive image-level scope. Do not add facts, change the operation, or change
the answer. If unique reference cannot be guaranteed, set safe_without_marks=false.
Also generate three plausible incorrect distractors in the same answer format.

Return JSON only:
{{"safe_without_marks": true, "natural_question": "...", "distractors": ["d1", "d2", "d3"]}}
"""
    text = call_chat(
        primary_api_url, primary_api_key, primary_model,
        [{"role": "user", "content": [image_message(image_b64),
          {"type": "text", "text": rewrite_prompt}]}],
        temperature=0.2, max_tokens=1024, enable_thinking=False,
    )
    obj = extract_json(text, dict)
    natural_question = str(obj.get("natural_question") or "").strip()
    if not obj.get("safe_without_marks") or not natural_question:
        return None, {"pass": False, "reason": "rewrite_not_safe"}
    qnorm = natural_question.lower()
    if any(phrase in qnorm for phrase in _UNMARKED_FORBIDDEN_PHRASES):
        return None, {"pass": False, "reason": "anchor_language_remains"}

    audit_prompt = f"""Audit this visual question using only the unmarked original image.

Anchored source question: {anchored_question}
Unmarked natural question: {natural_question}

Independently answer it, then judge its references. A question is unambiguous only if
every mentioned target or exhaustive target set is uniquely determined in this image.
It must also genuinely require inspecting multiple spatially separated image regions.
Be strict: uncertainty, multiple plausible referents, or a scope that could include
unintended objects is ambiguous.

The intended reasoning uses {len(expected_regions)} distinct evidence areas. Ground the
actual image areas you used, one box per area, as [x1,y1,x2,y2] normalized to 0..1000.
Do not infer these boxes from the wording; locate them from pixels. Also judge whether
the natural question preserves the source task and whether every grounded area is
required for the exact answer.

Return JSON only:
{{"answer": "short answer", "reference_status": "unique or ambiguous", "same_task_as_source": true, "all_targets_identifiable": true, "requires_multiple_regions": true, "every_evidence_area_required": true, "evidence_boxes": [{{"description": "...", "bbox_2d": [0,0,100,100]}}], "reason": "short reason"}}
"""

    model_specs = [
        ("primary", primary_api_url, primary_api_key, primary_model),
        ("secondary", audit_api_url, audit_api_key, audit_model),
    ]

    def run_audit(spec):
        family, url, key, model = spec
        audit_text = call_chat(
            url, key, model,
            [{"role": "user", "content": [image_message(image_b64),
              {"type": "text", "text": audit_prompt}]}],
            temperature=0.0, max_tokens=1024, enable_thinking=False,
        )
        audit = extract_json(audit_text, dict)
        audit_answer = str(audit.get("answer") or "").strip()
        grounding_match = _audit_grounding_matches(
            audit, expected_regions, image_width, image_height
        )
        passed = (
            str(audit.get("reference_status") or "").strip().lower() == "unique"
            and audit.get("same_task_as_source") is True
            and audit.get("all_targets_identifiable") is True
            and audit.get("requires_multiple_regions") is True
            and audit.get("every_evidence_area_required") is True
            and grounding_match
            and norm_match(audit_answer, answer)
        )
        audit.update({
            "model_family": family,
            "model": model,
            "pass": passed,
            "grounding_match": grounding_match,
        })
        return audit

    jobs = [spec for spec in model_specs for _ in range(max(1, audit_samples))]
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        audits = list(ex.map(run_audit, jobs))
    audit_summary = {
        "pass": bool(audits) and all(a.get("pass") for a in audits),
        "expected_answer": answer,
        "required_passes": len(jobs),
        "passed_count": sum(bool(a.get("pass")) for a in audits),
        "audits": audits,
    }
    if not audit_summary["pass"]:
        audit_summary["reason"] = "independent_unmarked_audit_failed"
        return None, audit_summary

    mcq = _assemble_mcq(natural_question, answer, obj.get("distractors") or [])
    if not mcq:
        audit_summary["pass"] = False
        audit_summary["reason"] = "invalid_distractors"
        return None, audit_summary
    return mcq, audit_summary


def extract_option_letter(text):
    text = (text or "").strip()
    boxed = re.search(r"\\boxed\{?\s*([ABCD])\s*\}?", text, re.I)
    if boxed:
        return boxed.group(1).upper()
    tag = re.search(r"<answer>\s*([ABCD])\s*</answer>", text, re.I)
    if tag:
        return tag.group(1).upper()
    first = re.search(r"\b([ABCD])\b", text, re.I)
    return first.group(1).upper() if first else ""


def run_mcq_rejection_sampling(image_path, mcq_question, correct_option, api_url, api_key,
                               model, samples=4, marked=True):
    image_b64 = file_to_b64(image_path)
    image_description = ("marked original image. The colored outlines are visible "
                         "reference anchors; inspect the image content inside them"
                         if marked else "unmarked original image")
    good_reasoning = ("The red box shows 3 cars and the blue box shows 2 motorcycles, "
                      "so the answer is A."
                      if marked else
                      "There are 3 cars and 2 motorcycles in the specified areas, so "
                      "the answer is A.")
    prompt = f"""Answer this multiple-choice visual question using only the provided {image_description}.

{mcq_question}

You MAY reason briefly step-by-step before answering. The reasoning goes in plain prose BEFORE the answer tag; the final answer MUST be wrapped in <answer>...</answer>. Format:

<your brief reasoning here, optional>
<answer>X</answer>

Rules:
- <answer>...</answer> MUST contain ONLY the option letter (A, B, C, or D) — no explanation, no text from the option.
- Do NOT output any visual grounding tokens or bounding box markers — no <|box_start|>, no <|box_end|>, no coordinate strings, no object_ref markers.
- Keep the reasoning TERSE (a few short sentences at most).
- The <answer>...</answer> tag is REQUIRED. If you omit it, the answer is discarded.

Good:
{good_reasoning}
<answer>A</answer>

<answer>B</answer>

Bad:
<answer>3 vehicles</answer>          (option text instead of letter)
The answer is A.                     (no <answer> tag at all)

Reply now:"""
    raw_answers = []
    parsed_options = []
    for _ in range(samples):
        text = call_chat(
            api_url, api_key, model,
            [{"role": "user", "content": [image_message(image_b64), {"type": "text", "text": prompt}]}],
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
        )
        raw_answers.append(text)
        parsed_options.append(extract_option_letter(text))

    correct_count = sum(1 for opt in parsed_options if opt == correct_option)
    valid_count = sum(1 for opt in parsed_options if opt in {"A", "B", "C", "D"})
    sampling_valid = valid_count == samples

    def difficulty_label(n_correct, n_total):
        if n_total <= 0:
            return "unknown"
        ratio = n_correct / n_total
        if ratio >= 1.0:
            return "too_easy"
        if ratio >= 0.75:
            return "easy"
        if ratio >= 0.5:
            return "medium"
        if ratio >= 0.25:
            return "hard"
        return "very_hard"

    return {
        "raw_answers": raw_answers,
        "parsed_options": parsed_options,
        "correct_option": correct_option,
        "correct_count": correct_count,
        "valid_count": valid_count,
        "sampling_valid": sampling_valid,
        "difficulty": (difficulty_label(correct_count, samples)
                       if sampling_valid else "invalid_sampling"),
        "keep": sampling_valid and correct_count < samples,
    }
