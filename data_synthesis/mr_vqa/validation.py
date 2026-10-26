import os
import re
import threading

from .image_utils import (draw_marked_image, file_to_b64, load_image,
                          make_layout_preserving_canvas, safe_name)
from .llm import call_chat, image_message


_CANONICAL_BAD = {"yes", "no", "true", "false", "insufficient information",
                  "unknown", "unclear", "not visible", "cannot determine"}
_materialize_lock = threading.Lock()

# Verbal lead-in patterns we strip from extracted candidates. Matched at start.
_LEAD_IN_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"^(?:the\s+)?(?:final\s+)?answer(?:'s|\s+is|\s+was|\s+should\s+be|\s+to\s+the\s+question\s+is)?\s*[:：]?\s*",
        r"^(?:the\s+)?(?:final\s+|correct\s+)?(?:answer|solution|result|value|number|count|total)\s+is\s*[:：]?\s*",
        r"^(?:so|thus|hence|therefore|finally|then|consequently|in\s+conclusion|conclusion\s*[:：]?|结论[是为：:]?)\s*[:，,]?\s*",
        r"^(?:the\s+)?(?:final\s+)?(?:answer|solution)\s*[:：]\s*",
        r"^(?:所以|因此|综上|结论是|最终答案[是为：:]?|答案是|答案为|所以答案是|因此答案为|最终结论是)\s*[:：]?\s*",
        r"^=\s*",
        r"^[:：=]\s*",
    )
]


def _strip_code_fence(text):
    if not text:
        return ""
    m = re.search(r"```[a-zA-Z]*\s*([\s\S]*?)\s*```", text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _strip_inline_code(text):
    if not text:
        return ""
    m = re.search(r"`([^`\n]{1,80})`", text)
    if m:
        return m.group(1).strip()
    return ""


def _short_tail(text, max_words=8, max_chars=60):
    """Return the LAST short segment of `text` when split on sentence delimiters."""
    if not text:
        return ""
    parts = re.split(r"[.!?。！？\n\r]+", text)
    for p in reversed(parts):
        p = p.strip().strip("`'\"“”‘’()[]{}").strip()
        if not p:
            continue
        if len(p) <= max_chars and len(p.split()) <= max_words:
            return p
        # Segment is long: try fine-grained split.
        sub = re.split(r"[,;:，；：\t]+|\s{2,}", p)
        for s in reversed(sub):
            s = s.strip().strip("`'\"“”‘’()[]{}").strip()
            if s and len(s) <= max_chars and len(s.split()) <= max_words:
                return s
    return text.strip()[:max_chars]


def _clean(cand):
    """Strip lead-in phrases, surrounding quotes, leading articles."""
    if not cand:
        return ""
    c = cand.strip()
    for _ in range(5):
        before = c
        for pat in _LEAD_IN_PATTERNS:
            c = pat.sub("", c).strip()
        # Strip surrounding quotes/brackets.
        c = c.strip("`'\"“”‘’()[]{}").strip()
        # Strip a leading article if there are more tokens after.
        c = re.sub(r"^(?:a|an|the)\s+(?=\S)", "", c, flags=re.IGNORECASE).strip()
        if c == before:
            break
    # Stop at any mid-sentence punctuation that survived.
    c = re.split(r"[.!?,;:。！？，；：\n\r\t]+", c)[0].strip()
    c = c.strip("`'\"“”‘’()[]{}").strip()
    return c[:60]


# Final-answer extraction patterns. We search the WHOLE text and take the LAST match,
# so trailing conclusions dominate earlier mentions.
_FINAL_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"(?:final\s+answer|the\s+final\s+answer|the\s+answer\s+is|answer\s+is|answer\s*[:：])\s*[:：]?\s*([^\n.;。！？!?]{1,80})",
        r"(?:所以|因此|综上|最终|结论)[^是为：:]{0,8}(?:是|为|：|:)\s*([^\n。！？!?]{1,80})",
        r"答案是\s*[:：]?\s*([^\n。！？!?]{1,80})",
        r"答案为\s*[:：]?\s*([^\n。！？!?]{1,80})",
        r"(?:the\s+)?(?:correct\s+)?(?:answer|solution|result|value|number|count|total)\s+is\s*[:：]?\s*([^\n.;。！？!?]{1,80})",
    )
]


def extract_canonical_answer(text):
    """Extract a single-token canonical answer from a model response.

    Search order:
      1. <answer>...</answer> tag (recursively re-extracted when verbose).
      2. Inline code `X`.
      3. Code fence ```X```.
      4. Verbal "final answer / answer is / 答案是" patterns (take the LAST match).
      5. Trailing "= X" (last occurrence).
      6. Last non-empty line → last short phrase.
    Every candidate is post-cleaned by `_clean`.
    """
    if not text:
        return ""
    t = text.strip()

    # 1. <answer>...</answer>
    m = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", t, re.IGNORECASE)
    if m:
        inner = m.group(1).strip()
        if inner:
            if len(inner) <= 60 and len(inner.split()) <= 8:
                c = _clean(inner)
            else:
                # Author wedged a sentence in the tag — re-extract from inner text.
                c = _clean(extract_canonical_answer(inner)) or _clean(_short_tail(inner))
            if c:
                return c

    # 2. Inline code (only if short).
    inline = _strip_inline_code(t)
    if inline:
        c = _clean(inline)
        if c:
            return c

    # 3. Code fence.
    fence_match = re.search(r"```[a-zA-Z]*\s*([\s\S]{1,200}?)\s*```", t)
    if fence_match:
        fenced = fence_match.group(1).strip()
        if len(fenced) <= 60 and len(fenced.split()) <= 8:
            c = _clean(fenced)
        else:
            c = _clean(extract_canonical_answer(fenced)) or _clean(_short_tail(fenced))
        if c:
            return c

    # 4. Verbal "answer is / 答案是" patterns — take the LAST match.
    last_match = None
    last_match_pos = -1
    last_match_pat = None
    for pat in _FINAL_PATTERNS:
        m = None
        pos = -1
        for mm in pat.finditer(t):
            m = mm
            pos = mm.start()
        if m and pos > last_match_pos:
            last_match = m
            last_match_pos = pos
            last_match_pat = pat
    if last_match:
        cand = last_match.group(1).strip()
        # If captured too much, take the FIRST short segment after the marker.
        c = _clean(re.split(r"[,;。；\n\r\t]|\s{2,}", cand)[0])
        if c:
            return c

    # 5. Trailing "= X" — take the LAST occurrence.
    eq_matches = list(re.finditer(r"=\s*([^\n.;。！？!?]{1,60})", t))
    if eq_matches:
        c = _clean(eq_matches[-1].group(1).strip())
        if c:
            return c

    # 6. Last non-empty line → last short phrase.
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        tail = _short_tail(lines[-1])
        c = _clean(tail) or _clean(_short_tail(t))
        if c:
            return c

    return _clean(t)[:60]


def answer_image(image_path, question, api_url, api_key, model, samples=4):
    image_b64 = file_to_b64(image_path)
    answers = []
    prompt = f"""Answer the question using only the provided image.

Question: {question}

Reply with ONLY the answer inside <answer>...</answer>. The answer MUST follow this canonical form:
- Any objective short answer under 10 words. Articles and short clauses are allowed; no explanation or multiple sentences.
- Colors: lowercase base color name (e.g. "red", "dark blue").
- Numbers: digits, no units (e.g. "3", not "three" or "3 people").
- Yes/no questions: answer "yes" or "no" (lowercase).
- Object identity or other semantic answer: concise natural phrase or clause under 10 words.
- Use the same language as the question (English unless the question is Chinese).
- If the image lacks enough information, answer <answer>insufficient information</answer>.
- Even if you reason internally, the content inside <answer>...</answer> MUST contain fewer than 10 words.
- DO NOT wrap a sentence, an explanation, or step-by-step reasoning inside <answer>...</answer>.

Good: <answer>red</answer>, <answer>3</answer>, <answer>yes</answer>, <answer>stop sign</answer>
Bad: <answer>The color is red.</answer>, <answer>a red sign</answer>, <answer>There are 3 people.</answer>, <answer>In the purple box we see a hand</answer>"""
    for _ in range(samples):
        text = call_chat(
            api_url, api_key, model,
            [{"role": "user", "content": [image_message(image_b64), {"type": "text", "text": prompt}]}],
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
        )
        answers.append(extract_canonical_answer(text))
    return answers


def consistency(question, answers, judge):
    numbered = []
    for idx, ans in enumerate(answers):
        a = (ans or "").strip()
        if len(a) > 200:
            a = a[:200] + "..."
        numbered.append(f"{idx + 1}. {a}")
    answer_text = "\n".join(numbered)
    prompt = f"""You are grouping semantically equivalent short answers to a visual question.

Question: {question}

Answers (some may be verbose reasoning; extract the FINAL short answer token from each before comparing):
{answer_text}

Rules:
- Each "answer" may be a long chain-of-thought. Read it, identify the final answer the assistant committed to (usually the last quoted word, the last short line, or the content after "answer is"), then compare those final tokens.
- Treat "unknown", "unclear", "not visible", "cannot determine", and "insufficient information" as insufficient.
- Do not merge different colors, numbers, object identities, or spatial relations.
- If the model never commits to a final answer (just keeps debating), treat it as insufficient.

Return exactly these three lines:
Most Common Answer: <the canonical short token, lowercase if yes/no/color/number>
Consistency Count: <integer from 1 to N>
Is Insufficient: Yes/No"""
    text = judge.ask(prompt, temperature=0.0, max_tokens=1024)
    answer = ""
    count = 0
    insufficient = False
    for line in text.splitlines():
        if line.startswith("Most Common Answer:"):
            answer = line.split(":", 1)[1].strip()
        elif line.startswith("Consistency Count:"):
            digits = "".join(ch for ch in line if ch.isdigit())
            count = int(digits) if digits else 0
        elif line.startswith("Is Insufficient:"):
            insufficient = "yes" in line.lower()
    return {
        "answers": answers,
        "most_common_answer": answer,
        "consistency_count": count,
        "consistency_score": count / max(1, len(answers)),
        "is_insufficient": insufficient or "insufficient" in answer.lower(),
        "raw": text,
    }


def equivalent(question, reference_answer, candidate_eval, judge):
    prompt = f"""Judge whether the candidate answer is semantically equivalent to the reference.

Question: {question}
Reference answer: {reference_answer}
Candidate answer: {candidate_eval.get('most_common_answer')}
Candidate insufficient: {candidate_eval.get('is_insufficient')}

Rules:
- If candidate is insufficient/unknown/unclear, it is NOT equivalent.
- For colors, numbers, object names, text, and spatial relations, require strict semantic match.
- Do not infer missing details.

Return exactly:
EQUIVALENT: Yes/No
REASON: brief reason"""
    text = judge.ask(prompt, temperature=0.0, max_tokens=512)
    return {"equivalent": "equivalent: yes" in text.lower(), "raw": text}


def _load_anchor_colors(candidate):
    """Pull anchor_colors stored on the candidate (set by stage-03 generation).

    Returns dict {region_id: (color_name, (r,g,b))} or None.
    Survives JSON round-trip (rgb tuples become lists).
    """
    ac = candidate.get("anchor_colors")
    if not isinstance(ac, dict) or not ac:
        return None
    out = {}
    for rid, v in ac.items():
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            return None
        color_name = v[0]
        rgb = v[1]
        if isinstance(rgb, (list, tuple)) and len(rgb) == 3:
            rgb = tuple(int(x) for x in rgb)
        out[rid] = (color_name, rgb)
    return out or None


def build_validation_images(record, candidate, evidence_dir, canvas_size=1024):
    image_id = record["image_id"]
    qid = candidate["candidate_id"]
    group = candidate["evidence_regions"]
    base = os.path.join(evidence_dir, image_id, qid)
    os.makedirs(base, exist_ok=True)
    paths = {
        "original": os.path.join(evidence_dir, image_id, "original.jpg"),
        "privileged_layout": os.path.join(base, "privileged_layout.jpg"),
        "marked_original": os.path.join(base, "marked_original.jpg"),
        "drop": {},
    }
    # Reuse the EXACT anchor_colors that stage-03 generation used, so the
    # color names referenced inside the question text (e.g. "the orange box")
    # match the colors painted on every layout / marked / drop-one view.
    # Falls back to seeded sampling only if the candidate lacks anchor_colors.
    anchor_colors = _load_anchor_colors(candidate)
    # Materialize remote originals so final datasets never depend on OSS access.
    with _materialize_lock:
        if not os.path.exists(paths["original"]):
            load_image(record["image_path"]).save(
                paths["original"], "JPEG", quality=98, subsampling=0
            )
    source_path = paths["original"]
    _, anchor_colors = make_layout_preserving_canvas(
        source_path, group, paths["privileged_layout"],
        canvas_size=canvas_size,
        anchor_colors=anchor_colors,
        color_seed=None if anchor_colors else f"{image_id}:{qid}",
    )
    draw_marked_image(source_path, group, paths["marked_original"], anchor_colors=anchor_colors)
    for region in group:
        rid = region["region_id"]
        paths["drop"][rid] = os.path.join(base, f"drop_{safe_name(rid)}_layout.jpg")
        make_layout_preserving_canvas(
            source_path, group, paths["drop"][rid],
            drop_region_id=rid, canvas_size=canvas_size,
            anchor_colors=anchor_colors,
        )
    paths["anchor_colors"] = anchor_colors
    return paths


def _single_answer_eval(image_path, question, api_url, api_key, model, judge, samples=3):
    return consistency(question, answer_image(image_path, question, api_url, api_key, model, samples=samples), judge)


def validate_candidate(record, candidate, paths, primary_api_url, primary_api_key, primary_model,
                       secondary_api_url, secondary_api_key, secondary_model, judge,
                       answer_samples=4, drop_samples=4, min_consistency=0.75,
                       drop_consistency=0.75, coarse_screen=True):
    """Legacy drop-one validator with necessity-mode-aware semantics.

    The reference answer is taken directly from the candidate (hypothetical_answer
    or answer_hint set during stage-03 generation). This skips the expensive and
    error-prone primary/secondary 4-shot anchor step: if the designer's answer is
    wrong the candidate will be filtered downstream by the drop-one test (no
    region can reproduce a wrong answer) or by the MCQ rejection sampler.

    Production stage-03 uses selfcheck.py. This compatibility path applies the
    same distinction between answer-sensitive and completeness/dependency modes.
    """
    question = candidate["question"]
    necessity_mode = candidate.get("necessity_mode", "answer_sensitive")
    privileged_path = paths["privileged_layout"]

    reference_answer = (candidate.get("hypothetical_answer")
                        or candidate.get("answer_hint")
                        or "").strip()
    if not reference_answer:
        return {"is_valid": False, "stage": "missing_reference_answer"}

    if reference_answer.lower() in {"insufficient information", "unknown",
                                    "unclear", "not visible", "cannot determine"}:
        return {"is_valid": False, "stage": "reference_answer_insufficient",
                "reference_answer": reference_answer}

    drop = {}
    for region in candidate["evidence_regions"]:
        rid = region["region_id"]
        ev = consistency(
            question,
            answer_image(paths["drop"][rid], question, primary_api_url,
                         primary_api_key, primary_model, drop_samples),
            judge,
        )
        eq = equivalent(question, reference_answer, ev, judge)
        drop[rid] = {"eval": ev, "equivalence": eq}
        if necessity_mode == "answer_sensitive":
            failed = eq["equivalent"] and ev["consistency_score"] >= drop_consistency
            failure_stage = "drop_one_sufficient"
        else:
            failed = (not ev["is_insufficient"] and
                      ev["consistency_score"] >= drop_consistency)
            failure_stage = "drop_one_still_answerable"
        if failed:
            return {
                "is_valid": False,
                "stage": failure_stage,
                "necessity_mode": necessity_mode,
                "region_id": rid,
                "answer": reference_answer,
                "drop": drop,
            }

    return {
        "is_valid": True,
        "answer": reference_answer,
        "drop": drop,
    }
