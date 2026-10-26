"""Stage-03 generation-time self check (merged with stage-04 drop-one necessity).

For each candidate question:
  1. Privileged voting: sample the privileged_layout view N times, split between
     the primary (designer) model and a secondary multimodal model. Each answer
     is compared (via _norm + mathruler for numeric) to the designer's
     `hypothetical_answer`. The candidate is kept only if at least
     `min_priv_votes` of the sampled answers match.
  2. Drop-one necessity: validate each candidate according to its declared mode:
     answer sensitivity, closed-set completeness, pair completeness, or chain
     dependency. Suspicious ablations are adaptively resampled.

The prompt enforces concise canonical answers. Closed outputs use a supplied
label vocabulary; open semantic/text outputs are clustered by a Judge when
equivalent wording is not string-identical.
"""
import os
import tempfile
import math
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .answer_norm import _norm, norm_match
from .image_utils import file_to_b64, load_image, make_layout_preserving_canvas
from .llm import call_chat, image_message


_materialize_lock = threading.Lock()


def _matches_designer(designer_hint, gt):
    """True if designer's hypothetical_answer matches the voted GT under
    norm_match (handles color-vs-color+box, numeric equivalent, etc.). Used
    to flag divergent designer answers so stage-04 can strip them.
    """
    if not designer_hint or not gt:
        return False
    try:
        return bool(norm_match(designer_hint, gt))
    except Exception:
        return False


_ANSWER_PROMPT_TPL = """Answer the question using only the provided image{caption_section}.
{ablation_instruction}

Question: {question}
{answer_vocabulary_section}

You MAY reason briefly step-by-step before answering. The reasoning goes in plain prose BEFORE the answer tag; the final answer MUST be wrapped in <answer>...</answer>. Format:

<your brief reasoning here, optional>
<answer>final short answer here</answer>

Rules for the content inside <answer>...</answer>:
- Answer naturally and concisely. Any objectively supported expression with fewer than 10 words is allowed, including a short clause. Do not force the answer into a predefined semantic category.
- Numbers: use plain Arabic digits with no units and no thousand separators. Example: "16", NOT "sixteen" or "16 people" or "1,600". Decimal like "0.5" when fractional.
- Yes/no questions -> "yes" or "no" (lowercase).
- Articles are allowed when natural. Do NOT include an explanation, multiple sentences, or step-by-step reasoning inside <answer>...</answer>.
- Do NOT output any visual grounding tokens or bounding box markers — no <|box_start|>, no <|box_end|>, no begin_of_box, no end_of_box, no coordinate strings, no object_ref markers. Output the literal answer only.
- Keep the reasoning TERSE (a few short sentences at most). The reasoning is for your own computation; the answer tag is the only output downstream consumers read.
- Use the same language as the question.
- If the image lacks enough information, the final answer must be <answer>insufficient information</answer>.

The <answer>...</answer> tag is REQUIRED. If you omit it, the answer is discarded.

Good:
Looking at the red box, I see 3 cars. The blue box has 2 motorcycles. Total = 5.
<answer>5</answer>

<answer>red</answer>

<answer>yes</answer>

<answer>stop sign</answer>

<answer>the woman is holding a cigarette</answer>

Bad:
<answer>The answer is 16.</answer>           (explanation inside the tag)
<answer>sixteen</answer>                      (use Arabic digits: "16")
<answer>the woman is probably waiting because the bus is late</answer> (speculative or 10+ words)
<answer>begin_of_box16end_of_box</answer>    (forbidden box markers)
I need to analyze this step by step.          (no <answer> tag — rejected)
The wheels in the green box are four.          (no <answer> tag — rejected)"""


import re as _re
_BOX_MARKER_PAT = _re.compile(
    r"(?:<\|box_start\|>|<\|object_ref_start\|>|begin_of_box)"
    r".*?"
    r"(?:<\|box_end\|>|<\|object_ref_end\|>|end_of_box)",
    _re.DOTALL,
)


def _strip_answer(text):
    """Extract the answer payload from a model reply.

    Accepts either:
      - "<reasoning prose> <answer>X</answer>": returns X (preserving original
        spacing/case so downstream _norm/judge handles it consistently).
      - "X" (bare short answer with no tag): returns _norm(X) — but ONLY when X
        is short enough to plausibly be an answer rather than reasoning prose.
        Long unlabeled prose (>= 10 tokens or > 120 chars) is treated as
        unparseable and returns empty string so it drops out of voting rather
        than being folded into a garbage char blob.
    """
    if not text:
        return ""
    t = text.strip()
    # Strip any visual-grounding box markers GLM/Qwen-VL family may emit.
    t = _BOX_MARKER_PAT.sub("", t).strip()
    lower = t.lower()
    s = lower.find("<answer>")
    e = lower.find("</answer>")
    if s >= 0 and e > s:
        return t[s + len("<answer>"):e].strip()
    # Fallback: no <answer> tag. Only trust as an answer if it's short.
    # Otherwise (long prose / reasoning-without-tag) return empty so it does
    # NOT pollute vote clustering with _norm'd garbage like "ineedtoanalyze…".
    candidate = t.strip()
    if not candidate:
        return ""
    token_count = len(candidate.split())
    if token_count >= 10 or len(candidate) > 120:
        return ""
    return _norm(candidate)


def _build_caption_section(group, anchor_colors):
    """Append per-region captions so the answering model has the SAME
    information the designer had when it wrote hypothetical_answer. Color
    anchors are referenced (not raw region IDs) so the prompt stays
    consistent with how the question text refers to regions.
    """
    if not group:
        return ""
    lines = []
    for r in group:
        rid = r["region_id"]
        cap = (r.get("caption") or "").strip()
        if not cap:
            continue
        color_name = None
        if anchor_colors and rid in anchor_colors:
            color_name = anchor_colors[rid][0]
        ref = f"the {color_name} box" if color_name else rid
        lines.append(f"- {ref}: {cap}")
    if not lines:
        return ""
    return " and the following region captions (use them as supplementary evidence; the image is the ground truth):\n" + "\n".join(lines)


def _answer_image_once(image_path, question, api_url, api_key, model,
                       caption_section="", ablation_instruction="",
                       answer_vocabulary_section=""):
    image_b64 = file_to_b64(image_path)
    prompt = _ANSWER_PROMPT_TPL.format(
        question=question,
        caption_section=caption_section,
        ablation_instruction=ablation_instruction,
        answer_vocabulary_section=answer_vocabulary_section,
    )
    text = call_chat(
        api_url, api_key, model,
        [{"role": "user", "content": [image_message(image_b64), {"type": "text", "text": prompt}]}],
        temperature=0.4,
        top_p=0.9,
        max_tokens=768,
        enable_thinking=False,
    )
    return _strip_answer(text), text


_INSUFFICIENT_PATTERNS = (
    "insufficientinformation", "cannotdetermine", "cannotbe determined",
    "not enough information", "unabletodetermine", "unknowable",
    "信息不足", "无法确定", "不能确定", "无法判断",
)


def _is_insufficient(answer):
    raw = str(answer or "").strip().lower()
    compact = _norm(raw)
    return any(p.replace(" ", "") in compact or p in raw
               for p in _INSUFFICIENT_PATTERNS)


def _drop_instruction(necessity_mode):
    if necessity_mode in {"set_complete", "pair_complete"}:
        return (
            "\nABLATION VIEW: one colored candidate box has been replaced by a gray "
            "masked box. The masked box is STILL a member of the explicitly closed "
            "candidate/pairing set; do not silently remove it from consideration. "
            "If its hidden content prevents proving the complete subset, unique "
            "anomaly, global extremum, unique match, or complete pairing, answer "
            "<answer>insufficient information</answer>.\n"
        )
    if necessity_mode == "chain_dependency":
        return (
            "\nABLATION VIEW: one required clue region has been replaced by a gray "
            "masked box. If that missing clue breaks the reasoning chain, answer "
            "<answer>insufficient information</answer>; do not guess the final answer.\n"
        )
    return (
        "\nABLATION VIEW: one required aggregate region has been replaced by a gray "
        "masked box. If the full aggregate cannot be computed, answer "
        "<answer>insufficient information</answer>; do not count only visible boxes.\n"
    )


def _extract_answer_label_sequence(answer, answer_vocabulary, anchor_colors):
    raw = str(answer or "").lower()
    found = []
    for label in answer_vocabulary or []:
        label_raw = str(label).strip().lower()
        pos = raw.find(label_raw)
        if label_raw and pos >= 0:
            found.append((pos, label_raw))
    if found:
        return [label for _, label in sorted(found)]
    for _, (color_name, _) in (anchor_colors or {}).items():
        match = _re.search(rf"\b{_re.escape(str(color_name).lower())}\b", raw)
        if match:
            found.append((match.start(), str(color_name).lower()))
    return [color for _, color in sorted(found)]


def _answer_matches(answer, gt, answer_semantics, anchor_colors, answer_vocabulary=None):
    if answer_semantics == "unordered_set":
        a = _extract_answer_label_sequence(answer, answer_vocabulary, anchor_colors)
        b = _extract_answer_label_sequence(gt, answer_vocabulary, anchor_colors)
        return bool(a and b and set(a) == set(b))
    if answer_semantics == "ordered_pair":
        a = _extract_answer_label_sequence(answer, answer_vocabulary, anchor_colors)
        b = _extract_answer_label_sequence(gt, answer_vocabulary, anchor_colors)
        return bool(len(a) == 2 and len(b) == 2 and a == b)
    return norm_match(answer, gt)


def _answer_vote_key(answer, answer_semantics, anchor_colors, answer_vocabulary=None):
    if _is_insufficient(answer):
        return _norm("insufficient information")
    labels = _extract_answer_label_sequence(answer, answer_vocabulary, anchor_colors)
    if answer_semantics == "unordered_set" and labels:
        return "+".join(sorted(set(labels)))
    if answer_semantics == "ordered_pair" and len(labels) == 2:
        return "+".join(labels)
    return _norm(answer)


_JUDGE_PROMPT_TPL = """You are an answer-equivalence judge for a VQA task.

Question: {question}

Below are {n} answers (one per line, in the form A1: ...). Group them by SEMANTIC EQUIVALENCE: answers that mean the same thing even if phrased differently (e.g. "16" and "sixteen"; "kneeling soldier" and "soldier kneeling"; "yes" and "Yes").
Answers that are clearly different (e.g. "3" vs "4"; "red" vs "blue"; "yes" vs "no") MUST go in separate groups.

Answers:
{answers_block}

Return STRICT JSON only, no prose. Schema:
{{
  "groups": [["<canonical form>", [<A-index as integer>, ...]], ...],
  "largest_group_size": <int>,
  "canonical": "<canonical form of the largest group>"
}}

Rules:
- Every A1..A{n} appears in exactly one group.
- Ties (two groups same size): pick the one with the lexicographically smallest canonical form.
- If all answers are different, largest_group_size = 1.
"""


def _llm_judge_largest_group(question, answers, api_url, api_key, model):
    """Use a text LLM to cluster answers by semantic equivalence.
    Returns (largest_size, canonical_form, one_based_member_indices).
    """
    answers = [str(a) for a in answers if a is not None]
    if not answers:
        return 0, "", []
    # Filter out empty / blank answers from clustering
    non_empty = [(i + 1, a) for i, a in enumerate(answers) if a.strip()]
    if not non_empty:
        return 0, "", []
    answers_block = "\n".join(f"A{i}: {a}" for i, a in non_empty)
    prompt = _JUDGE_PROMPT_TPL.format(
        question=question,
        n=len(non_empty),
        answers_block=answers_block,
    )
    try:
        text = call_chat(
            api_url, api_key, model,
            [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            temperature=0.0,
            top_p=None,
            max_tokens=512,
            enable_thinking=False,
        )
    except Exception:
        return 0, "", []
    # Parse JSON from text (may be wrapped in code fences)
    import json as _json
    import re as _re
    m = _re.search(r"\{[\s\S]*\}", text or "")
    if not m:
        return 0, "", []
    try:
        obj = _json.loads(m.group(0))
    except Exception:
        return 0, "", []
    size = int(obj.get("largest_group_size", 0) or 0)
    canonical = str(obj.get("canonical", "") or "")
    members = []
    for group in obj.get("groups") or []:
        if not isinstance(group, list) or len(group) != 2:
            continue
        group_canonical, group_members = group
        if _norm(str(group_canonical)) != _norm(canonical):
            continue
        if isinstance(group_members, list):
            members = [
                int(i) for i in group_members
                if isinstance(i, int) and 1 <= i <= len(answers)
            ]
        break
    if len(members) != size:
        members = []
    return size, canonical, members


def _load_anchor_colors(candidate, group):
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


def self_check_candidate(record, candidate,
                         primary_api_url, primary_api_key, primary_model,
                         secondary_api_url, secondary_api_key, secondary_model,
                         primary_samples=2, secondary_samples=2,
                         min_priv_votes=3, drop_samples=1,
                         drop_confirm_samples=3, drop_min_matches=2,
                         drop_fail_threshold=2,
                         canvas_size=1024, work_dir=None,
                         judge_api_url=None, judge_api_key=None,
                         judge_model=None):
    """Return dict with pass / reason / primary_answers / secondary_answers /
    drop_answers / gt / vote_match_count.

    Voting stage (GT derived from votes, NOT from designer's hypothetical_answer):
      - privileged_layout sampled `primary_samples` times with primary_model
      - privileged_layout sampled `secondary_samples` times with secondary_model
      - The GT is the most common normalized answer among the votes.
      - Candidate passes voting only if the largest equivalent-answer group has
        at least `min_priv_votes` votes. Equivalence is decided first by fast
        _norm string equality; if that doesn't reach the threshold, an LLM
        judge (judge_model, default GLM-5.2) re-clusters the answers by
        semantic equivalence.
    Drop-one stage (primary model only):
      - answer_sensitive: reject stable GT recovery after masking a contributor
      - set_complete / pair_complete / chain_dependency: require the model to
        recognize that the masked member/clue makes the answer insufficient
    The designer's `hypothetical_answer` is logged for diagnostics only.
    """
    group = candidate["evidence_regions"]
    question = candidate["question"]
    image_id = record["image_id"]
    qid = candidate["candidate_id"]
    designer_hint = (candidate.get("hypothetical_answer") or "").strip()
    necessity_mode = (candidate.get("necessity_mode") or "answer_sensitive").strip().lower()
    answer_semantics = (candidate.get("answer_semantics") or "scalar").strip().lower()
    answer_vocabulary = [
        str(x).strip() for x in (candidate.get("answer_vocabulary") or [])
        if str(x).strip()
    ]
    open_answer_space = candidate.get("answer_space") == "open"
    answer_vocabulary_section = ""
    if answer_vocabulary:
        answer_vocabulary_section = (
            "\nAllowed canonical answer labels (this is a neutral vocabulary, not "
            "the correct answer): " + " | ".join(answer_vocabulary) +
            ". Use these labels exactly. For multiple labels, join them with +.\n"
        )
    valid_modes = {"answer_sensitive", "set_complete", "pair_complete", "chain_dependency"}
    if necessity_mode not in valid_modes:
        return {"pass": False, "reason": "invalid_necessity_mode",
                "necessity_mode": necessity_mode, "drop_answers": {}}
    if answer_semantics not in {"scalar", "unordered_set", "ordered_pair"}:
        return {"pass": False, "reason": "invalid_answer_semantics",
                "answer_semantics": answer_semantics, "drop_answers": {}}
    base_dir = work_dir or os.path.join(tempfile.mkdtemp(prefix="selfcheck_"))
    base = os.path.join(base_dir, image_id, qid)
    os.makedirs(base, exist_ok=True)

    source_path = os.path.join(base_dir, image_id, "original.jpg")
    with _materialize_lock:
        if not os.path.exists(source_path):
            load_image(record["image_path"]).save(
                source_path, "JPEG", quality=98, subsampling=0
            )

    # Build privileged_layout + drop-one views with the SAME anchor color
    # palette the question text was generated against.
    anchor_colors = _load_anchor_colors(candidate, group)
    priv_path = os.path.join(base, "privileged_layout.jpg")
    _, anchor_colors = make_layout_preserving_canvas(
        source_path, group, priv_path, canvas_size=canvas_size,
        anchor_colors=anchor_colors,
        color_seed=None if anchor_colors else f"{record['image_id']}:{qid}",
    )
    drop_paths = {}
    for region in group:
        rid = region["region_id"]
        dp = os.path.join(base, f"drop_{rid}.jpg")
        make_layout_preserving_canvas(
            source_path, group, dp, drop_region_id=rid,
            canvas_size=canvas_size, anchor_colors=anchor_colors,
        )
        drop_paths[rid] = dp

    # Voting: privileged pixels only. Region captions are proposal metadata and
    # may contain model errors or answer-bearing details, so exposing them here
    # would make answer anchoring partially text-grounded.
    # All vote requests are independent — fan them out in a thread pool to
    # amortize latency. Each call_chat uses the multi-key pool internally so
    # multiple keys + per-key semaphore keep QPS under control.
    def _vote_primary(_):
        return _answer_image_once(priv_path, question,
                                  primary_api_url, primary_api_key,
                                  primary_model,
                                  caption_section="",
                                  answer_vocabulary_section=answer_vocabulary_section)

    def _vote_secondary(_):
        return _answer_image_once(priv_path, question,
                                  secondary_api_url, secondary_api_key,
                                  secondary_model,
                                  caption_section="",
                                  answer_vocabulary_section=answer_vocabulary_section)

    def _run_vote_round(n_primary, n_secondary):
        fns = [_vote_primary] * n_primary + [_vote_secondary] * n_secondary
        if not fns:
            return [], []
        with ThreadPoolExecutor(max_workers=len(fns)) as ex:
            pairs = list(ex.map(lambda fn, i: fn(i), fns, range(len(fns))))
        return pairs[:n_primary], pairs[n_primary:]

    # Broad generation creates many weak candidates. Use a 2+2 privileged
    # precheck, then spend the remaining 2+2 calls only on candidates with an
    # early cross-model consensus. Accepted candidates still satisfy the full
    # production 4+4 criterion below.
    effective_secondary_samples = secondary_samples if secondary_api_url else 0
    # Open-vocabulary answers may be semantically equivalent without being
    # string-identical. Give them the full vote round so the semantic judge can
    # cluster paraphrases before any candidate is rejected.
    initial_primary = primary_samples if open_answer_space else min(2, primary_samples)
    initial_secondary = (
        effective_secondary_samples if open_answer_space
        else min(2, effective_secondary_samples)
    )
    primary_pairs, secondary_pairs = _run_vote_round(
        initial_primary, initial_secondary
    )
    if (primary_samples > initial_primary or
            effective_secondary_samples > initial_secondary):
        pre_primary = [p[0] for p in primary_pairs]
        pre_secondary = [p[0] for p in secondary_pairs]
        pre_keys = [
            _answer_vote_key(a, answer_semantics, anchor_colors, answer_vocabulary)
            for a in pre_primary + pre_secondary
        ]
        pre_keys = [k for k in pre_keys if k]
        pre_counter = Counter(pre_keys)
        pre_top, pre_count = pre_counter.most_common(1)[0] if pre_counter else ("", 0)
        primary_support = sum(
            _answer_vote_key(a, answer_semantics, anchor_colors,
                             answer_vocabulary) == pre_top
            for a in pre_primary
        )
        secondary_support = sum(
            _answer_vote_key(a, answer_semantics, anchor_colors,
                             answer_vocabulary) == pre_top
            for a in pre_secondary
        )
        if (not pre_top or pre_top == _norm("insufficient information") or
                pre_count < 3 or primary_support < 1 or
                (pre_secondary and secondary_support < 1)):
            return {
                "pass": False,
                "reason": "privileged_precheck_disagreement",
                "necessity_mode": necessity_mode,
                "answer_semantics": answer_semantics,
                "primary_answers": pre_primary,
                "secondary_answers": pre_secondary,
                "precheck_top": pre_top,
                "precheck_count": pre_count,
                "drop_answers": {},
            }
        more_primary, more_secondary = _run_vote_round(
            primary_samples - initial_primary,
            effective_secondary_samples - initial_secondary,
        )
        primary_pairs.extend(more_primary)
        secondary_pairs.extend(more_secondary)

    # Each pair is (stripped_answer, raw_model_text).
    primary_answers = [p[0] for p in primary_pairs]
    secondary_answers = [p[0] for p in secondary_pairs]
    primary_raw = [p[1] for p in primary_pairs]
    secondary_raw = [p[1] for p in secondary_pairs]
    votes = primary_answers + secondary_answers
    vote_norms = [
        v for v in (
            _answer_vote_key(a, answer_semantics, anchor_colors, answer_vocabulary)
            for a in votes
        ) if v
    ]
    if not vote_norms:
        return {"pass": False, "reason": "privileged_all_empty",
                "designer_hint": designer_hint,
                "designer_hint_matches_voted_gt": False,
                "primary_answers": primary_answers,
                "secondary_answers": secondary_answers,
                "primary_raw": primary_raw,
                "secondary_raw": secondary_raw,
                "drop_answers": {}}

    # Insufficient-majority check: if half or more of votes are
    # "insufficient information", the question is unanswerable even on the
    # privileged view — fail.
    insuf_norm = _norm("insufficient information")
    insuf = sum(1 for v in vote_norms if v == insuf_norm)
    if insuf >= max(2, len(vote_norms) // 2 + 1):
        return {"pass": False, "reason": "privileged_insufficient_majority",
                "designer_hint": designer_hint,
                "designer_hint_matches_voted_gt": _matches_designer(designer_hint, insuf_norm),
                "primary_answers": primary_answers,
                "secondary_answers": secondary_answers,
                "primary_raw": primary_raw,
                "secondary_raw": secondary_raw,
                "drop_answers": {}}

    # GT = most common normalized answer. First count by exact _norm equality
    # (fast path). If that doesn't reach min_priv_votes, fall back to an LLM
    # judge that clusters the answers by semantic equivalence — this lets
    # natural-language answers phrased differently still count as the same GT.
    counter = Counter(vote_norms)
    top_norm, top_count = counter.most_common(1)[0]
    judge_used = False
    judge_canonical = ""
    judge_members = []
    if top_count < min_priv_votes and judge_api_url and judge_api_key and judge_model:
        size, canonical, members = _llm_judge_largest_group(
            question, votes,
            judge_api_url, judge_api_key, judge_model,
        )
        if size > top_count:
            top_count = size
            judge_used = True
            judge_canonical = canonical
            judge_members = members
            # If the judge returned a non-empty canonical, prefer it as GT;
            # else fall back to top_norm.
            if canonical.strip():
                top_norm = _norm(canonical) or top_norm
    if top_count < min_priv_votes:
        return {"pass": False, "reason": "vote_no_majority",
                "designer_hint": designer_hint,
                "designer_hint_matches_voted_gt": _matches_designer(designer_hint, top_norm),
                "primary_answers": primary_answers,
                "secondary_answers": secondary_answers,
                "primary_raw": primary_raw,
                "secondary_raw": secondary_raw,
                "vote_top_norm": top_norm,
                "vote_top_count": top_count,
                "min_required": min_priv_votes,
                "judge_used": judge_used,
                "judge_canonical": judge_canonical,
                "judge_members": judge_members,
                "drop_answers": {}}
    # Reject "insufficient information" as the winning GT — the question is
    # unanswerable even if 3+ votes say so.
    if top_norm == insuf_norm:
        return {"pass": False, "reason": "privileged_insufficient_majority",
                "designer_hint": designer_hint,
                "designer_hint_matches_voted_gt": False,
                "primary_answers": primary_answers,
                "secondary_answers": secondary_answers,
                "judge_used": judge_used,
                "judge_canonical": judge_canonical,
                "drop_answers": {}}
    # Find a raw answer (un-normalized) corresponding to top_norm, prefer
    # primary model's wording when available; if the judge produced a
    # canonical form, prefer that when no raw answer matched.
    gt = next((a for a in primary_answers
               if _answer_vote_key(a, answer_semantics, anchor_colors,
                                   answer_vocabulary) == top_norm), None) \
        or next((a for a in secondary_answers
                 if _answer_vote_key(a, answer_semantics, anchor_colors,
                                     answer_vocabulary) == top_norm), None) \
        or (judge_canonical if judge_used and judge_canonical.strip() else None) \
        or top_norm

    # Cross-confirmation means both model families must independently support
    # the anchor, not merely contribute to one pooled majority. With the
    # production 4+4 setting this requires at least 3 matching votes per model.
    if judge_used and judge_members:
        primary_match_count = sum(i <= len(primary_answers) for i in judge_members)
        secondary_match_count = sum(i > len(primary_answers) for i in judge_members)
    else:
        primary_match_count = sum(
            _answer_matches(a, gt, answer_semantics, anchor_colors, answer_vocabulary)
            for a in primary_answers
        )
        secondary_match_count = sum(
            _answer_matches(a, gt, answer_semantics, anchor_colors, answer_vocabulary)
            for a in secondary_answers
        )
    primary_required = max(1, math.ceil(primary_samples * 0.75))
    secondary_required = max(1, math.ceil(secondary_samples * 0.75)) if secondary_answers else 0
    if (primary_match_count < primary_required or
            (secondary_answers and secondary_match_count < secondary_required)):
        return {
            "pass": False,
            "reason": "cross_model_disagreement",
            "necessity_mode": necessity_mode,
            "answer_semantics": answer_semantics,
            "designer_hint": designer_hint,
            "designer_hint_matches_voted_gt": _matches_designer(designer_hint, gt),
            "voted_gt": gt,
            "primary_answers": primary_answers,
            "secondary_answers": secondary_answers,
            "primary_raw": primary_raw,
            "secondary_raw": secondary_raw,
            "primary_match_count": primary_match_count,
            "secondary_match_count": secondary_match_count,
            "primary_required": primary_required,
            "secondary_required": secondary_required,
            "vote_top_count": top_count,
            "judge_used": judge_used,
            "judge_canonical": judge_canonical,
            "judge_members": judge_members,
            "drop_answers": {},
        }

    # Drop-one necessity uses adaptive confirmation. The suspicious condition
    # depends on the question's necessity mode.
    ablation_instruction = _drop_instruction(necessity_mode)
    drop_tasks = []
    for region in group:
        rid = region["region_id"]
        for _ in range(drop_samples):
            drop_tasks.append((rid, drop_paths[rid]))
    drop_answers = {}
    drop_raw = {}
    fail_region = None
    fail_reason = None
    per_region = {}
    per_region_raw = {}
    if drop_tasks:
        with ThreadPoolExecutor(max_workers=min(len(drop_tasks), 8)) as ex:
            drop_results = list(ex.map(
                lambda item: (item[0], _answer_image_once(item[1], question,
                                                          primary_api_url,
                                                          primary_api_key,
                                                          primary_model,
                                                          ablation_instruction=ablation_instruction,
                                                          answer_vocabulary_section=answer_vocabulary_section)),
                drop_tasks
            ))
        # Each result is now (rid, (stripped_answer, raw_model_text)).
        for rid, (ans, raw) in drop_results:
            per_region.setdefault(rid, []).append(ans)
            per_region_raw.setdefault(rid, []).append(raw)

        # Across all modes, a region is suspicious only when the ablated view
        # still recovers the original reference (or returns no parseable
        # answer). A changed answer and "insufficient" both demonstrate that
        # removing the region affected the task.
        suspects = [
            rid for rid, answers in per_region.items()
            if any(not str(a).strip() or
                   _answer_matches(a, gt, answer_semantics, anchor_colors,
                                   answer_vocabulary)
                   for a in answers)
        ]
        confirm_tasks = []
        target_samples = max(drop_samples, drop_confirm_samples)
        for rid in suspects:
            for _ in range(max(0, target_samples - len(per_region[rid]))):
                confirm_tasks.append((rid, drop_paths[rid]))
        if confirm_tasks:
            with ThreadPoolExecutor(max_workers=min(len(confirm_tasks), 8)) as ex:
                confirm_results = list(ex.map(
                    lambda item: (item[0], _answer_image_once(
                        item[1], question, primary_api_url, primary_api_key,
                        primary_model,
                        ablation_instruction=ablation_instruction,
                        answer_vocabulary_section=answer_vocabulary_section)),
                    confirm_tasks,
                ))
            for rid, (ans, raw) in confirm_results:
                per_region.setdefault(rid, []).append(ans)
                per_region_raw.setdefault(rid, []).append(raw)

        for rid, ans_list in per_region.items():
            drop_answers[rid] = ans_list[0] if len(ans_list) == 1 else ans_list
            raw_list = per_region_raw[rid]
            drop_raw[rid] = raw_list[0] if len(raw_list) == 1 else raw_list
        # Evaluate each region according to its necessity semantics.
        # B-option tolerance: a question is rejected only when at least
        # `drop_fail_threshold` regions are individually sufficient (i.e. masking
        # each one still recovers the GT). A single redundant region is tolerated,
        # since a small region may coincidentally leak the answer through a
        # cross-region cue without making the question truly single-region.
        # drop_one_inconclusive (empty/unparseable) still fails immediately — that
        # is a measurement problem, not redundancy.
        sufficient_regions = []
        inconclusive_region = None
        for region in group:
            rid = region["region_id"]
            answers = per_region.get(rid, [])
            non_empty = [a for a in answers if str(a).strip()]
            match_count = sum(
                _answer_matches(a, gt, answer_semantics, anchor_colors,
                                answer_vocabulary)
                for a in non_empty
            )
            if match_count >= drop_min_matches:
                sufficient_regions.append(rid)
            elif not non_empty:
                if inconclusive_region is None:
                    inconclusive_region = rid
        fail_region = None
        fail_reason = None
        if len(sufficient_regions) >= drop_fail_threshold:
            fail_region = ",".join(sufficient_regions)
            fail_reason = "drop_one_sufficient"
        elif sufficient_regions:
            # Below threshold: tolerate the redundant region(s) but record them.
            pass
        elif inconclusive_region is not None:
            fail_region = inconclusive_region
            fail_reason = "drop_one_inconclusive"

    drop_match_counts = {
        rid: sum(_answer_matches(a, gt, answer_semantics, anchor_colors,
                                 answer_vocabulary)
                 for a in answers)
        for rid, answers in per_region.items()
    }
    drop_insufficient_counts = {
        rid: sum(_is_insufficient(a) for a in answers if str(a).strip())
        for rid, answers in per_region.items()
    }

    if fail_region is not None:
        return {"pass": False, "reason": f"{fail_reason}:{fail_region}",
                "necessity_mode": necessity_mode,
                "answer_semantics": answer_semantics,
                "designer_hint": designer_hint,
                "designer_hint_matches_voted_gt": _matches_designer(designer_hint, gt),
                "voted_gt": gt,
                "primary_answers": primary_answers,
                "secondary_answers": secondary_answers,
                "primary_raw": primary_raw,
                "secondary_raw": secondary_raw,
                "primary_match_count": primary_match_count,
                "secondary_match_count": secondary_match_count,
                "vote_top_count": top_count,
                "judge_used": judge_used,
                "judge_canonical": judge_canonical,
                "drop_match_counts": drop_match_counts,
                "drop_insufficient_counts": drop_insufficient_counts,
                "drop_min_matches": drop_min_matches,
                "drop_answers": drop_answers,
                "drop_raw": drop_raw}

    return {"pass": True, "reason": "ok",
            "necessity_mode": necessity_mode,
            "answer_semantics": answer_semantics,
            "designer_hint": designer_hint,
            "designer_hint_matches_voted_gt": _matches_designer(designer_hint, gt),
            "voted_gt": gt,
            "primary_answers": primary_answers,
            "secondary_answers": secondary_answers,
            "primary_raw": primary_raw,
            "secondary_raw": secondary_raw,
            "primary_match_count": primary_match_count,
            "secondary_match_count": secondary_match_count,
            "vote_top_count": top_count,
            "judge_used": judge_used,
            "judge_canonical": judge_canonical,
            "drop_match_counts": drop_match_counts,
            "drop_insufficient_counts": drop_insufficient_counts,
            "drop_min_matches": drop_min_matches,
            "privileged_layout_path": priv_path,
            "drop_answers": drop_answers,
            "drop_raw": drop_raw}
