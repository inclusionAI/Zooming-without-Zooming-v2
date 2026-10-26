import ast
import json
import os
import random
import re
import tempfile
from decimal import Decimal, InvalidOperation
from itertools import combinations

from .answer_norm import _norm
from .image_utils import assign_anchor_colors, file_to_b64, make_layout_preserving_canvas
from .llm import call_chat, extract_json, image_message
from .question_templates import select_question_templates
from .reasoning_certificate import validate_reasoning_certificate


QUESTION_TYPE_TO_MODE = {
    "aggregate_count": "answer_sensitive",
    "aggregate_attribute_count": "answer_sensitive",
    "aggregate_numeric": "answer_sensitive",
    "set_filtering": "set_complete",
    "anomaly_detection": "set_complete",
    "extremum": "set_complete",
    "candidate_matching": "set_complete",
    "cross_region_pairing": "pair_complete",
    "ocr_visual_verification": "chain_dependency",
    "spatial_chain": "chain_dependency",
    "dependent_chain": "chain_dependency",
    "viewpoint_transform": "chain_dependency",
    "agent_relative_direction": "chain_dependency",
    "geometric_line_relation": "set_complete",
    "topological_relation": "set_complete",
    "depth_ordering": "set_complete",
    "spatial_transitivity": "chain_dependency",
    "cross_image_composition": "chain_dependency",
    "arithmetic_composition": "answer_sensitive",
    "conditional_composition": "chain_dependency",
    "boolean_composition": "chain_dependency",
    "relational_composition": "chain_dependency",
    "multi_hop_reasoning": "chain_dependency",
}

_SEMANTIC_CLASS_TERMS = {
    "person": ("person", "people", "woman", "man", "pedestrian", "girl", "boy", "child"),
    "vehicle": ("car", "vehicle", "truck", "bus", "van", "volvo", "mercedes", "lift"),
    "window": ("window",),
    "sign": ("sign", "billboard", "placard"),
    "door": ("door", "gate"),
    "shoe": ("shoe", "sneaker", "sandal", "footwear"),
    "rock": ("rock", "stone", "boulder"),
}

ANCHOR_COLOR_WORDS = {
    "red", "orange", "yellow", "green", "blue", "purple", "pink",
    "magenta", "cyan", "white", "black", "gray", "grey", "brown",
    "红", "橙", "黄", "绿", "蓝", "紫", "粉", "白", "黑", "灰", "棕", "褐",
    "红色", "橙色", "黄色", "绿色", "蓝色", "紫色", "粉色", "白色", "黑色",
}

SUBJECTIVE_QUESTION_PHRASES = (
    "significantly different", "appears to be", "seems to be",
    "oriented differently", "different orientation", "orientation differs",
    "prominent rock", "surface area based on", "largest vertical span",
    "looks larger", "looks smaller", "most noticeable", "least noticeable",
)

OPEN_ANSWER_KINDS = {"semantic", "text"}

QUESTION_TYPE_ALIASES = {
    "comparison": "relational_composition",
    "relation_comparison": "relational_composition",
    "multi_hop": "multi_hop_reasoning",
    "multihop_reasoning": "multi_hop_reasoning",
    "spatial_reasoning": "spatial_chain",
    "counting": "aggregate_count",
    "attribute_counting": "aggregate_attribute_count",
    "odd_one_out": "anomaly_detection",
    "matching": "candidate_matching",
}


def uses_open_answer_space(answer_semantics, final_answer_kind):
    """Return whether strong models should answer without a supplied label set."""
    return (
        str(answer_semantics or "").strip().lower() == "scalar"
        and str(final_answer_kind or "").strip().lower() in OPEN_ANSWER_KINDS
    )


def open_answer_word_count(answer):
    """Count English words and approximate CJK length in two-character units."""
    tokens = re.findall(
        r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[\u3400-\u9fff]+",
        str(answer or "").strip(),
    )
    return sum(
        (len(token) + 1) // 2 if re.fullmatch(r"[\u3400-\u9fff]+", token) else 1
        for token in tokens
    )


def classify_final_answer_kind(answer, anchor_colors=None):
    """Classify the observable answer form, independent of question_type."""
    raw = str(answer or "").strip().lower()
    if not raw:
        return "unknown"
    if re.fullmatch(r"-?\d+(?:\.\d+)?", raw):
        return "numeric"
    if raw in {
        "both true", "only the first", "only the second", "neither",
        "yes", "no", "true", "false",
    }:
        return "boolean"
    colors = {
        str(value[0] if isinstance(value, (list, tuple)) else value).lower()
        for value in (anchor_colors or {}).values()
    } or ANCHOR_COLOR_WORDS
    atoms = [x.strip() for x in re.split(r"\s*\+\s*", raw) if x.strip()]

    def is_anchor_atom(atom):
        if atom in colors:
            return True
        return any(re.search(
            rf"\b{re.escape(color)}\b\s+(?:box|region|outline)", atom
        ) for color in colors if re.fullmatch(r"[a-z]+", color))

    if atoms and all(is_anchor_atom(atom) for atom in atoms):
        return "anchor"
    return "semantic"


def has_anchor_attribute_leak(question, answer, anchor_colors=None):
    """Detect cases where an answer color is stated as an object clue, not an anchor."""
    raw_answer = str(answer or "").strip().lower()
    colors = {
        str(value[0] if isinstance(value, (list, tuple)) else value).lower()
        for value in (anchor_colors or {}).values()
    } or ANCHOR_COLOR_WORDS
    if raw_answer not in colors or not re.fullmatch(r"[a-z]+", raw_answer):
        return False
    q = str(question or "").lower()
    anchor_uses = (
        rf"\b{re.escape(raw_answer)}\b\s+(?:(?:marked|outlined|colored)\s+)?(?:box|region|outline)",
        rf"(?:box|region|outline)\s+(?:in\s+)?\b{re.escape(raw_answer)}\b",
    )
    for pattern in anchor_uses:
        q = re.sub(pattern, " ", q)
    following_words = re.findall(
        rf"\b{re.escape(raw_answer)}\b\s+([a-z-]+)", q
    )
    anchor_or_list_words = {
        "box", "boxes", "region", "regions", "outline", "outlines",
        "outlined", "marked", "colored", "and", "or",
    }
    return any(word not in anchor_or_list_words for word in following_words)


def answer_kind_matches_declared(declared_kind, observed_kind):
    declared_kind = str(declared_kind or "").strip().lower()
    observed_kind = str(observed_kind or "").strip().lower()
    if declared_kind in {"anchor", "numeric", "boolean"}:
        return declared_kind == observed_kind
    if declared_kind in {"semantic", "semantic_set", "semantic_pair"}:
        return observed_kind == "semantic"
    # A text-reading template may legitimately return a digit string.
    if declared_kind == "text":
        return observed_kind in {"semantic", "numeric"}
    return False


def normalize_question_type(question_type):
    """Map common model-generated aliases onto the audited question ontology."""
    raw = str(question_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    return QUESTION_TYPE_ALIASES.get(raw, raw)


def infer_answer_semantics(question_type, final_answer_kind):
    """Infer the structural answer shape from the normalized type and answer kind."""
    qtype = normalize_question_type(question_type)
    answer_kind = str(final_answer_kind or "").strip().lower()
    if qtype == "cross_region_pairing" or answer_kind == "semantic_pair":
        return "ordered_pair"
    if qtype == "set_filtering" or answer_kind == "semantic_set":
        return "unordered_set"
    return "scalar"


_OUTPUT_REQUEST_VERBS = (
    "arrange", "order", "rank", "list", "enumerate", "name", "identify",
    "state", "report", "give", "provide", "specify",
)


def _has_multiple_output_requests(question):
    """Detect coordinated imperatives that explicitly request two outputs."""
    verbs = "|".join(_OUTPUT_REQUEST_VERBS)
    return bool(re.search(
        rf"^\s*(?:please\s+)?(?:first\s+)?(?:{verbs})\b"
        rf".*?\b(?:and|then)\s+(?:also\s+)?(?:{verbs})\b",
        str(question or "").strip().lower(),
        flags=re.DOTALL,
    ))


def validate_question_answer_contract(question, answer, declared_kind,
                                      anchor_colors=None,
                                      answer_semantics="scalar"):
    """Check that the answer form satisfies what the question explicitly requests.

    This deliberately checks response form rather than visual correctness. It catches
    failures such as a question asking for text while the answer names an image/box.
    """
    qtext = str(question or "").strip().lower()
    raw_answer = str(answer or "").strip().lower()
    if not qtext or not raw_answer:
        return False, ["empty_question_or_answer"]

    observed_kind = classify_final_answer_kind(raw_answer, anchor_colors)
    location_answer = observed_kind == "anchor" or bool(re.fullmatch(
        r"(?:the\s+)?(?:image|panel|region|box|outline)\s*(?:number\s*)?\d+",
        raw_answer,
    ))
    text_request = any(pattern in qtext for pattern in (
        "what text", "which text", "identify the text", "read the text",
        "what word", "which word", "identify the word", "read the word",
        "what letter", "which letter", "what character", "which character",
        "text or number", "word or number", "visible text", "visible word",
        "什么文字", "哪个字", "哪些字", "读取文字",
    ))
    numeric_request = (
        "how many" in qtext
        or any(pattern in qtext for pattern in (
            "what is the total", "what is the result", "calculate the total",
            "calculate the value", "总共有多少", "一共有多少", "结果是多少",
        ))
    )
    region_request = bool(re.search(
        r"\bwhich\s+(?:(?:marked|outlined|colored)\s+)?(?:one|region|box|outline)\b",
        qtext,
    )) or "which outline color" in qtext

    reasons = []
    if (str(answer_semantics or "scalar").strip().lower() == "scalar"
            and _has_multiple_output_requests(qtext)):
        reasons.append("multiple_output_requests_for_scalar_answer")
    if text_request and location_answer:
        reasons.append("text_request_answered_with_location")
    if numeric_request and observed_kind != "numeric":
        reasons.append("numeric_request_answered_with_non_numeric")
    if (region_request and str(declared_kind or "").strip().lower() == "anchor"
            and observed_kind != "anchor"):
        reasons.append("region_request_answered_with_non_anchor")
    if not answer_kind_matches_declared(declared_kind, observed_kind):
        reasons.append("declared_answer_kind_mismatch")
    return not reasons, reasons


def _coarse_semantic_class(region):
    text = f"{region.get('label', '')} {region.get('caption', '')}".lower()
    for class_name, terms in _SEMANTIC_CLASS_TERMS.items():
        if any(re.search(rf"\b{re.escape(term)}s?\b", text) for term in terms):
            return class_name
    return ""


def _evaluate_region_expression(expression, region_values):
    """Evaluate a small arithmetic expression over declared region IDs."""
    values = {rid: Decimal(str(value)) for rid, value in region_values.items()}

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Name) and node.id in values:
            return values[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Decimal(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "abs" and len(node.args) == 1
                and not node.keywords):
            return abs(visit(node.args[0]))
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        raise ValueError("unsupported arithmetic expression")

    return visit(ast.parse(expression, mode="eval"))


def _expression_root_is_abs(expression):
    try:
        root = ast.parse(expression, mode="eval").body
    except (SyntaxError, ValueError, TypeError):
        return False
    return (
        isinstance(root, ast.Call)
        and isinstance(root.func, ast.Name)
        and root.func.id == "abs"
        and len(root.args) == 1
        and not root.keywords
    )


def choose_region_groups(regions, image_width, image_height, min_regions=2, max_regions=7,
                         max_regions_for_groups=10, max_groups=5, min_spread=0.08):
    regions = sorted(
        regions,
        key=lambda r: (r.get("grounding_confidence") or 0, -abs((r.get("area_ratio") or 0) - 0.03)),
        reverse=True,
    )[:max_regions_for_groups]
    by_size = {s: [] for s in range(min_regions, max_regions + 1)}
    for size in range(min_regions, max_regions + 1):
        for combo in combinations(regions, size):
            centers = [r["center"] for r in combo]
            max_dist = max(
                (((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)
                for a, b in combinations(centers, 2)
            )
            if max_dist / max(image_width, image_height) < min_spread:
                continue
            if len({r["label"].lower() for r in combo}) < min(2, size):
                continue
            by_size[size].append(list(combo))
    # Prefer larger groups: the user's target is multi-region perception where
    # more necessary regions are better. Smaller groups are fallback only.
    out = []
    for size in sorted(by_size.keys(), reverse=True):
        bucket = by_size[size]
        random.shuffle(bucket)
        remaining = max_groups - len(out)
        if remaining <= 0:
            break
        out.extend(bucket[:remaining])
    random.shuffle(out)
    return out[:max_groups]


def group_summary(group, image_width, image_height, anchor_colors=None):
    lines = []
    for r in group:
        x1, y1, x2, y2 = r["bbox"]
        caption = r.get("caption", "")
        natural_referent = r.get("natural_referent", "")
        color_part = ""
        if anchor_colors and r["region_id"] in anchor_colors:
            color_name, _ = anchor_colors[r["region_id"]]
            color_part = f" outline_color={color_name};"
        source_part = ""
        if r.get("source_image_id"):
            source_part = (
                f" source_image=Image {r.get('source_image_index', '?')}"
                f" ({r['source_image_id']});"
            )
        lines.append(
            f"- {r['region_id']}: label={r['label']};{color_part}{source_part} "
            f"bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] in image {image_width}x{image_height}; "
            f"center=({r['center'][0]:.0f},{r['center'][1]:.0f}); caption={caption}; "
            f"natural_referent={natural_referent}"
        )
    return "\n".join(lines)


def propose_region_groups(record, api_url, api_key, model,
                          min_regions=2, max_regions=7, max_groups=4,
                          max_candidate_regions=10, return_specs=False):
    """Ask the strong model to pick coherent multi-region evidence sets.

    Returns a list of region groups (each a list of region dicts from `record['regions']`).
    The caller may optionally fall back to choose_region_groups.
    """
    regions = sorted(
        record.get("regions", []),
        key=lambda r: (r.get("grounding_confidence") or 0, -abs((r.get("area_ratio") or 0) - 0.03)),
        reverse=True,
    )[:max_candidate_regions]
    if len(regions) < min_regions:
        return []
    evidence_scope = record.get("evidence_scope", "single_image")
    region_list = "\n".join(
        f"- {r['region_id']}: source=Image {r.get('source_image_index', 1)}; "
        f"label={r['label']}; caption={r.get('caption','')}"
        for r in regions
    )
    prompt = f"""You are an expert VQA data curator. Below are {len(regions)} candidate evidence regions detected in one image.

Image caption (if available): {record.get('caption','')}

Regions:
{region_list}

Propose up to {max_groups} COMBINATIONS of regions that are GOOD for designing natural questions that require many regions.

A GOOD combination is a coherent candidate/evidence set, not a bag of unrelated boxes. Prefer the largest valid set, but every chosen region must have a real role.

Allowed task families and necessity modes:
- aggregate_count: count clearly visible target instances across all regions; every
  region contributes at least one instance. mode=answer_sensitive.
- aggregate_attribute_count: count instances with one objective attribute across all
  regions; every region contributes at least one qualifying instance. mode=answer_sensitive.
- aggregate_numeric: combine one clearly visible number from every region when that is
  natural for the depicted content. mode=answer_sensitive.
- set_filtering: all regions form a closed candidate set and the answer is the complete
  subset satisfying an objective attribute. Candidates must be the same object class
  or one clearly coherent semantic class. mode=set_complete.
- anomaly_detection: identify the unique objectively different member of a closed set.
  mode=set_complete.
- extremum: identify the global maximum/minimum among a closed comparable set.
  mode=set_complete.
- candidate_matching: use a visible reference/clue to select from a closed candidate set.
  mode=set_complete.
- cross_region_pairing: pair coherent people/objects/text regions by an objective rule.
  mode=pair_complete.
- ocr_visual_verification: text in one region constrains visual matching in others.
  mode=chain_dependency.
- spatial_chain / dependent_chain: each region supplies a required next-hop constraint.
  mode=chain_dependency.
- viewpoint_transform: establish a clearly visible person/object/opposite-side frame,
  transform target positions into it, and answer in that frame. Ambiguous orientation
  or rotation axes are forbidden. mode=chain_dependency.
- agent_relative_direction: answer left/right/front/behind relative to a clearly
  oriented person, vehicle, boat, or object, not relative to the camera.
  mode=chain_dependency.
- geometric_line_relation: use a closed set of crisp edges, paths, rails, or structural
  lines to identify parallel, perpendicular, intersecting, converging, or collinear
  members. Distinguish apparent 2D from physical 3D geometry. mode=set_complete.
- topological_relation: compare a closed set for touching, overlap, containment,
  connection, crossing, or disconnection. mode=set_complete.
- depth_ordering: use robust occlusion, ground-contact, and perspective evidence to
  select nearest/farthest or front/behind; pixel size alone is invalid. mode=set_complete.
- spatial_transitivity: compose two or more directional relations through shared
  entities to derive a final relation. mode=chain_dependency.
- cross_image_composition: only for evidence_scope=cross_image; connect evidence from
  at least two labeled source images through correspondence, comparison, matching,
  OCR-to-visual lookup, or within-image relation comparison. mode=chain_dependency.
- arithmetic_composition: read or count one objective value in each region and combine
  them with an explicit short formula. Artificial arithmetic is allowed. Every operand
  region must affect the final value. Prefer heterogeneous facts like people count +
  word count, or OCR number * object count + stripe count. Avoid all-one operands whose
  result is just the number of marked boxes. mode=answer_sensitive.
- conditional_composition: regions determine a condition, while the selected branch
  combines evidence from the remaining regions. Both branches must use the same payload
  regions so the actually executed path still needs every selected region.
  mode=chain_dependency.
- boolean_composition: evaluate two objective predicates over different regions and
  classify the truth pattern as both true, only the first, only the second, or neither.
  This four-state answer is MCQ-compatible. mode=chain_dependency.
- relational_composition: combine spatial, part-whole, material, identity, or matching
  relations across multiple regions into one short answer. mode=chain_dependency.
- multi_hop_reasoning: follow a genuine clue chain, such as OCR -> candidate match ->
  attribute lookup, reference object -> related object -> final property, or derive a
  value in early regions and use it to select/read a later region. mode=chain_dependency.

Selection rules:
1. Has between {min_regions} and {max_regions} regions. Prefer the upper end when the regions form one coherent set.
2. Every region must satisfy its mode: change an aggregate answer, belong to the closed
   candidate/pair set, or supply an indispensable chain constraint.
3. Prefer objective evidence: clear text/numbers, countable objects, repeated object classes, comparable fine parts, or unambiguous attributes.
4. For filtering/anomaly/extremum, candidates must be the same class or clearly
   comparable kind. A mixed people+cars set is forbidden; use cross_region_pairing
   only when unlike classes have an objective relation.
5. Use aggregate_count only when every selected region has a positive contribution.
   If some members contribute zero, use set_filtering or anomaly_detection instead.
6. Spans spatially distinct locations when possible, but coherence is more important than spread.
7. evidence_scope={evidence_scope}. For cross_image, every group must span 2-5 source
   images. Never assert a direct left/right/above/below relation between objects from
   independent images; compare within-image relations or use explicit correspondence.

Return JSON only, an array of objects:
[
  {{"region_ids": ["r000","r001","r003"], "task_family": "anomaly_detection", "necessity_mode": "set_complete", "reason": "all are comparable candidates"}},
  ...
]

Rules:
- Only use region IDs from the list above.
- Don't repeat the same combination or its superset.
- Diversify task_family across proposed groups when the image supports it; do not
  default every repeated-object group to aggregate_count.
- Propose at most one arithmetic_composition group per image. It is a valuable
  minority family, not the default for every heterogeneous region set.
- When possible, propose at least four distinct task families before adding a second
  group from any already-used family.
- necessity_mode must exactly match the task family definition above.
- Prefer 5-7 region groups for coherent counting, filtering, anomaly, extremum, matching,
  arithmetic, and boolean sets when every member is visually useful.
- Use 3-6 regions for pairing, OCR verification, relational composition, conditional
  composition, and dependent or multi-hop chains.
- Use 3-5 regions for viewpoint and agent-relative tasks, with an unmistakable facing
  direction or object heading. Prefer front/back views over ambiguous side views.
- Use 3-7 regions for geometric/topological closed sets only when all relevant edges or
  boundaries are crisp. Depth tasks need at least two independent depth cues.
- Skip if no good combination exists for this image; return [].
"""
    text = call_chat(
        api_url, api_key, model,
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        temperature=0.3,
        top_p=0.9,
        max_tokens=1500,
        enable_thinking=False,
    )
    items = extract_json(text, list)
    if isinstance(items, dict):
        items = items.get("groups") or items.get("region_groups") or items.get("proposals") or []
    if not isinstance(items, list):
        items = []
    valid_ids = {r["region_id"] for r in regions}
    by_id = {r["region_id"]: r for r in regions}
    seen_keys = set()
    groups = []
    # Keep distinct task families first. Repeated families remain available as
    # fallbacks when an image cannot support enough genuinely different tasks.
    diverse_items = []
    duplicate_family_items = []
    proposed_families = set()
    family_aliases = {
        "counting": "aggregate_count",
        "attribute_counting": "aggregate_attribute_count",
        "numeric_aggregation": "aggregate_numeric",
        "filtering": "set_filtering",
        "anomaly": "anomaly_detection",
        "matching": "candidate_matching",
        "pairing": "cross_region_pairing",
        "ocr_verification": "ocr_visual_verification",
        "spatial_reasoning": "spatial_chain",
        "perspective_taking": "viewpoint_transform",
        "viewpoint_reasoning": "viewpoint_transform",
        "agent_relative": "agent_relative_direction",
        "object_relative_direction": "agent_relative_direction",
        "line_relation": "geometric_line_relation",
        "geometric_relation": "geometric_line_relation",
        "topology": "topological_relation",
        "depth_reasoning": "depth_ordering",
        "relative_depth": "depth_ordering",
        "spatial_relation_transitivity": "spatial_transitivity",
        "cross_image": "cross_image_composition",
        "cross_image_reasoning": "cross_image_composition",
        "dependent_reasoning": "dependent_chain",
        "arithmetic": "arithmetic_composition",
        "conditional": "conditional_composition",
        "boolean": "boolean_composition",
        "comparison": "relational_composition",
        "relational_reasoning": "relational_composition",
        "multi_hop": "multi_hop_reasoning",
    }

    def canonical_family(item):
        family = (str(item.get("task_family") or item.get("question_type") or "")
                  .strip().lower())
        return family_aliases.get(family, family)

    for item in items[:max_groups * 4]:
        family = (canonical_family(item)
                  if isinstance(item, dict) else "")
        if family and family not in proposed_families:
            diverse_items.append(item)
            proposed_families.add(family)
        else:
            duplicate_family_items.append(item)
    for item in diverse_items + duplicate_family_items:
        if not isinstance(item, dict):
            continue
        rids = item.get("region_ids") or []
        if not isinstance(rids, list):
            continue
        rids = [str(x) for x in rids if str(x) in valid_ids]
        # dedupe + bound
        rids = list(dict.fromkeys(rids))
        if not (min_regions <= len(rids) <= max_regions):
            continue
        key = tuple(sorted(rids))
        if key in seen_keys:
            continue
        regions_for_group = [by_id[r] for r in rids]
        if evidence_scope == "cross_image":
            source_ids = {
                str(r.get("source_image_id") or "") for r in regions_for_group
                if r.get("source_image_id")
            }
            if not (2 <= len(source_ids) <= 5):
                continue
        task_family = canonical_family(item)
        expected_mode = QUESTION_TYPE_TO_MODE.get(task_family)
        if not expected_mode:
            continue
        if task_family in {"set_filtering", "anomaly_detection", "extremum"}:
            known_classes = {
                c for c in (_coarse_semantic_class(r) for r in regions_for_group) if c
            }
            if len(known_classes) > 1:
                continue
        seen_keys.add(key)
        if return_specs:
            groups.append({
                "regions": regions_for_group,
                "task_family": task_family,
                "necessity_mode": expected_mode,
                "reason": str(item.get("reason") or "").strip(),
            })
        else:
            groups.append(regions_for_group)
        if len(groups) >= max_groups:
            break
    return groups


def generate_questions(record, group, api_url, api_key, model, questions_per_group=3,
                       canvas_size=1536, layout_workdir=None,
                       preferred_task_family=None, preferred_necessity_mode=None,
                       return_diagnostics=False, generation_response_text=None,
                       generation_callable=None, generation_source="qwen"):
    """Design questions giving the strong model ONLY the privileged_layout view.

    The original full image is intentionally NOT shown, so the designer's visual
    information is exactly what the answering model will see at validation time
    (and at self-check). This forces every question to genuinely depend on the
    region crops, not on background pixels stolen from the full image.
    """
    num_regions = len(group)
    evidence_scope = record.get("evidence_scope", "single_image")
    source_image_ids = list(dict.fromkeys(
        str(r.get("source_image_id")) for r in group if r.get("source_image_id")
    ))
    # Stable per (image, group) color anchors so question text and downstream
    # privileged_layout / drop-one views all see the same colors.
    seed = f"{record['image_id']}:{'|'.join(r['region_id'] for r in group)}"
    anchor_colors = assign_anchor_colors(group, seed=seed)
    color_to_id = {color_name: rid for rid, (color_name, _) in anchor_colors.items()}
    color_list_str = ", ".join(f"{c}=box of {color_to_id[c]}" for c in color_to_id)
    freeform_slots = min(4, max(1, (questions_per_group + 1) // 2))
    template_slots = max(1, questions_per_group - freeform_slots)
    template_specs = select_question_templates(
        num_regions, template_slots, seed,
        preferred_type=preferred_task_family,
        evidence_scope=record.get("evidence_scope", "single_image"),
    )
    selected_templates = {spec["template_id"]: spec for spec in template_specs}
    template_block = "\n".join(
        f"- template_id={spec['template_id']}; type={spec['question_type']}; "
        f"mode={spec.get('necessity_mode') or QUESTION_TYPE_TO_MODE[spec['question_type']]}; "
        f"answer_semantics={spec['answer_semantics']}; "
        f"final_answer_kind={spec['final_answer_kind']}; blueprint={spec['instruction']}"
        for spec in template_specs
    )

    # Build the privileged layout image first; this is the ONLY image input.
    base_dir = layout_workdir or os.path.join(tempfile.mkdtemp(prefix="qgen_"))
    base = os.path.join(base_dir, record["image_id"],
                        "|".join(r["region_id"] for r in group))
    if generation_source != "qwen":
        base = os.path.join(base, generation_source)
    os.makedirs(base, exist_ok=True)
    layout_path = os.path.join(base, "privileged_layout.jpg")
    if generation_response_text is None or not os.path.exists(layout_path):
        make_layout_preserving_canvas(
            record["image_path"], group, layout_path,
            canvas_size=canvas_size, color_seed=seed,
        )
    layout_b64 = file_to_b64(layout_path)

    prompt = f"""Create {questions_per_group} diverse multi-region VQA questions.

NON-NEGOTIABLE SCOPE CHECK FOR EVERY OUTPUT ITEM:
- This group contains exactly {num_regions} selected regions: {', '.join(r['region_id'] for r in group)}.
- Each individual question must genuinely use ALL {num_regions} regions, not a useful
  subset. evidence_region_ids, region_contributions, answer_if_dropped, and the
  reasoning certificate must therefore cover exactly this full ID set.
- If no valid all-region question exists for a blueprint, skip that question. Never
  fill unused regions with decorative observations.
- evidence_scope={evidence_scope}. The selected regions come from
  {len(source_image_ids) if source_image_ids else 1} source image(s).
- For cross_image evidence, use every represented Image panel and return exactly these
  source_image_ids: {source_image_ids}. Image-plane coordinates are local to each panel:
  never claim that an object in one independent image is literally left/right/above/
  below an object in another image. Cross-image comparison and correspondence are valid.

You see a PRIVILEGED EVIDENCE LAYOUT containing only {num_regions} crops at
their original positions. Each crop has a unique colored outline. The original
full image is unavailable. Questions must be answerable from visible pixels in
this layout, not from the supplied captions alone.

Visible crop anchors (the same outline colors will appear on the marked original
image used by the final VQA sample):
{group_summary(group, record['image_width'], record['image_height'], anchor_colors=anchor_colors)}

Anchor mapping: {color_list_str}.

This region group was selected primarily for task_family={preferred_task_family or 'auto'}
with necessity_mode={preferred_necessity_mode or 'auto'}. Prioritize that family, but
also generate other allowed families when the same visible evidence genuinely supports
them. Do not force an unsupported family merely to follow the proposal metadata.

Question-mix policy:
- When the evidence supports it, cover at least four distinct question types among
  the generated questions. Do not emit many paraphrases of the same reasoning program.
- Try to cover multiple families: closed-set perception, comparison/matching,
  OCR/spatial/relational chains, and compositional reasoning.
- Generate at most two arithmetic_composition questions in this batch, and downstream
  selection will retain at most one per region group.
- Generate arithmetic_composition outside an arithmetic-preferred group only when at
  least three regions provide clear, independent numeric operands.
- Keep arithmetic as roughly 10-20% of the overall candidate pool. Use the remaining
  capacity for filtering, anomaly, matching, OCR, spatial, relational, boolean, and
  other multi-hop structures.
- Within arithmetic_composition, prefer Case-17-style heterogeneous evidence such as
  people count + people count + word count, or OCR number * object count + stripe count.
- Outline-color answers are valid when the marked candidate set is explicit, exhaustive,
  and unambiguous. Keep them as roughly 25-40% of the batch rather than making them the
  default. Aim for 35-50% semantic object, attribute, action, role, relation, pair, or
  short-text answers, and at most 25% purely numeric answers. Never force a weaker
  semantic question when a localization question is more objective and verifiable.
- Include open-vocabulary short answers when pixels support them. The answer may be
  ANY objectively verifiable expression of fewer than 10 words; object, action, role,
  state, material, relation, clothing, text, and short descriptive clauses are examples,
  not an exhaustive taxonomy. Do not constrain these answers to a supplied label set.
  Open vocabulary does NOT mean subjective, speculative, explanatory, or long-form.
- Aim for at least two scalar semantic/text candidates per batch when supported, with
  different answer domains. Do not repeat the same color, count, or object label merely
  through paraphrasing.
- Use two complementary synthesis tracks rather than forcing every question into a
  rigid program: roughly 60-70% may use executable certificates for deterministic
  operations, while roughly 30-40% should explore open semantic or unusual multi-hop
  questions backed by evidence_graph certificates. Accuracy requirements are identical.
- Prefer select-then-report programs: use several regions to identify a target, then
  report a second visible property of it. The final property must not be usable to
  select the target without completing the earlier multi-region reasoning.

Selected reasoning templates for this region group:
{template_block}

Template rules:
- Treat each template as a reasoning blueprint, never as evidence that a fact is visible.
- Instantiate a selected template only when every required fact is objectively visible.
- Use each selected template_id at most once and copy its ID exactly into the output.
- Copy each selected template's final_answer_kind exactly. Do not turn a semantic-output
  template into an outline-color answer.
- If a template is unsupported, skip it rather than fabricating an object, count, text,
  relation, or answer.
- You may generate at most {freeform_slots} additional well-grounded questions with
  template_id="freeform" when the image supports a useful structure not covered above.
- Return at most {questions_per_group} questions total.

Every question must need all regions, but necessity has three valid forms:
- answer_sensitive: every region contributes a positive value, so masking it changes
  an aggregate answer or makes the full aggregate unknowable.
- set_complete / pair_complete: all marked regions are members of an explicitly closed
  candidate or pairing set. Masking any member makes the complete subset, unique
  anomaly, global extremum, unique match, or complete pairing impossible to prove,
  even when the most likely answer text would remain unchanged.
- chain_dependency: every region supplies an indispensable clue or hop. Masking it
  makes the final answer unknowable.

Merely mentioning every color does not count. Explicitly describe every region's role
and what becomes unknowable when it is masked.

CRITICAL — every region must be INDEPENDENTLY necessary, not redundant:
- The correct answer must NOT remain the same when any single region is masked. If
  masking region X still leaves the answer solvable from the other regions, then X is
  redundant and the question is invalid.
- Do NOT design questions whose answer is visible in / inferable from multiple regions
  (e.g. "what object appears in front of the ruins?" where the object appears in every
  region). Each region must contribute a distinct, non-overlapping piece of evidence.
- For count/aggregate questions, every region's individual count must contribute to the
  total; masking one must change the total. A region whose count is 0 or already
  derivable from others is redundant.
- For each region, `answer_if_dropped` must predict an answer that DIFFERS from the
  reference answer (a genuinely different value or "insufficient information"), never
  the same value. If you cannot honestly do this for some region, drop that region or
  do not produce the question.

Reasoning-certificate rules:
- First design the question freely from visible evidence. Then provide a certificate
  proving how the selected regions lead to its answer. The certificate is a proof for
  validation, NOT a template from which the question should be mechanically rendered.
- Use mode="executable" for deterministic counting, arithmetic, comparison, filtering,
  matching, boolean, OCR-number, or lookup programs.
- In executable mode, use concise canonical operations such as observe, count,
  count_objects, read_text, read_number, read_attribute, get_attribute, classify,
  detect, select, filter, filter_by_attribute, add, subtract, multiply, divide, sum,
  absolute_difference, compare, argmax, argmin, match, select_unique, lookup, pair,
  boolean_and, boolean_or, boolean_not, return_value, or return_description.
- Use mode="evidence_graph" for open-vocabulary descriptions, semantic relations,
  actions, roles, unusual visual reasoning, or any valid question that does not fit a
  closed executable ontology. Evidence-graph op names may be freely chosen.
- Every step has id, op, regions, inputs, and query. inputs may only reference earlier
  step IDs. Every selected region must lie on a dependency path to output_step; no
  decorative or disconnected evidence is allowed.
- Keep the question diverse and natural. Do not translate certificates into repetitive
  "first...then...finally" wording unless that wording is genuinely useful.

Prefer these task families:
1. aggregate_count: total clearly bounded target instances across all boxes. Every
   box must contribute at least one target.
2. aggregate_attribute_count: total instances with one objective attribute. Every
   box must contribute at least one qualifying instance.
3. aggregate_numeric: combine one visible number from every box only when numbers
   are intrinsic to the scene, such as prices, scores, quantities, or dates.
4. set_filtering: return the complete subset satisfying one objective attribute.
5. anomaly_detection: identify one objectively unique member among 3-6 comparable boxes.
6. extremum: select the global max/min among clearly comparable values or counts.
7. candidate_matching: match a visible text/object clue against a closed candidate set.
8. cross_region_pairing: select a person-object, object-text, or object-object pair using
   one objective relation; every region must participate as a candidate or reference.
9. ocr_visual_verification: read clear text and use it to verify a visual candidate.
10. spatial_chain / dependent_chain: use 2-4 genuinely dependent location clues.
11. arithmetic_composition: combine objective OCR values, counts, or visible measurements
    from 3-7 regions with a short explicit formula. Cross-category operands are allowed
    and encouraged. Emit operation_expression using every region ID exactly as a numeric
    variable, for example "r000 + r001 + r002" or "r000 * r001 + r002". Use
    "abs(r000 - (r001 + r002))" for an absolute difference. region_values stores
    the numeric operand read from each region. Do not use a function such as sum().
    Avoid cases where every operand is 1 and the answer merely equals the number of boxes.
12. conditional_composition: evaluate a visible condition in some regions, then apply
    the selected operation to shared payload evidence in the remaining regions. The
    actual branch must still require every region; do not add an unused branch-only box.
13. boolean_composition: evaluate two independently visible predicates and ask which
    truth pattern holds. Use exactly these four answer_vocabulary entries:
    "both true", "only the first", "only the second", "neither". Do not ask for yes/no.
14. relational_composition: combine part-whole, spatial, material, identity, comparison,
    or person-object relations and return one short scalar or ordered-pair answer.
15. multi_hop_reasoning: construct a 2-5 hop dependency chain such as OCR -> matching ->
    attribute lookup, reference -> related object -> property, or value -> indexed target.
16. viewpoint_transform: transform locations from the camera frame into a clearly
    defined person, object, or opposite-side viewpoint.
17. agent_relative_direction: determine left/right/front/behind in an oriented agent or
    object's local frame, never silently using camera coordinates.
18. geometric_line_relation: classify or select parallel, perpendicular, intersecting,
    converging, or collinear lines from a closed marked set.
19. topological_relation: reason about touching, overlap, containment, connection,
    crossing, and disconnection over a closed marked set.
20. depth_ordering: compare front/behind or nearest/farthest using robust depth cues.
21. spatial_transitivity: compose directional relations through intermediate objects.
22. cross_image_composition: connect evidence from 2-5 labeled source images through
    correspondence, comparison, OCR-to-visual lookup, matching, or comparison of
    relations that are each established within one image.

Priority order is: (1) objectively valid and uniquely answerable from visible pixels,
(2) diverse reasoning structure, and (3) conversational naturalness. Mildly synthetic
logic and arithmetic are acceptable, but never trade correctness for diversity. Avoid
decorative premises: every region must be used on the actual reasoning path. Avoid
subjective comparisons, vague background counts, or facts that cannot be read from pixels.

Wording rules:
- Ask for one final answer. It may require multiple operations, predicates, or hops.
- Do not coordinate two output commands in one scalar-answer question. BAD: "Arrange
  the objects and state the middle object's shape" because it requests both the order
  and the shape. GOOD: "After ordering the objects from left to right, what shape is
  the middle one?" because ordering is an internal reasoning step and only the shape
  is requested as output. Use ordered_pair or unordered_set semantics only when a
  genuinely multi-item answer is intended.
- Aggregate questions must explicitly say "in total", "altogether", or naturally
  bound the full target set so the operation cannot be read as a per-object count.
- Aggregate questions that use every evidence region should normally say "across all
  marked regions" instead of mechanically enumerating every outline color.
- Never create a count where every region contains exactly one counted target and the
  answer can therefore be recovered by counting color names or boxes in the question.
- The final question is paired with the marked original image. Use outline colors
  whenever they prevent ambiguous reference. Never expose raw region IDs or coordinates.
- Spatial tasks must explicitly use one coordinate frame: image_plane, camera, agent,
  object, or scene_3d. For line geometry, say whether the relation is apparent_2d or
  physical_3d. Camera-left and agent-left are not interchangeable. Lines that converge
  in the image may still be physically parallel under perspective.
- It is natural to say "among all marked boxes" for a closed set; do not mechanically
  repeat every color when one collective phrase is clearer.
- Optionally combine a color anchor with a content description, such as "the woman
  in the red box", when that makes the question easier to read.
- For scalar final_answer_kind="semantic" or "text", use answer_vocabulary=[] so the
  validation models answer open-vocabulary from pixels. Keep the answer objective and
  fewer than 10 words. Articles and short clauses are allowed. For closed-set, pair,
  boolean, and anchor outputs, provide short
  canonical labels covering every possible answer without identifying the correct one.
  Semantic labels must never be outline colors. Set answers join labels with " + ".
- Never ask about a red/green/blue/etc. object when the correct answer is the same
  outline color; this leaks the answer through wording. Avoid orientation, apparent
  pixel size, prominence, or vague thresholds such as "significantly different".
- The answer must be short, objective, unique, and absent from the question text.
- Generate broadly and diversify formulations. Mild uncertainty is acceptable because
  strict privileged voting and ablation checks run later; do not fabricate invisible facts.

Output format — return JSON only, an array of objects each with this schema:
[
  {{
    "template_id": "selected template ID or freeform",
    "question": "...",
    "answer_hint": "expected short answer",
    "question_type": "aggregate_count | aggregate_attribute_count | aggregate_numeric | set_filtering | anomaly_detection | extremum | candidate_matching | cross_region_pairing | ocr_visual_verification | spatial_chain | dependent_chain | viewpoint_transform | agent_relative_direction | geometric_line_relation | topological_relation | depth_ordering | spatial_transitivity | cross_image_composition | arithmetic_composition | conditional_composition | boolean_composition | relational_composition | multi_hop_reasoning",
    "necessity_mode": "answer_sensitive | set_complete | pair_complete | chain_dependency",
    "answer_semantics": "scalar | unordered_set | ordered_pair",
    "final_answer_kind": "anchor | numeric | boolean | semantic | semantic_set | semantic_pair | text",
    "interrogative": "what | how | which",
    "evidence_region_ids": ["r000", "r001", ...],
    "evidence_scope": "single_image | cross_image",
    "source_image_ids": ["exact source image ids represented by the evidence regions"],
    "region_contributions": {{"r000": "positive fact/value used", "r001": "positive fact/value used"}},
    "region_values": {{"r000": 2, "r001": 1}},
    "aggregation": "sum | formula | boolean_and | boolean_or | compare | concatenate | dependent_chain",
    "operation_expression": "for arithmetic_composition only, e.g. r000 * r001 + r002",
    "candidate_region_ids": ["all region ids for set_complete; otherwise empty"],
    "closed_candidate_set": true,
    "pair_region_ids": [["r000", "r001"], ["r002", "r003"]],
    "chain_region_ids": ["ordered ids used by chain_dependency"],
    "coordinate_frame": "image_plane | camera | agent | object | scene_3d | not_applicable",
    "relation_basis": "apparent_2d | physical_3d | not_applicable",
    "spatial_relation": "canonical short relation label, or empty for non-spatial tasks",
    "natural_referents": {{"r000": "optional content description", "r001": "optional content description"}},
    "reference_scope": "optional natural phrase that bounds the target set",
    "answer_vocabulary": ["short candidate label 1", "short candidate label 2"],
    "answer_if_dropped": {{"r000": "changed answer or insufficient information", "r001": "changed answer or insufficient information"}},
    "hypothetical_answer": "short canonical answer the designer expects",
    "reasoning": "explain how EACH listed region is required to derive the answer",
    "reasoning_certificate": {{
      "mode": "executable | evidence_graph",
      "steps": [
        {{"id": "s1", "op": "open operation name", "regions": ["r000"],
          "inputs": [], "query": "what this step obtains from visible evidence"}},
        {{"id": "s2", "op": "compose or open operation", "regions": ["r001"],
          "inputs": ["s1"], "query": "how evidence is combined"}}
      ],
      "output_step": "s2"
    }},
    "naturalness_score": 1,
    "visual_confidence": 1,
    "multi_region_score": 1
  }}
]"""
    if generation_response_text is None and generation_callable is not None:
        text = generation_callable(prompt, layout_path)
    elif generation_response_text is None:
        text = call_chat(
            api_url, api_key, model,
            [{"role": "user", "content": [image_message(layout_b64), {"type": "text", "text": prompt}]}],
            temperature=0.5,
            top_p=0.9,
            max_tokens=8192,
            enable_thinking=False,
        )
    else:
        text = generation_response_text
    # Preserve raw output so zero-candidate groups can be diagnosed without
    # another model request.
    try:
        with open(os.path.join(base, "generation_response.txt"), "w", encoding="utf-8") as f:
            f.write(text or "")
    except OSError:
        pass
    items = extract_json(text, list)
    diagnostics = {
        "raw_count": len(items) if isinstance(items, list) else 0,
        "accepted_before_limit": 0,
        "rejection_counts": {},
        "repair_counts": {},
        "soft_flag_counts": {},
        "rejection_examples": {},
    }
    current_item = {"value": None}

    def _count(bucket, reason):
        counts = diagnostics[bucket]
        counts[reason] = counts.get(reason, 0) + 1

    def reject(reason):
        _count("rejection_counts", reason)
        examples = diagnostics["rejection_examples"].setdefault(reason, [])
        item = current_item["value"]
        if len(examples) < 3 and isinstance(item, dict):
            examples.append({
                "question": str(item.get("question") or "")[:500],
                "answer": str(item.get("hypothetical_answer") or item.get("answer_hint") or "")[:200],
                "question_type": str(item.get("question_type") or ""),
            })

    def repair(item, reason):
        repairs = item.setdefault("local_gate_repairs", [])
        if reason not in repairs:
            repairs.append(reason)
            _count("repair_counts", reason)

    def soft_flag(item, reason):
        flags = item.setdefault("local_gate_soft_flags", [])
        if reason not in flags:
            flags.append(reason)
            _count("soft_flag_counts", reason)

    valid_ids = {r["region_id"] for r in group}
    color_names = [color_name for color_name, _ in anchor_colors.values()]
    seen_types = {}
    seen_interrogatives = {}
    color_only_count = 0
    numeric_only_count = 0
    out = []
    used_template_ids = set()
    freeform_count = 0
    # Conditional phrasing that REQUIRES an else branch to pass.
    _COND_OPENERS = (
        "if ", "in case", "assuming ",
        "in the case that", "in case of",
    )
    # Else-branch markers that satisfy the conditional constraint.
    _ELSE_MARKERS = (
        "otherwise", "else", "if not", "否则", "不然",
    )
    # Ambiguous-background counting phrasing.
    _AMBIG_BG = (
        "background", "in the distance", "far away",
        "trees in ", "buildings in ",
        "people in the distance", "stars in",
    )
    # Extended color set — single-token answers that are JUST a color are capped.
    _EXT_COLORS = ANCHOR_COLOR_WORDS
    # Allowed interrogative tags (lower-cased). "how many" collapses to "how".
    _VALID_INTERROGATIVES = {"what", "who", "when", "why", "how", "where", "which"}
    _TYPE_TO_MODE = QUESTION_TYPE_TO_MODE
    _TYPE_TO_ANSWER_SEMANTICS = {
        **{qtype: {"scalar"} for qtype in _TYPE_TO_MODE},
        "set_filtering": {"unordered_set"},
        "anomaly_detection": {"scalar", "unordered_set"},
        "extremum": {"scalar", "unordered_set"},
        "candidate_matching": {"scalar", "unordered_set"},
        "cross_region_pairing": {"ordered_pair"},
        "relational_composition": {"scalar", "ordered_pair"},
        "multi_hop_reasoning": {"scalar", "ordered_pair"},
        "cross_image_composition": {"scalar", "unordered_set", "ordered_pair"},
        "geometric_line_relation": {"scalar", "unordered_set", "ordered_pair"},
        "topological_relation": {"scalar", "unordered_set", "ordered_pair"},
        "depth_ordering": {"scalar", "ordered_pair"},
    }
    _STRICT_TYPES = set(_TYPE_TO_MODE)
    _AGGREGATE_TYPES = {
        "aggregate_count", "aggregate_attribute_count", "aggregate_numeric",
    }
    _COMPOSITION_TYPES = {
        "arithmetic_composition", "conditional_composition",
        "boolean_composition", "relational_composition", "multi_hop_reasoning",
        "cross_image_composition",
    }
    _SPATIAL_FRAME_TYPES = {
        "viewpoint_transform", "agent_relative_direction",
        "geometric_line_relation", "topological_relation", "depth_ordering",
        "spatial_transitivity",
    }
    _COORDINATE_FRAMES = {"image_plane", "camera", "agent", "object", "scene_3d"}
    _RELATION_BASES = {"apparent_2d", "physical_3d"}
    _TOTAL_MARKERS = (
        "in total", "altogether", "total number", "total count",
        "across all", "总共", "一共", "合计",
    )
    _LOW_NATURALNESS_PHRASES = (
        "given that", "considering the", "considering that",
        "taking into account", "based on the fact that",
        "otherwise", "else", "if not", "sum of", "add ", "adding ",
        "minus", "subtract", "subtracting", "multiplied", "multiplying",
        "product of", "difference between", "combined with the word count",
    )
    _MULTI_QUESTION_MARKERS = (
        ", and is ", ", and are ", ", and what ", ", and where ",
        "; and ", "? and ", " as well as whether ",
    )
    _TYPE_MISMATCH_QUESTION_WORDS = (
        ("which letter", "word_or_phrase"),
        ("which object", "number"),
        ("what object", "number"),
        ("which person", "number"),
    )
    _NUMBER_WORDS = {
        "zero", "one", "two", "three", "four", "five", "six", "seven",
        "eight", "nine", "ten", "十一", "十二", "一", "二", "三", "四", "五",
        "六", "七", "八", "九", "十",
    }

    def _infer_interrogative(q):
        raw = (q or "").strip().lower()
        # Strip leading "in / on / for / from / by / at / to / of / about"prenouns.
        first = raw.split(None, 1)[0] if raw else ""
        # Common Chinese question particles.
        for cn in ("什么", "谁", "哪", "在哪", "何时", "为什么", "怎么", "如何"):
            if cn in raw:
                cn_map = {"什么": "what", "谁": "who", "为什么": "why",
                          "怎么": "how", "如何": "how", "何时": "when",
                          "在哪": "where", "哪": "which"}
                return cn_map.get(cn, "what")
        if first in {"how"}:
            return "how"
        if first.startswith("how ") and "many" in raw[:12]:
            return "how"
        return first if first in _VALID_INTERROGATIVES else "what"

    def _looks_number_answer(ans):
        raw = (ans or "").strip().lower()
        if not raw:
            return False
        return raw.isdigit() or raw in _NUMBER_WORDS

    def _looks_word_or_phrase_answer(ans):
        raw = (ans or "").strip()
        return bool(raw and not _looks_number_answer(raw) and len(raw.split()) <= 5)

    def _is_low_naturalness(item, qtext, ans_hint):
        qtype = (item.get("question_type") or "").strip().lower()
        is_composition = qtype in _COMPOSITION_TYPES
        if any(p in qtext for p in _MULTI_QUESTION_MARKERS) and not is_composition:
            return True
        if qtext.count("?") > 1:
            return True
        # Conditional and arithmetic-heavy prompts were the dominant source of
        # synthetic, unnatural failures in manual inspection.
        arithmetic_hits = sum(1 for p in _LOW_NATURALNESS_PHRASES if p in qtext)
        if arithmetic_hits >= 2 and not is_composition:
            return True
        if ("sum of" in qtext or "minus" in qtext or "multiply" in qtext or "product of" in qtext):
            if not is_composition:
                return True
        if ("word count" in qtext or "number of letters" in qtext) and not is_composition:
            return True
        for phrase, expected in _TYPE_MISMATCH_QUESTION_WORDS:
            if phrase in qtext:
                if expected == "number" and _looks_number_answer(ans_hint):
                    return True
                if expected == "word_or_phrase" and _looks_word_or_phrase_answer(ans_hint):
                    return True
        # Reject obvious two-part answer strings. Short comma phrases are OK for
        # object names only when the question type explicitly supports them; most
        # generated cases with commas were actually two independent answers.
        raw_ans = (ans_hint or "").strip().lower()
        if raw_ans.count(",") >= 1 or " yes" in raw_ans or " no" in raw_ans:
            if qtype not in {
                "set_filtering", "cross_region_pairing",
                "relational_composition", "multi_hop_reasoning",
            }:
                return True
            if len(raw_ans.split()) > 5:
                return True
        return False

    for item in items:
        current_item["value"] = item
        if not isinstance(item, dict) or not item.get("question"):
            reject("missing_question_or_object")
            continue
        qtext = item["question"].lower()
        # Reject raw region IDs (r000 etc.) leaking into the question text.
        if any(rid.lower() in qtext for rid in valid_ids):
            reject("raw_region_id_leak")
            continue
        referenced_colors = [
            c for c in color_names
            if re.search(rf"\b{re.escape(c.lower())}\b", qtext)
        ]
        collective_reference = any(p in qtext for p in (
            "all the marked", "all marked", "marked boxes", "marked regions",
            "outlined boxes", "outlined regions", "colored boxes", "colored regions",
            "which marked region", "which outlined region", "which colored region",
            "which marked box", "which outlined box", "which colored box",
            "in which marked region", "in which outlined region",
            "among the marked", "among the outlined", "among the colored",
            "所有标记", "全部标记", "所有框", "全部框",
        )) or bool(re.search(r"\b(?:marked|outlined)\b", qtext))
        if not referenced_colors and not collective_reference:
            reject("missing_marked_region_reference")
            continue
        raw_qtype = (item.get("question_type") or "").strip().lower()
        qtype = normalize_question_type(raw_qtype)
        if qtype != raw_qtype:
            repair(item, f"question_type:{raw_qtype}->{qtype}")
        if qtype not in _STRICT_TYPES:
            reject("unsupported_question_type")
            continue
        declared_scope = str(item.get("evidence_scope") or evidence_scope).strip().lower()
        if declared_scope != evidence_scope:
            reject("evidence_scope_mismatch")
            continue
        declared_source_ids = {
            str(x) for x in (item.get("source_image_ids") or []) if str(x)
        }
        expected_source_ids = {
            str(r.get("source_image_id")) for r in group if r.get("source_image_id")
        }
        if evidence_scope == "cross_image":
            if not (2 <= len(expected_source_ids) <= 5):
                reject("invalid_cross_image_source_count")
                continue
            if declared_source_ids != expected_source_ids:
                reject("cross_image_source_ids_mismatch")
                continue
        elif qtype == "cross_image_composition":
            reject("cross_image_type_in_single_image")
            continue
        coordinate_frame = str(item.get("coordinate_frame") or "").strip().lower()
        relation_basis = str(item.get("relation_basis") or "").strip().lower()
        spatial_relation = str(item.get("spatial_relation") or "").strip().lower()
        if qtype in _SPATIAL_FRAME_TYPES:
            if coordinate_frame not in _COORDINATE_FRAMES:
                reject("invalid_spatial_coordinate_frame")
                continue
            if relation_basis not in _RELATION_BASES or not spatial_relation:
                reject("missing_spatial_relation_basis")
                continue
            if qtype in {"viewpoint_transform", "agent_relative_direction"}:
                if coordinate_frame not in {"agent", "object"}:
                    reject("invalid_agent_coordinate_frame")
                    continue
                if relation_basis != "physical_3d":
                    reject("invalid_agent_relation_basis")
                    continue
            if qtype == "geometric_line_relation":
                if coordinate_frame not in {"image_plane", "scene_3d"}:
                    reject("invalid_geometry_coordinate_frame")
                    continue
                if ((coordinate_frame == "image_plane") !=
                        (relation_basis == "apparent_2d")):
                    reject("geometry_frame_basis_mismatch")
                    continue
            if qtype == "depth_ordering":
                if coordinate_frame not in {"camera", "scene_3d"}:
                    reject("invalid_depth_coordinate_frame")
                    continue
                if relation_basis != "physical_3d":
                    reject("invalid_depth_relation_basis")
                    continue
        else:
            coordinate_frame = coordinate_frame or "not_applicable"
            relation_basis = relation_basis or "not_applicable"
        template_id = str(item.get("template_id") or "").strip()
        if not template_id:
            compatible_ids = [
                tid for tid, spec in selected_templates.items()
                if spec["question_type"] == qtype and tid not in used_template_ids
            ]
            template_id = compatible_ids[0] if len(compatible_ids) == 1 else "freeform"
        # Template IDs guide diversity; they are not evidence. Preserve an otherwise
        # valid model-designed question when the model paraphrases or invents the ID.
        if template_id != "freeform":
            template_spec = selected_templates.get(template_id)
            if (not template_spec or template_spec["question_type"] != qtype
                    or template_id in used_template_ids):
                template_id = "freeform"
                repair(item, "template_downgraded_to_freeform")
        if template_id == "freeform":
            if freeform_count >= freeform_slots:
                soft_flag(item, "freeform_quota_exceeded")
        else:
            template_spec = selected_templates.get(template_id)
        final_answer_kind = str(item.get("final_answer_kind") or "").strip().lower()
        allowed_answer_kinds = {
            "anchor", "numeric", "boolean", "semantic", "semantic_set",
            "semantic_pair", "text",
        }
        if final_answer_kind not in allowed_answer_kinds:
            inferred_answer = classify_final_answer_kind(
                item.get("hypothetical_answer") or item.get("answer_hint"), anchor_colors
            )
            if inferred_answer in {"anchor", "numeric", "boolean", "semantic"}:
                final_answer_kind = inferred_answer
                repair(item, f"answer_kind_inferred:{inferred_answer}")
            else:
                reject("invalid_final_answer_kind")
                continue
        if template_id != "freeform" and final_answer_kind != template_spec["final_answer_kind"]:
            template_id = "freeform"
            repair(item, "template_answer_kind_downgraded_to_freeform")
            if freeform_count >= freeform_slots:
                soft_flag(item, "freeform_quota_exceeded")
        necessity_mode = (item.get("necessity_mode") or "").strip().lower()
        if necessity_mode != _TYPE_TO_MODE[qtype]:
            necessity_mode = _TYPE_TO_MODE[qtype]
            repair(item, f"necessity_mode_inferred:{necessity_mode}")
        answer_semantics = (item.get("answer_semantics") or "").strip().lower()
        if answer_semantics not in _TYPE_TO_ANSWER_SEMANTICS[qtype]:
            answer_semantics = infer_answer_semantics(qtype, final_answer_kind)
            if answer_semantics not in _TYPE_TO_ANSWER_SEMANTICS[qtype]:
                answer_semantics = sorted(_TYPE_TO_ANSWER_SEMANTICS[qtype])[0]
            repair(item, f"answer_semantics_inferred:{answer_semantics}")
        declared_ids = {str(x) for x in (item.get("evidence_region_ids") or [])}
        contributions = item.get("region_contributions") or {}
        dropped_answers = item.get("answer_if_dropped") or {}
        if declared_ids != valid_ids:
            reject("evidence_region_ids_mismatch")
            continue
        if not isinstance(contributions, dict):
            contributions = {}
        if set(contributions) != valid_ids or any(
                not str(contributions.get(rid, "")).strip() for rid in valid_ids):
            contributions = {
                rid: str(contributions.get(rid) or
                         f"Required evidence from {rid} in the reasoning certificate.")
                for rid in valid_ids
            }
            repair(item, "region_contributions_completed")
        if not isinstance(dropped_answers, dict):
            dropped_answers = {}
        if set(dropped_answers) != valid_ids:
            dropped_answers = {
                rid: str(dropped_answers.get(rid) or "insufficient information")
                for rid in valid_ids
            }
            repair(item, "answer_if_dropped_completed")
        item["region_contributions"] = contributions
        item["answer_if_dropped"] = dropped_answers
        cert_ok, cert_reasons, cert_details = validate_reasoning_certificate(
            item.get("reasoning_certificate"), valid_ids
        )
        if (not cert_ok
                and set(cert_reasons) == {"unsupported_executable_operation"}
                and qtype not in (_AGGREGATE_TYPES | {
                    "arithmetic_composition", "boolean_composition",
                })):
            certificate = dict(item.get("reasoning_certificate") or {})
            certificate["mode"] = "evidence_graph"
            cert_ok, cert_reasons, cert_details = validate_reasoning_certificate(
                certificate, valid_ids
            )
            if cert_ok:
                item["reasoning_certificate"] = certificate
                repair(item, "certificate_downgraded_to_evidence_graph")
        if not cert_ok:
            reject("invalid_reasoning_certificate:" + "+".join(cert_reasons))
            continue
        if qtype in (_AGGREGATE_TYPES | {
            "arithmetic_composition", "boolean_composition",
        }) and cert_details.get("mode") != "executable":
            reject("deterministic_task_requires_executable_certificate")
            continue
        answer_vocabulary = item.get("answer_vocabulary") or []
        if not isinstance(answer_vocabulary, list):
            reject("invalid_answer_vocabulary")
            continue
        answer_vocabulary = [str(x).strip() for x in answer_vocabulary if str(x).strip()]
        if not answer_vocabulary and final_answer_kind == "anchor" and qtype in {
            "set_filtering", "anomaly_detection", "extremum",
            "candidate_matching", "cross_region_pairing",
        }:
            answer_vocabulary = list(color_names)
        if len({_norm(x) for x in answer_vocabulary}) != len(answer_vocabulary):
            deduped = []
            seen_vocabulary = set()
            for value in answer_vocabulary:
                normalized = _norm(value)
                if normalized not in seen_vocabulary:
                    seen_vocabulary.add(normalized)
                    deduped.append(value)
            answer_vocabulary = deduped
            repair(item, "answer_vocabulary_deduplicated")
        if answer_semantics in {"unordered_set", "ordered_pair"} and len(answer_vocabulary) < 2:
            reject("insufficient_closed_answer_vocabulary")
            continue
        answer_space = (
            "open" if uses_open_answer_space(answer_semantics, final_answer_kind)
            else "closed"
        )
        if answer_space == "open":
            # Do not expose designer-proposed alternatives to privileged voters.
            # Agreement must come from independently reading the pixels.
            answer_vocabulary = []
        if qtype == "boolean_composition":
            required_truth_states = {
                "both true", "only the first", "only the second", "neither",
            }
            if {_norm(x) for x in answer_vocabulary} != {
                _norm(x) for x in required_truth_states
            }:
                reject("invalid_boolean_answer_vocabulary")
                continue
            expected = str(
                item.get("hypothetical_answer") or item.get("answer_hint") or ""
            ).strip()
            if _norm(expected) not in {_norm(x) for x in answer_vocabulary}:
                reject("boolean_answer_outside_vocabulary")
                continue
        if qtype in _AGGREGATE_TYPES:
            # A color-by-color list makes one-target-per-box counts solvable
            # from the question text. Require a natural collective scope.
            if not collective_reference:
                reject("aggregate_missing_collective_scope")
                continue
            region_values = item.get("region_values") or {}
            if not isinstance(region_values, dict) or set(region_values) != valid_ids:
                reject("aggregate_region_values_mismatch")
                continue
            if str(item.get("aggregation") or "").strip().lower() != "sum":
                reject("aggregate_operation_not_sum")
                continue
            try:
                values = [Decimal(str(region_values[rid])) for rid in valid_ids]
                answer_value = Decimal(str(
                    item.get("hypothetical_answer") or item.get("answer_hint")
                ).strip())
            except (InvalidOperation, ValueError, TypeError):
                reject("aggregate_non_numeric_value")
                continue
            if any(v <= 0 for v in values) or sum(values) != answer_value:
                reject("aggregate_answer_inconsistent")
                continue
            if all(v == 1 for v in values):
                reject("aggregate_equals_box_count")
                continue
            if "across the images" in qtext or "outlined images" in qtext:
                reject("cross_image_wording_in_single_image")
                continue
        elif qtype == "arithmetic_composition":
            region_values = item.get("region_values") or {}
            if not isinstance(region_values, dict) or set(region_values) != valid_ids:
                reject("arithmetic_region_values_mismatch")
                continue
            if any(not str(region_values[rid]).strip() for rid in valid_ids):
                reject("arithmetic_empty_region_value")
                continue
            if not str(item.get("aggregation") or "").strip():
                reject("arithmetic_missing_aggregation")
                continue
            expression = str(item.get("operation_expression") or "").strip()
            if not expression or any(
                not re.search(rf"\b{re.escape(rid)}\b", expression)
                for rid in valid_ids
            ):
                reject("arithmetic_expression_missing_region")
                continue
            try:
                values = [Decimal(str(region_values[rid])) for rid in valid_ids]
                computed = _evaluate_region_expression(expression, region_values)
                answer_value = Decimal(str(
                    item.get("hypothetical_answer") or item.get("answer_hint")
                ).strip())
            except (InvalidOperation, ValueError, TypeError, ZeroDivisionError, SyntaxError):
                reject("arithmetic_expression_invalid")
                continue
            if computed != answer_value or all(v == 1 for v in values):
                reject("arithmetic_answer_inconsistent_or_trivial")
                continue
            if (template_id == "arithmetic.absolute_difference"
                    and not _expression_root_is_abs(expression)):
                reject("absolute_difference_expression_mismatch")
                continue
            if (template_id == "arithmetic.absolute_difference"
                    and "absolute_difference" not in cert_details.get("operations", [])):
                reject("absolute_difference_certificate_mismatch")
                continue
        elif necessity_mode == "set_complete":
            candidate_ids = {str(x) for x in (item.get("candidate_region_ids") or [])}
            closed = item.get("closed_candidate_set")
            if candidate_ids != valid_ids or closed is not True:
                item["candidate_region_ids"] = sorted(valid_ids)
                item["closed_candidate_set"] = True
                repair(item, "closed_candidate_set_completed")
        elif necessity_mode == "pair_complete":
            pairs = item.get("pair_region_ids") or []
            if not isinstance(pairs, list) or not pairs:
                reject("missing_pair_region_ids")
                continue
            pair_ids = {
                str(rid) for pair in pairs if isinstance(pair, list)
                for rid in pair
            }
            if pair_ids != valid_ids:
                reject("pair_region_ids_mismatch")
                continue
        elif necessity_mode == "chain_dependency":
            chain_ids = {str(x) for x in (item.get("chain_region_ids") or [])}
            # chain_region_ids may list only the ordered backbone while closed
            # negative controls branch into it. The validated reasoning
            # certificate is the authoritative all-region connectivity proof.
            if chain_ids - valid_ids:
                reject("chain_contains_unknown_region")
                continue
        # Interrogative: prefer explicit field, fall back to first-token inference.
        interrogative = (item.get("interrogative") or "").strip().lower()
        if interrogative not in _VALID_INTERROGATIVES:
            interrogative = _infer_interrogative(item["question"])
        # Keep conditional logic explicit and correctly typed. Both branches
        # operate on the same payload regions; drop-one verifies the actual path.
        has_condition = any(p in qtext for p in _COND_OPENERS)
        if qtype == "conditional_composition":
            if not has_condition or not any(m in qtext for m in _ELSE_MARKERS):
                reject("conditional_missing_explicit_branches")
                continue
        elif has_condition:
            reject("conditional_wording_for_nonconditional_type")
            continue
        # Reject ambiguous-background counting phrasing.
        if any(p in qtext for p in _AMBIG_BG):
            reject("ambiguous_background_counting")
            continue
        ans_hint = (item.get("hypothetical_answer") or item.get("answer_hint") or "").strip().lower()
        if not ans_hint:
            reject("missing_answer_hint")
            continue
        if answer_space == "open" and not (0 < open_answer_word_count(ans_hint) < 10):
            reject("open_answer_length_out_of_range")
            continue
        observed_answer_kind = classify_final_answer_kind(ans_hint, anchor_colors)
        if not answer_kind_matches_declared(final_answer_kind, observed_answer_kind):
            explicit_region_request = bool(re.search(
                r"\bwhich\s+(?:(?:marked|outlined|colored)\s+)?(?:region|box|outline)\b",
                qtext,
            )) or "which outline color" in qtext
            if (final_answer_kind == "anchor" and observed_answer_kind == "semantic"
                    and not explicit_region_request):
                if not (0 < open_answer_word_count(ans_hint) < 10):
                    reject("promoted_semantic_answer_length_out_of_range")
                    continue
                final_answer_kind = "semantic"
                answer_semantics = "scalar"
                answer_space = "open"
                answer_vocabulary = []
                repair(item, "answer_kind_promoted_from_anchor_to_semantic")
                if template_id != "freeform":
                    template_id = "freeform"
                    repair(item, "promoted_answer_template_downgraded_to_freeform")
            else:
                reject("declared_answer_kind_mismatch")
                continue
        contract_ok, contract_reasons = validate_question_answer_contract(
            item.get("question"), ans_hint, final_answer_kind, anchor_colors,
            answer_semantics=answer_semantics,
        )
        if not contract_ok:
            reject("question_answer_contract:" + "+".join(contract_reasons))
            continue
        if has_anchor_attribute_leak(qtext, ans_hint, anchor_colors):
            reject("anchor_attribute_leak")
            continue
        if any(phrase in qtext for phrase in SUBJECTIVE_QUESTION_PHRASES):
            reject("subjective_question_phrase")
            continue
        # The designer must explicitly predict a changed answer under every
        # ablation. This is a cheap structural screen before strong-model calls.
        # A region whose masked answer equals the reference answer is REDUNDANT:
        # the question would survive drop-one on that region. Reject such
        # questions up front instead of paying for an expensive self-check that
        # will only fail anyway. Tolerate up to 0 redundant regions (any single
        # redundant region breaks the all-region-necessity contract).
        norm_hint = _norm(ans_hint)
        redundant_regions = [
            rid for rid in valid_ids
            if str(dropped_answers[rid]).strip()
            and _norm(str(dropped_answers[rid])) == norm_hint
        ]
        if redundant_regions:
            reject(
                "redundant_region_in_answer_if_dropped:"
                + ",".join(sorted(redundant_regions))
            )
            continue
        # After the redundancy screen, treat empty/missing predictions as
        # "insufficient information" and flag for the visual drop-one test to
        # confirm (the designer simply did not commit a predicted value).
        if any(not str(dropped_answers[rid]).strip() for rid in valid_ids):
            for rid in valid_ids:
                if not str(dropped_answers[rid]).strip():
                    dropped_answers[rid] = "insufficient information"
            item["answer_if_dropped"] = dropped_answers
            soft_flag(item, "answer_if_dropped_repaired_pending_visual_check")
        if _is_low_naturalness(item, qtext, ans_hint):
            soft_flag(item, "low_naturalness")
        ans_first_token = ans_hint.split(None, 1)[0] if ans_hint else ""
        is_color_only = observed_answer_kind == "anchor"
        is_numeric_only = (
            observed_answer_kind == "numeric"
        )
        max_anchor_answers = max(1, (questions_per_group * 3 + 7) // 8)
        max_numeric_answers = max(1, questions_per_group // 4)
        if is_color_only and color_only_count >= max_anchor_answers:
            soft_flag(item, "anchor_quota_exceeded")
        if is_numeric_only and numeric_only_count >= max_numeric_answers:
            soft_flag(item, "numeric_quota_exceeded")
        seen_types[qtype] = seen_types.get(qtype, 0) + 1
        seen_interrogatives[interrogative] = seen_interrogatives.get(interrogative, 0) + 1
        if is_color_only:
            color_only_count += 1
        if is_numeric_only:
            numeric_only_count += 1
        item["interrogative"] = interrogative
        item["evidence_region_ids"] = [r["region_id"] for r in group]
        item["anchor_colors"] = dict(anchor_colors)
        item["question_type"] = qtype
        item["evidence_scope"] = evidence_scope
        item["source_image_ids"] = sorted(expected_source_ids)
        item["template_id"] = template_id
        item["necessity_mode"] = necessity_mode
        item["answer_semantics"] = answer_semantics
        item["final_answer_kind"] = final_answer_kind
        item["observed_answer_kind"] = observed_answer_kind
        item["answer_space"] = answer_space
        item["answer_vocabulary"] = answer_vocabulary
        item["coordinate_frame"] = coordinate_frame
        item["relation_basis"] = relation_basis
        item["spatial_relation"] = spatial_relation
        item["certificate_validation"] = {
            "passed": True,
            "reasons": cert_reasons,
            **cert_details,
        }
        item["generation_source"] = generation_source
        for score_name in ("naturalness_score", "visual_confidence", "multi_region_score"):
            try:
                item[score_name] = max(1.0, min(5.0, float(item.get(score_name, 3))))
            except (TypeError, ValueError):
                item[score_name] = 3.0
        out.append(item)
        if template_id == "freeform":
            freeform_count += 1
        else:
            used_template_ids.add(template_id)
    diagnostics["accepted_before_limit"] = len(out)
    try:
        with open(os.path.join(base, "generation_diagnostics.json"), "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, ensure_ascii=False, indent=2)
    except OSError:
        pass
    if return_diagnostics:
        return out, diagnostics
    return out


def judge_question_quality(record, group, candidates, api_url, api_key, model):
    """Text-only quality gate for naturalness and multi-region necessity.

    This is intentionally cheap: it judges the question wording and declared
    region roles, not visual correctness. Visual correctness remains the job of
    privileged voting and drop-one self-check.
    """
    if not candidates:
        return []
    region_block = group_summary(group, record["image_width"], record["image_height"])
    cand_block = "\n".join(
        f"{i}. type={c.get('question_type','')}; mode={c.get('necessity_mode','')}; "
        f"answer_kind={c.get('final_answer_kind','')}; "
        f"answer={c.get('hypothetical_answer') or c.get('answer_hint','')}; "
        f"question={c.get('question','')}"
        for i, c in enumerate(candidates)
    )
    prompt = f"""You are a strict data-quality judge for natural multi-region VQA.

Image caption: {record.get('caption','')}

Evidence regions:
{region_block}

Candidate questions:
{cand_block}

For each candidate, decide whether to KEEP it.

KEEP only if:
- the question is understandable; controlled math/logic constructions are allowed;
- necessity matches the declared mode: answer_sensitive changes the aggregate;
  set_complete/pair_complete needs every closed-set member to prove the complete or
  unique answer; chain_dependency needs every clue hop;
- it produces one short final answer, even if several operations or hops are required;
- objects being aggregated are a coherent class or semantic set;
- the answer type matches the question wording.
- semantic/text output questions use outline colors only for reference and return a
  genuine object, attribute, action, role, relation, or visible text.

REJECT if:
- any region is only decorative context;
- a set_complete question does not clearly define the marked regions as the full
  candidate set, or uses subjective/weak comparison criteria;
- the question says "given/considering A and B" but A/B do not constrain the answer;
- a conditional branch leaves a selected region unused on the actual reasoning path;
- it combines two independent questions;
- it relies on subjective attributes or vague thresholds.
- its final answer is merely an outline color despite declaring a semantic answer kind;
- an object clue repeats the correct outline color and thereby reveals the answer.

Return JSON only:
[
  {{"index": 0, "keep": true, "reason": "short reason"}},
  {{"index": 1, "keep": false, "reason": "short reason"}}
]"""
    text = call_chat(
        api_url, api_key, model,
        [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        temperature=0.0,
        max_tokens=2048,
        enable_thinking=False,
    )
    items = extract_json(text, list)
    decisions = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("index"))
            except Exception:
                continue
            decisions[idx] = {
                "keep": bool(item.get("keep")),
                "reason": str(item.get("reason") or "").strip(),
            }
    out = []
    for i, cand in enumerate(candidates):
        dec = decisions.get(i)
        if not dec:
            cand["quality_judge"] = {"keep": False, "reason": "missing_judge_decision"}
            continue
        cand["quality_judge"] = dec
        if dec["keep"]:
            out.append(cand)
    return out
