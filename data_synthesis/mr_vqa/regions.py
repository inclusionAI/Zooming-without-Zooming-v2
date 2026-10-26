from .image_utils import bbox_area, bbox_center, bbox_iou, image_to_b64, load_image
from .llm import call_chat, extract_json, image_message


def _as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _clamp_bbox(bbox, width, height):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1, x2 = sorted([max(0.0, min(width, x1)), max(0.0, min(width, x2))])
    y1, y2 = sorted([max(0.0, min(height, y1)), max(0.0, min(height, y2))])
    return [x1, y1, x2, y2]


def _maybe_denormalize_bbox(bbox, width, height):
    vals = [float(v) for v in bbox]
    if max(vals) <= 1.5:
        return [vals[0] * width, vals[1] * height, vals[2] * width, vals[3] * height]
    if max(vals) <= 1000.0:
        return [vals[0] / 1000.0 * width, vals[1] / 1000.0 * height,
                vals[2] / 1000.0 * width, vals[3] / 1000.0 * height]
    return vals


def clean_grounded_regions(items, width, height, min_area_ratio=0.0004, max_area_ratio=0.18,
                           max_iou=0.65, max_regions=24):
    regions = []
    for item in items:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("name") or item.get("object") or "").strip()
        bbox = item.get("bbox") or item.get("box") or item.get("bbox_2d")
        if not label or not isinstance(bbox, list) or len(bbox) != 4:
            continue
        bbox = _clamp_bbox(_maybe_denormalize_bbox(bbox, width, height), width, height)
        area_ratio = bbox_area(bbox) / max(1, width * height)
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        if any(bbox_iou(bbox, r["bbox"]) >= max_iou for r in regions):
            continue
        regions.append({
            "region_id": f"r{len(regions):03d}",
            "label": label,
            "bbox": bbox,
            "center": bbox_center(bbox),
            "caption": str(item.get("caption") or item.get("description") or "").strip(),
            "natural_referent": str(item.get("natural_referent") or "").strip(),
            "role": str(item.get("role") or "").strip(),
            "grounding_confidence": item.get("confidence"),
            "area_ratio": area_ratio,
        })
        if len(regions) >= max_regions:
            break
    return regions


def propose_grounded_regions(record, api_url, api_key, model, max_regions=18,
                             min_area_ratio=0.0004, max_area_ratio=0.18, max_iou=0.65,
                             return_metadata=False,
                             max_concurrent_per_key=12,
                             dense_inventory=False):
    image = load_image(record["image_path"])
    width, height = image.size
    image_b64 = image_to_b64(image)
    caption = record.get("caption", "")

    if dense_inventory:
        task_description = f"""Build a dense, high-quality object inventory for a
long-context visual graph. Enumerate as many distinct useful entities and fine
details as the image objectively supports, up to {max_regions}; aim for 12-18
regions on a rich image. Do not stop after finding one small coherent set."""
        collection_policy = """Cover different locations and semantic roles. Include
repeated comparable objects, people, products, signs, readable text, components,
and small discriminative details when each has a precise bbox. Standalone objects
are useful too when their category or visible attributes can connect to entities
in other images. Mark the image suitable when it contains at least eight concrete,
question-worthy regions; do not require strong physical relations within one image."""
    else:
        task_description = f"""Decide whether this image can support natural questions
whose answer genuinely depends on 3-5 distinct regions, then propose up to
{max_regions} useful evidence regions."""
        collection_policy = """Prioritize groups of related regions over isolated
interesting objects. The image is suitable only when it contains at least one
coherent set that can support a natural all-region task. Strong examples are
repeated comparable objects, a reference plus multiple plausible candidates,
clear OCR/price regions, or objects linked by objective pairing/spatial clues.
A bag of unrelated objects is not sufficient."""

    prompt = f"""You are a precise visual grounding annotator and data-quality gate.

Task: {task_description}

Image caption, if available: {caption}

Return regions that are:
- visually concrete and clearly localizable;
- useful for multi-region reasoning, including people, animals, objects, signs, screens, text regions, fine details, and distinctive parts;
- preferably small or medium-sized, spatially diverse, and not overly overlapping;
- suitable for coherent multi-region tasks: positive-contribution counting,
  attribute filtering over a closed set, unique anomaly, extremum, candidate
  matching, cross-region pairing, OCR-to-visual verification, or a short
  dependent spatial clue chain.

{collection_policy}

Avoid vague regions such as "background", "scene", "sky", "room", or the whole image.

For each region, provide:
- label: short name;
- bbox: [x1, y1, x2, y2] in absolute pixel coordinates. If you cannot use pixels, use normalized 0-1000 coordinates;
- caption: one concise sentence describing visible attributes/text/context;
- natural_referent: a concise description intended to identify this target in the
  original unmarked image without coordinates, boxes, or region IDs;
- role: one of candidate, reference, comparator, text_region, count_target, fine_part, distractor.

Return JSON only:
{{
  "suitable_for_multi_region_vqa": true,
  "richness_reason": "one short objective reason",
  "coherent_sets": [
    {{"labels": ["window 1", "window 2", "window 3"],
      "task_family": "aggregate_count"}}
  ],
  "regions": [
    {{
      "label": "red sign",
      "bbox": [120, 85, 260, 150],
      "caption": "A red sign with white text above the shop.",
      "natural_referent": "the red sign above the shop",
      "role": "text_region",
      "confidence": 0.87
    }}
  ]
}}"""

    text = call_chat(
        api_url, api_key, model,
        [{"role": "user", "content": [image_message(image_b64), {"type": "text", "text": prompt}]}],
        temperature=0.4,
        max_tokens=4096,
        max_concurrent_per_key=max_concurrent_per_key,
    )
    parsed = extract_json(text, dict)
    if isinstance(parsed, dict):
        items = parsed.get("regions") or []
        suitable = _as_bool(parsed.get("suitable_for_multi_region_vqa"))
        richness_reason = str(parsed.get("richness_reason") or "").strip()
        coherent_sets = parsed.get("coherent_sets") or []
    else:
        # Backward compatibility for providers that still return a bare list.
        items = extract_json(text, list)
        suitable = bool(items)
        richness_reason = "legacy_list_response"
        coherent_sets = []
    regions = clean_grounded_regions(
        items, width, height,
        min_area_ratio=min_area_ratio,
        max_area_ratio=max_area_ratio,
        max_iou=max_iou,
        max_regions=max_regions,
    )
    if not return_metadata:
        return regions
    return {
        "regions": regions,
        "suitable_for_multi_region_vqa": suitable,
        "richness_reason": richness_reason,
        "coherent_sets": coherent_sets if isinstance(coherent_sets, list) else [],
    }
