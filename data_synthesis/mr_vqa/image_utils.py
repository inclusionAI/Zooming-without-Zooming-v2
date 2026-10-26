import base64
import io
import os
import random
import re

from PIL import Image, ImageDraw

ANCHOR_PALETTE = [
    ("red",     (220, 30, 30)),
    ("blue",    (30, 90, 220)),
    ("green",   (30, 170, 60)),
    ("orange",  (240, 130, 0)),
    ("purple",  (140, 30, 200)),
    ("brown",   (130, 80, 40)),
    ("pink",    (240, 120, 180)),
]


def assign_anchor_colors(regions, seed=None):
    """Randomly sample k unique colors for k regions, deterministically by seed.

    Returns dict mapping region_id -> (color_name, rgb_tuple).
    """
    if not regions:
        return {}
    if len(regions) > len(ANCHOR_PALETTE):
        raise ValueError(
            f"at most {len(ANCHOR_PALETTE)} evidence regions are supported; "
            f"got {len(regions)}"
        )
    rng = random.Random(seed)
    k = len(regions)
    chosen = rng.sample(ANCHOR_PALETTE, k)
    return {r["region_id"]: chosen[i] for i, r in enumerate(regions)}

def load_image(path):
    """Load a local image file as an RGB PIL.Image.

    The pipeline reads images from the local filesystem (a manifest of image
    paths prepared by the user). Remote object-store backends were part of the
    internal deployment and are intentionally not shipped with this release.
    """
    return Image.open(path).convert("RGB")


def image_to_b64(image, fmt="JPEG"):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format=fmt, quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def file_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = bbox_area([ix1, iy1, ix2, iy2])
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def expand_bbox(bbox, image_size, ratio=0.08):
    x1, y1, x2, y2 = bbox
    w, h = image_size
    bw, bh = x2 - x1, y2 - y1
    return [
        max(0, x1 - bw * ratio),
        max(0, y1 - bh * ratio),
        min(w, x2 + bw * ratio),
        min(h, y2 + bh * ratio),
    ]


def crop_bbox(image, bbox, expand_ratio=0.08, upscale=1):
    x1, y1, x2, y2 = [int(round(v)) for v in expand_bbox(bbox, image.size, expand_ratio)]
    crop = image.crop((x1, y1, x2, y2))
    if upscale > 1:
        crop = crop.resize((crop.width * upscale, crop.height * upscale), Image.LANCZOS)
    return crop


def safe_name(text):
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())[:96] or "item"


def union_bbox(regions, image_size, padding_ratio=0.12):
    width, height = image_size
    x1 = min(r["bbox"][0] for r in regions)
    y1 = min(r["bbox"][1] for r in regions)
    x2 = max(r["bbox"][2] for r in regions)
    y2 = max(r["bbox"][3] for r in regions)
    bw, bh = x2 - x1, y2 - y1
    return [
        max(0, x1 - bw * padding_ratio),
        max(0, y1 - bh * padding_ratio),
        min(width, x2 + bw * padding_ratio),
        min(height, y2 + bh * padding_ratio),
    ]


def _merge_intervals(intervals):
    """Merge overlapping/adjacent intervals. Input: list of (start, end). Returns sorted disjoint list."""
    if not intervals:
        return []
    s = sorted(intervals, key=lambda t: t[0])
    merged = [list(s[0])]
    for a, b in s[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [tuple(x) for x in merged]


def _build_axis_remap(intervals, gap):
    """Build coordinate remap: each interval keeps its length, gaps between intervals become `gap` pixels.

    Returns (total_length, remap_fn). remap_fn(x) maps original coord → compressed coord.
    For x outside any interval, returns the nearest endpoint's compressed coord.
    """
    new_cursor = 0.0
    spans = []  # (orig_a, orig_b, new_a, new_b)
    for a, b in intervals:
        w = b - a
        spans.append((a, b, new_cursor, new_cursor + w))
        new_cursor += w + gap
    total = max(0.0, new_cursor - gap) if spans else 0.0

    def remap(x):
        if not spans:
            return 0.0
        # Before first interval
        if x <= spans[0][0]:
            return spans[0][2]
        # After last interval
        if x >= spans[-1][1]:
            return spans[-1][3]
        # Inside an interval
        for a, b, na, nb in spans:
            if a <= x <= b:
                return na + (x - a)
        # Between intervals → map to gap center
        for i in range(len(spans) - 1):
            if spans[i][1] < x < spans[i + 1][0]:
                frac = (x - spans[i][1]) / (spans[i + 1][0] - spans[i][1])
                return spans[i][3] + frac * gap
        return spans[0][2]

    return total, remap


def make_layout_preserving_canvas(image_path, regions, output_path, drop_region_id=None,
                                  canvas_size=None, expand_ratio=0.08,
                                  max_crop_size=None, padding_ratio=0.12,
                                  max_canvas_size=4096, gap=0, title_pad=0,
                                  margin=0, anchor_colors=None, color_seed=None):
    """Remove empty rows/columns: paint regions on original-resolution canvas, delete all-white rows & cols.

    Equivalent to:
      1. Create white canvas at original image resolution.
      2. Paste each region crop at its original bbox coordinates.
      3. Scan every row: delete if entirely white. Scan every column: delete if entirely white.
      4. Result canvas contains only rows/cols that have region pixels.

    Implemented via interval-union math (no pixel scan needed):
      - Union all expanded-bbox x-intervals → these are the columns to keep.
      - Union all expanded-bbox y-intervals → these are the rows to keep.
      - Each kept interval preserves its original pixel span.
      - Original spacing BETWEEN intervals (empty space between two disjoint
        regions that happen to share the same row or column band) is kept as-is.
        If you want to compress that inter-region gap to `gap` pixels, see the
        `compress_inter_gap` parameter logic below.

    Region topology is perfectly preserved: if region A was to the right of
    and above region B in the original image, it remains so in the output.

    The final canvas may be up to `max_canvas_size` on its long side; if it
    would exceed that, a uniform shrink is applied.
    """
    image = load_image(image_path)
    if not regions:
        return None

    img_w, img_h = image.size

    expanded_bboxes = []
    for region in regions:
        eb = expand_bbox(region["bbox"], image.size, ratio=expand_ratio)
        x1, y1, x2, y2 = [max(0, min(v, s)) for v, s in zip(eb, [img_w, img_h, img_w, img_h])]
        if x2 <= x1:
            x1, x2 = region["bbox"][0], region["bbox"][2]
        if y2 <= y1:
            y1, y2 = region["bbox"][1], region["bbox"][3]
        expanded_bboxes.append((x1, y1, x2, y2))

    # Build interval unions for x and y axes
    x_intervals = _merge_intervals([(x1, x2) for x1, _, x2, _ in expanded_bboxes])
    y_intervals = _merge_intervals([(y1, y2) for _, y1, _, y2 in expanded_bboxes])

    # Map from original coordinate → compressed coordinate.
    # Each interval keeps its length; space between consecutive intervals
    # is set to `gap` (default 0 = edges touch, which is the "delete empty
    # rows/cols" semantics with no extra padding).
    x_total, x_remap = _build_axis_remap(x_intervals, gap)
    y_total, y_remap = _build_axis_remap(y_intervals, gap)

    canvas_w_raw = int(round(x_total)) + 2 * margin
    canvas_h_raw = int(round(y_total)) + 2 * margin + title_pad

    shrink = 1.0
    if max(canvas_w_raw, canvas_h_raw) > max_canvas_size:
        shrink = max_canvas_size / float(max(canvas_w_raw, canvas_h_raw))
    canvas_w = max(1, int(round(canvas_w_raw * shrink)))
    canvas_h = max(1, int(round(canvas_h_raw * shrink)))

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    if title_pad > 0:
        draw.text((margin, 6), "Layout-preserving evidence view", fill="black")

    # Anchor colors: 12 high-contrast palette, randomly sampled k=group size.
    if anchor_colors is None:
        anchor_colors = assign_anchor_colors(regions, seed=color_seed)
    color_for = dict(anchor_colors)

    for region, (x1, y1, x2, y2) in zip(regions, expanded_bboxes):
        rid = region["region_id"]

        # Compressed coordinates
        cx1 = (margin + x_remap(x1)) * shrink
        cx2 = (margin + x_remap(x2)) * shrink
        cy1 = (margin + title_pad + y_remap(y1)) * shrink
        cy2 = (margin + title_pad + y_remap(y2)) * shrink
        cx1, cx2 = sorted([cx1, cx2])
        cy1, cy2 = sorted([cy1, cy2])

        px = int(round(cx1))
        py = int(round(cy1))
        pw = max(1, int(round(cx2 - cx1)))
        ph = max(1, int(round(cy2 - cy1)))

        if rid == drop_region_id:
            # Drop region: solid gray block + red X marker. No text.
            draw.rectangle([px, py, px + pw, py + ph], fill=(220, 220, 220),
                           outline=(180, 180, 180), width=max(1, int(round(2 * shrink))))
            diag_w = max(2, int(round(6 * shrink)))
            draw.line([px, py, px + pw, py + ph], fill=(200, 0, 0), width=diag_w)
            draw.line([px + pw, py, px, py + ph], fill=(200, 0, 0), width=diag_w)
        else:
            crop = crop_bbox(image, region["bbox"], expand_ratio, upscale=1)
            if (pw, ph) != crop.size:
                crop = crop.resize((pw, ph), Image.LANCZOS)
            canvas.paste(crop, (px, py))
            _, rgb = color_for[rid]
            lw = max(3, int(round(8 * shrink)))
            draw.rectangle([px, py, px + pw, py + ph], outline=rgb, width=lw)

        # Cross-image records retain panel membership after empty rows/columns
        # are compressed. The label is metadata, never answer-bearing content.
        source_index = region.get("source_image_index")
        if source_index is not None:
            label = f"Image {source_index}"
            tx, ty = px + 6, py + 6
            label_box = draw.textbbox((tx, ty), label)
            draw.rectangle(
                [label_box[0] - 3, label_box[1] - 2,
                 label_box[2] + 3, label_box[3] + 2],
                fill=(255, 255, 255),
            )
            draw.text((tx, ty), label, fill=(15, 18, 24))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    canvas.save(output_path, "JPEG", quality=95)
    return output_path, dict(color_for)


def draw_marked_image(image_path, regions, output_path, anchor_colors=None):
    """Draw bbox rectangles on the original image.

    If `anchor_colors` is provided (dict region_id -> (color_name, rgb_tuple)),
    reuse those so the marked original matches the privileged_layout / drop-one
    views that the question was written against. Otherwise fall back to the
    legacy fixed palette.
    """
    image = load_image(image_path)
    draw = ImageDraw.Draw(image)
    legacy_colors = ["red", "lime", "blue", "orange", "magenta", "cyan"]
    for idx, region in enumerate(regions):
        x1, y1, x2, y2 = region["bbox"]
        rid = region["region_id"]
        if anchor_colors and rid in anchor_colors:
            _, rgb = anchor_colors[rid]
            color = rgb
        else:
            color = legacy_colors[idx % len(legacy_colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=6)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    image.save(output_path, "JPEG", quality=92)
    return output_path
