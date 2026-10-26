"""Explicit GUI coordinate contracts; no gold-dependent coordinate guessing.

The native XML parser is retained exactly. The API adapter's tolerant parser
is a separately declared protocol, never a silent replacement for old scores.
"""
from __future__ import annotations

import json
import math
import re


COORDINATE_SPACES = ('normalized_1000', 'normalized_1', 'image_pixels')
PARSERS = ('native_xml', 'tool_call')


def legacy_point(text):
    """Historical native XML parser, including its exact newline requirement."""
    try:
        action = json.loads(text.split('<tool_call>\n')[-1].split('\n</tool_call>')[-2])
        values = action['arguments']['coordinate']
        if len(values) == 2:
            x, y = values
        elif len(values) == 4:
            a, b, c, d = values
            x, y = (a + c) / 2, (b + d) / 2
        else:
            return None
        return [x / 1000, y / 1000]
    except (IndexError, KeyError, TypeError, ValueError):
        return None


def _coordinate(action):
    if not isinstance(action, dict) or action.get('name') != 'computer_use':
        return None
    args = action.get('arguments')
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if not isinstance(args, dict) or args.get('action') != 'left_click':
        return None
    values = args.get('coordinate')
    if not isinstance(values, list) or len(values) not in (2, 4):
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        return None
    if len(values) == 4:
        a, b, c, d = values
        return [(a + c) / 2, (b + d) / 2]
    return list(values)


def extract_coordinate(text):
    """Accept XML-wrapped JSON or a JSON call; reject multiple/ambiguous calls.

    No free-text number regex, reasoning-channel search, bbox-based selection,
    or per-example unit inference. Native API function calls are serialized by
    the transport into this same JSON structure before parsing.
    """
    blocks = re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>', text or '', re.S)
    if len(blocks) > 1:
        return None
    candidate = blocks[0] if blocks else (text or '').strip()
    if candidate.startswith('```json') and candidate.endswith('```'):
        candidate = candidate[7:-3].strip()
    try:
        return _coordinate(json.loads(candidate))
    except (ValueError, TypeError):
        return None


def normalize_coordinate(point, space, sent_size):
    if space not in COORDINATE_SPACES:
        raise ValueError('Declare one coordinate space for the entire run')
    if point is None:
        return None
    width, height = sent_size
    if width <= 0 or height <= 0:
        raise ValueError('Image dimensions must be positive')
    sx, sy = {'normalized_1000': (1000, 1000), 'normalized_1': (1, 1),
              'image_pixels': (width, height)}[space]
    x, y = point
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (x, y)):
        return None
    if not (0 <= x <= sx and 0 <= y <= sy):
        return None
    return [x / sx, y / sy]


def parse_point(text, *, parser, coordinate_space, sent_size):
    if parser == 'native_xml':
        if coordinate_space != 'normalized_1000':
            raise ValueError('native_xml always divides coordinates by 1000')
        return legacy_point(text)
    if parser != 'tool_call':
        raise ValueError('Unknown GUI parser')
    return normalize_coordinate(extract_coordinate(text), coordinate_space, sent_size)


def point_in_box(point, bbox, original_size, *, reorder=False):
    if point is None or len(point) != 2 or any(not math.isfinite(v) for v in point):
        return False
    x1, y1, x2, y2 = bbox
    if reorder:
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
    width, height = original_size
    # Match the official normalized comparison, including inclusive boundaries.
    return x1 / width <= point[0] <= x2 / width and y1 / height <= point[1] <= y2 / height


def summarize(items, predictions, benchmark):
    """Incomplete runs have diagnostics but no publishable primary accuracy."""
    expected_ids = {str(item['id']) for item in items}
    if len(expected_ids) != len(items):
        raise ValueError('Duplicate annotation IDs')
    latest = {}
    for row in predictions:
        uid = str(row['id'])
        if uid not in expected_ids:
            raise ValueError('Prediction ID absent from annotations')
        latest[uid] = row
    splits = {}
    diagnostics = {'unparseable_tool_call': 0, 'coordinate_out_of_range': 0,
                   'format_only_legacy_failure': 0}
    for item in items:
        result = splits.setdefault(item['subset'], {'expected': 0, 'returned': 0, 'correct': 0,
                                                   'invalid_format': 0, 'transport_failed': 0})
        result['expected'] += 1
        row = latest.get(str(item['id']))
        if row is None:
            continue
        if row.get('error'):
            result['transport_failed'] += 1
            continue
        result['returned'] += 1
        if 'coordinate_space' in row and 'image_input' in row:
            raw = extract_coordinate(row.get('raw_response') or '')
            if raw is None:
                diagnostics['unparseable_tool_call'] += 1
            elif normalize_coordinate(raw, row['coordinate_space'], row['image_input']['size']) is None:
                diagnostics['coordinate_out_of_range'] += 1
            elif legacy_point(row.get('raw_response') or '') is None:
                diagnostics['format_only_legacy_failure'] += 1
        result['correct'] += int(bool(row['correct']))
        result['invalid_format'] += int(row.get('point') is None)
    for result in splits.values():
        result['missing'] = result['expected'] - result['returned'] - result['transport_failed']
        result['accuracy'] = result['correct'] / result['expected'] if result['returned'] == result['expected'] else None
        result['correct_over_expected_lower_bound'] = result['correct'] / result['expected']
    complete = bool(items) and all(s['returned'] == s['expected'] for s in splits.values())
    correct = sum(s['correct'] for s in splits.values())
    if benchmark == 'ui_vision':
        complete = complete and set(splits) == {'basic', 'functional', 'spatial'}
    primary = None
    if complete:
        primary = (sum(s['accuracy'] for s in splits.values()) / 3
                   if benchmark == 'ui_vision' else correct / len(items))
    return {'benchmark': benchmark, 'status': 'complete' if complete else 'incomplete',
            'expected': len(items), 'returned': sum(s['returned'] for s in splits.values()),
            'correct': correct, 'accuracy': primary, 'splits': splits,
            'diagnostics': diagnostics,
            'aggregation': 'three-split macro' if benchmark == 'ui_vision' else 'sample micro',
            'judge': 'deterministic inclusive point-in-box; no LLM judge'}
