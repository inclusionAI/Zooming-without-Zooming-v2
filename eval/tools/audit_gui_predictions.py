#!/usr/bin/env python3
"""Read-only coordinate/coverage audit of historical GLM prediction journals.

All alternative units are diagnostic only. This never chooses units per target
or writes an official score. Original rows and original metrics remain intact.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zwz_eval.gui_grounding import extract_coordinate, legacy_point, normalize_coordinate, point_in_box
from zwz_eval.io import atomic_json, file_hash, read_journal
from zwz_eval.protocols import GUI_EXPECTED


def audit(path, benchmark):
    rows = read_journal(path)
    latest = {str(row['id']): row for row in rows}
    valid = [r for r in latest.values() if not r.get('error')]
    counts = Counter()
    split_counts = Counter()
    for row in valid:
        point = extract_coordinate(row.get('raw_response', ''))
        size = row['size']
        bbox = row['gold_bbox']
        counts['saved_correct'] += bool(row.get('correct'))
        counts['saved_null_point'] += row.get('point') is None
        counts['finish_length'] += row.get('finish_reason') == 'length'
        counts['raw_coordinates_gt_1000'] += bool(point and max(point) > 1000)
        counts['tool_call_parsed'] += point is not None
        split_counts[row['subset']] += 1
        for space in ('normalized_1000', 'image_pixels'):
            predicted = normalize_coordinate(point, space, size)
            counts[space + '_diagnostic_hits'] += point_in_box(predicted, bbox, size)
        counts['native_xml_diagnostic_hits'] += point_in_box(legacy_point(row.get('raw_response', '')), bbox, size)
    expected = sum(GUI_EXPECTED[benchmark].values())
    return {'benchmark': benchmark, 'source_sha256': file_hash(path), 'journal_rows': len(rows),
            'unique_ids': len(latest), 'successful_returns': len(valid), 'expected': expected,
            'unresolved_transport_errors': len(latest) - len(valid), 'unseen_ids': expected - len(latest),
            'coverage_by_subset': dict(split_counts), 'counts': dict(counts),
            'primary_score': None, 'status': 'diagnostic_only',
            'warning': 'Unit interpretations are run-wide alternatives, NOT scores to maximize per example. '
                       'image_pixels assumes original image dimensions; old resized inputs may differ. '
                       'Stored correctness fields may have been produced by different historical parser revisions.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--bench', choices=GUI_EXPECTED, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.predictions, args.bench)
    if args.output:
        if args.output.resolve() == args.predictions.resolve() or args.output.exists():
            raise ValueError('Audit must write a new file, never overwrite evidence')
        atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
