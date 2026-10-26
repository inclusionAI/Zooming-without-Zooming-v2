"""Content-free historical input contracts; no questions or gold answers embedded."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

from .data import rows as read_rows
from .io import read_json
from .protocols import RESOURCES


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def identities(rows, image_prefix=None):
    """Preserve list order and repeated references; only explicit path prefix is relocatable."""
    non_images, ordered_images = [], []
    for row in rows:
        non_images.append({k: v for k, v in row.items() if k != 'images'})
        paths = row.get('images')
        if not isinstance(paths, list) or not all(isinstance(p, str) and p for p in paths):
            raise ValueError('Ordered image lists required')
        images = []
        for value in paths:
            p = PurePosixPath(value)
            if image_prefix:
                try:
                    p = p.relative_to(PurePosixPath(image_prefix))
                except ValueError:
                    raise ValueError('Image lies outside the explicit source image prefix') from None
            if p.is_absolute() or '..' in p.parts or '\\' in str(p):
                raise ValueError('Use relative image paths or supply the exact original --image-prefix')
            images.append(str(p))
        ordered_images.append(images)
    return {'rows': len(rows), 'semantic_sha256': fingerprint(non_images),
            'ordered_images_sha256': fingerprint(ordered_images),
            'image_references': sum(map(len, ordered_images))}


def verify_rows(bench, rows, task=None, image_prefix=None):
    contract = read_json(RESOURCES / 'normalization_contracts.json')
    key = 'mimic-' + task if bench == 'mimic' and task else bench
    if key not in contract['inputs']:
        raise ValueError('No historical input contract for this benchmark/task')
    expected = contract['inputs'][key]
    actual = identities(rows, image_prefix)
    mismatches = [name for name, value in actual.items() if expected[name] != value]
    if mismatches:
        raise ValueError('Historical input mismatch (' + key + '): ' + ', '.join(mismatches) +
                         '; do not run this as a same-protocol reproduction')
    return {'benchmark': key, **actual, 'matches_historical_annotations': True,
            'image_bytes_verified': False, 'model_or_score_parity_verified': False}


def verify_source(bench, source, image_prefix=None):
    source = Path(source)
    if bench == 'mimic':
        return [verify_rows(bench, read_rows(source / ('mimic_' + task + '.json')),
                            task, image_prefix) for task in ('common', 'counting', 'listing', 'oddone')]
    return verify_rows(bench, read_rows(source), image_prefix=image_prefix)
