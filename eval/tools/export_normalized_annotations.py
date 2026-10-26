#!/usr/bin/env python3
"""Export already-normalized inputs; never reconstruct questions from raw HF data."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def semantic(rows):
    content = [{k: v for k, v in row.items() if k != 'images'} for row in rows]
    return sha(json.dumps(content, sort_keys=True, ensure_ascii=False).encode())


def export(source, target, image_prefix=None):
    source, target = Path(source), Path(target)
    if target.exists():
        raise ValueError('Target must be new')
    raw = source.read_bytes()
    rows = (json.loads(raw) if source.suffix == '.json' else
            [json.loads(line) for line in raw.splitlines() if line.strip()])
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty row list')
    paths = [p for row in rows for p in row['images']]
    if not all(isinstance(p, str) and p.startswith('/') for p in paths):
        raise ValueError('Expected original absolute ordered image lists')
    prefix = image_prefix or os.path.commonpath([str(PurePosixPath(p).parent) for p in paths])
    if prefix == '/':
        raise ValueError('No unambiguous common image root')
    mapping = {p: str(PurePosixPath(p).relative_to(prefix)) for p in paths}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError('Image relocation collision')
    result = copy.deepcopy(rows)
    for old, new in zip(rows, result):
        new['images'] = [mapping[p] for p in old['images']]
        assert len(new['images']) == len(old['images'])
    assert semantic(rows) == semantic(result)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(result, ensure_ascii=False, indent=2) + '\n' if source.suffix == '.json'
               else ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in result)).encode()
    target.write_bytes(encoded)
    manifest = {
        'source_sha256': sha(raw), 'export_sha256': sha(encoded), 'rows': len(rows),
        'semantic_sha256_excluding_images': semantic(rows),
        'image_references': len(paths), 'unique_images': len(mapping),
        'image_order_sha256': sha(json.dumps([row['images'] for row in result]).encode()),
        'relocation': 'Only images[] prefix removed; all other row fields and row order unchanged.',
        'source_path_hash_to_relative': {sha(p.encode()): v for p, v in mapping.items()},
        'image_bytes_verified': False,
        'note': 'Annotation parity only; populate exact image files and verify their provenance separately.'}
    target.with_name(target.name + '.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--image-prefix', help='Original common image root; only this prefix is removed')
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.target, args.image_prefix), indent=2))
