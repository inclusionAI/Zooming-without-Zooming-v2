"""Convert a local official MIMIC snapshot, without changing its questions.

The four historical annotation fingerprints are mandatory acceptance gates.
Raw ``idx`` values are not unique across Counting's two halves: never dedupe.
Images are copied byte-for-byte, not decoded and re-encoded.  This module does
not download anything and does not infer an image's identity from its position.
"""
from __future__ import annotations

import hashlib
import io
import json
import numbers
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from .io import atomic_json, file_hash, read_json
from .mimic import COUNTS
from .normalization import verify_rows
from .protocols import RESOURCES

SHARD = re.compile(r'test-(\d+)-of-(\d+)\.parquet\Z')


def source_files(source, task):
    """Accept the author's HF layout or an explicit raw JSON metadata export."""
    source = Path(source)
    directory = source / task
    parquet = list(directory.glob('test-*-of-*.parquet'))
    exports = [p for p in (directory / 'test.json', directory / 'test.jsonl') if p.is_file()]
    if parquet and exports or len(exports) > 1:
        raise ValueError(f'MIMIC {task}: ambiguous raw formats; select one source snapshot')
    if exports:
        return exports
    if not parquet:
        raise ValueError(f'MIMIC {task}: missing {task}/test-*-of-*.parquet (or raw test.jsonl)')
    parts = []
    for path in parquet:
        match = SHARD.fullmatch(path.name)
        if not match:
            raise ValueError(f'MIMIC {task}: invalid shard filename')
        parts.append((int(match[1]), int(match[2]), path))
    totals = {total for _, total, _ in parts}
    total = next(iter(totals))
    if len(totals) != 1 or total < 1 or sorted(index for index, _, _ in parts) != list(range(total)):
        raise ValueError(f'MIMIC {task}: missing or duplicate Parquet shards')
    return [path for _, _, path in sorted(parts)]


def raw_rows(paths):
    """Iterate HF row order with at most one Parquet image-bearing row at a time."""
    for path in paths:
        if path.suffix == '.jsonl':
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    if line.strip():
                        yield json.loads(line)
        elif path.suffix == '.json':
            data = json.loads(path.read_text(encoding='utf-8'))
            if not isinstance(data, list):
                raise ValueError('Raw MIMIC JSON must be a list, not normalized annotations')
            yield from data
        else:
            try:
                import pyarrow.parquet as pq
            except ImportError:
                raise RuntimeError("Reading raw MIMIC Parquet requires pip install 'zwz-eval[data]'") from None
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1, use_threads=False):
                yield from batch.to_pylist()


def normalize_row(raw, task, images):
    """Only rename official columns and add the saved project's fixed metadata."""
    if not isinstance(raw, dict):
        raise ValueError('MIMIC raw rows must be objects')
    required = {'question', 'images', 'gt', 'task'} | ({'idx', 'class_id'} if task == 'counting' else {'id'})
    if not required.issubset(raw):
        raise ValueError(f'MIMIC {task}: expected original HF columns {sorted(required)}')
    if not isinstance(raw['question'], str) or not isinstance(raw['task'], str):
        raise ValueError('MIMIC question and task must be original strings')
    if task == 'counting':
        if isinstance(raw['idx'], bool) or not isinstance(raw['idx'], numbers.Integral):
            raise ValueError('MIMIC Counting idx must be an integer; never renumber it')
        if isinstance(raw['gt'], bool) or not isinstance(raw['gt'], numbers.Integral):
            raise ValueError('MIMIC Counting gt must be an integer')
        if not isinstance(raw['class_id'], str) or not raw['class_id']:
            raise ValueError('MIMIC Counting class_id is required')
        identity = f"counting_{raw['idx']}"
    else:
        if not isinstance(raw['id'], str) or not raw['id'] or not isinstance(raw['gt'], str):
            raise ValueError('MIMIC id and gt must be original strings')
        identity = raw['id']
    result = {'id': identity, 'sample_id': identity, 'query': raw['question'],
              'images': images, 'response': str(raw['gt']), 'category': task,
              'task': raw['task'], 'split': 'test'}
    if task == 'counting':
        result['meta'] = {'class_id': raw['class_id']}
    return result


def _image_name(value, expected=None):
    path = value.get('path') if isinstance(value, dict) else value
    if path is None and isinstance(value, dict) and value.get('bytes') is not None:
        if not expected:
            raise ValueError('Missing original image path and no historical identity recipe')
        # HF Image features may embed bytes with path=None. This metadata-only
        # recipe supplies filenames, never question text, gold or image bytes.
        return expected
    if not isinstance(path, str) or not path or '\\' in path or '\x00' in path:
        raise ValueError('Raw MIMIC image needs its original path/basename; image identities cannot be guessed')
    parsed = PurePosixPath(path)
    if '..' in parsed.parts or parsed.name in ('', '.') or '://' in path:
        raise ValueError('Unsafe/remote MIMIC image path; use the local HF snapshot')
    return parsed.name


def _image_bytes(value, name, task, source, image_root):
    embedded = value.get('bytes') if isinstance(value, dict) else None
    if embedded is not None:
        if not isinstance(embedded, (bytes, bytearray, memoryview)) or not embedded:
            raise ValueError('MIMIC embedded image must contain nonempty original bytes')
        return bytes(embedded)
    path = value.get('path') if isinstance(value, dict) else value
    roots = [Path(image_root)] if image_root is not None else [Path(source)]
    candidates = []
    for root in roots:
        root = root.resolve()
        for relative in (Path(task) / name, Path(name), Path(path)):
            if relative.is_absolute():
                continue
            candidate = (root / relative).resolve()
            if candidate.is_relative_to(root) and candidate.is_file() and candidate not in candidates:
                candidates.append(candidate)
    if not candidates:
        raise ValueError(f'MIMIC image {task}/{name} has no bytes and no readable local file; pass --image-root')
    hashes = {file_hash(path) for path in candidates}
    if len(hashes) != 1:
        raise ValueError(f'MIMIC image {task}/{name}: ambiguous local files have different contents')
    return candidates[0].read_bytes()


def convert(source, target, image_root=None):
    """Convert local HF raw data and publish only after all historical hashes match.

    The result contains mimic_*.json and images/{task}/{original basename}.
    Use ``prepare --bench mimic --source TARGET --image-root TARGET/images``
    afterwards to create runtime sample_uids and absolute local image paths.
    """
    source, target = Path(source), Path(target)
    if not source.is_dir():
        raise ValueError('MIMIC source must be an extracted local HF snapshot directory')
    if target.exists() or target.is_symlink():
        raise ValueError('MIMIC normalization target must not already exist')
    if image_root is not None and not Path(image_root).is_dir():
        raise ValueError('MIMIC image-root must be an existing local directory')
    inputs = {task: source_files(source, task) for task in COUNTS}
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.' + target.name + '-mimic-', dir=target.parent))
    try:
        from PIL import Image
        recipe = read_json(RESOURCES / 'normalization/mimic_images.json')['tasks']
        all_images, reports, recipe_references = {}, [], 0
        for task, expected in COUNTS.items():
            normalized = []
            for index, raw in enumerate(raw_rows(inputs[task])):
                if index >= expected:
                    raise ValueError(f'MIMIC {task}: more than the expected {expected} rows; do not mix extensions')
                values = raw.get('images') if isinstance(raw, dict) else None
                if not isinstance(values, list) or not values or len(values) > 256:
                    raise ValueError('MIMIC requires a nonempty ordered raw images list (at most 256)')
                paths = []
                if len(recipe.get(task, [])) != expected or len(recipe[task][index]) != len(values):
                    raise ValueError(f'MIMIC {task}: raw row/image count differs from historical identity recipe')
                for image_index, value in enumerate(values):
                    expected_name = recipe[task][index][image_index]
                    name = _image_name(value, expected_name)
                    if name != expected_name:
                        raise ValueError(f'MIMIC {task}: raw image identity/order differs from historical recipe')
                    if isinstance(value, dict) and value.get('path') is None:
                        recipe_references += 1
                    relative = f'{task}/{name}'
                    content = _image_bytes(value, name, task, source, image_root)
                    sha = hashlib.sha256(content).hexdigest()
                    if relative in all_images:
                        if all_images[relative] != sha:
                            raise ValueError(f'MIMIC image identity collision: {relative}')
                    else:
                        with Image.open(io.BytesIO(content)) as image:
                            image.verify()
                        output = stage / 'images' / relative
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_bytes(content)
                        all_images[relative] = sha
                    paths.append(relative)
                normalized.append(normalize_row(raw, task, paths))
            if len(normalized) != expected:
                raise ValueError(f'MIMIC {task}: expected {expected} rows, got {len(normalized)}')
            # Checks every preserved non-image field, row order and original
            # ordered image identity, including both 2500-row Counting halves.
            reports.append(verify_rows('mimic', normalized, task=task))
            atomic_json(stage / f'mimic_{task}.json', normalized)
        report = {'benchmark': 'mimic', 'raw_source': 'anurag4446/mimic',
                  'transform': 'Original HF fields renamed; image bytes copied verbatim; no ID deduplication',
                  'tasks': reports, 'images': all_images,
                  'matches_historical_annotations': True,
                  'historical_image_bytes_verified': False,
                  'image_paths_restored_from_recipe': recipe_references,
                  'note': 'Image byte hashes record this raw snapshot, not a comparison to historical server image bytes.',
                  'source_files': [str(p.relative_to(source)) for paths in inputs.values() for p in paths]}
        atomic_json(stage / 'normalization.json', report)
        if target.exists() or target.is_symlink():
            raise ValueError('MIMIC normalization target appeared during conversion; refusing overwrite')
        stage.rename(target)
        return report
    except BaseException:
        shutil.rmtree(stage)
        raise


def normalize(bench, source, target, image_root=None):
    """Shared CLI dispatcher signature; no other benchmark is guessed here."""
    if bench != 'mimic':
        raise ValueError('raw_mimic only converts the MIMIC benchmark')
    return convert(source, target, image_root=image_root)
