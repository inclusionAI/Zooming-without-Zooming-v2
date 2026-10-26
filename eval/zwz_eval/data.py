"""Explicit layouts and fail-closed annotation/image validation."""
from __future__ import annotations

import importlib.util
import json
import random
from collections import Counter
from pathlib import Path

from .io import atomic_json, digest, file_hash, read_json
from .protocols import GUI_EXPECTED, MEM_SPLITS, MILE_TASKS, RESOURCES, mml_specs, PLATFORM_BENCHMARKS


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rows(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        with path.open(encoding='utf-8') as stream:
            return [json.loads(line) for line in stream if line.strip()]
    value = read_json(path)
    if not isinstance(value, list):
        raise ValueError(f'Expected a JSON list: {path}; use prepare for official raw formats')
    return value


def sample_uid(row, index):
    for key in ('sample_uid', 'question_id', 'sample_id', 'id', 'doc_id'):
        if row.get(key) is not None:
            return f'{key}:{row[key]}'
    return 'sha256:' + digest(row)


def image_references(row):
    paths = list(row.get('images') or [])
    for message in row.get('messages') or []:
        for block in message.get('content', []) if isinstance(message.get('content'), list) else []:
            if block.get('type') == 'image_path':
                paths.append(block['path'])
    return paths


def resolve_image(value, image_root):
    path = Path(value)
    if not path.is_absolute():
        path = Path(image_root) / path
    if not path.is_file():
        raise FileNotFoundError(f'Image missing: {path}')
    return path.resolve()


def image_root_for(bench, data):
    """One image-prefix policy for audit, inference and service command plans.

    Paths are relocated only; image names/order/pixels are never guessed here.
    """
    if 'image_root' in data:
        return Path(data['image_root'])
    root = Path(data['root'])
    if bench == 'mmlongbench':
        return root / 'images/mmlb_image'
    if bench in PLATFORM_BENCHMARKS or bench in ('mmsi', 'viewspatial', 'mindcube', 'chartography', 'spar', 'erqa', 'mimic'):
        return root
    return root / 'images'


def annotation_sets(bench, root):
    root = Path(root)
    if bench in PLATFORM_BENCHMARKS or bench in ('mmsi', 'viewspatial', 'mindcube', 'chartography', 'spar', 'erqa'):
        return {'all': root / 'records.json'}
    if bench == 'mimic':
        return {f'mimic-{s}': root / f'mimic_{s}.json' for s in ('common', 'counting', 'listing', 'oddone')}
    if bench == 'memlens':
        return {f'memlens-{s}': root / f'memlens_{s}.json' for s in MEM_SPLITS}
    if bench == 'milebench':
        return {f'milebench-{t}-test': root / f'milebench_{t}_test.json' for t in MILE_TASKS}
    if bench == 'agentvista':
        return {'all': root / 'val.json'}
    if bench == 'screenspot_pro':
        files = sorted((root / 'annotations').glob('*.json'))
        if not files:
            raise FileNotFoundError('ScreenSpot annotations/*.json missing')
        return {p.stem: p for p in files}
    if bench == 'ui_vision':
        return {s: root / 'annotations/element_grounding' / f'element_grounding_{s}.json' for s in GUI_EXPECTED[bench]}
    if bench == 'mmlongbench':
        return {f"{s['family']}:K{s['k']}": root / 'mmlb_data' / s['file'] for s in mml_specs()}
    raise ValueError(bench)


def audit(bench, data, verify_images=True):
    root = Path(data['root'])
    image_root = image_root_for(bench, data)
    sets = annotation_sets(bench, root)
    platform_mode = None
    report = {'benchmark': bench, 'sets': {}, 'images_checked': 0, 'image_sha256': {}}
    if (bench in PLATFORM_BENCHMARKS or bench in ('mmsi', 'viewspatial', 'mindcube', 'chartography', 'spar', 'erqa', 'mimic')) and (root / 'preparation.json').exists():
        preparation = read_json(root / 'preparation.json')
        records_sha = ({p.name: file_hash(p) for p in sets.values()} if bench == 'mimic'
                       else file_hash(root / 'records.json'))
        if preparation['benchmark'] != bench or preparation['records_sha256'] != records_sha:
            raise ValueError('Prepared annotation provenance mismatch')
        report['preparation'] = preparation
    for task, path in sets.items():
        if not path.is_file():
            raise FileNotFoundError(f'Missing annotation: {path}')
        annotations = rows(path)
        if not annotations:
            raise ValueError(f'Empty annotation: {path}')
        if bench in PLATFORM_BENCHMARKS:
            from .general_visual import COUNTS, validate_row, preparation_mode
            if len(annotations) != COUNTS[bench]:
                raise ValueError(f'{bench}: wrong fixed denominator')
            if not (root / 'preparation.json').is_file():
                raise ValueError('These profiles require the audited Qwen preprocessing manifest')
            for row in annotations:
                validate_row(row)
            policy = preparation['preprocessing']
            if (policy['min_pixels'], policy['max_pixels'], policy['image_patch_size']) != (65536, 16777216, 14):
                raise ValueError('Image preprocessing policy differs from the reference')
            platform_mode = preparation_mode(preparation)
            if preparation.get('rows') != len(annotations):
                raise ValueError('Prepared row count differs from its manifest')
            if platform_mode == 'materialized':
                for row in annotations:
                    for image in row['images']:
                        asset = preparation['assets'].get(image)
                        if not isinstance(asset, dict) or not isinstance(asset.get('prepared_sha256'), str):
                            raise ValueError('Prepared image is absent from its preprocessing manifest')
        if bench == 'mimic':
            from .mimic import COUNTS, validate_row
            if data.get('mathruler_mode') not in ('disabled', 'required'):
                raise ValueError('Explicit MIMIC mathruler_mode required')
            if len(annotations) != COUNTS[task.removeprefix('mimic-')]:
                raise ValueError('MIMIC requires 1000/5000/1000/1000 records')
            for row in annotations:
                validate_row(row, task)
        if bench == 'erqa':
            from .erqa import EXPECTED, validate_row
            if len(annotations) != EXPECTED:
                raise ValueError('ERQA requires 400 annotations')
            for row in annotations:
                validate_row(row)
        if bench in ('chartography', 'spar'):
            from .structured import validate_set
            validate_set(annotations, bench)
        if bench in ('mmsi', 'viewspatial', 'mindcube'):
            from .spatial import EXPECTED, validate_row
            if len(annotations) != EXPECTED[bench]:
                raise ValueError(f'{bench}: expected {EXPECTED[bench]}, got {len(annotations)}')
            for row in annotations:
                validate_row(row, bench)
            if bench == 'mindcube':
                from .references import load
                settings = Counter(load('mindcube').get_setting_from_id(row['id']) for row in annotations)
                if settings != {'around': 250, 'rotation': 200, 'among': 600}:
                    raise ValueError('MindCube Tiny settings differ from the verified project 1050-row profile')
        ids = [sample_uid(row, i) for i, row in enumerate(annotations)]
        if bench != 'mmlongbench' and len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate sample IDs in {task}')
        if bench == 'memlens' and len(annotations) != 789:
            raise ValueError(f'{task}: expected 789, got {len(annotations)}')
        if bench == 'agentvista' and len(annotations) != 209:
            raise ValueError('AgentVista requires 209 questions')
        if bench == 'ui_vision' and len(annotations) != GUI_EXPECTED[bench][task]:
            raise ValueError(f'UI-Vision {task}: wrong denominator')
        selected = annotations
        if bench == 'mmlongbench':
            spec = next(s for s in mml_specs() if f"{s['family']}:K{s['k']}" == task)
            if spec['family'] in ('infoseek', 'viquae'):
                selected = [r for r in selected if '"' not in r['image']]
            keys = sorted({r['id'] for r in selected})
            keys = set(random.Random(42).sample(keys, min(spec['max_test_samples'], len(keys))))
            selected = [r for r in selected if r['id'] in keys]
            if not selected:
                raise ValueError(f'{task}: selected dataset is empty')
        if verify_images:
            paths = set()
            for row in selected:
                if bench == 'screenspot_pro':
                    paths.add(str(image_root / row['img_filename']))
                elif bench == 'ui_vision':
                    paths.add(str(image_root / row['image_path']))
                elif bench == 'mmlongbench':
                    # V-RAG one image; NIAH ctxs structures vary by family.
                    if 'image' in row and isinstance(row['image'], str):
                        paths.add(str(image_root / row['image']))
                    for image in row.get('image_list', []):
                        paths.add(str(image_root / image))
                    for image in row.get('choices_image', []):
                        paths.add(str(image_root / image))
                    if task.startswith('vh_multi:'):
                        for image in row.get('ctxs', []):
                            paths.add(str(image_root / image))
                else:
                    paths.update(str(resolve_image(p, image_root)) for p in image_references(row))
            for value in paths:
                p = Path(value)
                if not p.is_file() or p.stat().st_size == 0:
                    raise FileNotFoundError(f'Missing/empty image: {p}')
                key = str(p.resolve())
                if key not in report['image_sha256']:
                    report['image_sha256'][key] = file_hash(p)
                if bench in PLATFORM_BENCHMARKS and platform_mode == 'materialized':
                    relative = str(p.resolve().relative_to(Path(image_root).resolve()))
                    expected_image = preparation['assets'].get(relative)
                    if expected_image is None or expected_image['prepared_sha256'] != report['image_sha256'][key]:
                        raise ValueError('Prepared image differs from its preprocessing manifest')
            report['images_checked'] += len(paths)
        report['sets'][task] = {'path': str(path.resolve()), 'sha256': file_hash(path), 'rows': len(annotations), 'unique_ids': len(set(ids)), 'ids_sha256': digest(ids)}
        if bench == 'mindcube':
            report['sets'][task]['setting_counts'] = dict(settings)
        if bench == 'mmlongbench':
            report['sets'][task]['selected_rows'] = len(selected)
            report['sets'][task]['selected_ids_sha256'] = digest([r['id'] for r in selected])
    if bench == 'screenspot_pro' and sum(v['rows'] for v in report['sets'].values()) != 1581:
        raise ValueError('ScreenSpot-Pro requires 1581 tasks')
    report['manifest_sha256'] = digest({'sets': report['sets'], 'images': report['image_sha256'],
                                       'preparation': report.get('preparation')})
    return report


def prepare_memlens(source, image_root, target):
    """Reuse the exact project role-preserving converter; verify every referenced image."""
    converter = load_module(RESOURCES / 'vendor/memlens/prepare.py', '_memlens_prepare')
    target = Path(target)
    for split in MEM_SPLITS:
        original = read_json(Path(source) / f'dataset_{split}.json')
        converted = [converter.convert_record(row, split, Path(image_root).resolve()) for row in original]
        if len(converted) != 789:
            raise ValueError(f'MemLens {split}: expected 789')
        for row in converted:
            for p in image_references(row):
                resolve_image(p, image_root)
        atomic_json(target / f'memlens_{split}.json', converted)


def prepare_milebench(source, target):
    """Official annotation -> project schema. No resizing, merging or context truncation.

    Exact historical claims still require matching the released normalized-data manifest.
    """
    source, target = Path(source), Path(target)
    for task in MILE_TASKS:
        raw_path = source / task / f'{task}.json'
        raw = read_json(raw_path)
        metadata, result = raw['meta_data'], []
        for row in raw['data']:
            item = row['task_instance']
            query = item['context']
            images = [str((source / task / 'images' / p).resolve()) for p in item['images_path']]
            for i in range(len(images)):
                query = query.replace('{image#%d}' % (i+1), '<image>').replace('{table#%d}' % (i+1), '<image>')
            choices = item.get('choice_list', [])
            if choices:
                query += '\nChoice list: \n' + '\n'.join((f'{chr(65+i)}. ' if task != 'GPR1200' else '') + str(c) for i, c in enumerate(choices)) + '\nYour answer is: '
            instruction = metadata['task_instruction'][row['task_instruction_id']]
            for path in images:
                resolve_image(path, source)
            result.append({'sample_id': row['sample_id'], 'query': query, 'images': images,
                           'response': str(row['response']), 'task': task, 'system_prompt': instruction,
                           'meta': {'question_type': metadata['question_type'], 'choice_list': choices, 'task_instruction': instruction}})
        atomic_json(target / f'milebench_{task}_test.json', result)


def prepare_ui_vision(source, target):
    source, target = Path(source), Path(target)
    for split, expected in GUI_EXPECTED['ui_vision'].items():
        original = read_json(source / 'annotations/element_grounding' / f'element_grounding_{split}.json')
        if len(original) != expected:
            raise ValueError(f'UI-Vision {split}: expected {expected}')
        converted = [{'img_filename': r['image_path'], 'img_size': r['image_size'], 'bbox': r['bbox'],
                      'instruction': r['prompt_to_evaluate'], 'platform': r['platform'], 'application': r['platform'],
                      'ui_type': r['element_type'], 'group': r['category']} for r in original]
        atomic_json(target / f'{split}.json', converted)


def prepare_agentvista(source, target):
    """Convert the pinned official Parquet release; retain encoded image bytes unchanged."""
    import io
    import pyarrow.parquet as pq
    from PIL import Image
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists():
        raise ValueError('AgentVista conversion requires a new target directory')
    files = sorted((source / 'data').glob('*.parquet'))
    if not files:
        files = sorted(source.glob('*.parquet'))
    if not files:
        raise FileNotFoundError('No official AgentVista Parquet shards found')
    annotations, blobs = [], []
    for file in files:
        for batch in pq.ParquetFile(file).iter_batches():
            for row in batch.to_pylist():
                index = len(annotations)
                names = []
                for i, value in enumerate(row['images']):
                    if not isinstance(value, dict):
                        raise ValueError('Expected the official encoded image structure')
                    blob = value.get('bytes')
                    if blob is None:
                        image = resolve_image(value['path'], source)
                        blob = image.read_bytes()
                    with Image.open(io.BytesIO(blob)) as image:
                        extension = (image.format or '').lower()
                    if extension not in ('png', 'jpeg', 'webp', 'gif', 'bmp', 'tiff'):
                        raise ValueError('Unrecognized encoded image format')
                    name = f'row{index:03d}_{i:02d}.{extension}'
                    blobs.append((name, blob))
                    names.append(name)
                annotations.append({'doc_id': f'agentvista_official_row_{index:03d}',
                    'problem': row['problem'], 'solution': row['answer'], 'images': names,
                    'domain': row.get('domain'), 'subdomain': row.get('subdomain')})
    if len(annotations) != 209:
        raise ValueError(f'Expected 209 official questions, got {len(annotations)}')
    images = target / 'images'
    images.mkdir(parents=True)
    for name, blob in blobs:
        (images / name).write_bytes(blob)
    atomic_json(target / 'val.json', annotations)
    atomic_json(target / 'conversion_manifest.json', {'source_files': {str(p.relative_to(source)): file_hash(p) for p in files},
        'questions': len(annotations), 'images': len(blobs), 'image_bytes_unchanged': True})
