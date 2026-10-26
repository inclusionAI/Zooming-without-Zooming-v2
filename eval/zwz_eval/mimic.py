"""Four-task MIMIC with full Counting identity and explicit mathruler policy."""
from __future__ import annotations

import importlib.metadata
from functools import lru_cache
from collections import Counter
from pathlib import Path

from .api import CompatibleAPI
from .data import annotation_sets, load_module, rows, sample_uid
from .io import atomic_json, digest, file_hash, read_journal
from .protocols import RESOURCES
from .references import load
from .structured import judge_jobs

COUNTS = {'common': 1000, 'counting': 5000, 'listing': 1000, 'oddone': 1000}


@lru_cache(maxsize=1)
def formatter():
    return load_module(RESOURCES / 'vendor/common/infer.py', '_mimic_shared_formatter')


def validate_row(row, task):
    if any(not isinstance(row.get(k), str) for k in ('query', 'response')):
        raise ValueError('MIMIC requires original query and textual reference response')
    if not isinstance(row.get('images'), list) or not row['images']:
        raise ValueError('MIMIC requires original ordered images')
    if not all(isinstance(p, str) and p for p in row['images']) or len(row['images']) > 256:
        raise ValueError('Invalid MIMIC image paths/count')
    if not isinstance(row.get('sample_uid'), str) or not row['sample_uid']:
        raise ValueError('MIMIC prepared records need explicit sample_uid')
    if task == 'mimic-counting' and row['sample_uid'] != formatter().make_sample_uid(row, task):
        raise ValueError('Counting sample_uid does not cover id/query/images')


def prepare(source, target, image_root=None, image_map=None):
    source, target = Path(source), Path(target)
    if target.exists():
        raise ValueError('Preparation target must not exist')
    from .data import resolve_image
    from .io import read_json
    remap = read_json(image_map) if image_map else {}
    if not isinstance(remap, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in remap.items()):
        raise ValueError('image-map must map exact original image paths to local files')
    prepared, provenance = {}, {}
    for name, count in COUNTS.items():
        path = source / f'mimic_{name}.json'
        data = rows(path)
        if len(data) != count:
            raise ValueError(f'MIMIC {name}: expected {count}')
        result = []
        for original in data:
            row = dict(original)
            if image_root or image_map:
                row['images'] = [str(resolve_image(remap.get(p, p), image_root or source)) for p in row['images']]
            row['sample_uid'] = (formatter().make_sample_uid(row, f'mimic-{name}') if name == 'counting'
                                 else row.get('sample_uid') or formatter().make_sample_uid(row, f'mimic-{name}'))
            validate_row(row, f'mimic-{name}')
            result.append(row)
        if len({r['sample_uid'] for r in result}) != count:
            raise ValueError('Duplicate full sample identities; never deduplicate raw Counting IDs')
        prepared[path.name] = result
        provenance[path.name] = file_hash(path)
    target.mkdir(parents=True)
    for name, result in prepared.items():
        atomic_json(target / name, result)
    atomic_json(target / 'preparation.json', {'benchmark': 'mimic', 'source_files': provenance,
        'image_map_sha256': file_hash(image_map) if image_map else None,
        'records_sha256': {name: file_hash(target / name) for name in prepared},
        'transform': 'preserved normalized annotations plus reference full sample_uid; no prompt/image edits'})


def math_grader(config):
    data = config['datasets']['mimic']
    mode = data.get('mathruler_mode')
    if mode not in ('disabled', 'required'):
        raise ValueError('Declare datasets.mimic.mathruler_mode as disabled or required; no environment-dependent default')
    if mode == 'disabled':
        return None
    expected = data.get('mathruler_version')
    if not expected:
        raise ValueError('Record the reference mathruler_version when enabling its branch')
    actual = importlib.metadata.version('mathruler')
    if actual != expected:
        raise ValueError('mathruler version differs from the declared reference')
    from mathruler.grader import grade_answer
    return grade_answer


def score(config, directory, client=None, grade=None):
    directory = Path(directory)
    expected = {}
    for task, path in annotation_sets('mimic', config['datasets']['mimic']['root']).items():
        for index, row in enumerate(rows(path)):
            uid = f'{task}/{sample_uid(row, index)}'
            if uid in expected:
                raise ValueError('Duplicate MIMIC identity')
            expected[uid] = (task, row)
    predictions = {r['uid']: r for r in read_journal(directory / 'predictions.jsonl')}
    if not expected or set(expected) != set(predictions):
        raise ValueError('MIMIC incomplete/foreign inference records')
    if any(predictions[uid].get('status') != 'ok' or predictions[uid]['input_sha256'] != digest(row)
           for uid, (_, row) in expected.items()):
        raise ValueError('MIMIC failed/stale predictions')
    grader = grade if grade is not None else math_grader(config)
    ref, listing = load('mimic_judge'), load('mimic_listing')
    requests, scored = {}, {}
    total_gt = total_pred = total_recall = 0
    for uid, (task, row) in expected.items():
        text = predictions[uid]['model_answer']
        if task == 'mimic-listing':
            truth = listing.gt_objects_from_response(row['response'])
            objects = listing.parse_model_answer_to_objects(text)
            hits = sum(any(obj == pred for pred in objects) for obj in truth)
            total_gt += len(truth)
            total_pred += len(objects)
            total_recall += hits
            scored[uid] = {'uid': uid, 'task': task, 'objects': objects,
                           'gold_objects': truth, 'matched': hits, 'source': 'micro_listing'}
            continue
        extracted = ref.extract_answer(text)
        correct = ref.string_match(extracted, row['response'])
        source = 'string' if correct else None
        if not correct and grader is not None:
            try:
                correct = bool(grader(extracted, row['response']))
            except Exception:
                correct = False  # Reference behavior for a failed equivalence check.
            if correct:
                source = 'mathruler'
        scored[uid] = {'uid': uid, 'task': task, 'extracted': extracted,
                       'correct': bool(correct), 'source': source}
        if not correct:
            prompt = ref.PROMPT_TEMPLATE.format(question=row['query'].replace('<image>', ''),
                                                 gt=row['response'], response=extracted)
            requests[uid] = [{'role': 'user', 'content': prompt}]
    client = client or CompatibleAPI(config['judge'], config['timeout'], config['retries'])

    def call(messages):
        response = client.complete(messages, thinking=False, max_tokens=8192)
        raw = response['choices'][0]['message'].get('content') or ''
        return {'verdict': ref.parse_llm_judge_response(raw), 'raw_response': raw}

    judged = judge_jobs(config, directory, 'mimic_judgments.jsonl', requests, call)
    for uid in requests:
        scored[uid].update(correct=judged[uid]['verdict'].lower() == 'yes', source='llm')
    if not total_gt:
        raise ValueError('Listing gold denominator cannot be empty')
    precision = total_recall / total_pred if total_pred else 0.0
    recall = total_recall / total_gt
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0
    tasks = {'mimic-listing': f1}
    for task in ('mimic-common', 'mimic-counting', 'mimic-oddone'):
        selected = [r for r in scored.values() if r['task'] == task]
        if not selected:
            raise ValueError('All four MIMIC tasks are required')
        tasks[task] = sum(r['correct'] for r in selected)/len(selected)
    result = {'benchmark': 'mimic', 'total': len(scored), 'per_task': tasks,
              'per_task_total': dict(Counter(task for task, _ in expected.values())),
              'source_counts': dict(Counter(r['source'] for r in scored.values())),
              'mathruler_mode': config['datasets']['mimic']['mathruler_mode'],
              'listing': {'total_gt': total_gt, 'total_pred': total_pred, 'total_recall': total_recall,
                          'precision': precision, 'recall': recall, 'f1': f1}}
    atomic_json(directory / 'scores/scored_records.json', list(scored.values()))
    atomic_json(directory / 'scores/metrics.json', result)
    return result
