"""Chartography and SPAR project profiles with preserved, distinct metrics."""
from __future__ import annotations

import base64
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .api import CompatibleAPI
from .data import resolve_image, rows, sample_uid
from .io import append_jsonl, atomic_json, digest, file_hash, read_journal, repair_torn_tail
from .references import load
from .spatial import indexed, infer as journal_infer

BENCHMARKS = ('chartography', 'spar')
EXPECTED = {'chartography': 100, 'spar': 7211}
CHART_SYSTEM = ('Return only the required verdict JSON. Keep extracted_answer concise '
                '(at most 200 characters), summarizing repetitive or irrelevant output '
                'rather than copying it. Keep rationale concise. Apply the supplied '
                'grading rubric without changing it.')


def task_names():
    m = load('spar')
    return m.MCA | m.NA | {'view_change_infer'}


def validate_row(row, bench):
    fields = ('prompt', 'golden_answer') if bench == 'chartography' else ('question', 'answer', 'task')
    if any(not isinstance(row.get(k), str) or not row[k].strip() for k in fields):
        raise ValueError(f'{bench}: missing original question/gold/task fields')
    if bench == 'spar' and row['task'] not in task_names():
        raise ValueError('Unknown SPAR task')
    images = row.get('images')
    if not isinstance(images, list) or not images or not all(isinstance(p, str) and p for p in images):
        raise ValueError('Ordered image paths required')
    if len(images) > (1 if bench == 'chartography' else 3):
        raise ValueError('Image count differs from reference profile')


def validate_set(records, bench):
    if len(records) != EXPECTED[bench]:
        raise ValueError(f'{bench}: requires {EXPECTED[bench]} records')
    ids = [sample_uid(row, i) for i, row in enumerate(records)]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate benchmark identity')
    for row in records:
        validate_row(row, bench)
    if bench == 'spar' and {row['task'] for row in records} != task_names():
        raise ValueError('SPAR requires all 20 task types')


def prepare(bench, source, target):
    source, target = Path(source), Path(target)
    if target.exists():
        raise ValueError('Preparation target must not exist')
    import pyarrow.parquet as pq
    files = sorted((source / 'data').glob('*.parquet')) if bench == 'spar' else [source]
    if bench == 'spar' and len(files) != 4:
        raise ValueError('SPAR requires four original Parquet shards')
    records, assets = [], []
    for index, row in enumerate(pq.read_table(files).to_pylist()):
        row = dict(row)
        if bench == 'chartography':
            row['id'] = row['task_id']
            row['images'] = [row['chart_path']]
        else:
            paths = []
            for position, image in enumerate(row.pop('image')):
                raw = image.get('bytes') if isinstance(image, dict) else None
                if not isinstance(raw, bytes) or not raw:
                    raise ValueError('SPAR expects original embedded image bytes')
                relative = f'images/{index:05d}-{position}.jpg'
                paths.append(relative)
                assets.append((relative, raw))
            row['images'] = paths
        records.append(row)
    validate_set(records, bench)
    target.mkdir(parents=True)
    for relative, raw in assets:
        path = target / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)
    atomic_json(target / 'records.json', records)
    atomic_json(target / 'preparation.json', {
        'benchmark': bench, 'source_files': {p.name: file_hash(p) for p in files},
        'records_sha256': file_hash(target / 'records.json'),
        'transform': 'schema/path projection only; embedded image bytes unchanged', 'rows': len(records)})


def messages(row, bench, image_root):
    images = []
    for value in row['images']:
        path = resolve_image(value, image_root)
        mime = 'image/jpeg'
        if bench == 'chartography':
            mime = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
                    '.gif': 'image/gif', '.webp': 'image/webp'}.get(path.suffix.lower(), 'image/png')
        url = 'data:' + mime + ';base64,' + base64.b64encode(path.read_bytes()).decode('ascii')
        images.append({'type': 'image_url', 'image_url': {'url': url}})
    if bench == 'chartography':
        return [{'role': 'user', 'content': [{'type': 'text', 'text': row['prompt']}, *images]}]
    ref = load('spar')
    return [{'role': 'system', 'content': ref.ANSWER_FORMAT_SYSTEM},
            {'role': 'user', 'content': [{'type': 'text', 'text': ref.official_prompt(row)}, *images]}]


def infer(config, bench, directory, client=None):
    options = ({'enable_thinking': False, 'add_generation_parameters': {'enable_thinking': False}}
               if bench == 'chartography' else {'top_p': 1.0})
    return journal_infer(config, bench, directory, client, message_builder=messages,
                         generation_options=options, strip_content=bench == 'chartography')


def checked_predictions(config, bench, directory):
    expected = indexed(config, bench)
    if not expected:
        raise ValueError('Cannot score an empty benchmark')
    predictions = {r['uid']: r for r in read_journal(Path(directory) / 'predictions.jsonl')}
    if set(expected) != set(predictions):
        raise ValueError('Missing/foreign predictions; partial scoring is not permitted')
    if any(predictions[uid].get('status') != 'ok' or predictions[uid]['input_sha256'] != digest(row)
           for uid, row in expected.items()):
        raise ValueError('Failed or stale predictions')
    return expected, predictions


def judge_jobs(config, directory, filename, requests, call):
    """Shared persistence only: each caller supplies its own prompt/rubric/parser."""
    journal = Path(directory) / 'scores' / filename
    journal.parent.mkdir(parents=True, exist_ok=True)
    repair_torn_tail(journal)
    prior = {r['uid']: r for r in read_journal(journal)}
    if set(prior) - set(requests):
        raise ValueError('Foreign review records')
    for uid, request in requests.items():
        if uid in prior and prior[uid]['request_sha256'] != digest(request):
            raise ValueError('Review request changed; choose a new run ID')

    def work(uid):
        record = {'uid': uid, 'request_sha256': digest(requests[uid])}
        try:
            record.update(call(requests[uid]))
            record['status'] = 'ok'
        except Exception as exc:
            record.update(status='error', error_type=type(exc).__name__)
            if hasattr(exc, 'judge_attempts'):
                record['judge_attempts'] = exc.judge_attempts
        return record

    pending = [uid for uid in requests if prior.get(uid, {}).get('status') != 'ok']
    with ThreadPoolExecutor(max_workers=config['workers']) as pool:
        for future in as_completed([pool.submit(work, uid) for uid in pending]):
            record = future.result()
            append_jsonl(journal, record)
            prior[record['uid']] = record
    if any(prior.get(uid, {}).get('status') != 'ok' for uid in requests):
        raise RuntimeError('Required judge records incomplete; no final score emitted')
    return prior


def chart_request(row, prediction):
    ref = load('chartography')
    gold = row['golden_answer'].strip()
    return {'gold': gold, 'empty': not prediction.strip(),
            'messages': [{'role': 'system', 'content': CHART_SYSTEM},
                         {'role': 'user', 'content': ref._judge_prompt(row['prompt'], gold, prediction)}]}


def score(config, bench, directory, client=None):
    directory = Path(directory)
    expected, predictions = checked_predictions(config, bench, directory)
    if bench == 'spar':
        return score_spar(directory, expected, predictions)
    client = client or CompatibleAPI(config['judge'], config['timeout'], config['retries'])
    requests = {uid: chart_request(row, predictions[uid]['model_answer']) for uid, row in expected.items()}

    def call(request):
        if request['empty']:
            return {'scores': [{'criterion_id': 'criterion_1', 'extracted_answer': '(empty)', 'score': 0,
                               'rationale': 'Model produced empty response; treated as incorrect.'}],
                    'raw_response': '(empty)', 'judge_attempts': []}
        ref, guard = load('chartography'), load('chartography_guard')
        attempts = []
        for _ in range(5):  # Reference MAX_RETRIES=4 plus initial attempt.
            try:
                response = client.complete(request['messages'], thinking=False, max_tokens=8192,
                                           response_format={'type': 'json_object'}, guided_json=ref.JUDGE_GUIDED_JSON)
            except Exception as error:
                error.judge_attempts = attempts
                raise
            raw = response['choices'][0]['message'].get('content') or ''
            entries = ref._parse_verdict(raw)
            attempts.append(raw)
            if entries and guard.valid_verdict(entries, request['gold']):
                return {'scores': entries, 'raw_response': raw, 'judge_attempts': attempts}
        error = ValueError('Unparseable or contradictory Chartography verdict')
        error.judge_attempts = attempts
        raise error

    judged = judge_jobs(config, directory, 'chart_judgments.jsonl', requests, call)
    scored = []
    for uid, row in expected.items():
        rule, reason = load('chartography_rule').score_one(predictions[uid]['model_answer'], row['golden_answer'].strip())
        correct = all(load('chartography')._verdict_passes(item['score']) for item in judged[uid]['scores'])
        scored.append({'uid': uid, 'correct': correct, 'rule_correct': rule, 'rule_reason': reason})
    metrics = {'benchmark': bench, 'total': len(scored),
               'primary_accuracy': sum(r['correct'] for r in scored)/len(scored),
               'rule_diagnostic_accuracy': sum(r['rule_correct'] for r in scored)/len(scored),
               'reasoning_channel_outputs': sum(bool(p.get('reasoning_content')) for p in predictions.values())}
    atomic_json(directory / 'scores/scored_records.json', scored)
    atomic_json(directory / 'scores/metrics.json', metrics)
    return metrics


def score_spar(directory, expected, predictions):
    ref = load('spar')
    grouped, scored = defaultdict(list), []
    for uid, row in expected.items():
        value = ref.official_score(row, predictions[uid]['model_answer'])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid fractional SPAR score')
        grouped[row['task']].append(value)
        scored.append({'uid': uid, 'task': row['task'], 'official_score': value})
    per_task = {task: sum(values)/len(values) for task, values in sorted(grouped.items())}
    metrics = {'benchmark': 'spar', 'total': len(scored), 'per_task': per_task,
               'primary_accuracy': sum(per_task.values())/len(per_task),
               'official_micro': sum(r['official_score'] for r in scored)/len(scored),
               'buckets': {name: sum(per_task[k] for k in keys if k in per_task)/len(keys & per_task.keys())
                           for name, keys in [('Low', ref.LOW), ('Middle', ref.MIDDLE), ('High', ref.HIGH)]
                           if keys & per_task.keys()}}
    atomic_json(Path(directory) / 'scores/scored_records.json', scored)
    atomic_json(Path(directory) / 'scores/metrics.json', metrics)
    return metrics


def review_spar(config, directory, client=None):
    directory = Path(directory)
    expected, predictions = checked_predictions(config, 'spar', directory)
    ref, judge_ref = load('spar'), load('spar_review')
    values = {uid: ref.official_score(row, predictions[uid]['model_answer']) for uid, row in expected.items()}
    requests = {}
    for uid, row in expected.items():
        if values[uid] < 1:
            user = (f"Task: {row['task']}\n\nQuestion:\n{row['question']}\n\n"
                    f"Reference answer:\n{row['answer']}\n\n"
                    f"Assistant response:\n{predictions[uid]['model_answer']}")
            requests[uid] = [{'role': 'system', 'content': judge_ref.JUDGE_SYSTEM},
                             {'role': 'user', 'content': user}]
    client = client or CompatibleAPI(config['judge'], config['timeout'], config['retries'])

    def call(messages):
        response = client.complete(messages, thinking=False, max_tokens=160)
        raw = response['choices'][0]['message'].get('content') or ''
        try:
            return {**judge_ref.parse_json(raw), 'raw_response': raw}
        except (ValueError, TypeError, AttributeError):
            error = ValueError('Invalid SPAR review JSON')
            error.judge_attempts = [raw]
            raise error

    reviewed = judge_jobs(config, directory, 'spar_reviews.jsonl', requests, call)
    sums = defaultdict(list)
    for uid, row in expected.items():
        corrected = 1.0 if uid in reviewed and reviewed[uid]['verdict'] == 'correct' else values[uid]
        sums[row['task']].append(corrected)
    report = {'protocol': 'non_official_llm_fallback_audit', 'samples': len(expected),
              'reviewed': len(requests), 'official_micro': sum(values.values())/len(values),
              'reviewed_micro': sum(sum(v) for v in sums.values())/len(values),
              'reviewed_task_macro': sum(sum(v)/len(v) for v in sums.values())/len(sums),
              'per_task': {task: sum(v)/len(v) for task, v in sorted(sums.items())}}
    atomic_json(directory / 'scores/spar_audit_summary.json', report)
    return report
