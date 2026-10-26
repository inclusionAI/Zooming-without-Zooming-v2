"""Portable ports of the audited MMSI/ViewSpatial/MindCube project clients.

Only transport, storage and validation are shared. Prompts, image transforms,
answer parsers and review routing remain benchmark-specific.
"""
from __future__ import annotations

import base64
import io
import mimetypes
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from pathlib import Path

from .api import CompatibleAPI
from .data import image_root_for, resolve_image, rows, sample_uid
from .io import append_jsonl, atomic_json, digest, file_hash, read_journal, repair_torn_tail
from .protocols import PROTOCOLS
from .references import load

BENCHMARKS = ('mmsi', 'viewspatial', 'mindcube')
EXPECTED = {'mmsi': 1000, 'viewspatial': 5712, 'mindcube': 1050}
MMSI_SUFFIX = "Answer with the option's letter from the given choices directly. Enclose the option's letter within ``."
VIEW_SUFFIX = '\nReply only to the corresponding option.\nAnswer:'


def gold(row, bench):
    value = row['gt_answer'] if bench == 'mindcube' else row['answer']
    if bench == 'mmsi':
        # Source scorer compares lowercased first characters without stripping.
        value = str(value)[:1].upper()
    elif bench == 'viewspatial':
        value = load('viewspatial').extract_answer(value)
    allowed = 'ABCDE' if bench == 'mindcube' else 'ABCD'
    if not isinstance(value, str) or len(value) != 1 or value not in allowed:
        raise ValueError(f'{bench}: gold must be a single valid option letter')
    return value


def validate_row(row, bench):
    gold(row, bench)
    required = {'mmsi': ('question', 'question_type'),
                'viewspatial': ('question', 'choices', 'question_type'),
                'mindcube': ('input_prompt', 'id')}[bench]
    if any(not isinstance(row.get(k), str) for k in required):
        raise ValueError(f'{bench}: missing/invalid original prompt fields')
    images = row.get('images')
    if not isinstance(images, list) or not images or not all(isinstance(p, str) and p for p in images):
        raise ValueError(f'{bench}: nonempty ordered image paths required')
    if len(images) > PROTOCOLS[bench]['max_images']:
        raise ValueError(f'{bench}: image count exceeds reference service limit')


def prepare(bench, source, target):
    """Path/schema projection only; no changes to question, choices or gold.

    MMSI Parquet image bytes are materialized verbatim. Other inputs retain
    image paths and require the original image root in the run configuration.
    Targets must be new, avoiding accidental replacement of useful data.
    """
    source, target = Path(source), Path(target)
    if target.exists():
        raise ValueError('Preparation target must not exist; existing data is retained')
    if bench == 'mmsi':
        import pyarrow.parquet as pq
        original = pq.read_table(source).to_pylist()
    else:
        original = rows(source)
    normalized, assets = [], []
    for index, row in enumerate(original):
        row = dict(row)
        if bench == 'mmsi':
            paths = []
            for position, image in enumerate(row.get('images') or []):
                raw = image.get('bytes') if isinstance(image, dict) else image
                if not isinstance(raw, bytes) or not raw:
                    raise ValueError('MMSI expects original embedded image bytes')
                relative = f'images/{index:05d}-{position:03d}.jpg'
                paths.append(relative)
                assets.append((relative, raw))
            row['images'] = paths
        elif bench == 'viewspatial':
            if not isinstance(row.get('image_path'), list):
                raise ValueError('ViewSpatial image_path must retain the ordered list')
            row['images'] = list(row['image_path'])
            row['source_id'] = row.get('id')
            row['id'] = index  # The project source uses annotation row index.
        validate_row(row, bench)
        normalized.append(row)
    ids = [sample_uid(row, i) for i, row in enumerate(normalized)]
    if len(normalized) != EXPECTED[bench] or len(ids) != len(set(ids)):
        raise ValueError('Preparation counts/IDs do not match the fixed benchmark')
    target.mkdir(parents=True)
    for relative, raw in assets:
        path = target / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(raw)
    atomic_json(target / 'records.json', normalized)
    atomic_json(target / 'preparation.json', {
        'benchmark': bench, 'source_sha256': file_hash(source),
        'records_sha256': file_hash(target / 'records.json'),
        'transform': 'schema/path projection; original image bytes; no prompt edits',
        'rows': len(normalized),
    })


def image_url(path, bench):
    if bench == 'viewspatial':
        from PIL import Image
        with Image.open(path) as image:
            image = image.convert('RGB').resize((420, 280), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=95)
        raw, mime = buffer.getvalue(), 'image/jpeg'
    else:
        raw = Path(path).read_bytes()
        mime = 'image/jpeg' if bench == 'mmsi' else (mimetypes.guess_type(str(path))[0] or 'image/jpeg')
    return 'data:' + mime + ';base64,' + base64.b64encode(raw).decode('ascii')


def messages(row, bench, image_root):
    content = [{'type': 'image_url', 'image_url': {
        'url': image_url(resolve_image(path, image_root), bench)}} for path in row['images']]
    text = {'mmsi': lambda: row['question'] + '\n' + MMSI_SUFFIX,
            'viewspatial': lambda: row['question'] + row['choices'] + VIEW_SUFFIX,
            'mindcube': lambda: row['input_prompt']}[bench]()
    content.append({'type': 'text', 'text': text})
    return [{'role': 'user', 'content': content}]


def indexed(config, bench):
    records = rows(Path(config['datasets'][bench]['root']) / 'records.json')
    result = {sample_uid(row, i): row for i, row in enumerate(records)}
    if len(result) != len(records):
        raise ValueError('Duplicate record identity')
    return result


def infer(config, bench, directory, client=None, *, message_builder=None,
          generation_options=None, strip_content=False):
    client = client or CompatibleAPI(config['main'], config['timeout'], config['retries'])
    data = config['datasets'][bench]
    expected = indexed(config, bench)
    journal = Path(directory) / 'predictions.jsonl'
    repair_torn_tail(journal)
    prior = {item['uid']: item for item in read_journal(journal)}
    if set(prior) - set(expected):
        raise ValueError('Inference journal has foreign sample IDs')
    pending = []
    for uid, row in expected.items():
        if uid in prior and prior[uid]['input_sha256'] != digest(row):
            raise ValueError('Input changed; choose a new run ID')
        if prior.get(uid, {}).get('status') != 'ok':
            pending.append(uid)

    def work(uid):
        record = {'uid': uid, 'input_sha256': digest(expected[uid])}
        try:
            builder = message_builder or messages
            request = builder(expected[uid], bench, image_root_for(bench, data))
            extra = ({'top_p': 1.0, 'reasoning_effort': 'low', 'stream': False}
                     if bench == 'mindcube' else {})
            extra.update(generation_options or {})
            response = client.complete(request, thinking=False,
                                       max_tokens=PROTOCOLS[bench]['max_tokens'], **extra)
            choice = response['choices'][0]
            message = choice['message']
            content = message.get('content') or ''
            reasoning = message.get('reasoning') if bench == 'mindcube' else None
            if reasoning is not None and not isinstance(reasoning, str):
                raise ValueError('Invalid reasoning channel type')
            selected = content or reasoning or ''
            if strip_content:
                selected = selected.strip()
            record.update(status='ok', raw_response=content, model_answer=selected,
                          response_channel='reasoning' if not content and reasoning else 'content',
                          reasoning_response=reasoning, finish_reason=choice.get('finish_reason'),
                          reasoning_content=message.get('reasoning_content'),
                          usage=response.get('usage', {}), request_sha256=digest(request))
        except Exception as exc:
            record.update(status='error', error_type=type(exc).__name__)
        return record

    with ThreadPoolExecutor(max_workers=config['workers']) as pool:
        for future in as_completed([pool.submit(work, uid) for uid in pending]):
            record = future.result()
            append_jsonl(journal, record)
            prior[record['uid']] = record
    failed = sum(prior.get(uid, {}).get('status') != 'ok' for uid in expected)
    summary = {'expected': len(expected), 'completed': len(expected)-failed, 'failed': failed,
               'empty_model_outputs': sum(prior.get(uid, {}).get('status') == 'ok' and
                                          not prior[uid]['model_answer'] for uid in expected)}
    atomic_json(Path(directory) / 'inference_summary.json', summary)
    if failed:
        raise RuntimeError(f'{bench}: {failed} unresolved inference errors; resume retains valid outputs')
    return summary


def view_review_messages(row, prediction):
    prompt = f"""You are a strict evaluator for a multiple-choice visual-spatial question.
You do NOT have the image. Evaluate only whether the candidate's FINAL answer is semantically
consistent with the reference answer. Ignore intermediate reasoning. The candidate may answer
with a letter or the option text. If there is no clear final answer, it is INCORRECT.

Question:
{row.get('question','')}

Choices:
{row.get('choices','')}

Reference answer:
{row.get('answer','')}

Candidate response:
{prediction}

Return exactly one word: CORRECT or INCORRECT."""
    return [{'role': 'system', 'content': 'Return only CORRECT or INCORRECT.'},
            {'role': 'user', 'content': prompt}]


def score(config, bench, directory, client=None):
    """Keep initial/reviewed metrics separate and require all review verdicts."""
    directory = Path(directory)
    (directory / 'scores').mkdir(parents=True, exist_ok=True)
    expected = indexed(config, bench)
    predictions = {item['uid']: item for item in read_journal(directory / 'predictions.jsonl')}
    if set(predictions) != set(expected) or any(p.get('status') != 'ok' for p in predictions.values()):
        raise ValueError('Incomplete inference; cannot score a subset')
    scored = []
    for uid, row in expected.items():
        if predictions[uid]['input_sha256'] != digest(row):
            raise ValueError('Prediction input hash differs from annotations')
        text = predictions[uid]['model_answer']
        if bench == 'mmsi':
            extracted, parser = load('mmsi_final').extract_choice(text)
            included = True
        elif bench == 'viewspatial':
            extracted, parser = load('viewspatial').extract_answer(text), 'project-tag-final-line'
            included = True
        else:
            extracted, parser = load('mindcube').extract_answer(text), 'project-pattern-priority'
            included = load('mindcube').get_setting_from_id(row['id']) != 'translation'
        scored.append({'uid': uid, 'extracted': extracted, 'parser': parser,
                       'gold': gold(row, bench), 'initial_correct': extracted == gold(row, bench),
                       'included_in_primary': included, 'reviewed_correct': extracted == gold(row, bench)})
    if bench == 'viewspatial':
        client = client or CompatibleAPI(config['judge'], config['timeout'], config['retries'])
        journal = directory / 'scores/reviews.jsonl'
        repair_torn_tail(journal)
        prior = {item['uid']: item for item in read_journal(journal)}
        requests = {item['uid']: view_review_messages(expected[item['uid']], predictions[item['uid']]['model_answer'])
                    for item in scored if not item['initial_correct']}
        if set(prior) - set(requests):
            raise ValueError('Review journal contains foreign or initially-correct records')
        for uid, request in requests.items():
            if uid in prior and prior[uid]['request_sha256'] != digest(request):
                raise ValueError('Review inputs changed; choose a new run ID')

        def review(uid):
            result = {'uid': uid, 'request_sha256': digest(requests[uid])}
            try:
                response = client.complete(requests[uid], thinking=False, max_tokens=16)
                raw = response['choices'][0]['message'].get('content') or ''
                result['raw_response'] = raw
                label = re.search(r'\b(INCORRECT|CORRECT)\b', raw, re.I)
                if not label:
                    raise ValueError('Unparseable judge verdict')
                result.update(status='ok', verdict=label.group(1).upper(), raw_response=raw)
            except Exception as exc:
                result.update(status='error', error_type=type(exc).__name__)
            return result

        pending = [uid for uid in requests if prior.get(uid, {}).get('status') != 'ok']
        with ThreadPoolExecutor(max_workers=config['workers']) as pool:
            for future in as_completed([pool.submit(review, uid) for uid in pending]):
                item = future.result()
                append_jsonl(journal, item)
                prior[item['uid']] = item
        if any(prior.get(uid, {}).get('status') != 'ok' for uid in requests):
            raise RuntimeError('Incomplete ViewSpatial review; initial scores preserved, final score withheld')
        for item in scored:
            if not item['initial_correct']:
                item['reviewed_correct'] = prior[item['uid']]['verdict'] == 'CORRECT'
    included = [item for item in scored if item['included_in_primary']]
    if not included:
        raise ValueError('No records included in primary metric')
    report = {'benchmark': bench, 'total': len(scored), 'primary_total': len(included),
              'initial_correct': sum(item['initial_correct'] for item in included),
              'initial_accuracy': sum(item['initial_correct'] for item in included) / len(included),
              'reviewed_correct': sum(item['reviewed_correct'] for item in included),
              'primary_accuracy': sum(item['reviewed_correct'] for item in included) / len(included),
              'unfiltered_accuracy': sum(item['reviewed_correct'] for item in scored) / len(scored),
              'missing_extractions': sum(item['extracted'] is None for item in scored),
              'reasoning_fallbacks': sum(p.get('response_channel') == 'reasoning' for p in predictions.values()),
              'length_finishes': sum(p.get('finish_reason') == 'length' for p in predictions.values())}
    groups = defaultdict(list)
    for item in scored:
        row = expected[item['uid']]
        kind = (load('mindcube').get_setting_from_id(row['id']) if bench == 'mindcube'
                else str(row.get('question_type', 'unknown')))
        groups[kind].append(item)
    report['by_type'] = {kind: {'total': len(items),
        'initial_correct': sum(item['initial_correct'] for item in items),
        'reviewed_correct': sum(item['reviewed_correct'] for item in items),
        'included_in_primary': items[0]['included_in_primary']}
        for kind, items in sorted(groups.items())}
    report['extraction_sources'] = dict(Counter(item['parser'] for item in scored))
    atomic_json(directory / 'scores/scored_records.json', scored)
    atomic_json(directory / 'scores/metrics.json', report)
    return report
