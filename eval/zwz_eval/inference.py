from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .api import CompatibleAPI
from .data import annotation_sets, image_root_for, load_module, rows, sample_uid
from .io import append_jsonl, atomic_json, digest, read_journal, repair_torn_tail
from .protocols import PROTOCOLS, RESOURCES


def infer_records(config, bench, directory, client=None, formatter=None):
    """Inference for the shared MileBench/MemLens project schema.

    Completed model outputs, including EOS-only generations, are not resampled
    merely because they are empty/wrong. Transport failures remain resumable.
    """
    protocol = PROTOCOLS[bench]
    data = config['datasets'][bench]
    image_root = str(image_root_for(bench, data))
    client = client or CompatibleAPI(config['main'], config['timeout'], config['retries'])
    if formatter is None:
        formatter = load_module(RESOURCES / 'vendor/common/infer.py', '_preserved_infer')
    journal = Path(directory) / 'predictions.jsonl'
    repair_torn_tail(journal)
    old = {r['uid']: r for r in read_journal(journal)}
    expected, pending = {}, []
    for task, path in annotation_sets(bench, data['root']).items():
        for i, row in enumerate(rows(path)):
            uid = f'{task}/{sample_uid(row, i)}'
            record_hash = digest(row)
            expected[uid] = (task, row, record_hash)
            prior = old.get(uid)
            if prior and prior.get('input_sha256') != record_hash:
                raise ValueError('Input record changed; choose a new run_id')
            if not prior or prior.get('status') != 'ok':
                pending.append(uid)
    if set(old) - set(expected):
        raise ValueError('Journal contains records outside the current dataset')

    def work(uid):
        task, row, source_hash = expected[uid]
        try:
            messages = formatter.build_record_messages(row, task, [image_root])
            if messages is None:
                query = row.get('query', '')
                system = row.get('system_prompt') or row.get('meta', {}).get('system_prompt')
                if bench == 'milebench' and system:
                    query = str(system).rstrip() + '\n\n' + str(query).lstrip()
                    system = None
                messages = []
                if system:
                    messages.append({'role': 'system', 'content': system})
                content = formatter.build_user_content(query, row.get('images', []), task, [image_root])
                messages.append({'role': 'user', 'content': content})
            response = client.complete(messages, thinking=protocol['thinking'], max_tokens=protocol['max_tokens'])
            choice = response['choices'][0]
            raw = choice['message'].get('content') or ''
            return {'uid': uid, 'input_sha256': source_hash, 'task': task, 'status': 'ok',
                    'raw_response': raw, 'model_answer': formatter.normalize_model_answer(raw),
                    'finish_reason': choice.get('finish_reason'), 'usage': response.get('usage', {}),
                    'request_sha256': digest(messages), 'request_roles': [m['role'] for m in messages]}
        except Exception as exc:
            return {'uid': uid, 'input_sha256': source_hash, 'task': task, 'status': 'error', 'error_type': type(exc).__name__}
    with ThreadPoolExecutor(max_workers=config['workers']) as pool:
        futures = [pool.submit(work, uid) for uid in pending]
        for i, future in enumerate(as_completed(futures), 1):
            record = future.result()
            append_jsonl(journal, record)
            old[record['uid']] = record
            print(f'[{bench}] {i}/{len(pending)} pending requests processed', flush=True)
    failures = [uid for uid in expected if old.get(uid, {}).get('status') != 'ok']
    summary = {'expected': len(expected), 'completed': len(expected)-len(failures), 'failed': len(failures),
               'empty_model_outputs': sum(old.get(uid, {}).get('status') == 'ok' and not old[uid]['raw_response'] for uid in expected)}
    atomic_json(Path(directory) / 'inference_summary.json', summary)
    if failures:
        raise RuntimeError(f'{bench}: {len(failures)} requests failed; valid records preserved for resume')
    # Export the exact legacy scorer input schema, without losing raw generations.
    for task in annotation_sets(bench, data['root']):
        target = Path(directory) / 'legacy' / config['model_id'] / 'model_answer' / task
        target.mkdir(parents=True, exist_ok=True)
        exported = []
        for uid, (which, source, _) in expected.items():
            if which == task:
                record = {**source, 'model_answer': old[uid]['model_answer'], 'model_answer_raw': old[uid]['raw_response']}
                if bench == 'milebench':
                    # Cache isolation only; retain original IDs for traceability.
                    record['original_sample_id'] = source.get('sample_id')
                    record['sample_id'] = uid
                exported.append(record)
        # The whole projection is deterministic and recoverable from the journal.
        atomic_json(target / 'records.json', exported)
        dest = target / f"{config['model_id']}_seed42_answer.jsonl"
        tmp = dest.with_suffix('.jsonl.tmp')
        with tmp.open('w', encoding='utf-8') as stream:
            from .io import canonical
            for record in exported:
                stream.write(canonical(record) + '\n')
        tmp.replace(dest)
    return summary
