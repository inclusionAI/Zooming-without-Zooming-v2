"""Fail-closed aggregation; public scores are percentages (0..100)."""
from __future__ import annotations

import math
from pathlib import Path

from .data import load_module
from .io import atomic_json, file_hash, read_json, read_journal
from .protocols import GUI_EXPECTED, MEM_SPLITS, MILE_TASKS, RESOURCES, mml_specs, PLATFORM_BENCHMARKS


def number(value, maximum=100):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= maximum:
        raise ValueError(f'Invalid score value: {value!r}')
    return float(value)


def aggregate(config, bench, directory):
    directory = Path(directory)
    tasks, diagnostics = {}, {}
    if bench in PLATFORM_BENCHMARKS:
        from .general_visual import COUNTS
        metrics = read_json(directory / 'scores/metrics.json')
        if metrics['total'] != COUNTS[bench]:
            raise ValueError('Incomplete project platform denominator')
        # Preserve evaluate_v3's rounded fraction before converting to percent.
        tasks['all'] = number(metrics['platform_reported_final_score'], 1)*100
        diagnostics.update(metrics)
    elif bench == 'mimic':
        from .mimic import COUNTS
        metrics = read_json(directory / 'scores/metrics.json')
        counts = {f'mimic-{name}': n for name, n in COUNTS.items()}
        if metrics['per_task_total'] != counts or metrics['total'] != 8000 or set(metrics['per_task']) != set(counts):
            raise ValueError('MIMIC requires all four fixed-denominator scores')
        tasks.update({task: number(value, 1)*100 for task, value in metrics['per_task'].items()})
        diagnostics.update(metrics)
    elif bench == 'erqa':
        metrics = read_json(directory / 'scores/metrics.json')
        if metrics['total'] != 400:
            raise ValueError('ERQA requires all 400 samples and reviews')
        tasks['all'] = number(metrics['primary_accuracy'], 1)*100
        diagnostics.update(metrics)
    elif bench in ('chartography', 'spar'):
        from .structured import EXPECTED, task_names
        metrics = read_json(directory / 'scores/metrics.json')
        if metrics['total'] != EXPECTED[bench]:
            raise ValueError('Incomplete Chartography/SPAR score denominator')
        if bench == 'spar':
            if set(metrics['per_task']) != task_names():
                raise ValueError('SPAR requires exactly 20 task scores')
            tasks.update({task: number(value, 1)*100 for task, value in metrics['per_task'].items()})
            audit_path = directory / 'scores/spar_audit_summary.json'
            if audit_path.exists():
                diagnostics['llm_audit_not_primary'] = read_json(audit_path)
        else:
            tasks['all'] = number(metrics['primary_accuracy'], 1)*100
        diagnostics.update(metrics)
    elif bench in ('mmsi', 'viewspatial', 'mindcube'):
        from .spatial import EXPECTED
        metrics = read_json(directory / 'scores/metrics.json')
        if metrics['total'] != EXPECTED[bench] or not 0 < metrics['primary_total'] <= metrics['total']:
            raise ValueError('Incomplete spatial benchmark score denominator')
        tasks['all'] = number(metrics['primary_accuracy'], 1) * 100
        diagnostics.update(metrics)
    elif bench == 'memlens':
        for split in MEM_SPLITS:
            task = f'memlens-{split}'
            overall = read_json(directory / 'scores' / task / 'judge_metrics.json')['overall']
            if overall['total'] != 789:
                raise ValueError(f'{task}: incomplete judge denominator')
            tasks[task] = number(overall['accuracy'], 1) * 100
    elif bench == 'milebench':
        data = read_json(directory / 'scores' / f"{config['model_id']}_llm.json")
        if set(data) != {f'milebench-{t}-test' for t in MILE_TASKS}:
            raise ValueError('MileBench requires exactly 29 test task scores')
        manifest = read_json(directory / 'dataset_manifest.json')['sets']
        for task, score in data.items():
            if score['n'] != manifest[task]['rows']:
                raise ValueError(f'{task}: scorer denominator differs from annotations')
            tasks[task] = number(score['mean'])
        diagnostics['invalid_extractor_outputs'] = sum(
            r.get('extracted_letter') == '?' for p in (directory / 'legacy').glob('llm_extract.shard*.jsonl')
            for r in read_journal(p))
    elif bench == 'mmlongbench':
        for item in mml_specs():
            name = f"{item['family']}_K{item['k']}"
            candidates = sorted((directory / 'tasks' / name).glob('*.json.score'))
            if len(candidates) != 1:
                raise ValueError(f'{name}: expected one completed score file')
            source = Path(str(candidates[0]).removesuffix('.score'))
            result = read_json(source)
            cache = read_journal(str(source) + '.cache')
            if not cache or len(cache) < len(result['data']):
                raise ValueError(f'{name}: missing full inference cache')
            if len(cache) != len(result['data']):
                raise ValueError(f'{name}: upstream excluded empty outputs; inspect cache before reporting a final score')
            tasks[name] = number(read_json(candidates[0])[item['metric']])
    elif bench == 'agentvista':
        result = read_json(directory / 'tasks/judge_summary.json')
        if result['expected_samples'] != 209 or result['missing_results'] or result['samples'] != 209:
            raise ValueError('AgentVista judge results do not cover all 209 questions')
        if result['judged'] + result['infer_failed'] != 209:
            raise ValueError('AgentVista incomplete judgment accounting')
        diagnostics.update(result)
        tasks['all'] = number(result['correct'] / 209, 1) * 100
    elif bench in GUI_EXPECTED:
        for split, expected in GUI_EXPECTED[bench].items():
            report = read_json(directory / 'tasks' / f'{split}.json')
            overall = report['metrics']['overall']
            if overall['num_total'] != expected or len(report['details']) != expected:
                raise ValueError(f'{bench}/{split}: incomplete grounding output')
            tasks[split] = number(overall['action_acc'], 1) * 100
    else:
        raise ValueError(f'No scorer for {bench}')
    report = {'benchmark': bench, 'model_id': config['model_id'], 'run_id': config['run_id'],
              'status': 'complete', 'scale': 'percentage', 'aggregation': 'equal-weight task macro',
              'macro': sum(tasks.values()) / len(tasks), 'tasks': tasks, 'diagnostics': diagnostics}
    contract = directory / 'contract.json'
    report['contract_sha256'] = read_json(contract)['sha256'] if contract.exists() else None
    report['source_outputs_sha256'] = {
        str(p.relative_to(directory)): file_hash(p)
        for folder in ('tasks', 'scores', 'legacy') for p in (directory / folder).rglob('*')
        if p.is_file() and p.suffix in ('.json', '.jsonl', '.score', '.cache')}
    # Include judge journals as well as predictions. Otherwise changing a root-level
    # verdict journal after scoring would not be detected by summarize.
    for journal in directory.glob('*.jsonl'):
        if journal.is_file():
            report['source_outputs_sha256'][journal.name] = file_hash(journal)
    atomic_json(directory / 'score.json', report)
    return report
