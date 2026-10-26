"""Isolated legacy adapters: no shell interpolation, no credential-bearing argv."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .data import image_root_for
from .io import atomic_json, digest, file_hash, read_json
from .protocols import MEM_SPLITS, MILE_TASKS, RESOURCES, mml_specs, PLATFORM_BENCHMARKS

VENDOR = RESOURCES / 'vendor'


def step(script, arguments, *, cwd=None, env=None, unset=()):
    return {'argv': [sys.executable, str(script), *map(str, arguments)],
            'cwd': str(cwd or Path(script).parent), 'env': env or {}, 'unset': list(unset)}


def endpoint_env(config, role):
    key = 'MAIN_API_KEY' if role == 'main' else 'JUDGE_API_KEY'
    # Reference, not a secret. Resolved only inside execute_steps.
    return {key: {'from_env': config[role]['api_key_env'], 'default': 'EMPTY'}}


def command_plan(config, bench, phase, directory):
    if bench in PLATFORM_BENCHMARKS or bench in ('mmsi', 'viewspatial', 'mindcube', 'chartography', 'spar', 'erqa', 'mimic'):
        return []  # In-process journal adapters; never a generic subprocess prompt.
    directory = Path(directory)
    root = Path(config['datasets'][bench]['root'])
    main = config.get('main', {})
    judge = config.get('judge', {})
    worker = config['workers']
    jobs = []
    if bench == 'memlens' and phase == 'score':
        for split in MEM_SPLITS:
            task = f'memlens-{split}'
            source = directory / 'legacy' / config['model_id'] / 'model_answer' / task / f"{config['model_id']}_seed42_answer.jsonl"
            jobs.append(step(VENDOR / 'memlens/judge.py', [
                '--infer_file', source, '--output_dir', directory / 'scores' / task,
                '--judge_base_url', ','.join(judge['base_urls']), '--judge_model', judge['model'],
                '--num_workers', worker, '--max_tokens_judge', 2048], env=endpoint_env(config, 'judge')))
    elif bench == 'milebench' and phase == 'score':
        jobs.append(step(VENDOR / 'milebench/score.py', [], env={
            **endpoint_env(config, 'judge'), 'EVAL_ROOT_MILEBENCH': str(directory / 'legacy'),
            'CACHE_DIR_MILEBENCH': str(directory / 'scores'), 'MILEBENCH_MODELS': config['model_id'],
            'JUDGE_URL': ','.join(judge['base_urls']), 'JUDGE_MODEL': judge['model'],
            'ZWZ_STRICT_ERRORS': '1',
            'SHARD_ID': '0', 'NUM_SHARDS': '1', 'AGGREGATE': '0'}))
    elif bench == 'mmlongbench' and phase == 'infer':
        for item in mml_specs():
            task = f"{item['family']}_K{item['k']}"
            jobs.append(step(VENDOR / 'mmlongbench/eval_api_batch.py', [
                '--model_name_or_path', main['model'], '--use_vllm',
                '--datasets', item['family'], '--test_file_root', root / 'mmlb_data',
                '--image_file_root', image_root_for(bench, config['datasets'][bench]), '--test_files', item['file'],
                '--output_dir', directory / 'tasks' / task, '--max_test_samples', item['max_test_samples'],
                '--input_max_length', item['k'] * 1024, '--test_length', item['k'],
                '--generation_max_length', 2048, '--temperature', 0, '--do_sample', 'False',
                '--docqa_llm_judge', 'False', '--seed', 42, '--batch_size', 16,
                '--num_workers', 8, '--preprocessing_num_workers', 4],
                env={**endpoint_env(config, 'main'), 'VLLM_BASE_URLS': ','.join(main['base_urls']),
                     'MMLONGBENCH_ROBUST_OUTPUT': '0', 'MMLONGBENCH_VRAG_REQUIRE_NONEMPTY_ANSWER': '0',
                     'ZWZ_STRICT_ERRORS': '1'}))
    elif bench == 'agentvista':
        if phase == 'infer':
            jobs.append(step(VENDOR / 'agentvista/infer.py', [
                '--input-file', root / 'val.json', '--image-folder', image_root_for(bench, config['datasets'][bench]),
                '--output-dir', directory / 'tasks', '--skip-completed', '--temperature', 0,
                '--top-p', 1, '--max-completion-tokens', 1024, '--max-turns', 30,
                '--max-images', 100, '--max-total-tokens', 65536,
                '--system-prompt-key', 'multi_tool_agent_search_only',
                '--web-search-max-calls', 100, '--image-search-max-calls', 100,
                '--tool-config-path', VENDOR / 'agentvista/configs/tool_configs.yaml'], env={
                    'REASONING_MODEL_NAME': main['model'], 'REASONING_END_POINT': main['base_urls'][0],
                    'REASONING_API_KEY': {'from_env': main['api_key_env'], 'default': 'EMPTY'},
                    'FREE_TOOLS': '1', 'AGENTVISTA_FREE_TOOLS': '1', 'ENABLE_FUNCTION_CALLING': 'true',
                    'ENABLED_TOOLS': 'web_search,image_search,visit', 'AGENTVISTA_TOTAL_TOOL_BUDGET': '12',
                    'AGENTVISTA_DISABLE_THINKING': '0', 'AGENTVISTA_ACCEPT_PLAIN_FINAL_TEXT': '1',
                    'AGENTVISTA_API_MAX_RETRIES': '5',
                    'AGENTVISTA_FREE_VISIT_CHARS': '150000',
                    'AGENTVISTA_DISABLE_THINKING_WHEN_NO_TOOLS': '1',
                    'AGENTVISTA_RATE_LIMIT_DIR': str(directory / 'rate_limits')},
                    unset=('VERIFIER_API_KEY', 'VERIFIER_END_POINT', 'VERIFIER_MODEL_NAME', 'REASONING_API_KEYS', 'REASONING_API_KEY_2', 'REASONING_EFFORT', 'REASONING_MAX_TOKENS')))
        else:
            jobs.append(step(VENDOR / 'agentvista/score_final_answers.py', [directory / 'tasks'], env={
                'VERIFIER_MODEL_NAME': judge['model'], 'VERIFIER_END_POINT': judge['base_urls'][0],
                'VERIFIER_API_KEY': {'from_env': judge['api_key_env'], 'default': 'EMPTY'}}))
    elif bench in ('screenspot_pro', 'ui_vision') and phase == 'infer':
        hardware = config.get('hardware', {})
        model_path = config.get('identity', {}).get('model_path')
        if not model_path:
            raise ValueError('Native GUI backend needs identity.model_path; no generic prompt substitution is allowed')
        splits = ['all'] if bench == 'screenspot_pro' else ['basic', 'functional', 'spatial']
        for split in splits:
            annotations = root / 'annotations' if bench == 'screenspot_pro' else directory / 'prepared'
            jobs.append(step(VENDOR / 'gui/scripts/eval_grounding_topology.py', [
                '--model_type', 'qwen3_5_vllm', '--model_name_or_path', model_path,
                '--screenspot_imgs', image_root_for(bench, config['datasets'][bench]), '--screenspot_test', annotations,
                '--task', split, '--inst_style', 'instruction', '--language', 'en', '--gt_type', 'positive',
                '--log_path', directory / 'tasks' / f'{split}.json',
                '--num_gpu', hardware.get('gpus', 16), '--tp', hardware.get('tp', 1)], env={
                    'QWEN3_5_ENABLE_THINKING': 'false', 'QWEN3_5_MAX_NEW_TOKENS': '8192', 'QWEN3_5_MAX_PIXELS': '8294400'}))
    elif phase != 'score':
        raise ValueError(f'No external adapter for {bench}/{phase}')
    if phase == 'score' and bench in ('memlens', 'milebench', 'agentvista'):
        score_cache_bindings(config, bench, directory, jobs)
    return jobs


def score_cache_bindings(config, bench, directory, jobs):
    """Bind retained native caches without changing their prompts or metric code."""
    signature = digest(jobs)  # Includes judge model/endpoint and flags, never key values.
    def attach(job, target, inputs, existing):
        def encode(value):
            return {k: str(v) for k, v in value.items()} if isinstance(value, dict) else str(value)
        job['cache_binding'] = {'path': str(target), 'inputs': [encode(v) for v in inputs],
                                'existing': [encode(v) for v in existing], 'signature': signature}
    if bench == 'memlens':
        for job in jobs:
            argv = job['argv']
            output = Path(argv[argv.index('--output_dir')+1])
            source = argv[argv.index('--infer_file')+1]
            attach(job, output / 'cache_inputs.json', [source],
                   [{'root': output, 'pattern': 'judge_metrics.json*'}, output / 'judge_details.json'])
    elif bench == 'milebench':
        legacy = directory / 'legacy'
        files = [legacy / config['model_id'] / 'model_answer' / f'milebench-{task}-test' /
                 f"{config['model_id']}_seed42_answer.jsonl" for task in MILE_TASKS]
        attach(jobs[0], directory / 'scores/milebench_cache_inputs.json', files,
               [{'root': legacy, 'pattern': 'llm_extract.shard*.jsonl'}, directory / 'scores' / f"{config['model_id']}_llm.json"])
    else:
        attach(jobs[0], directory / 'tasks/cache_inputs.json',
               [{'root': directory / 'tasks', 'pattern': '*/result.json'}],
               [{'root': directory / 'tasks', 'pattern': '*/judge.json'}, directory / 'tasks/judgments.jsonl',
                directory / 'tasks/judge_summary.json'])


def cache_paths(spec):
    # Roots/ordinary filenames are literal paths, including spaces or '[' / ']'.
    # Only explicitly declared relative patterns are interpreted as globs.
    if isinstance(spec, str):
        return [Path(spec)] if Path(spec).exists() else []
    if not isinstance(spec, dict) or set(spec) != {'root', 'pattern'}:
        raise ValueError('Invalid scorer cache file specification')
    pattern = Path(spec['pattern'])
    if pattern.is_absolute() or '..' in pattern.parts:
        raise ValueError('Scorer cache glob must stay inside its declared root')
    return sorted(Path(spec['root']).glob(spec['pattern']))


def bind_score_cache(binding):
    """Fail closed for changed inputs or unbound legacy verdicts; never delete them."""
    files = {}
    for pattern in binding['inputs']:
        matches = cache_paths(pattern)
        if not matches:
            raise FileNotFoundError(f'Missing scorer input: {pattern}')
        for match in matches:
            path = Path(match)
            if not path.is_file():
                raise ValueError('Scorer inputs must be regular files')
            files[str(path.resolve())] = file_hash(path)
    contract = {'version': 1, 'signature': binding['signature'], 'inputs_sha256': files}
    target = Path(binding['path'])
    if target.exists():
        if read_json(target) != contract:
            raise ValueError('Scorer input/config changed; refusing stale verdict cache. Use a new run_id; old results are untouched.')
    else:
        if any(cache_paths(pattern) for pattern in binding['existing']):
            raise ValueError('Unbound legacy verdict cache; cannot certify reuse. Use a new run_id; existing files are untouched.')
        atomic_json(target, contract)


def execute_steps(jobs, directory):
    """Log only command metadata. API credentials never enter serialized plans."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / 'command_plan.json', jobs)
    for index, job in enumerate(jobs):
        if job.get('cache_binding'):
            bind_score_cache(job['cache_binding'])
        from .io import repair_torn_tail
        if '--infer_file' in job['argv'] and '--output_dir' in job['argv']:
            judge_output = Path(job['argv'][job['argv'].index('--output_dir')+1])
            repair_torn_tail(judge_output / 'judge_metrics.json.jsonl')
        if '--generation_max_length' in job['argv'] and '--output_dir' in job['argv']:
            task_output = Path(job['argv'][job['argv'].index('--output_dir')+1])
            for cache in task_output.glob('*.json.cache'):
                repair_torn_tail(cache)
        if '--log_path' in job['argv']:
            from .io import read_json
            output = Path(job['argv'][job['argv'].index('--log_path')+1])
            expected = {'all': 1581, 'basic': 1772, 'functional': 1772, 'spatial': 1935}[output.stem]
            if output.is_file():
                try:
                    report = read_json(output)
                    if report['metrics']['overall']['num_total'] == expected and len(report['details']) == expected:
                        print(f'Skipping completed grounding split: {output.stem}', flush=True)
                        continue
                except (ValueError, KeyError, TypeError):
                    pass
        env = os.environ.copy()
        for key in job['unset']:
            env.pop(key, None)
        for key, value in job['env'].items():
            env[key] = str(os.environ.get(value['from_env'], value.get('default', '')) if isinstance(value, dict) else value)
        # A caller's inherited cpp/upb setting must not reactivate the known
        # old-generated-proto incompatibility in preserved native adapters.
        env['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
        log = directory / f'step-{index:03d}.log'
        print(f'Running step {index+1}/{len(jobs)}; log: {log}', flush=True)
        # Mode 0600: dataset text and model generations can be sensitive too.
        fd = os.open(log, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            result = subprocess.run(job['argv'], cwd=job['cwd'], env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Adapter step {index+1} exited {result.returncode}; inspect {log}')
        if job.get('cache_binding'):
            bind_score_cache(job['cache_binding'])  # Inputs may not change mid-judge.
