"""Two-stage orchestration: adapters preserve prompts, scorers preserve metrics."""
from __future__ import annotations

from pathlib import Path
import importlib.metadata
import json
import platform

from .api import CompatibleAPI
from .config import public_config, run_directory, for_benchmark
from .data import annotation_sets, audit, image_root_for, prepare_ui_vision, resolve_image, rows
from .execution import command_plan, execute_steps
from .inference import infer_records
from .io import append_jsonl, atomic_json, file_hash, lock_contract, read_journal, run_lock
from .protocols import PROTOCOLS, RESOURCES, PLATFORM_BENCHMARKS
from .scoring import aggregate


def execution_manifest(config, bench, directory):
    """Validate annotations on each phase, not every byte of every image.

    Full asset hashing belongs to explicit ``doctor --audit-data``. Adapters
    still read/validate the actual images when making inference requests.
    Preserve a previously locked full manifest when its annotations match.
    """
    manifest = audit(bench, config['datasets'][bench], verify_images=False)
    target = directory / 'contract.json'
    if target.exists():
        previous = json.loads(target.read_text())['contract']['dataset']
        def annotations(value):
            return {k: v for k, v in value.items()
                    if k not in ('images_checked', 'image_sha256', 'manifest_sha256')}
        if annotations(previous) != annotations(manifest):
            raise ValueError('Dataset annotations changed; use a new run_id')
        return previous
    return manifest


def execution_contract(directory, contract):
    """An explicitly authorized execution-only update must not discard answers.

    Protocol, runtime and dataset identities remain immutable. Reviewed source
    updates require exact file-pair receipts; no implicit contract migration.
    """
    target = directory / 'contract.json'
    receipt = directory / 'orchestration_update.json'
    if target.exists() and receipt.exists():
        from .io import read_json, digest
        previous = read_json(target)
        permission = read_json(receipt)
        old = previous['contract']
        old_hash = old['framework'].get('runner.py')
        new_hash = contract['framework'].get('runner.py')
        expected = {**old, 'framework': {**old['framework'], 'runner.py': new_hash}}
        # Explicit MME dispatch-capacity correction, never an arbitrary config
        # migration. Pin both effective configs; retain the original contract.
        concurrency = permission.get('request_concurrency_update')
        if concurrency is not None:
            if (not isinstance(concurrency, dict)
                    or set(concurrency) != {'previous_workers', 'workers',
                                           'previous_config_sha256', 'config_sha256'}
                    or set(old.get('config', {}).get('datasets', {})) not in
                       ({'mme_en'}, {'mme_cn'})
                    or type(concurrency['previous_workers']) is not int
                    or type(concurrency['workers']) is not int
                    or concurrency['previous_workers'] != old['config'].get('workers')
                    or not 1 <= concurrency['workers'] <= 256
                    or concurrency['previous_config_sha256'] != digest(old['config'])):
                raise ValueError('Unauthorized request concurrency update')
            updated_config = {**old['config'], 'workers': concurrency['workers']}
            if (concurrency['config_sha256'] != digest(updated_config)
                    or contract.get('config') != updated_config):
                raise ValueError('Unauthorized configuration change outside request concurrency')
            expected['config'] = updated_config
        if 'cli_sha256' in permission:
            if (permission.get('previous_cli_sha256') != old['framework'].get('cli.py')
                    or permission['cli_sha256'] != contract['framework'].get('cli.py')):
                raise ValueError('Unauthorized CLI orchestration update')
            expected['framework']['cli.py'] = permission['cli_sha256']
        # These reviewed files only add GUI worker startup/error/exit handling.
        # Pin exact before/after hashes; no generic scorer or protocol override.
        gui_execution_files = {
            'vendor/gui/scripts/eval_grounding_topology.py',
            'vendor/gui/scripts/replica_lifecycle.py',
            'vendor/gui/official/screenspot-pro/models/qwen3_5.py',
        }
        updates = permission.get('gui_execution_updates', {})
        if updates:
            expected['scorer_sources'] = dict(old.get('scorer_sources', {}))
            for name, update in updates.items():
                if (name not in gui_execution_files
                        or update.get('previous_sha256') != old.get('scorer_sources', {}).get(name)
                        or not update.get('sha256')
                        or update['sha256'] != contract.get('scorer_sources', {}).get(name)):
                    raise ValueError('Unauthorized GUI execution update')
                expected['scorer_sources'][name] = update['sha256']
        # Local service transport only. New helper files are pinned explicitly;
        # no wildcard framework/config migration or model-setting override.
        service_updates = permission.get('service_execution_updates', {})
        for name, update in service_updates.items():
            if (name not in {'serving.py', 'service_lifecycle.py'}
                    or update.get('previous_sha256') != old['framework'].get(name)
                    or not update.get('sha256')
                    or update['sha256'] != contract['framework'].get(name)):
                raise ValueError('Unauthorized service execution update')
            expected['framework'][name] = update['sha256']
        # Reviewed platform execution support only. Image preparation may move
        # to request time, but receipts cannot alter the declared dataset mode,
        # pixel policy, prompts, generation parameters, or scoring identity.
        platform_updates = permission.get('platform_execution_updates', {})
        if not isinstance(platform_updates, dict):
            raise ValueError('Unauthorized platform execution update')
        for name, update in platform_updates.items():
            if (name not in {'general_visual.py', 'data.py', 'workflow.py', 'preflight.py'}
                    or not isinstance(update, dict)
                    or set(update) != {'previous_sha256', 'sha256'}
                    or not isinstance(update['previous_sha256'], str)
                    or not update['previous_sha256']
                    or update['previous_sha256'] != old['framework'].get(name)
                    or not isinstance(update['sha256'], str)
                    or not update['sha256']
                    or update['sha256'] != contract['framework'].get(name)):
                raise ValueError('Unauthorized platform execution update')
            expected['framework'][name] = update['sha256']
        if (previous['sha256'] != digest(old)
                or permission.get('previous_runner_sha256') != old_hash
                or permission.get('runner_sha256') != new_hash
                or expected != contract):
            raise ValueError('Unauthorized change outside the declared orchestration update')
        atomic_json(directory / 'execution_provenance.json', {
            'contract_sha256': previous['sha256'], 'runner_sha256': new_hash,
            'previous_runner_sha256': old_hash,
            'cli_sha256': contract['framework'].get('cli.py'),
            'gui_execution_updates': updates,
            'service_execution_updates': service_updates,
            'platform_execution_updates': platform_updates,
            'request_concurrency_update': concurrency,
            'change': ('explicit platform execution update; image preparation timing follows the declared dataset mode; '
                       'Qwen pixel policy, prompts, generation and scoring unchanged' if platform_updates else
                       'explicit execution-only update; prompts, image handling, generation and scoring unchanged')})
        return previous['sha256']
    return lock_contract(directory, contract)


def runtime_fingerprint():
    """Prevent cross-version resume, including runtime ViewSpatial JPEG changes.

    Read distribution metadata only; never import torch/TF to inspect versions.
    """
    result = {'python': platform.python_version()}
    for name in ('Pillow', 'pyarrow', 'openai', 'requests', 'numpy', 'torch', 'vllm', 'tensorflow', 'mathruler', 'qwen-vl-utils', 'transformers'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def runtime_inputs(config, bench, directory):
    """Only path projection / environment references; never change question text."""
    data = config['datasets'][bench]
    if bench == 'ui_vision':
        prepare_ui_vision(data['root'], directory / 'prepared')


def validate_inference(config, bench, directory, manifest):
    """Do not trust an adapter's exit code alone (several upstreams swallow errors)."""
    if bench in PLATFORM_BENCHMARKS or bench in ('milebench', 'memlens', 'mmsi', 'viewspatial', 'mindcube', 'chartography', 'spar', 'erqa', 'mimic'):
        records = {r['uid']: r for r in read_journal(directory / 'predictions.jsonl')}
        expected = sum(r['rows'] for r in manifest['sets'].values())
        if len(records) != expected or any(r['status'] != 'ok' for r in records.values()):
            raise ValueError('Incomplete API inference journal')
    elif bench == 'mmlongbench':
        from .protocols import mml_specs
        for item in mml_specs():
            name = f"{item['family']}_K{item['k']}"
            scores = list((directory / 'tasks' / name).glob('*.json.score'))
            if len(scores) != 1:
                raise ValueError(f'{name}: missing/ambiguous score after inference')
            source = Path(str(scores[0]).removesuffix('.score'))
            cache = read_journal(str(source) + '.cache')
            info = manifest['sets'][f"{item['family']}:K{item['k']}"]
            from .io import digest
            if len(cache) != info['selected_rows'] or digest([r['id'] for r in cache]) != info['selected_ids_sha256']:
                raise ValueError(f'{name}: cached rows differ from deterministic official sample selection')
    elif bench == 'agentvista':
        source = rows(Path(config['datasets'][bench]['root']) / 'val.json')
        expected = {str(r.get('doc_id', r.get('question_id', f'sample_{i}'))) for i, r in enumerate(source)}
        found = list((directory / 'tasks').glob('*/result.json'))
        if {p.parent.name for p in found} != expected:
            raise ValueError('AgentVista result identities do not cover the 209 questions')
    else:
        aggregate(config, bench, directory)  # Grounding scoring is part of native inference.
    atomic_json(directory / 'inference_complete.json', {'status': 'complete', 'benchmark': bench})


def run(config, bench, phase):
    config = for_benchmark(config, bench)
    directory = run_directory(config, bench)
    manifest = execution_manifest(config, bench, directory)
    sources = {str(p.relative_to(RESOURCES)): file_hash(p) for p in RESOURCES.rglob('*') if p.is_file() and p.suffix in ('.py', '.md', '.json', '.yaml')}
    framework = {p.name: file_hash(p) for p in Path(__file__).parent.glob('*.py')}
    contract = {'config': public_config(config), 'protocol': PROTOCOLS[bench],
                'dataset': manifest, 'scorer_sources': sources, 'framework': framework,
                'runtime': runtime_fingerprint()}
    if bench in PLATFORM_BENCHMARKS:
        from .general_visual import tokenizer_manifest
        contract['judge_tokenizer_files'] = tokenizer_manifest(config)
    if bench == 'mimic':
        from .mimic import math_grader
        math_grader(config)  # Fail before expensive inference if a required dependency is unavailable.
    with run_lock(directory):
        execution_contract(directory, contract)
        atomic_json(directory / 'dataset_manifest.json', manifest)
        runtime_inputs(config, bench, directory)
        for current in (('infer', 'score') if phase == 'all' else (phase,)):
            if current == 'infer':
                if bench not in ('screenspot_pro', 'ui_vision'):
                    CompatibleAPI(config['main'], config['timeout'], config['retries']).health()
                if bench in ('milebench', 'memlens', 'mimic'):
                    infer_records(config, bench, directory)
                elif bench in ('mmsi', 'viewspatial', 'mindcube'):
                    from .spatial import infer
                    infer(config, bench, directory)
                elif bench in ('chartography', 'spar'):
                    from .structured import infer
                    infer(config, bench, directory)
                elif bench == 'erqa':
                    from .erqa import infer
                    infer(config, directory)
                elif bench in PLATFORM_BENCHMARKS:
                    from .general_visual import infer
                    infer(config, bench, directory)
                else:
                    execute_steps(command_plan(config, bench, 'infer', directory), directory / 'logs/infer')
                validate_inference(config, bench, directory, manifest)
            else:
                validate_inference(config, bench, directory, manifest)
                if bench in PLATFORM_BENCHMARKS or bench in ('milebench', 'memlens', 'agentvista', 'viewspatial', 'chartography', 'erqa', 'mimic') or current == 'review':
                    CompatibleAPI(config['judge'], config['timeout'], config['retries']).health()
                (directory / 'scores').mkdir(exist_ok=True)
                if bench in ('mmsi', 'viewspatial', 'mindcube'):
                    from .spatial import score
                    score(config, bench, directory)
                elif bench in ('chartography', 'spar'):
                    from .structured import score, review_spar
                    score(config, bench, directory)
                    if current == 'review':
                        if bench != 'spar':
                            raise ValueError('Only SPAR exposes this diagnostic review phase')
                        review_spar(config, directory)
                elif bench == 'erqa':
                    from .erqa import score
                    score(config, directory)
                elif bench == 'mimic':
                    from .mimic import score
                    score(config, directory)
                elif bench in PLATFORM_BENCHMARKS:
                    from .general_visual import score
                    score(config, bench, directory)
                else:
                    execute_steps(command_plan(config, bench, 'score', directory), directory / 'logs/score')
                aggregate(config, bench, directory)
    return directory
