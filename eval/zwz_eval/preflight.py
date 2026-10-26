"""Read-only preflight. Never install dependencies, allocate GPUs or generate answers."""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import re
from pathlib import Path

from .api import CompatibleAPI
from .config import load_config, for_benchmark, run_directory
from .io import file_hash, read_json
from .protocols import PLATFORM_BENCHMARKS, PROTOCOLS
from .workflow import selection

JUDGED = set(PLATFORM_BENCHMARKS) | {
    'chartography', 'mimic', 'erqa', 'viewspatial', 'milebench', 'memlens', 'agentvista'}
GUI = {'screenspot_pro', 'ui_vision'}


def requirements(bench, phase, *, data=None):
    names = set()
    if phase in ('infer', 'all'):
        names.add('Pillow')
        if bench in PLATFORM_BENCHMARKS and data is not None:
            preparation = Path(data['root']) / 'preparation.json'
            if preparation.is_file():
                from .general_visual import preparation_mode
                if preparation_mode(read_json(preparation)) == 'on-demand':
                    # Only metadata is read here. Existing materialized bundles
                    # still need no image preprocessor on the inference client.
                    names.update(('qwen-vl-utils', 'torch'))
        if bench in ('milebench', 'memlens', 'mimic'):
            names.update(('openai', 'tqdm'))  # Eager imports in the preserved shared formatter.
        if bench in GUI:
            names.update(('torch', 'vllm', 'transformers'))
        elif bench == 'mmlongbench':
            names.update(('torch', 'transformers', 'numpy', 'datasets', 'rouge-score', 'openai', 'tqdm', 'requests', 'PyYAML'))
        elif bench == 'agentvista':
            names.update(('openai', 'requests', 'PyYAML', 'tqdm', 'duckduckgo-search', 'trafilatura', 'beautifulsoup4'))
    if phase in ('score', 'all'):
        if bench in PLATFORM_BENCHMARKS:
            names.update(('transformers', 'tokenizers'))
        elif bench in ('milebench', 'memlens'):
            names.add('openai')
        elif bench in ('agentvista'):
            names.update(('openai', 'requests', 'PyYAML'))
        elif bench == 'mimic':
            names.update(('Pillow', 'openai', 'tqdm'))  # Counting identity validation loads the formatter.
    return names


def inspect_checkpoint(directory):
    """Inspect metadata only; never load tensors, remote code or a tokenizer."""
    root = Path(directory)
    if not root.is_dir() or not (root / 'config.json').is_file():
        raise ValueError('Local checkpoint requires a readable directory and config.json')
    config = read_json(root / 'config.json')
    if not isinstance(config, dict):
        raise ValueError('Checkpoint config.json must be an object')
    result = {'config_sha256': file_hash(root / 'config.json'), 'chat_template_sha256': None,
              'chat_template_source': None, 'model_type': config.get('model_type')}
    template = root / 'chat_template.jinja'
    if template.is_file():
        result.update(chat_template_sha256=file_hash(template), chat_template_source='chat_template.jinja')
    elif (root / 'tokenizer_config.json').is_file():
        tokenizer = read_json(root / 'tokenizer_config.json')
        if not isinstance(tokenizer, dict):
            raise ValueError('tokenizer_config.json must be an object')
        text = tokenizer.get('chat_template')
        if isinstance(text, str) and text:
            result.update(chat_template_sha256=hashlib.sha256(text.encode()).hexdigest(),
                          chat_template_source='tokenizer_config.json:chat_template')
    return result


def doctor(config_path, selected='all', phase='infer', *, audit_data=False, check_endpoints=False, check_tools=False):
    if phase not in ('infer', 'score', 'all'):
        raise ValueError('Doctor phase must be infer, score or all')
    config = load_config(config_path)
    names = selection(selected, config['datasets'])
    report = {'model_id': config['model_id'], 'run_id': config['run_id'], 'phase': phase,
              'data_audit_requested': audit_data, 'endpoint_checks_requested': check_endpoints,
              'tool_checks_requested': check_tools,
              'inference_started': False, 'gpu_runtime_tested': False, 'results': []}
    checked_services = {}
    for bench in names:
        item = {'benchmark': bench, 'protocol': PROTOCOLS[bench], 'errors': [], 'warnings': [],
                'output': str(run_directory(config, bench)), 'services': {}, 'dependencies': {}}
        report['results'].append(item)
        try:
            effective = for_benchmark(config, bench)
            data = effective['datasets'][bench]
            output = run_directory(effective, bench).resolve()
            for key in ('root', 'image_root'):
                if key not in data:
                    continue
                directory = Path(data[key])
                if not directory.is_dir():
                    item['errors'].append(f'datasets.{bench}.{key} is missing or not a directory: {directory}')
                if output.is_relative_to(directory) or directory.is_relative_to(output):
                    item['errors'].append('Output run overlaps source dataset; choose a separate output_root')
            ancestor = output
            while not ancestor.exists() and ancestor != ancestor.parent:
                ancestor = ancestor.parent
            if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
                item['errors'].append('Output parent is not a writable directory')
            for name in sorted(requirements(bench, phase, data=data)):
                try:
                    item['dependencies'][name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    item['dependencies'][name] = None
                    item['errors'].append(f'Missing runtime distribution: {name}; see installation guide')
            if bench == 'mimic':
                from .mimic import math_grader
                math_grader(effective)  # Require the declared branch/version, never infer it from the environment.
            identity = effective.get('identity', {})
            if phase in ('infer', 'all'):
                local_model = identity.get('model_path')
                if local_model and Path(local_model).is_dir():
                    item['checkpoint'] = inspect_checkpoint(local_model)
                    expected = identity.get('chat_template_sha256')
                    actual = item['checkpoint']['chat_template_sha256']
                    if expected and re.fullmatch(r'[0-9a-fA-F]{64}', expected):
                        if not actual or expected.lower() != actual:
                            item['errors'].append('Declared main chat-template hash differs from local checkpoint')
                    else:
                        item['warnings'].append('Record the actual chat-template hash before publishing comparable results')
                elif bench in GUI:
                    item['errors'].append('Native GUI inference requires an existing identity.model_path')
                else:
                    item['warnings'].append('No local checkpoint metadata checked; remote model/template identity is not attested')
                if not identity.get('model_revision') or identity['model_revision'].startswith('RECORD_'):
                    item['warnings'].append('Record the actual main model revision/checkpoint identity')
            if bench in PLATFORM_BENCHMARKS:
                from .general_visual import tokenizer_manifest
                item['judge_tokenizer'] = tokenizer_manifest(effective)
            if audit_data:
                from .data import audit
                item['data_audit'] = audit(bench, data)
            if bench == 'agentvista' and phase in ('infer', 'all'):
                if check_tools:
                    from .agent_tools import check_live_tools
                    item['tool_checks'] = check_live_tools()
                    if not item['tool_checks']['checks_passed']:
                        item['errors'].append('AgentVista free tools are unavailable; fix network/tool access before inference')
                else:
                    item['warnings'].append('AgentVista live tool access not checked; run doctor --check-tools before inference')
            roles = []
            if phase in ('infer', 'all') and bench not in GUI:
                roles.append('main')
            if phase in ('score', 'all') and bench in JUDGED:
                roles.append('judge')
            for role in roles:
                endpoint = effective.get(role)
                if not endpoint:
                    item['errors'].append(f'Missing required {role} endpoint')
                    continue
                item['services'][role] = {'model': endpoint['model'], 'checked': False}
                if check_endpoints:
                    key = (endpoint['model'], tuple(endpoint['base_urls']), endpoint['api_key_env'])
                    if key not in checked_services:
                        try:
                            CompatibleAPI(endpoint, min(effective['timeout'], 30), 1).health()
                            checked_services[key] = None
                        except RuntimeError as exc:
                            checked_services[key] = str(exc)
                    if checked_services[key]:
                        item['errors'].append(f'{role}: {checked_services[key]}')
                    else:
                        item['services'][role]['checked'] = True
            if 'judge' in roles and (not identity.get('judge_revision') or identity['judge_revision'].startswith('RECORD_')):
                item['warnings'].append('Record the actual benchmark-specific judge revision; a served alias is not a weight identity')
            if bench == 'mmlongbench':
                item['warnings'].append('file:// images require the service to see and allow the same local image root')
            item['warnings'].append('Dependency metadata and /v1/models do not prove a working GPU runtime or serving template')
        except (ValueError, RuntimeError, OSError, ImportError, KeyError, TypeError) as exc:
            item['errors'].append(f'{type(exc).__name__}: {exc}')
        item['checks_passed'] = not item['errors']
    report['checks_passed'] = all(item['checks_passed'] for item in report['results'])
    return report
