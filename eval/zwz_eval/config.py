from __future__ import annotations

import copy
import math
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

from .io import read_json
from .protocols import PROTOCOLS, PLATFORM_BENCHMARKS


def expand(value):
    if isinstance(value, str):
        def replace(match):
            key = match.group(1)
            if key not in os.environ:
                raise ValueError(f'Missing environment variable: {key}')
            return os.environ[key]
        return re.sub(r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}', replace, value)
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value


def identifier(value, label):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value):
        raise ValueError(f'{label} must be a safe single path component')
    return value


def gpu_groups(gpus, tp, *, environment=None):
    """Map replicas to the caller's visible device identifiers, without CUDA calls.

    CUDA renumbers visible devices inside a process. Child processes receive the
    original identifiers, not those logical indices, so an allocation such as
    ``CUDA_VISIBLE_DEVICES=8,9`` can never silently become physical GPUs 0,1.
    An absent variable keeps the historical 0..gpus-1 assignment; an explicitly
    empty/disabled or insufficient allocation is an error, including in plans.
    """
    if (isinstance(gpus, bool) or isinstance(tp, bool)
            or not isinstance(gpus, int) or not isinstance(tp, int)
            or tp < 1 or gpus < tp or gpus % tp):
        raise ValueError('gpus must be positive and divisible by tp')
    environment = os.environ if environment is None else environment
    if 'CUDA_VISIBLE_DEVICES' not in environment:
        devices = [str(index) for index in range(gpus)]
    else:
        visible = environment['CUDA_VISIBLE_DEVICES']
        if not isinstance(visible, str) or not visible.strip():
            raise ValueError('CUDA_VISIBLE_DEVICES disables all GPUs')
        devices = []
        for value in visible.split(','):
            token = value.strip()
            if re.fullmatch(r'[0-9]+', token):
                token = str(int(token))
            elif not re.fullmatch(r'(?:GPU-[A-Za-z0-9-]+|MIG-[A-Za-z0-9./-]+)', token):
                raise ValueError('Invalid CUDA_VISIBLE_DEVICES device identifier')
            if token in devices:
                raise ValueError('CUDA_VISIBLE_DEVICES contains duplicate devices')
            devices.append(token)
        if len(devices) < gpus:
            raise ValueError(f'CUDA_VISIBLE_DEVICES exposes {len(devices)} devices, but hardware.gpus requests {gpus}')
        devices = devices[:gpus]
    return [devices[start:start + tp] for start in range(0, gpus, tp)]


def validate_endpoint(endpoint, role):
    if not isinstance(endpoint, dict):
        raise ValueError(f'{role} must be an object')
    unexpected = set(endpoint) - {'model', 'base_urls', 'api_key_env'}
    if unexpected:
        raise ValueError(f'Unknown {role} fields: {sorted(unexpected)}')
    endpoint.setdefault('api_key_env', 'MAIN_API_KEY' if role == 'main' else 'JUDGE_API_KEY')
    if not isinstance(endpoint['api_key_env'], str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', endpoint['api_key_env']):
        raise ValueError('api_key_env must be an environment variable name')
    if not isinstance(endpoint.get('model'), str) or not endpoint['model'].strip():
        raise ValueError(f'{role}.model must be a nonempty served-model name')
    urls = endpoint.get('base_urls', [])
    if not isinstance(urls, list) or not urls or not all(isinstance(u, str) for u in urls):
        raise ValueError(f'{role}.base_urls required')
    for url in urls:
        parsed = urlsplit(url)
        if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(f'Invalid {role} base URL (credentials/query are prohibited)')
        if not parsed.path.rstrip('/').endswith('/v1'):
            raise ValueError('Use a base URL ending /v1, not /chat/completions')


def for_benchmark(config, bench):
    result = copy.deepcopy(config)
    name = result['datasets'][bench].get('judge_profile')
    if name:
        profiles = result.get('judge_profiles', {})
        if name not in profiles:
            raise ValueError('Unknown benchmark judge_profile')
        profile = profiles[name]
        result['judge'] = copy.deepcopy(profile['endpoint'])
        identity = result.setdefault('identity', {})
        for key in ('judge_model_path', 'judge_tokenizer_path', 'judge_revision', 'judge_family'):
            identity.pop(key, None)  # Never inherit another judge family's tokenizer.
        for source, target in (('model_path', 'judge_model_path'), ('tokenizer_path', 'judge_tokenizer_path'),
                               ('revision', 'judge_revision'), ('family', 'judge_family')):
            if source in profile:
                identity[target] = profile[source]
    if bench in PLATFORM_BENCHMARKS and result.get('identity', {}).get('judge_family') != 'qwen3.5-35b-a3b':
        raise ValueError('The project evaluate_v3 profile requires declared qwen3.5-35b-a3b judge family; select its named judge profile')
    return result


def load_config(path):
    source = Path(path).resolve()
    value = expand(read_json(source))
    if not isinstance(value, dict):
        raise ValueError('Run config must be a JSON object')
    unknown = set(value) - {'model_id', 'run_id', 'output_root', 'main', 'judge', 'judge_profiles', 'datasets', 'workers', 'timeout', 'retries', 'identity', 'hardware'}
    if unknown:
        raise ValueError(f'Unknown config fields: {sorted(unknown)}')
    for name in ('model_id', 'run_id'):
        identifier(value[name], name)
    def absolute(text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Local paths must be nonempty strings')
        p = Path(text).expanduser()
        return str((source.parent / p).resolve() if not p.is_absolute() else p.resolve())
    value['output_root'] = absolute(value['output_root'])
    identity = value.get('identity', {})
    if not isinstance(identity, dict):
        raise ValueError('identity must be an object')
    allowed_identity = {'model_path', 'judge_model_path', 'judge_tokenizer_path', 'model_revision',
                        'judge_revision', 'judge_family', 'chat_template_sha256'}
    if set(identity) - allowed_identity:
        raise ValueError('Unknown identity fields: ' + ', '.join(sorted(set(identity)-allowed_identity)))
    if any(not isinstance(item, str) or not item.strip() for item in identity.values()):
        raise ValueError('Identity fields must be nonempty strings')
    for key in ('model_path', 'judge_model_path', 'judge_tokenizer_path'):
        if key in identity:
            if not isinstance(identity[key], str) or not identity[key].strip():
                raise ValueError(f'identity.{key} must be a nonempty local path')
            identity[key] = absolute(identity[key])
    hardware = value.get('hardware', {})
    if not isinstance(hardware, dict):
        raise ValueError('hardware must be an object')
    if set(hardware) - {'gpus', 'tp', 'judge_tp', 'memory_utilization'}:
        raise ValueError('Unknown hardware fields')
    for key in ('gpus', 'tp', 'judge_tp'):
        if key in hardware and (isinstance(hardware[key], bool) or not isinstance(hardware[key], int) or hardware[key] < 1):
            raise ValueError(f'hardware.{key} must be a positive integer')
    if 'memory_utilization' in hardware:
        memory = hardware['memory_utilization']
        if isinstance(memory, bool) or not isinstance(memory, (int, float)) or not math.isfinite(memory) or not 0 < memory <= 1:
            raise ValueError('hardware.memory_utilization must be in (0, 1]')
    datasets = value.get('datasets', {})
    if not isinstance(datasets, dict):
        raise ValueError('datasets must be an object')
    for bench, data in datasets.items():
        if bench not in PROTOCOLS:
            raise ValueError(f'Unknown benchmark {bench}')
        if not isinstance(data, dict) or 'root' not in data:
            raise ValueError(f'datasets.{bench}.root required')
        allowed = {'root', 'image_root', 'judge_profile'}
        if bench == 'mimic':
            allowed |= {'mathruler_mode', 'mathruler_version'}
        if set(data) - allowed:
            raise ValueError(f'Unknown datasets.{bench} fields: {sorted(set(data)-allowed)}')
        for key in ('root', 'image_root'):
            if key in data:
                if not isinstance(data[key], str) or not data[key].strip():
                    raise ValueError(f'datasets.{bench}.{key} must be a nonempty path')
                data[key] = absolute(data[key])
    for role in ('main', 'judge'):
        endpoint = value.get(role)
        if endpoint is None:
            continue
        validate_endpoint(endpoint, role)
    profiles = value.get('judge_profiles', {})
    if not isinstance(profiles, dict):
        raise ValueError('judge_profiles must be an object')
    for name, profile in profiles.items():
        identifier(name, 'judge profile name')
        if not isinstance(profile, dict) or set(profile) - {'endpoint', 'model_path', 'tokenizer_path', 'revision', 'family'}:
            raise ValueError('Invalid judge profile fields')
        validate_endpoint(profile.get('endpoint'), 'judge profile')
        for key in set(profile) - {'endpoint'}:
            if not isinstance(profile[key], str) or not profile[key].strip():
                raise ValueError(f'judge_profiles.{name}.{key} must be a nonempty string')
        for key in ('model_path', 'tokenizer_path'):
            if profile.get(key):
                profile[key] = absolute(profile[key])
    for bench, data in datasets.items():
        if 'judge_profile' in data and (not isinstance(data['judge_profile'], str) or data['judge_profile'] not in profiles):
            raise ValueError(f'datasets.{bench}.judge_profile must name an existing profile')
    value.setdefault('workers', 16)
    value.setdefault('timeout', 600)
    value.setdefault('retries', 3)
    if isinstance(value['workers'], bool) or not isinstance(value['workers'], int) or not 1 <= value['workers'] <= 256:
        raise ValueError('workers must be 1..256')
    if isinstance(value['retries'], bool) or not isinstance(value['retries'], int) or not 1 <= value['retries'] <= 10:
        raise ValueError('retries must be 1..10')
    if isinstance(value['timeout'], bool) or not isinstance(value['timeout'], (int, float)) or not 0 < value['timeout'] <= 86400:
        raise ValueError('timeout must be greater than zero and at most 86400 seconds')
    value['_config_path'] = str(source)
    return value


def run_directory(config, bench):
    return Path(config['output_root']) / config['model_id'] / config['run_id'] / bench


def public_config(config):
    result = copy.deepcopy(config)
    result.pop('_config_path', None)
    return result
