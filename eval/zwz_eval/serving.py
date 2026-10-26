"""Generate commands only; never kill existing services or install platform runtimes."""
from __future__ import annotations

import sys
from urllib.parse import urlsplit

from .config import gpu_groups
from .data import image_root_for
from .protocols import PROTOCOLS


def plan(config, bench, role='main', *, main_internal_port_base=25000,
         judge_internal_port_base=29000, internal_port_block_size=32,
         service_start_attempts=3):
    if role not in ('main', 'judge'):
        raise ValueError('role must be main or judge')
    if bench in ('screenspot_pro', 'ui_vision'):
        raise ValueError('Native GUI adapter starts local GPU replicas itself; do not start a duplicate API server')
    protocol = PROTOCOLS[bench]
    hardware = config.get('hardware', {})
    tp = hardware.get('judge_tp', 2) if role == 'judge' else hardware.get('tp', 1)
    gpus = hardware.get('gpus', 16)
    groups = gpu_groups(gpus, tp)
    endpoints = config[role]['base_urls']
    if len(endpoints) * tp > gpus:
        raise ValueError('Configured replicas exceed available GPUs')
    model_path = config.get('identity', {}).get('model_path' if role == 'main' else 'judge_model_path')
    if not model_path:
        raise ValueError('Set identity.model_path / identity.judge_model_path explicitly')
    thinking = protocol['thinking'] if role == 'main' else protocol['judge_thinking']
    # vLLM uses VLLM_PORT, not MASTER_PORT, for get_open_port(). Each replica
    # gets separate startup-attempt blocks; HTTP --port is unrelated. These
    # execution-only options deliberately do not alter the evaluation config.
    options = {'main_internal_port_base': main_internal_port_base,
               'judge_internal_port_base': judge_internal_port_base,
               'internal_port_block_size': internal_port_block_size,
               'service_start_attempts': service_start_attempts}
    if any(isinstance(v, bool) or not isinstance(v, int) for v in options.values()):
        raise ValueError('Internal port options must be integers, not booleans')
    if not 8 <= internal_port_block_size <= 1024 or not 1 <= service_start_attempts <= 10:
        raise ValueError('Internal port block size must be 8..1024; startup attempts must be 1..10')
    ranges = {}
    for name, base in (('main', main_internal_port_base), ('judge', judge_internal_port_base)):
        count = len(config.get(name, {}).get('base_urls', []))
        end = base + count * internal_port_block_size * service_start_attempts
        if not 1024 <= base <= 65535 or end > 65536:
            raise ValueError(f'{name} internal port range is outside 1024..65535')
        if count:
            ranges[name] = (base, end)
    if len(ranges) == 2 and max(ranges['main'][0], ranges['judge'][0]) < min(ranges['main'][1], ranges['judge'][1]):
        raise ValueError('Main and judge internal port ranges overlap')
    for name in ('main', 'judge'):
        for url in config.get(name, {}).get('base_urls', []):
            parsed = urlsplit(url)
            if parsed.hostname in ('127.0.0.1', 'localhost') and parsed.port:
                if any(start <= parsed.port < end for start, end in ranges.values()):
                    raise ValueError('An HTTP API port overlaps an internal port range')
    jobs = []
    ports = set()
    for index, url in enumerate(endpoints):
        parsed = urlsplit(url)
        if parsed.hostname not in ('127.0.0.1', 'localhost') or not parsed.port or parsed.port in ports:
            raise ValueError('Server plans require unique explicit localhost ports')
        ports.add(parsed.port)
        argv = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', model_path,
                '--served-model-name', config[role]['model'], '--host', '127.0.0.1', '--port', str(parsed.port),
                '--tensor-parallel-size', str(tp), '--max-model-len', str(protocol.get('judge_context', protocol['context']) if role == 'judge' else protocol['context']),
                '--gpu-memory-utilization', str(hardware.get('memory_utilization', 0.9)),
                '--limit-mm-per-prompt', '{"image":%d,"video":0}' % (protocol.get('judge_max_images', protocol['max_images']) if role == 'judge' else protocol['max_images'])]
        if thinking is not None:
            import json
            argv.extend(['--default-chat-template-kwargs', json.dumps({'enable_thinking': thinking})])
        if role == 'main' and bench == 'mmlongbench':
            argv.extend(['--allowed-local-media-path', str(image_root_for(bench, config['datasets'][bench]))])
        base = ranges[role][0] + index * service_start_attempts * internal_port_block_size
        blocks = [{'start': base + attempt * internal_port_block_size,
                   'end': base + (attempt + 1) * internal_port_block_size - 1}
                  for attempt in range(service_start_attempts)]
        jobs.append({'env': {'CUDA_VISIBLE_DEVICES': ','.join(groups[index]),
                             'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'python',
                             'VLLM_HOST_IP': '127.0.0.1', 'VLLM_LOOPBACK_IP': '127.0.0.1',
                             'VLLM_PORT': str(base)}, 'argv': argv,
                     'unset_env': ['VLLM_DP_MASTER_PORT'],
                     'service': {'api_port': parsed.port, 'model': config[role]['model'],
                                 'internal_port_blocks': blocks}})
    return {'role': role, 'commands': jobs, 'executed': False,
            'warning': 'Single-node, data-parallel-size=1 only. Apply unset_env as well as env. VLLM_PORT is a scan start, not an OS reservation or hard range limit. Use ServicePool for preflight and bounded startup retries. Verify flags against your installed vLLM build. Start main and judge sequentially if they share GPUs. No automatic context extension or truncation.'}
