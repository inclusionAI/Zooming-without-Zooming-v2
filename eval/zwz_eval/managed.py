"""One-terminal local main/infer/judge/score orchestration; protocols stay intact."""
from __future__ import annotations

from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from .config import for_benchmark, gpu_groups, load_config, run_directory
from .io import atomic_json, run_lock
from .protocols import PLATFORM_BENCHMARKS
from .service_lifecycle import ServicePool, _group_alive, _signal_owned
from .serving import plan
from .workflow import selection

NATIVE_GPU = {'screenspot_pro', 'ui_vision'}
JUDGED = set(PLATFORM_BENCHMARKS) | {
    'milebench', 'memlens', 'agentvista', 'viewspatial', 'chartography', 'erqa', 'mimic'}


def _stop_phase(process, timeout=10):
    """Clean only the child session we started, including adopted descendants."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not _group_alive(process.pid):
            process.wait()
            return
        _signal_owned(process.pid, sig)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            process.poll()  # Reap the leader so zombie-only groups can disappear.
            if not _group_alive(process.pid):
                process.wait()
                return
            time.sleep(0.1)
    raise RuntimeError('Owned evaluation process group did not exit; refusing the next phase')


def _phase(config_path, config, bench, phase, pool=None):
    env = dict(os.environ)
    # The locally managed servers have no API-key flag; do not propagate a
    # missing secret requirement to these unauthenticated loopback clients.
    for role in ('main', 'judge'):
        if role in config:
            env.setdefault(config[role].get('api_key_env', role.upper() + '_API_KEY'), 'EMPTY')
    child = subprocess.Popen(
        [sys.executable, '-m', 'zwz_eval', phase, '--config', str(config_path), '--bench', bench],
        env=env, start_new_session=True)
    try:
        while True:
            if pool:
                pool.check_running()
            code = child.poll()
            if code is not None:
                if code:
                    raise RuntimeError(f'{bench} {phase} failed with exit code {code}; see preceding output')
                return
            time.sleep(0.5)
    finally:
        _stop_phase(child)


def _plans(config, names, phase):
    """Validate the entire queue before starting any process."""
    result = {}
    for bench in names:
        effective = for_benchmark(config, bench)
        phases = ('infer', 'score') if phase == 'all' else (phase,)
        for current in phases:
            role = 'main' if current == 'infer' else 'judge'
            needed = bench not in NATIVE_GPU and (current == 'infer' or bench in JUDGED)
            jobs = plan(effective, bench, role)['commands'] if needed else []
            if jobs:
                hardware = effective.get('hardware', {})
                tp = hardware.get('tp', 1) if role == 'main' else hardware.get('judge_tp', 2)
                if len(jobs) * tp != hardware.get('gpus', 16):
                    raise ValueError(f'{bench} {role}: configure exactly gpus/tp loopback endpoints to use the requested devices')
                path = effective.get('identity', {}).get('model_path' if role == 'main' else 'judge_model_path')
                if not path or not Path(path).is_dir():
                    raise ValueError(f'{bench} {role}: local model directory does not exist')
            result[bench, current] = (effective, jobs)
    return result


def evaluate(config_path, selected='all', phase='all', *, startup_timeout=1800):
    """Serialize selected benchmarks and roles, holding host-local device locks.

    Existing endpoints are never reused or killed in local mode. External API
    execution remains run-suite, without service ownership or automatic changes.
    """
    if phase not in ('all', 'infer', 'score'):
        raise ValueError('Managed evaluation supports all, infer or score')
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    names = selection(selected, config['datasets'])
    plans = _plans(config, names, phase)
    devices = sum(gpu_groups(config.get('hardware', {}).get('gpus', 16), 1), [])
    lock_root = Path(tempfile.gettempdir()) / ('zwz-eval-devices-' + str(os.getuid()))
    if lock_root.is_symlink():
        raise ValueError('Device lock directory must not be a symlink')
    lock_root.mkdir(mode=0o700, exist_ok=True)
    if lock_root.stat().st_uid != os.getuid():
        raise ValueError('Device lock directory has another owner')
    def terminate(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGTERM, terminate)
    results = []
    report_path = (Path(config['output_root']) / config['model_id'] / config['run_id'] /
                   ('managed_' + phase + '.json'))
    try:
        with ExitStack() as locks:
            # Per-device, not per-allocation: overlapping selections conflict.
            for device in sorted(devices):
                locks.enter_context(run_lock(lock_root / hashlib.sha256(device.encode()).hexdigest()))
            for bench in names:
                directory = run_directory(config, bench)
                with run_lock(directory / '.managed'):
                    for current in (('infer', 'score') if phase == 'all' else (phase,)):
                        atomic_json(report_path, {'complete': False, 'status': 'running',
                                                  'benchmark': bench, 'phase': current, 'results': results})
                        effective, jobs = plans[bench, current]
                        role = 'main' if current == 'infer' else 'judge'
                        print(json.dumps({'benchmark': bench, 'phase': current,
                                          'service_role': role if jobs else 'none'}), flush=True)
                        pool = None
                        try:
                            if jobs:
                                pool = ServicePool(jobs, directory / 'logs/services' / role,
                                                   readiness_timeout=startup_timeout, log_prefix=role)
                                pool.start()
                            _phase(config_path, effective, bench, current, pool)
                        finally:
                            if pool is not None:
                                pool.stop()
                results.append({'benchmark': bench, 'status': 'complete', 'phase': phase})
        report = {'complete': True, 'results': results, 'services': 'local',
                  'checked_at': datetime.now(timezone.utc).isoformat()}
        atomic_json(report_path, report)
        return report
    except BaseException as exc:
        atomic_json(report_path, {'complete': False, 'status': 'interrupted' if isinstance(exc, KeyboardInterrupt)
                                  else 'failed', 'error_type': type(exc).__name__, 'results': results})
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
