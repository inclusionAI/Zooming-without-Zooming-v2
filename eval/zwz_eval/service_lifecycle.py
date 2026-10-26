"""Owned, single-node vLLM service startup; never executes inference or scoring.

Port blocks reduce vLLM's bind/close/bind race. Preflight is not an OS port
reservation: a confirmed startup EADDRINUSE can retry in the next block. Model
arguments are never changed, and a service which reached ready is never retried.
"""
from __future__ import annotations

import copy
import errno
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.request

from .io import atomic_json


def _check_ports(ports):
    """Check simultaneous bindability, then release; this is NOT a reservation."""
    sockets = []
    try:
        for port in ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sockets.append(sock)
            sock.bind(('', port))  # Same IPv4 bind used by vLLM get_open_port.
    finally:
        for sock in sockets:
            sock.close()


def _group_alive(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    # Container PID 1 can leave adopted zombies behind. They hold no sockets
    # and cannot be killed; do not confuse a zombie-only group with live workers.
    proc = Path('/proc')
    if proc.is_dir():
        observed = False
        for entry in proc.iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                if int(fields[2]) == pgid:
                    observed = True
                    if fields[0] != 'Z':
                        return True
            except (OSError, ValueError, IndexError):
                continue
        if observed:
            return False
    return True


def _ephemeral_port_range():
    """Read Linux's actual ephemeral range; never modify host networking."""
    path = Path('/proc/sys/net/ipv4/ip_local_port_range')
    if not path.exists():
        return None
    values = [int(value) for value in path.read_text().split()]
    if len(values) != 2 or not 1 <= values[0] <= values[1] <= 65535:
        raise ValueError('Cannot validate the host ephemeral port range')
    return values


def _signal_owned(pgid, sig):
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


class ServicePool:
    """Launch all replicas before readiness waits; clean up only owned groups.

    ``jobs`` are commands from ``serving.plan`` (callers may append runtime
    flags before construction). ``start`` returns current launch records;
    ``wait`` monitors already-ready services until interrupted or a child exits.
    Use the context manager or ``finally: pool.stop()`` to close a healthy pool.
    """

    def __init__(self, jobs, log_dir, *, readiness_timeout=1800,
                 max_start_retries=2, on_event=None, base_env=None, cwd=None,
                 log_prefix='service', poll_interval=1.0):
        if not hasattr(os, 'killpg'):
            raise RuntimeError('ServicePool requires POSIX process groups')
        if not isinstance(max_start_retries, int) or isinstance(max_start_retries, bool) or not 0 <= max_start_retries <= 9:
            raise ValueError('max_start_retries must be 0..9')
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value <= 0
               for value in (readiness_timeout, poll_interval)):
            raise ValueError('Service timeouts must be positive')
        if not re.fullmatch(r'[A-Za-z0-9_-]+', log_prefix):
            raise ValueError('log_prefix must be a safe filename prefix')
        self.jobs = copy.deepcopy(jobs)
        self.log_dir = Path(log_dir)
        self.readiness_timeout = readiness_timeout
        self.max_start_retries = max_start_retries
        self.on_event = on_event
        self.base_env = dict(os.environ if base_env is None else base_env)
        self.cwd = cwd
        self.log_prefix = log_prefix
        self.poll_interval = poll_interval
        self.records = []
        self._current = {}
        self._owned = []
        self._started = False
        self._validate_jobs()

    def _validate_jobs(self):
        if not isinstance(self.jobs, list) or not self.jobs:
            raise ValueError('ServicePool requires a nonempty command list')
        api_ports = set()
        blocks = []
        for job in self.jobs:
            # vLLM can fall back to topology environment variables even when
            # the CLI omits DP flags. Do not silently reinterpret a multi-node
            # caller as one of our independent, owned DP=1 local services.
            child_env = {**self.base_env, **job.get('env', {})}
            topology = {'VLLM_DP_SIZE': 1, 'VLLM_DP_RANK': 0, 'VLLM_DP_RANK_LOCAL': 0}
            for key, expected in topology.items():
                if key in child_env:
                    value = child_env[key]
                    try:
                        valid = not isinstance(value, bool) and int(value) == expected
                    except (TypeError, ValueError):
                        valid = False
                    if not valid:
                        raise ValueError(f'ServicePool requires single-node DP=1; incompatible {key}')
            service = job['service']
            port = service['api_port']
            if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535 or port in api_ports:
                raise ValueError('Service API ports must be unique integers in 1..65535')
            api_ports.add(port)
            argv = job['argv']
            if int(argv[argv.index('--port') + 1]) != port or argv[argv.index('--served-model-name') + 1] != service['model']:
                raise ValueError('Service metadata does not match argv')
            if argv[argv.index('--host') + 1] != '127.0.0.1':
                raise ValueError('ServicePool only supports single-node loopback services')
            topology_flags = {'--data-parallel-size': 1, '--data-parallel-size-local': 1,
                              '--nnodes': 1, '--node-rank': 0, '--data-parallel-rank': 0,
                              '--data-parallel-start-rank': 0}
            for index, token in enumerate(argv):
                key, separator, value = token.partition('=')
                if key in topology_flags:
                    if not separator:
                        value = argv[index + 1]
                    if int(value) != topology_flags[key]:
                        raise ValueError(f'ServicePool requires single-node DP=1; incompatible {key}')
            if not service['internal_port_blocks']:
                raise ValueError('Internal startup port blocks are required')
            for block in service['internal_port_blocks']:
                start, end = block['start'], block['end']
                if any(isinstance(p, bool) or not isinstance(p, int) for p in (start, end)) or not 1024 <= start <= end <= 65535:
                    raise ValueError('Invalid internal port block')
                if any(max(start, other_start) <= min(end, other_end) for other_start, other_end in blocks):
                    raise ValueError('Internal startup port blocks overlap')
                blocks.append((start, end))
        if any(start <= port <= end for port in api_ports for start, end in blocks):
            raise ValueError('HTTP API and internal ports overlap')

    def _event(self, event, record, **extra):
        if self.on_event:
            self.on_event({'event': event, 'replica': record['replica'],
                           'attempt': record['attempt'], 'pid': record.get('pid'), **extra})

    def _save_record(self, record):
        public = {key: value for key, value in record.items() if not key.startswith('_')}
        stem = self.log_dir / f"{self.log_prefix}-{record['replica']:02d}"
        atomic_json(stem.with_suffix('.command.json'), public)
        atomic_json(stem.with_name(stem.name + f".attempt-{record['attempt']:02d}.json"), public)
        self.records = [{key: value for key, value in item.items() if not key.startswith('_')}
                        for _, item in sorted(self._current.items())]

    def _launch(self, index, attempt):
        job = self.jobs[index]
        service = job['service']
        attempts = min(len(service['internal_port_blocks']), self.max_start_retries + 1)
        while attempt <= attempts:
            # Never reuse an existing API server, even when its model ID matches.
            try:
                _check_ports([service['api_port']])
            except OSError as exc:
                raise RuntimeError(f"API port {service['api_port']} is unavailable; refusing reuse or termination") from exc
            block = service['internal_port_blocks'][attempt - 1]
            try:
                _check_ports(range(block['start'], block['end'] + 1))
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise
                record = {'replica': index, 'attempt': attempt, 'pid': None}
                self._event('port_block_busy', record, internal_port_start=block['start'], internal_port_end=block['end'])
                attempt += 1
                continue
            path = self.log_dir / f'{self.log_prefix}-{index:02d}.log'
            stream = path.open('ab', buffering=0)
            offset = stream.tell()
            child_env = {**self.base_env, **job['env'],
                         'VLLM_PORT': str(block['start']), 'VLLM_HOST_IP': '127.0.0.1',
                         'VLLM_LOOPBACK_IP': '127.0.0.1'}
            # This launcher starts independent DP=1 servers. An inherited DP
            # reserved-port range can make get_open_port loop on VLLM_PORT.
            child_env.pop('VLLM_DP_MASTER_PORT', None)
            try:
                process = subprocess.Popen(job['argv'], env=child_env, cwd=self.cwd,
                                           stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            except BaseException:
                stream.close()
                raise
            record = {'replica': index, 'attempt': attempt, 'pid': process.pid,
                      'hostname': socket.gethostname(), 'argv': job['argv'],
                      'env': {key: child_env[key] for key in ('CUDA_VISIBLE_DEVICES',
                              'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'VLLM_PORT', 'VLLM_HOST_IP',
                              'VLLM_LOOPBACK_IP') if key in child_env},
                      'unset_env': ['VLLM_DP_MASTER_PORT'],
                      'api_port': service['api_port'], 'model': service['model'],
                      'internal_port_start': block['start'], 'internal_port_end': block['end'],
                      'log_path': str(path), 'log_offset': offset, 'started_at': time.time(),
                      'ready': False, '_process': process, '_stream': stream,
                      '_deadline': time.monotonic() + self.readiness_timeout, '_cleaned': False}
            self._owned.append(record)
            self._current[index] = record
            self._save_record(record)
            self._event('started', record)
            return
        raise RuntimeError(f'Replica {index}: no available internal port block within {attempts} startup attempts')

    def _current_address_in_use(self, record):
        with Path(record['log_path']).open('rb') as stream:
            stream.seek(record['log_offset'])
            text = stream.read().decode('utf-8', errors='replace')
        collision = r'EADDRINUSE|\[Errno (?:98|48)\]|address already in use'
        if not re.search(collision, text, re.IGNORECASE):
            return False
        # A recoverable diagnostic earlier in this attempt is not evidence for
        # retrying a later OOM/checkpoint/import failure. Only accept exception
        # chains whose concrete errors are port conflicts. vLLM may append this
        # generic parent-process wrapper after the EngineCore root exception.
        fatal_port_error = False
        for match in re.finditer(r'\b[\w.]*(?:Error|Exception):\s*([^\n]*)', text):
            message = match.group(0)
            if re.search(collision, message, re.IGNORECASE):
                fatal_port_error = True
                continue
            if 'RuntimeError: Engine core initialization failed.' in message:
                continue
            return False
        return fatal_port_error

    def _ready(self, record):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{record['api_port']}/v1/models", timeout=3) as response:
                ids = {item['id'] for item in json.load(response)['data']}
        except (OSError, ValueError, KeyError, TypeError):
            return False
        if record['model'] not in ids:
            raise RuntimeError(f"Replica {record['replica']}: wrong served model ID")
        return True

    def start(self):
        if self._started:
            raise RuntimeError('A ServicePool instance can only be started once')
        self._started = True
        self.log_dir.mkdir(parents=True, exist_ok=True)
        try:
            ephemeral = _ephemeral_port_range()
            if ephemeral:
                for job in self.jobs:
                    for block in job['service']['internal_port_blocks']:
                        if max(ephemeral[0], block['start']) <= min(ephemeral[1], block['end']):
                            raise ValueError('Internal port block overlaps the host ephemeral port range; choose other execution port bases')
            # Detect occupied APIs before starting any child. Model warm-up is
            # intentionally parallel, not serialized behind each /v1/models.
            try:
                _check_ports([job['service']['api_port'] for job in self.jobs])
            except OSError as exc:
                raise RuntimeError('An API port is unavailable; refusing reuse or termination') from exc
            for index in range(len(self.jobs)):
                self._launch(index, 1)
            while True:
                for index in range(len(self.jobs)):
                    record = self._current[index]
                    process = record['_process']
                    if process.poll() is not None:
                        collision = not record['ready'] and self._current_address_in_use(record)
                        attempts = min(len(self.jobs[index]['service']['internal_port_blocks']), self.max_start_retries + 1)
                        if collision and record['attempt'] < attempts:
                            self._event('retrying_address_in_use', record)
                            self._stop_record(record)
                            self._launch(index, record['attempt'] + 1)
                            continue
                        raise RuntimeError(f"Replica {index} exited {process.returncode}; inspect {record['log_path']} at offset {record['log_offset']}")
                    if not record['ready']:
                        if time.monotonic() >= record['_deadline']:
                            raise TimeoutError(f"Replica {index} not ready after {self.readiness_timeout}s; inspect {record['log_path']}")
                        if self._ready(record):
                            record['ready'] = True
                            record['ready_at'] = time.time()
                            self._save_record(record)
                            self._event('ready', record)
                if all(record['ready'] for record in self._current.values()):
                    # Do not return a pool whose earlier ready replica died
                    # while the remaining replicas were loading.
                    if any(record['_process'].poll() is not None for record in self._current.values()):
                        continue
                    return self.records
                time.sleep(self.poll_interval)
        except BaseException:
            self.stop()
            raise

    def _stop_record(self, record):
        if record['_cleaned']:
            return
        process = record['_process']
        # The API parent may already have exited, leaving EngineCore children
        # in our session. Always signal our saved PGID, not just a live parent.
        _signal_owned(process.pid, signal.SIGTERM)
        deadline = time.monotonic() + 10
        while True:
            process.poll()  # Reap our parent, including one already exited.
            if not _group_alive(process.pid):
                break
            if time.monotonic() >= deadline:
                _signal_owned(process.pid, signal.SIGKILL)
                deadline = time.monotonic() + 5
                while _group_alive(process.pid) and time.monotonic() < deadline:
                    process.poll()
                    time.sleep(min(self.poll_interval, 0.1))
                if _group_alive(process.pid):
                    raise RuntimeError(f'Owned process group {process.pid} did not exit; refusing startup retry')
                break
            time.sleep(min(self.poll_interval, 0.1))
        process.wait(timeout=5)
        record['_stream'].close()
        record['_cleaned'] = True
        record['stopped_at'] = time.time()
        self._save_record(record)
        self._event('stopped', record)

    def stop(self):
        errors = []
        # Signal the fleet first so shutdown is not serialized behind one slow
        # model. _stop_record also signals on the single-replica retry path.
        for record in self._owned:
            if not record['_cleaned']:
                try:
                    _signal_owned(record['pid'], signal.SIGTERM)
                except OSError as exc:
                    errors.append(exc)
        for record in self._owned:
            try:
                self._stop_record(record)
            except BaseException as exc:
                # A denied signal/inspection must stay an explicit failure,
                # but it must not leak this parent's log file descriptor.
                record['_stream'].close()
                errors.append(exc)
        if errors:
            raise RuntimeError(f'Could not clean up {len(errors)} owned process group(s)') from errors[0]

    def check_running(self):
        """Fail on a ready replica's exit; never restart it behind a running client."""
        if not self._started or not self._current or not all(record['ready'] for record in self._current.values()):
            raise RuntimeError('Call start() successfully before wait()')
        for record in self._current.values():
            if record['_process'].poll() is not None:
                raise RuntimeError(f"Ready replica {record['replica']} exited; no post-readiness retry is allowed")

    def wait(self):
        while True:
            self.check_running()
            time.sleep(self.poll_interval)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.stop()
