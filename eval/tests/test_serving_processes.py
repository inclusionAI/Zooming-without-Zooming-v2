"""Opt-in CPU-only process/socket checks (no torch, vLLM, data or model loads).

Run with RUN_SERVICE_PROCESS_TESTS=1. All listeners use loopback; shutdown is
restricted to process groups created by this test or by its ServicePool.
"""
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request

from zwz_eval.service_lifecycle import ServicePool, _check_ports, _group_alive


FAKE_SERVICE = r'''
import http.server, json, os, pathlib, socket, subprocess, sys, time
def arg(name):
    return sys.argv[sys.argv.index(name) + 1]
root = pathlib.Path(arg('--case-dir'))
replica = arg('--replica')
port = int(os.environ['VLLM_PORT'])
info = {'pid': os.getpid(), 'pgid': os.getpgrp(), 'internal_port': port}
marker = root / ('started-' + replica + '.json')
if str(port) == os.environ.get('CPU_TEST_FAIL_PORT'):
    child_code = """
import json, os, pathlib, socket, sys, time
s = socket.socket(); s.bind(('127.0.0.1', int(sys.argv[1]))); s.listen()
pathlib.Path(sys.argv[2]).write_text(json.dumps({'pid': os.getpid(), 'pgid': os.getpgrp()}))
while True: time.sleep(1)
"""
    orphan = root / 'orphan.json'
    child = subprocess.Popen([sys.executable, '-u', '-c', child_code, str(port), str(orphan)])
    deadline = time.monotonic() + 5
    while not orphan.exists():
        if child.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError('CPU fake orphan did not start')
        time.sleep(.01)
    marker.write_text(json.dumps(info))
    print('torch.distributed.DistNetworkError: TCPStore EADDRINUSE: address already in use', flush=True)
    print('RuntimeError: Engine core initialization failed. See root cause above.', flush=True)
    os._exit(17)  # Deliberately leave a child in this attempt's owned group.
internal = socket.socket(); internal.bind(('127.0.0.1', port)); internal.listen()
marker.write_text(json.dumps(info))
expected = int(os.environ.get('CPU_TEST_BARRIER_COUNT', '1'))
deadline = time.monotonic() + 8
while len(list(root.glob('started-*.json'))) < expected:
    if time.monotonic() > deadline:
        raise RuntimeError('Replicas were not started in parallel')
    time.sleep(.01)
model = os.environ.get('CPU_TEST_RETURN_MODEL', arg('--served-model-name'))
class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def do_GET(self):
        if self.path != '/v1/models':
            self.send_error(404); return
        payload = json.dumps({'data': [{'id': model}], 'pid': os.getpid(),
                              'internal_port': port}).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Connection', 'close')
        self.end_headers(); self.wfile.write(payload)
    def log_message(self, *args):
        pass
http.server.HTTPServer(('127.0.0.1', int(arg('--port'))), Handler).serve_forever()
'''


def models(port):
    with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=1) as response:
        return json.load(response)


def process_live(pid):
    """A reparented Linux zombie holds no CPU/socket resources."""
    stat = Path(f'/proc/{pid}/stat')
    if stat.exists():
        try:
            return stat.read_text().rsplit(')', 1)[1].split()[0] != 'Z'
        except FileNotFoundError:
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@unittest.skipUnless(os.environ.get('RUN_SERVICE_PROCESS_TESTS') == '1' and os.name == 'posix',
                     'Opt-in POSIX CPU process/socket tests: RUN_SERVICE_PROCESS_TESTS=1')
class ServiceProcesses(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='zwz-cpu-service-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Keep the whole test range below the actual Linux ephemeral range.
        limit = 32768
        ephemeral = Path('/proc/sys/net/ipv4/ip_local_port_range')
        if ephemeral.exists():
            limit = min(limit, int(ephemeral.read_text().split()[0]))
        candidates = list(range(12000, limit - 128, 64))
        random.SystemRandom().shuffle(candidates)
        for base in candidates:
            try:
                _check_ports(range(base, base + 64))
            except OSError:
                continue
            self.base = base
            break
        else:
            self.fail('No isolated CPU test port range is available below ephemeral ports')
        self.events = []

    def jobs(self, count=2):
        jobs = []
        for replica in range(count):
            port = self.base + replica
            start = self.base + 8 + replica * 24
            jobs.append({
                'argv': [sys.executable, '-u', '-c', FAKE_SERVICE,
                         '--host', '127.0.0.1', '--port', str(port),
                         '--served-model-name', 'cpu-test-model',
                         '--case-dir', str(self.root), '--replica', str(replica)],
                'env': {'CPU_TEST_BARRIER_COUNT': str(count)},
                'service': {'api_port': port, 'model': 'cpu-test-model',
                            'internal_port_blocks': [
                                {'start': start + attempt * 8, 'end': start + attempt * 8 + 7}
                                for attempt in range(3)]},
            })
        return jobs

    def pool(self, jobs):
        pool = ServicePool(jobs, self.root / 'logs', readiness_timeout=12,
                           poll_interval=.02, on_event=self.events.append)
        self.addCleanup(pool.stop)
        return pool

    def assert_stopped(self, pool):
        pool.stop()
        for record in pool._owned:
            self.assertIsNotNone(record['_process'].poll())
            self.assertFalse(_group_alive(record['pid']))
            self.assertTrue(record['_cleaned'])
        for job in pool.jobs:
            with socket.socket() as probe:
                probe.settimeout(.2)
                self.assertNotEqual(probe.connect_ex(('127.0.0.1', job['service']['api_port'])), 0)
        # The fake internal sockets never accept connections, so no TIME_WAIT.
        _check_ports(range(self.base + 8, self.base + 56))

    def test_two_replicas_start_in_parallel_and_return_exact_model(self):
        pool = self.pool(self.jobs())
        records = pool.start()
        self.assertEqual(len(records), 2)
        self.assertEqual([event['event'] for event in self.events[:2]], ['started', 'started'])
        for record in records:
            reply = models(record['api_port'])
            self.assertEqual(reply['data'], [{'id': 'cpu-test-model'}])
            self.assertEqual(reply['pid'], record['pid'])
            self.assertEqual(reply['internal_port'], record['internal_port_start'])
        self.assert_stopped(pool)

    def test_failed_attempt_orphan_cleaned_healthy_replica_not_restarted(self):
        jobs = self.jobs()
        first_start = jobs[0]['service']['internal_port_blocks'][0]['start']
        jobs[0]['env']['CPU_TEST_FAIL_PORT'] = str(first_start)
        pool = self.pool(jobs)
        records = pool.start()
        self.assertEqual([record['attempt'] for record in records], [2, 1])
        started = [event for event in self.events if event['event'] == 'started']
        self.assertEqual(len(started), 3)
        healthy = [event for event in started if event['replica'] == 1]
        self.assertEqual(len(healthy), 1)
        self.assertEqual(healthy[0]['pid'], records[1]['pid'])
        self.assertEqual(models(records[1]['api_port'])['pid'], healthy[0]['pid'])
        failed = pool._owned[0]
        orphan = json.loads((self.root / 'orphan.json').read_text())
        self.assertEqual(orphan['pgid'], failed['pid'])
        self.assertFalse(process_live(orphan['pid']))
        self.assertFalse(_group_alive(failed['pid']))
        self.assertTrue(failed['_cleaned'])
        self.assertEqual(records[0]['internal_port_start'], first_start + 8)
        self.assertEqual(pool._owned[0]['argv'], pool._owned[2]['argv'])
        self.assert_stopped(pool)

    def test_existing_api_owner_not_reused_or_terminated(self):
        job = self.jobs(1)[0]
        owner = subprocess.Popen(job['argv'], env={**os.environ, **job['env'],
                                 'VLLM_PORT': str(self.base + 60)},
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                 start_new_session=True)
        def cleanup_owner():
            if owner.poll() is None:
                os.killpg(owner.pid, signal.SIGTERM)
            owner.wait(timeout=5)
        self.addCleanup(cleanup_owner)
        deadline = time.monotonic() + 8
        while True:
            try:
                before = models(job['service']['api_port']); break
            except OSError:
                if owner.poll() is not None or time.monotonic() >= deadline:
                    self.fail('CPU external test owner did not become ready')
                time.sleep(.02)
        self.assertEqual(before['pid'], owner.pid)
        pool = self.pool([job])
        with self.assertRaisesRegex(RuntimeError, 'refusing reuse or termination'):
            pool.start()
        self.assertEqual(pool._owned, [])
        self.assertIsNone(owner.poll())
        self.assertEqual(models(job['service']['api_port'])['pid'], owner.pid)
        cleanup_owner()
        self.assertFalse(_group_alive(owner.pid))

    def test_wrong_model_fails_without_retry_and_cleans_children(self):
        jobs = self.jobs(1)
        jobs[0]['env']['CPU_TEST_RETURN_MODEL'] = 'wrong-model'
        pool = self.pool(jobs)
        with self.assertRaisesRegex(RuntimeError, 'wrong served model ID'):
            pool.start()
        self.assertEqual(len(pool._owned), 1)
        self.assert_stopped(pool)


if __name__ == '__main__':
    unittest.main()
