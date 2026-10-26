import copy
import errno
import io
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zwz_eval.protocols import PROTOCOLS
from zwz_eval.service_lifecycle import ServicePool, _check_ports, _group_alive
from zwz_eval.serving import plan


def configuration(main_count=16, judge_count=8):
    return {'identity': {'model_path': '/model', 'judge_model_path': '/judge'},
            'hardware': {'gpus': 16, 'tp': 1, 'judge_tp': 2},
            'main': {'model': 'model', 'base_urls': [f'http://127.0.0.1:{23100+i}/v1' for i in range(main_count)]},
            'judge': {'model': 'judge', 'base_urls': [f'http://127.0.0.1:{24100+i}/v1' for i in range(judge_count)]},
            'datasets': {bench: {'root': '/data', 'image_root': '/images'} for bench in PROTOCOLS}}


class ServingPorts(unittest.TestCase):
    def setUp(self):
        # Plans use a synthetic 16-device allocation, not the test host's GPUs.
        env = patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                    if k != 'CUDA_VISIBLE_DEVICES'}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def test_16_main_8_judge_all_attempts_disjoint_and_gpu_groups_preserved(self):
        cfg = configuration()
        original = copy.deepcopy(cfg)
        ports = set()
        for role, count, tp in (('main', 16, 1), ('judge', 8, 2)):
            jobs = plan(cfg, 'spar', role)['commands']
            self.assertEqual(len(jobs), count)
            for index, job in enumerate(jobs):
                self.assertEqual(job['env']['CUDA_VISIBLE_DEVICES'], ','.join(map(str, range(index*tp, (index+1)*tp))))
                self.assertEqual(job['env']['VLLM_HOST_IP'], '127.0.0.1')
                self.assertEqual(job['env']['VLLM_LOOPBACK_IP'], '127.0.0.1')
                self.assertEqual(job['unset_env'], ['VLLM_DP_MASTER_PORT'])
                self.assertNotIn('MASTER_PORT', job['env'])
                self.assertEqual(len(job['service']['internal_port_blocks']), 3)
                for block in job['service']['internal_port_blocks']:
                    block_ports = set(range(block['start'], block['end'] + 1))
                    self.assertEqual(len(block_ports), 32)
                    self.assertFalse(ports & block_ports)
                    ports |= block_ports
                self.assertEqual(int(job['env']['VLLM_PORT']), job['service']['internal_port_blocks'][0]['start'])
        self.assertFalse(ports & (set(range(23100, 23116)) | set(range(24100, 24108))))
        self.assertEqual(cfg, original)

    def test_argv_protocol_parameters_unchanged_for_every_non_gui_bench(self):
        cfg = configuration(1, 1)
        for bench, protocol in PROTOCOLS.items():
            if bench in ('screenspot_pro', 'ui_vision'):
                continue
            for role, tp, port in (('main', 1, 23100), ('judge', 2, 24100)):
                with self.subTest(bench=bench, role=role):
                    argv = plan(cfg, bench, role)['commands'][0]['argv']
                    context = protocol.get('judge_context', protocol['context']) if role == 'judge' else protocol['context']
                    images = protocol.get('judge_max_images', protocol['max_images']) if role == 'judge' else protocol['max_images']
                    expected = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
                                '--model', '/model' if role == 'main' else '/judge',
                                '--served-model-name', 'model' if role == 'main' else 'judge',
                                '--host', '127.0.0.1', '--port', str(port),
                                '--tensor-parallel-size', str(tp), '--max-model-len', str(context),
                                '--gpu-memory-utilization', '0.9', '--limit-mm-per-prompt', '{"image":%d,"video":0}' % images]
                    thinking = protocol['thinking'] if role == 'main' else protocol['judge_thinking']
                    if thinking is not None:
                        expected += ['--default-chat-template-kwargs', json.dumps({'enable_thinking': thinking})]
                    if role == 'main' and bench == 'mmlongbench':
                        expected += ['--allowed-local-media-path', '/images']
                    self.assertEqual(argv, expected)

    def test_invalid_port_options_fail_closed(self):
        invalid = [dict(main_internal_port_base=True), dict(judge_internal_port_base='29000'),
                   dict(main_internal_port_base=1000), dict(main_internal_port_base=65530),
                   dict(judge_internal_port_base=25000), dict(internal_port_block_size=0),
                   dict(internal_port_block_size=3.5), dict(service_start_attempts=False),
                   dict(service_start_attempts=0), dict(service_start_attempts=11)]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                plan(configuration(), 'spar', **options)

    def test_both_roles_api_ports_checked_against_full_ranges(self):
        for role, port in (('main', 25100), ('judge', 25100), ('main', 29100), ('judge', 29100)):
            cfg = configuration()
            cfg[role]['base_urls'][0] = f'http://127.0.0.1:{port}/v1'
            with self.subTest(role=role, port=port), self.assertRaisesRegex(ValueError, 'overlaps'):
                plan(cfg, 'spar')

    def test_options_only_affect_execution_plan_not_config_or_model_arguments(self):
        cfg = configuration()
        before = copy.deepcopy(cfg)
        default = plan(cfg, 'spar')['commands']
        custom = plan(cfg, 'spar', main_internal_port_base=15000, judge_internal_port_base=18000,
                      internal_port_block_size=16, service_start_attempts=2)['commands']
        self.assertEqual([job['argv'] for job in default], [job['argv'] for job in custom])
        self.assertEqual(cfg, before)


class FakeProcess:
    def __init__(self, pid, returncode=None):
        self.pid = pid
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise AssertionError('Test must clean up the owned process before wait')
        return self.returncode


class ServiceLifecycle(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                    if k != 'CUDA_VISIBLE_DEVICES'}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log_dir = Path(self.tmp.name)
        self.launches = []
        self.signals = []
        self.processes = {}
        self.behaviors = []
        self.events = []
        self.patches = [patch('zwz_eval.service_lifecycle._check_ports'),
                        patch('zwz_eval.service_lifecycle._ephemeral_port_range', return_value=[32768, 60999]),
                        patch('zwz_eval.service_lifecycle.subprocess.Popen', side_effect=self.popen),
                        patch('zwz_eval.service_lifecycle._signal_owned', side_effect=self.signal),
                        patch('zwz_eval.service_lifecycle._group_alive', return_value=False),
                        patch('zwz_eval.service_lifecycle.time.sleep')]
        self.mocks = [patcher.start() for patcher in self.patches]
        for patcher in self.patches:
            self.addCleanup(patcher.stop)

    def popen(self, argv, **kwargs):
        pid = 1000 + len(self.launches)
        behavior = self.behaviors.pop(0) if self.behaviors else (None, b'')
        if isinstance(behavior, Exception):
            raise behavior
        returncode, text = behavior
        kwargs['stdout'].write(text)
        process = FakeProcess(pid, returncode)
        self.processes[pid] = process
        self.launches.append((argv, kwargs, process))
        return process

    def signal(self, pid, sig):
        self.signals.append((pid, sig))
        self.processes[pid].returncode = self.processes[pid].returncode or -sig

    def pool(self, count=2, **kwargs):
        jobs = plan(configuration(count, 1), 'spar')['commands']
        return ServicePool(jobs, self.log_dir, on_event=self.events.append,
                           poll_interval=0.001, **kwargs)

    def test_parallel_launch_before_readiness_and_environment_overrides(self):
        base = {'VLLM_PORT': '123', 'VLLM_HOST_IP': '0.0.0.0', 'VLLM_LOOPBACK_IP': 'bad',
                'VLLM_DP_MASTER_PORT': '25000', 'SECRET_TEST_KEY': 'not-for-records'}
        pool = self.pool(base_env=base)
        def ready(record):
            self.assertEqual(len(self.launches), 2)
            return True
        with patch.object(pool, '_ready', side_effect=ready):
            records = pool.start()
        self.assertEqual(len(records), 2)
        for index, (_, kwargs, _) in enumerate(self.launches):
            self.assertTrue(kwargs['start_new_session'])
            self.assertEqual(kwargs['stdin'], subprocess.DEVNULL)
            self.assertEqual(kwargs['env']['VLLM_HOST_IP'], '127.0.0.1')
            self.assertEqual(kwargs['env']['VLLM_LOOPBACK_IP'], '127.0.0.1')
            self.assertEqual(kwargs['env']['VLLM_PORT'], str(25000 + index*96))
            self.assertNotIn('VLLM_DP_MASTER_PORT', kwargs['env'])
        self.assertEqual(base['VLLM_PORT'], '123')
        self.assertEqual(base['VLLM_DP_MASTER_PORT'], '25000')
        record_text = (self.log_dir / 'service-00.command.json').read_text()
        self.assertNotIn('not-for-records', record_text)
        for key in ('argv', 'env', 'pid', 'hostname', 'attempt', 'log_offset'):
            self.assertIn(key, json.loads(record_text))
        pool.stop()

    def test_only_failed_replica_retries_after_cleaning_exited_parent_group(self):
        self.behaviors = [(1, b'torch.distributed.DistNetworkError: TCPStore EADDRINUSE address already in use\n'),
                          (None, b''), (None, b'')]
        pool = self.pool()
        with patch.object(pool, '_ready', return_value=True):
            records = pool.start()
        self.assertEqual(len(self.launches), 3)
        self.assertIn((1000, signal.SIGTERM), self.signals)
        self.assertFalse(any(pid == 1001 for pid, _ in self.signals))
        self.assertEqual([record['attempt'] for record in records], [2, 1])
        self.assertEqual(self.launches[2][1]['env']['VLLM_PORT'], '25032')
        self.assertEqual(self.launches[0][0], self.launches[2][0])
        self.assertTrue(pool._owned[0]['_cleaned'])
        pool.stop()

    def test_retry_waits_for_orphan_group_to_exit_before_relaunch(self):
        self.behaviors = [(1, b'OSError: EADDRINUSE\n'), (None, b'')]
        pool = self.pool(1)
        self.mocks[4].side_effect = [True, False, False]
        with patch.object(pool, '_ready', return_value=True):
            pool.start()
        self.assertEqual(len(self.launches), 2)
        self.assertIn((1000, signal.SIGTERM), self.signals)
        self.assertGreaterEqual(self.mocks[5].call_count, 1)
        pool.stop()

    def test_old_collision_log_does_not_retry_current_unrelated_failure(self):
        (self.log_dir / 'service-00.log').write_text('old attempt: TCPStore EADDRINUSE\n')
        self.behaviors = [(1, b'new attempt: CUDA out of memory\n')]
        pool = self.pool(1)
        with self.assertRaisesRegex(RuntimeError, 'exited'):
            pool.start()
        self.assertEqual(len(self.launches), 1)
        self.assertGreater(pool.records[0]['log_offset'], 0)

    def test_collision_text_while_process_alive_is_not_a_retry_signal(self):
        self.behaviors = [(None, b'EADDRINUSE in a previous recoverable diagnostic\n')]
        pool = self.pool(1)
        with patch.object(pool, '_ready', return_value=True):
            pool.start()
        self.assertEqual(len(self.launches), 1)
        pool.stop()

    def test_current_recoverable_collision_then_other_terminal_error_not_retried(self):
        self.behaviors = [(1, b'recoverable diagnostic: EADDRINUSE\ntorch.OutOfMemoryError: CUDA out of memory\n')]
        pool = self.pool(1)
        with self.assertRaisesRegex(RuntimeError, 'exited'):
            pool.start()
        self.assertEqual(len(self.launches), 1)

    def test_diagnostic_keyword_without_fatal_exception_is_not_retried(self):
        self.behaviors = [(1, b'recovered earlier EADDRINUSE\nservice later failed without a confirmed cause\n')]
        with self.assertRaisesRegex(RuntimeError, 'exited'):
            self.pool(1).start()
        self.assertEqual(len(self.launches), 1)

    def test_engine_core_port_root_cause_with_generic_parent_wrapper_retries(self):
        self.behaviors = [(1, b'torch.distributed.DistNetworkError: TCPStore EADDRINUSE\n'
                              b'RuntimeError: Engine core initialization failed. See root cause above.\n'), (None, b'')]
        pool = self.pool(1)
        with patch.object(pool, '_ready', return_value=True):
            pool.start()
        self.assertEqual(len(self.launches), 2)
        pool.stop()

    def test_collision_retries_are_bounded(self):
        self.behaviors = [(1, b'OSError: EADDRINUSE\n')] * 3
        pool = self.pool(1)
        with self.assertRaisesRegex(RuntimeError, 'exited'):
            pool.start()
        self.assertEqual(len(self.launches), 3)
        self.assertEqual([item[1]['env']['VLLM_PORT'] for item in self.launches], ['25000', '25032', '25064'])
        self.assertTrue(all(record['_cleaned'] for record in pool._owned))

    def test_ready_replica_never_retried_even_if_later_log_says_collision(self):
        self.behaviors = [(None, b''), (None, b'')]
        pool = self.pool()
        def ready(record):
            if record['replica'] == 1:
                self.processes[1000].returncode = 1
                pool._current[0]['_stream'].write(b'EADDRINUSE\n')
                return False
            return True
        with patch.object(pool, '_ready', side_effect=ready), self.assertRaisesRegex(RuntimeError, 'exited'):
            pool.start()
        self.assertEqual(len(self.launches), 2)

    def test_non_port_failure_does_not_retry_and_cleans_healthy_owned(self):
        self.behaviors = [(None, b''), (1, b'model checkpoint invalid\n')]
        pool = self.pool()
        with patch.object(pool, '_ready', return_value=True), self.assertRaises(RuntimeError):
            pool.start()
        self.assertEqual(len(self.launches), 2)
        self.assertTrue({1000, 1001} <= {pid for pid, _ in self.signals})

    def test_api_occupied_refuses_launch_and_does_not_signal_owner(self):
        self.mocks[0].side_effect = OSError(errno.EADDRINUSE, 'address already in use')
        with self.assertRaisesRegex(RuntimeError, 'refusing reuse or termination'):
            self.pool().start()
        self.assertFalse(self.launches)
        self.assertFalse(self.signals)

    def test_preflight_busy_internal_block_uses_next_block_only(self):
        def check(ports):
            if list(ports) == list(range(25000, 25032)):
                raise OSError(errno.EADDRINUSE, 'address already in use')
        self.mocks[0].side_effect = check
        pool = self.pool(1)
        with patch.object(pool, '_ready', return_value=True):
            records = pool.start()
        self.assertEqual(len(self.launches), 1)
        self.assertEqual(records[0]['attempt'], 2)
        self.assertEqual(self.launches[0][1]['env']['VLLM_PORT'], '25032')
        self.assertFalse(self.signals)
        pool.stop()

    def test_permission_error_not_retried(self):
        def check(ports):
            if isinstance(ports, range):
                raise OSError(errno.EACCES, 'permission denied')
        self.mocks[0].side_effect = check
        with self.assertRaises(PermissionError):
            self.pool(1).start()
        self.assertFalse(self.launches)

    def test_ephemeral_range_overlap_rejected_before_launch(self):
        self.mocks[1].return_value = [24000, 61000]
        with self.assertRaisesRegex(ValueError, 'ephemeral'):
            self.pool().start()
        self.assertFalse(self.launches)

    def test_start_timeout_not_retried_and_owned_cleaned(self):
        pool = self.pool(1, readiness_timeout=0.1)
        with patch.object(pool, '_ready', return_value=False), patch('zwz_eval.service_lifecycle.time.monotonic', side_effect=[0, 1, 1]):
            with self.assertRaises(TimeoutError):
                pool.start()
        self.assertEqual(len(self.launches), 1)
        self.assertTrue(pool._owned[0]['_cleaned'])

    def test_wrong_model_is_not_ready_and_not_retried(self):
        pool = self.pool(1)
        response = io.BytesIO(b'{"data":[{"id":"not-the-model"}]}')
        with patch('zwz_eval.service_lifecycle.urllib.request.urlopen', return_value=response), self.assertRaisesRegex(RuntimeError, 'wrong served model'):
            pool.start()
        self.assertEqual(len(self.launches), 1)

    def test_wait_detects_failure_without_restarting(self):
        pool = self.pool(1)
        with patch.object(pool, '_ready', return_value=True):
            pool.start()
        self.processes[1000].returncode = 1
        with self.assertRaisesRegex(RuntimeError, 'no post-readiness retry'):
            pool.wait()
        self.assertEqual(len(self.launches), 1)
        pool.stop()

    def test_popen_failure_cleans_previously_started_owned(self):
        self.behaviors = [(None, b''), OSError('cannot start')]
        with self.assertRaises(OSError):
            self.pool().start()
        self.assertEqual(len(self.launches), 1)
        self.assertIn((1000, signal.SIGTERM), self.signals)

    def test_dp_multi_node_or_overlapping_metadata_rejected(self):
        for mutation in ('dp', 'nodes', 'rank', 'host', 'blocks'):
            jobs = plan(configuration(2, 1), 'spar')['commands']
            if mutation == 'dp':
                jobs[0]['argv'] += ['--data-parallel-size', '2']
            elif mutation == 'nodes':
                jobs[0]['argv'] += ['--nnodes=2']
            elif mutation == 'rank':
                jobs[0]['argv'] += ['--data-parallel-rank', '1']
            elif mutation == 'host':
                jobs[0]['argv'][jobs[0]['argv'].index('--host')+1] = '0.0.0.0'
            else:
                jobs[1]['service']['internal_port_blocks'] = jobs[0]['service']['internal_port_blocks']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                ServicePool(jobs, self.log_dir)

    def test_inherited_or_job_multi_node_topology_rejected_without_rewriting(self):
        for key, value in (('VLLM_DP_SIZE', '2'), ('VLLM_DP_RANK', '1'),
                           ('VLLM_DP_RANK_LOCAL', '1'),
                           ('VLLM_DP_SIZE', 'invalid')):
            for location in ('base', 'job'):
                jobs = plan(configuration(1, 1), 'spar')['commands']
                base = {}
                if location == 'base':
                    base[key] = value
                else:
                    jobs[0]['env'][key] = value
                with self.subTest(key=key, value=value, location=location), self.assertRaisesRegex(ValueError, key):
                    ServicePool(jobs, self.log_dir, base_env=base)
        self.assertFalse(self.launches)


FAKE_SERVICE = r'''
import http.server, json, os, signal, subprocess, sys
signal.alarm(30)  # Bound test-only lifetime even if a sandbox rejects cleanup.
def argument(name):
    return sys.argv[sys.argv.index(name) + 1]
if os.environ.get('FAKE_FAIL_START') == os.environ['VLLM_PORT']:
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    print('owned orphan pid=' + str(child.pid), flush=True)
    print('torch.distributed.DistNetworkError: TCPStore EADDRINUSE address already in use', flush=True)
    sys.exit(1)
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps({'data': [{'id': argument('--served-model-name')}]}).encode())
    def log_message(self, *args):
        pass
server = http.server.HTTPServer(('127.0.0.1', int(argument('--port'))), Handler)
server.serve_forever()
'''


@unittest.skipUnless(os.environ.get('RUN_SERVICE_PROCESS_TESTS') == '1',
                     'Set RUN_SERVICE_PROCESS_TESTS=1 in an environment allowing local sockets/process groups')
class ServiceLifecycleIntegration(unittest.TestCase):
    def setUp(self):
        # These CPU fake services do not use or initialize the listed devices.
        env = patch.dict(os.environ, {k: v for k, v in os.environ.items()
                                    if k != 'CUDA_VISIBLE_DEVICES'}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    """Actual local child/HTTP tests; no vLLM, models, GPUs or external network."""
    def test_real_listener_is_not_reused_or_terminated(self):
        with tempfile.TemporaryDirectory() as tmp, socket.socket() as owner:
            owner.bind(('127.0.0.1', 0))
            owner.listen()
            jobs = plan(configuration(1, 1), 'spar')['commands']
            api_port = owner.getsockname()[1]
            jobs[0]['service']['api_port'] = api_port
            jobs[0]['argv'][jobs[0]['argv'].index('--port') + 1] = str(api_port)
            pool = ServicePool(jobs, tmp)
            with patch('zwz_eval.service_lifecycle._signal_owned') as signals:
                with self.assertRaisesRegex(RuntimeError, 'refusing reuse or termination'):
                    pool.start()
                signals.assert_not_called()
            # The test-owned listener is still reachable and has not been closed.
            with socket.create_connection(('127.0.0.1', api_port), timeout=1):
                pass

    def test_real_exited_parent_orphan_cleanup_then_retry_and_http_ready(self):
        with tempfile.TemporaryDirectory() as tmp, socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            api_port = probe.getsockname()[1]
            probe.close()
            for base in range(12000, 20000, 100):
                try:
                    _check_ports(range(base, base + 96))
                    break
                except OSError:
                    continue
            else:
                self.fail('No test-only internal port block available')
            cfg = configuration(1, 1)
            cfg['main']['base_urls'] = [f'http://127.0.0.1:{api_port}/v1']
            job = plan(cfg, 'spar', main_internal_port_base=base)['commands'][0]
            job['argv'] = [sys.executable, '-u', '-c', FAKE_SERVICE,
                           '--host', '127.0.0.1', '--port', str(api_port),
                           '--served-model-name', 'model']
            pool = ServicePool([job], tmp, base_env={**os.environ, 'FAKE_FAIL_START': str(base)},
                               readiness_timeout=10, poll_interval=0.02)
            try:
                records = pool.start()
                self.assertEqual(records[0]['attempt'], 2)
                self.assertTrue(records[0]['ready'])
                self.assertFalse(_group_alive(pool._owned[0]['pid']))
                self.assertTrue(pool._owned[0]['_cleaned'])
                self.assertTrue(Path(records[0]['log_path']).read_text().startswith('owned orphan pid='))
            finally:
                pool.stop()
            self.assertFalse(_group_alive(records[0]['pid']))


if __name__ == '__main__':
    unittest.main()
