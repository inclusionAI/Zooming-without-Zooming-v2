"""Synthetic service orchestration tests, not real-model/GPU acceptance."""
import contextlib
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from zwz_eval import managed
from zwz_eval.cli import main


class ManagedTests(unittest.TestCase):
    def exercise(self, fail=None, start_fail=False, bench='memlens', phase='all'):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            config = {'model_id': 'test', 'run_id': 'run', 'output_root': directory,
                      'hardware': {'gpus': 1}, 'datasets': {bench: {}},
                      'main': {}, 'judge': {}}
            plans = {(bench, p): (config, [{'test': True}]) for p in ('infer', 'score')}
            if bench == 'mmsi':
                plans[bench, 'score'] = config, []
            class Pool:
                def __init__(self, jobs, logs, **kwargs):
                    self.role = kwargs['log_prefix']
                def start(self):
                    events.append('start-' + self.role)
                    if start_fail:raise RuntimeError('startup failed')
                def stop(self):events.append('stop-' + self.role)
            def phase_run(path, cfg, name, phase, pool):
                events.append(phase)
                if fail:raise fail
            previous = signal.getsignal(signal.SIGTERM)
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), \
                 patch.object(managed, 'load_config', return_value=config), \
                 patch.object(managed, '_plans', return_value=plans), \
                 patch.object(managed, 'ServicePool', Pool), \
                 patch.object(managed, '_phase', side_effect=phase_run), \
                 contextlib.redirect_stdout(io.StringIO()):
                try:
                    result = managed.evaluate('unused.json', bench, phase)
                except (RuntimeError, KeyboardInterrupt) as exc:
                    result = exc
            self.assertEqual(signal.getsignal(signal.SIGTERM), previous)
        return events, result

    def test_main_exits_before_judge_starts(self):
        events, report = self.exercise()
        self.assertEqual(events, ['start-main', 'infer', 'stop-main', 'start-judge', 'score', 'stop-judge'])
        self.assertTrue(report['complete'])

    def test_no_judge_for_rule_scoring(self):
        events, _ = self.exercise(bench='mmsi')
        self.assertEqual(events, ['start-main', 'infer', 'stop-main', 'score'])

    def test_infer_failure_stops_queue_and_services(self):
        events, result = self.exercise(fail=RuntimeError('inference failed'))
        self.assertIsInstance(result, RuntimeError)
        self.assertEqual(events, ['start-main', 'infer', 'stop-main'])

    def test_interrupt_cleans_services(self):
        events, result = self.exercise(fail=KeyboardInterrupt())
        self.assertIsInstance(result, KeyboardInterrupt)
        self.assertEqual(events, ['start-main', 'infer', 'stop-main'])

    def test_partial_start_failure_cleans(self):
        events, _ = self.exercise(start_fail=True)
        self.assertEqual(events, ['start-main', 'stop-main'])

    def test_score_only_does_not_launch_main(self):
        events, _ = self.exercise(phase='score')
        self.assertEqual(events, ['start-judge', 'score', 'stop-judge'])

    def test_external_mode_never_launches_pool(self):
        with patch('zwz_eval.workflow.run_suite', return_value={'complete': True}) as suite, \
             patch.object(managed, 'ServicePool') as pool, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['evaluate', '--config', 'unused', '--bench', 'memlens', '--services', 'external']), 0)
        suite.assert_called_once_with('unused', 'memlens', 'all')
        pool.assert_not_called()

    def test_native_gui_does_not_launch_duplicate_pool(self):
        config = {'datasets': {'screenspot_pro': {}}, 'hardware': {'gpus': 16}}
        with patch.object(managed, 'for_benchmark', return_value=config), patch.object(managed, 'plan') as plan:
            result = managed._plans(config, ['screenspot_pro'], 'all')
        plan.assert_not_called()
        self.assertTrue(all(not jobs for _, jobs in result.values()))

    def test_underfilled_devices_rejected(self):
        config = {'datasets': {'memlens': {}}, 'hardware': {'gpus': 16, 'tp': 1}}
        with patch.object(managed, 'for_benchmark', return_value=config), \
             patch.object(managed, 'plan', return_value={'commands': [{}]}):
            with self.assertRaisesRegex(ValueError, 'exactly gpus/tp'):
                managed._plans(config, ['memlens'], 'infer')

    @unittest.skipUnless(sys.platform == 'linux', 'Real process-group lifecycle acceptance runs on Linux')
    def test_real_child_exit_and_failure(self):
        real_popen = subprocess.Popen
        for code in (0, 3):
            children = []
            def child(argv, **kwargs):
                p = real_popen([sys.executable, '-c', f'raise SystemExit({code})'], **kwargs)
                children.append(p)
                return p
            with patch.object(managed.subprocess, 'Popen', side_effect=child):
                if code:
                    with self.assertRaisesRegex(RuntimeError, 'exit code 3'):
                        managed._phase('unused', {}, 'memlens', 'infer')
                else:
                    managed._phase('unused', {}, 'memlens', 'infer')
            self.assertIsNotNone(children[0].poll())

    @unittest.skipUnless(sys.platform == 'linux', 'Real process-group lifecycle acceptance runs on Linux')
    def test_dead_service_cancels_real_evaluation_child(self):
        real_popen = subprocess.Popen
        children = []
        def child(argv, **kwargs):
            p = real_popen([sys.executable, '-c', 'import time;time.sleep(60)'], **kwargs)
            children.append(p)
            return p
        pool = MagicMock()
        pool.check_running.side_effect = RuntimeError('ready service died')
        with patch.object(managed.subprocess, 'Popen', side_effect=child):
            with self.assertRaisesRegex(RuntimeError, 'service died'):
                managed._phase('unused', {}, 'memlens', 'infer', pool)
        self.assertIsNotNone(children[0].poll())

    def test_overlapping_gpu_allocation_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = {'model_id': 'test', 'run_id': 'run', 'output_root': directory,
                      'hardware': {'gpus': 1}, 'datasets': {'mmsi': {}}}
            plans = {('mmsi', 'score'): (config, [])}
            def nested(*args):
                with self.assertRaisesRegex(RuntimeError, 'Another process owns'):
                    managed.evaluate('unused', 'mmsi', 'score')
            with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), \
                 patch.object(managed, 'load_config', return_value=config), \
                 patch.object(managed, '_plans', return_value=plans), \
                 patch.object(managed, '_phase', side_effect=nested), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(managed.evaluate('unused', 'mmsi', 'score')['complete'])

    def test_all_plans_validate_before_startup(self):
        config = {'datasets': {'memlens': {}, 'mmsi': {}}}
        with patch.object(managed, 'load_config', return_value=config), \
             patch.object(managed, '_plans', side_effect=ValueError('bad second benchmark')), \
             patch.object(managed, 'ServicePool') as pool:
            with self.assertRaisesRegex(ValueError, 'bad second'):
                managed.evaluate('unused', 'memlens,mmsi')
        pool.assert_not_called()


if __name__ == '__main__':
    unittest.main()
