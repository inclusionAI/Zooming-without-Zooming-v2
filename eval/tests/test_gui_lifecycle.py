"""CPU-only regressions for native GUI worker/engine ownership."""
import ast
import multiprocessing as mp
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / 'zwz_eval/resources/vendor/gui/scripts'
sys.path.insert(0, str(SCRIPTS))
from replica_lifecycle import run_replicas


def fixture_initialize(group, args):
    global fixture_group, fixture_args
    fixture_group, fixture_args = group, args
    if args['mode'] in ('orphan', 'init_failure'):
        child = subprocess.Popen([sys.executable, '-c',
            'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(120)'])
        (Path(args['root']) / f'child-{group[0]}.pid').write_text(str(child.pid))
        # Let the child's SIGTERM handler install before the parent cleans up.
        time.sleep(.1)
    if args['mode'] == 'init_failure':
        raise RuntimeError('synthetic model initialization failure')
    if args['mode'] == 'init_hang':
        time.sleep(120)


def fixture_infer(sample):
    if sample == 'crash':
        os._exit(17)
    if sample == 'hang':
        time.sleep(120)
    return {'sample': sample, 'group': fixture_group}


def fixture_close():
    if 'fixture_args' in globals():
        (Path(fixture_args['root']) / f'closed-{fixture_group[0]}').write_text('closed')
        if fixture_args['mode'] == 'close_failure':
            raise RuntimeError('synthetic shutdown failure')


def process_active(pid):
    probe = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)],
                           text=True, capture_output=True)
    return bool(probe.stdout.strip()) and not probe.stdout.strip().startswith('Z')


def fixture_supervisor(root):
    list(run_replicas(['hang'], [[0]], fixture_initialize, fixture_infer,
        fixture_close, {'root': root, 'mode': 'orphan'}, startup_timeout=10,
        task_timeout=120, shutdown_timeout=.3))


@unittest.skipUnless(hasattr(os, 'setsid'), 'native GUI process ownership requires POSIX')
class GUILifecycle(unittest.TestCase):
    def run_fixture(self, root, samples, mode='normal', groups=None, **kwargs):
        return list(run_replicas(samples, groups or [[0], [1]], fixture_initialize,
            fixture_infer, fixture_close, {'root': str(root), 'mode': mode},
            startup_timeout=kwargs.get('startup_timeout', 10),
            task_timeout=kwargs.get('task_timeout', 5), shutdown_timeout=.3))

    def assert_children_stopped(self, root):
        files = list(Path(root).glob('child-*.pid'))
        self.assertTrue(files)
        for path in files:
            self.assertFalse(process_active(int(path.read_text())), str(path))

    def test_success_keeps_every_sample_once_and_closes_each_replica(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = self.run_fixture(tmp, list(range(20)), groups=[[2, 3], [6, 7]])
            self.assertEqual(sorted(row['sample'] for row in results), list(range(20)))
            self.assertTrue(all(row['group'] in ([2, 3], [6, 7]) for row in results))
            self.assertTrue((Path(tmp) / 'closed-2').exists())
            self.assertTrue((Path(tmp) / 'closed-6').exists())

    def test_success_removes_engine_descendants_but_leaves_unrelated_process(self):
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])
        try:
            with tempfile.TemporaryDirectory() as tmp:
                self.run_fixture(tmp, [0, 1], mode='orphan')
                self.assert_children_stopped(tmp)
                self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_initializer_failure_propagates_and_does_not_respawn_or_hang(self):
        with tempfile.TemporaryDirectory() as tmp:
            started = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, 'synthetic model initialization failure'):
                self.run_fixture(tmp, [0], mode='init_failure')
            self.assertLess(time.monotonic() - started, 10)
            self.assert_children_stopped(tmp)

    def test_initializer_hang_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(TimeoutError, 'model initialization'):
                self.run_fixture(tmp, [0], mode='init_hang', startup_timeout=.5)

    def test_worker_crash_propagates_without_resampling(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'worker exited'):
                self.run_fixture(tmp, ['crash'], groups=[[0]])

    def test_inference_hang_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(TimeoutError, 'inference'):
                self.run_fixture(tmp, ['hang'], groups=[[0]], task_timeout=.3)

    def test_cleanup_error_cannot_be_reported_as_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'synthetic shutdown failure'):
                self.run_fixture(tmp, [0], mode='close_failure', groups=[[0]])

    def test_parent_sigterm_cleans_its_engine_descendants(self):
        with tempfile.TemporaryDirectory() as tmp:
            supervisor = mp.get_context('spawn').Process(target=fixture_supervisor, args=(tmp,))
            supervisor.start()
            try:
                deadline = time.monotonic() + 10
                while not (Path(tmp) / 'child-0.pid').exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertTrue((Path(tmp) / 'child-0.pid').exists())
                os.kill(supervisor.pid, signal.SIGTERM)
                supervisor.join(timeout=10)
                self.assertEqual(supervisor.exitcode, 128 + signal.SIGTERM)
                self.assert_children_stopped(tmp)
            finally:
                if supervisor.is_alive():
                    supervisor.kill()
                supervisor.join(timeout=5)

    def test_model_close_supports_v1_engine_and_is_idempotent(self):
        path = SCRIPTS.parent / 'official/screenspot-pro/models/qwen3_5.py'
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Qwen3_5_VLLM_Model')
        close = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'close')
        namespace = {}
        exec(compile(ast.Module(body=[close], type_ignores=[]), str(path), 'exec'), namespace)
        calls = []
        model = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda: calls.append(1))))
        replica = SimpleNamespace(model=model)
        namespace['close'](replica)
        namespace['close'](replica)
        self.assertEqual(calls, [1])
        self.assertIsNone(replica.model)
