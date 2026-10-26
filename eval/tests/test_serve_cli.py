import io
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

from zwz_eval.cli import main


class ServeCliTests(unittest.TestCase):
    def run_cli(self, command, pool=None):
        config = {'datasets': {'mmsi': {}}, 'output_root': '/tmp/out',
                  'model_id': 'model', 'run_id': 'test'}
        with tempfile.TemporaryDirectory() as logs, \
                patch('zwz_eval.cli.load_config', return_value=config), \
                patch('zwz_eval.cli.for_benchmark', return_value=config), \
                patch('zwz_eval.serving.plan', return_value={'commands': []}) as plan, \
                patch('zwz_eval.service_lifecycle.ServicePool', return_value=pool) as factory, \
                redirect_stdout(io.StringIO()):
            args = [command, '--config', 'unused', '--bench', 'mmsi']
            if command == 'serve':
                args += ['--log-dir', logs]
            result = main(args)
        return result, plan, factory

    def test_plan_stays_read_only(self):
        result, plan, factory = self.run_cli('serve-plan')
        self.assertEqual(result, 0)
        factory.assert_not_called()
        self.assertEqual(plan.call_args.kwargs['service_start_attempts'], 3)

    def test_explicit_serve_starts_and_cleans_up(self):
        pool = MagicMock()
        pool.start.return_value = []
        result, plan, factory = self.run_cli('serve', pool)
        self.assertEqual(result, 0)
        pool.start.assert_called_once()
        pool.wait.assert_called_once()
        pool.stop.assert_called_once()
        self.assertEqual(factory.call_args.kwargs['max_start_retries'], 2)

    def test_interrupt_always_cleans_owned_services(self):
        pool = MagicMock()
        pool.start.side_effect = KeyboardInterrupt
        result, _, _ = self.run_cli('serve', pool)
        self.assertEqual(result, 130)
        pool.stop.assert_called_once()
        pool.wait.assert_not_called()

    def test_start_failure_cleans_owned_services(self):
        pool = MagicMock()
        pool.start.side_effect = RuntimeError('synthetic start failure')
        result, _, _ = self.run_cli('serve', pool)
        self.assertEqual(result, 2)
        pool.stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
