import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from zwz_eval.execution import command_plan, execute_steps
from test_adapters import configuration


class NativeEnvironment(unittest.TestCase):
    def test_native_environment_resolves_secrets_only_at_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = {'argv': ['python', 'fixture.py'], 'cwd': tmp, 'unset': ['UNWANTED'],
                   'env': {'MAIN_API_KEY': {'from_env': 'PRIVATE_TEST_KEY', 'default': 'EMPTY'}}}
            with patch.dict(os.environ, {'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION': 'cpp',
                                         'UNWANTED': 'stale', 'PRIVATE_TEST_KEY': 'test-secret-value'}):
                with patch('zwz_eval.execution.subprocess.run',
                           return_value=types.SimpleNamespace(returncode=0)) as run:
                    execute_steps([job], root)
            effective = run.call_args.kwargs['env']
            self.assertEqual(effective['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'], 'python')
            self.assertNotIn('UNWANTED', effective)
            self.assertEqual(effective['MAIN_API_KEY'], 'test-secret-value')
            self.assertNotIn('test-secret-value', (root / 'command_plan.json').read_text())

    def test_mml_reference_explicitly_disables_adapted_prompt_flags(self):
        jobs = command_plan(configuration('/data'), 'mmlongbench', 'infer', Path('/out'))
        self.assertEqual(len(jobs), 25)
        for job in jobs:
            self.assertEqual(job['env']['MMLONGBENCH_ROBUST_OUTPUT'], '0')
            self.assertEqual(job['env']['MMLONGBENCH_VRAG_REQUIRE_NONEMPTY_ANSWER'], '0')

    def test_agent_inference_cannot_inherit_inline_verifier_or_extra_key_pool(self):
        job, = command_plan(configuration('/data'), 'agentvista', 'infer', Path('/out'))
        self.assertTrue({'VERIFIER_API_KEY', 'VERIFIER_END_POINT', 'VERIFIER_MODEL_NAME',
                         'REASONING_API_KEYS', 'REASONING_API_KEY_2', 'REASONING_EFFORT',
                         'REASONING_MAX_TOKENS'}.issubset(job['unset']))
