import copy
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path

from zwz_eval.config import load_config
from zwz_eval.io import atomic_json
from zwz_eval.protocols import PROTOCOLS
from test_adapters import configuration


class ConfigurationValidation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.value = configuration(self.root)
        self.path = self.root / 'config.json'

    def load(self, value):
        atomic_json(self.path, value)
        return load_config(self.path)

    def test_local_paths_are_relative_to_config_not_working_directory(self):
        self.value['identity']['model_path'] = 'models/checkpoint'
        loaded = self.load(self.value)
        self.assertEqual(loaded['identity']['model_path'], str((self.root / 'models/checkpoint').resolve()))

    def test_nested_typo_and_invalid_types_fail_closed(self):
        cases = [
            ('hardware', {'gpu': 16}),
            ('hardware', {'memory_utilization': True}),
            ('hardware', {'memory_utilization': float('nan')}),
            ('datasets', []),
            ('datasets', {'memlens': {'root': str(self.root), 'max_toknes': 512}}),
            ('datasets', {'memlens': {'root': ''}}),
            ('datasets', {'memlens': {'root': str(self.root), 'judge_profile': 'typo'}}),
            ('retries', True),
            ('main', {'model': {}, 'base_urls': ['http://127.0.0.1:8000/v1']}),
            ('output_root', []),
            ('identity', {'model_paht': '/models/typo'}),
            ('identity', {'model_revision': []}),
            ('judge_profiles', {'bad': {'endpoint': {'model': 'judge', 'base_urls': ['http://127.0.0.1:9000/v1']}, 'tokenizer_path': []}}),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                config = copy.deepcopy(self.value)
                config[key] = value
                with self.assertRaises(ValueError):
                    self.load(config)

    def test_complete_example_covers_exactly_the_requested_protocols(self):
        example = Path(__file__).resolve().parents[1] / 'examples/all.json'
        config = load_config(example)
        self.assertEqual(set(config['datasets']), set(PROTOCOLS))

    def test_module_entrypoint_propagates_failure_to_shell(self):
        project = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, '-m', 'zwz_eval', 'release-check'],
                                cwd=project, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn('"release_ready": false', result.stdout)
        result = subprocess.run([sys.executable, '-m', 'zwz_eval', 'plan',
                                 '--config', str(self.root / 'missing.json'), '--bench', 'memlens'],
                                cwd=project, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
