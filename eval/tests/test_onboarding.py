import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zwz_eval.cli import main
from zwz_eval.config import load_config
from zwz_eval.io import atomic_json, read_json
from zwz_eval.onboarding import initialize
from zwz_eval.preflight import doctor, inspect_checkpoint, requirements
from zwz_eval.protocols import PROTOCOLS, RESOURCES
from test_adapters import configuration


class Onboarding(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def test_installed_templates_match_reviewed_examples(self):
        root = Path(__file__).resolve().parents[1]
        for packaged, example in (('config.json', 'all.json'), ('datasets.json', 'datasets.json')):
            self.assertEqual(read_json(RESOURCES / 'templates' / packaged), read_json(root / 'examples' / example))

    def test_init_all_25_and_subset_with_only_three_configuration_files(self):
        report = initialize(self.root / 'all')
        config = load_config(report['config'])
        self.assertEqual(set(config['datasets']), set(PROTOCOLS))
        self.assertEqual(set(read_json(report['manifest'])['datasets']), set(PROTOCOLS))
        self.assertEqual({entry.name for entry in Path(report['directory']).iterdir()},
                         {'model.json', 'datasets.json', 'README.md'})
        self.assertFalse((Path(report['directory']) / 'data').exists())
        report = initialize(self.root / 'subset', 'memlens,mmlongbench', model_id='custom-model',
                            model_path=self.root / 'my weights', data_root=self.root / 'storage',
                            output_root=self.root / 'results')
        config = load_config(report['config'])
        manifest = read_json(report['manifest'])
        self.assertEqual(list(config['datasets']), ['memlens', 'mmlongbench'])
        self.assertEqual(config['main']['model'], 'custom-model')
        self.assertEqual(config['identity']['model_path'], str(self.root / 'my weights'))
        self.assertEqual(manifest['datasets']['memlens']['source'], str(self.root / 'storage/downloads/memlens'))
        self.assertEqual(config['output_root'], str(self.root / 'results'))
        self.assertNotIn('judge_profiles', config)

    def test_existing_target_symlink_and_invalid_selection_are_not_overwritten(self):
        destination = self.root / 'existing'
        destination.mkdir()
        (destination / 'keep').write_text('keep')
        with self.assertRaises(ValueError):
            initialize(destination)
        dangling = self.root / 'dangling'
        dangling.symlink_to(self.root / 'untouched')
        with self.assertRaises(ValueError):
            initialize(dangling)
        self.assertFalse((self.root / 'untouched').exists())
        with self.assertRaises(ValueError):
            initialize(self.root / 'bad', 'memlens,typo')
        self.assertFalse((self.root / 'bad').exists())
        self.assertEqual((destination / 'keep').read_text(), 'keep')

    def fixture(self, bench='mmsi'):
        data = self.root / 'data'
        model = self.root / 'checkpoint'
        data.mkdir(exist_ok=True)
        model.mkdir(exist_ok=True)
        atomic_json(model / 'config.json', {'model_type': 'qwen'})
        (model / 'chat_template.jinja').write_text('native template')
        value = configuration(self.root / 'outputs')
        value['datasets'] = {bench: {'root': str(data)}}
        value['identity'] = {'model_path': str(model), 'model_revision': 'local-checkpoint-1',
                             'chat_template_sha256': hashlib.sha256(b'native template').hexdigest()}
        config = self.root / 'config.json'
        atomic_json(config, value)
        return config

    def test_doctor_is_offline_read_only_unless_requested(self):
        config = self.fixture()
        before = {str(entry.relative_to(self.root)) for entry in self.root.rglob('*')}
        with patch('zwz_eval.preflight.importlib.metadata.version', return_value='synthetic'), \
             patch('zwz_eval.preflight.CompatibleAPI.health') as health, \
             patch('zwz_eval.data.audit') as audit:
            report = doctor(config)
        self.assertTrue(report['checks_passed'])
        self.assertFalse(report['inference_started'])
        self.assertFalse(report['gpu_runtime_tested'])
        self.assertEqual(before, {str(entry.relative_to(self.root)) for entry in self.root.rglob('*')})
        health.assert_not_called()
        audit.assert_not_called()

    def test_doctor_data_audit_and_endpoint_probes_follow_phase(self):
        config = self.fixture('memlens')
        with patch('zwz_eval.preflight.importlib.metadata.version', return_value='synthetic'), \
             patch('zwz_eval.preflight.CompatibleAPI.health', return_value=True) as health, \
             patch('zwz_eval.data.audit', return_value={'manifest_sha256': 'synthetic'}) as audit:
            report = doctor(config, phase='infer', audit_data=True, check_endpoints=True)
        self.assertTrue(report['checks_passed'])
        self.assertEqual(set(report['results'][0]['services']), {'main'})
        self.assertEqual(health.call_count, 1)
        audit.assert_called_once()
        with patch('zwz_eval.preflight.importlib.metadata.version', return_value='synthetic'), \
             patch('zwz_eval.preflight.CompatibleAPI.health', return_value=True) as health:
            report = doctor(config, phase='score', check_endpoints=True)
        self.assertTrue(report['checks_passed'])
        self.assertEqual(set(report['results'][0]['services']), {'judge'})
        self.assertEqual(health.call_count, 1)

    def test_doctor_catches_missing_data_template_mismatch_and_service_failure(self):
        config = self.fixture()
        (self.root / 'checkpoint/chat_template.jinja').write_text('wrong template')
        (self.root / 'data').rmdir()
        with patch('zwz_eval.preflight.importlib.metadata.version', return_value='synthetic'), \
             patch('zwz_eval.preflight.CompatibleAPI.health', side_effect=RuntimeError('HTTP 401 from model service')):
            report = doctor(config, check_endpoints=True)
        self.assertFalse(report['checks_passed'])
        errors = '\n'.join(report['results'][0]['errors'])
        self.assertIn('missing or not a directory', errors)
        self.assertIn('chat-template hash differs', errors)
        self.assertIn('HTTP 401', errors)

    def test_agent_tool_checks_are_opt_in_and_block_failed_live_access(self):
        config = self.fixture('agentvista')
        with patch('zwz_eval.preflight.importlib.metadata.version', return_value='synthetic'), \
             patch('zwz_eval.agent_tools.check_live_tools', return_value={'checks_passed': False}) as tools:
            self.assertTrue(doctor(config)['checks_passed'])
            tools.assert_not_called()
            report = doctor(config, check_tools=True)
            self.assertFalse(report['checks_passed'])
            self.assertIn('free tools are unavailable', '\n'.join(report['results'][0]['errors']))
            tools.assert_called_once()
            tools.reset_mock()
            doctor(config, phase='score', check_tools=True)
            tools.assert_not_called()

    def test_native_templates_are_hashed_not_imported(self):
        config = self.fixture()
        checkpoint = self.root / 'checkpoint'
        (checkpoint / 'chat_template.jinja').unlink()
        atomic_json(checkpoint / 'tokenizer_config.json', {'chat_template': 'native embedded'})
        info = inspect_checkpoint(checkpoint)
        self.assertEqual(info['chat_template_sha256'], hashlib.sha256(b'native embedded').hexdigest())
        atomic_json(checkpoint / 'tokenizer_config.json', [])
        with self.assertRaises(ValueError):
            inspect_checkpoint(checkpoint)

    def test_missing_dependency_and_cli_error_return_nonzero(self):
        import importlib.metadata
        config = self.fixture()
        with patch('zwz_eval.preflight.importlib.metadata.version', side_effect=importlib.metadata.PackageNotFoundError), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['doctor', '--config', str(config)]), 2)
        self.assertIn('torch', requirements('mmlongbench', 'infer'))
        self.assertNotIn('torch', requirements('mmlongbench', 'score'))
        self.assertIn('transformers', requirements('mmstar', 'score'))
        self.assertEqual(requirements('mmsi', 'score'), set())

    def test_shared_formatter_imports_are_checked_before_inference(self):
        import importlib.metadata
        for bench, phase in (('milebench', 'infer'), ('memlens', 'infer'), ('mimic', 'score')):
            with self.subTest(benchmark=bench, phase=phase):
                config = self.fixture(bench)
                if bench == 'mimic':
                    value = read_json(config)
                    value['datasets'][bench]['mathruler_mode'] = 'disabled'
                    atomic_json(config, value)
                def version(name):
                    if name == 'openai':
                        raise importlib.metadata.PackageNotFoundError(name)
                    return 'synthetic'
                with patch('zwz_eval.preflight.importlib.metadata.version', side_effect=version):
                    report = doctor(config, phase=phase)
                self.assertFalse(report['checks_passed'])
                self.assertIn('openai', report['results'][0]['dependencies'])
                self.assertIn('tqdm', report['results'][0]['dependencies'])

    def test_doctor_requires_explicit_mimic_math_policy_and_matching_version(self):
        config = self.fixture('mimic')
        with patch('zwz_eval.preflight.importlib.metadata.version', return_value='synthetic'):
            report = doctor(config)
        self.assertFalse(report['checks_passed'])
        self.assertIn('mathruler_mode', '\n'.join(report['results'][0]['errors']))
        value = read_json(config)
        value['datasets']['mimic'].update(mathruler_mode='required', mathruler_version='expected-reference')
        atomic_json(config, value)
        with patch('zwz_eval.preflight.importlib.metadata.version', return_value='different-version'):
            report = doctor(config)
        self.assertFalse(report['checks_passed'])
        self.assertIn('mathruler version differs', '\n'.join(report['results'][0]['errors']))
