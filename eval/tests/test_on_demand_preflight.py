"""On-demand dependencies are checked without decoding or auditing images."""
import importlib.metadata
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zwz_eval.io import atomic_json
from zwz_eval.preflight import doctor, requirements
from test_adapters import configuration


class OnDemandPreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = {'root': str(self.root / 'data')}

    def preparation(self, value):
        atomic_json(Path(self.data['root']) / 'preparation.json', value)

    def test_on_demand_inference_requires_qwen_and_torch(self):
        self.preparation({'image_mode': 'on-demand'})
        for bench in ('mme_en', 'mme_cn', 'mmstar'):
            for phase in ('infer', 'all'):
                with self.subTest(bench=bench, phase=phase):
                    found = requirements(bench, phase, data=self.data)
                    self.assertTrue({'qwen-vl-utils', 'torch', 'Pillow'} <= found)

    def test_materialized_and_legacy_inference_do_not_add_preprocessor(self):
        for preparation in ({'image_mode': 'materialized', 'assets': {}}, {'assets': {}}):
            with self.subTest(preparation=preparation):
                self.preparation(preparation)
                found = requirements('mme_en', 'infer', data=self.data)
                self.assertNotIn('qwen-vl-utils', found)
                self.assertNotIn('torch', found)
        self.assertNotIn('qwen-vl-utils', requirements('mme_en', 'infer'))

    def test_score_and_unrelated_bench_do_not_read_preparation(self):
        self.preparation({'image_mode': 'on-demand'})
        with patch('zwz_eval.preflight.read_json', side_effect=AssertionError('unexpected manifest read')):
            self.assertNotIn('qwen-vl-utils', requirements('mme_en', 'score', data=self.data))
            self.assertNotIn('qwen-vl-utils', requirements('mmsi', 'infer', data=self.data))

    def test_unknown_mode_fails_closed(self):
        self.preparation({'image_mode': 'raw-passthrough'})
        with self.assertRaisesRegex(ValueError, 'Unknown platform'):
            requirements('mme_en', 'infer', data=self.data)

    def test_doctor_reports_missing_qwen_without_image_reads_or_api_calls(self):
        self.preparation({'image_mode': 'on-demand'})
        config = configuration(self.root / 'outputs')
        config['datasets'] = {'mme_en': self.data}
        config['identity']['judge_family'] = 'qwen3.5-35b-a3b'
        config_path = self.root / 'config.json'
        atomic_json(config_path, config)
        def version(name):
            if name == 'qwen-vl-utils':
                raise importlib.metadata.PackageNotFoundError(name)
            return 'synthetic'
        with patch('zwz_eval.preflight.importlib.metadata.version', side_effect=version), \
             patch('zwz_eval.general_visual.tokenizer_manifest', return_value={}), \
             patch('zwz_eval.general_visual.preprocess') as preprocess, \
             patch('zwz_eval.data.audit') as audit, \
             patch('zwz_eval.preflight.CompatibleAPI.health') as health, \
             patch.object(Path, 'read_bytes', side_effect=AssertionError('unexpected binary read')):
            report = doctor(config_path)
        self.assertFalse(report['checks_passed'])
        item = report['results'][0]
        self.assertIn('qwen-vl-utils', item['dependencies'])
        self.assertTrue(any('Missing runtime distribution: qwen-vl-utils' in error for error in item['errors']))
        preprocess.assert_not_called()
        audit.assert_not_called()
        health.assert_not_called()


if __name__ == '__main__':
    unittest.main()
