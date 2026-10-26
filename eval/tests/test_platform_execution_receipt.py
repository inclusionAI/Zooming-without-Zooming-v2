"""Exact platform execution receipts never authorize data/protocol migration."""
import copy
import tempfile
import unittest
from pathlib import Path

from zwz_eval.io import atomic_json, digest, lock_contract, read_json
from zwz_eval.runner import execution_contract


class PlatformExecutionReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.files = ('general_visual.py', 'data.py', 'workflow.py', 'preflight.py')
        self.old = {
            'framework': {name: digest(['old', name]) for name in
                          ('runner.py', 'cli.py', 'scoring.py', 'serving.py', *self.files)},
            'protocol': {'max_tokens': 8192},
            'dataset': {'preparation': {'image_mode': 'materialized', 'records_sha256': 'fixed'}},
            'config': {'datasets': {'mme_en': {'root': '/original/prepared'}}, 'workers': 16},
            'runtime': {'qwen-vl-utils': 'fixed'},
            'scorer_sources': {'native.py': 'fixed'},
        }
        lock_contract(self.root, self.old)
        self.original_bytes = (self.root / 'contract.json').read_bytes()
        self.new = copy.deepcopy(self.old)
        for name in ('runner.py', 'cli.py', *self.files):
            self.new['framework'][name] = digest(['new', name])
        self.receipt = {
            'previous_runner_sha256': self.old['framework']['runner.py'],
            'runner_sha256': self.new['framework']['runner.py'],
            'previous_cli_sha256': self.old['framework']['cli.py'],
            'cli_sha256': self.new['framework']['cli.py'],
            'platform_execution_updates': {
                name: {'previous_sha256': self.old['framework'][name],
                       'sha256': self.new['framework'][name]} for name in self.files},
        }

    def assert_original_contract(self):
        self.assertEqual((self.root / 'contract.json').read_bytes(), self.original_bytes)

    def test_exact_update_keeps_original_contract_and_records_timing_change(self):
        atomic_json(self.root / 'orchestration_update.json', self.receipt)
        self.assertEqual(execution_contract(self.root, self.new), digest(self.old))
        self.assert_original_contract()
        provenance = read_json(self.root / 'execution_provenance.json')
        self.assertEqual(provenance['platform_execution_updates'], self.receipt['platform_execution_updates'])
        self.assertIn('image preparation timing', provenance['change'])
        self.assertIn('Qwen pixel policy', provenance['change'])
        self.assertNotIn('image handling,', provenance['change'])

    def test_missing_receipt_or_missing_file_is_rejected(self):
        with self.assertRaises(ValueError):
            execution_contract(self.root, self.new)
        receipt = copy.deepcopy(self.receipt)
        del receipt['platform_execution_updates']['workflow.py']
        atomic_json(self.root / 'orchestration_update.json', receipt)
        with self.assertRaises(ValueError):
            execution_contract(self.root, self.new)
        self.assert_original_contract()

    def test_wrong_empty_missing_and_extra_hash_fields_are_rejected(self):
        for key, value in [('previous_sha256', 'wrong'), ('sha256', 'wrong'),
                           ('previous_sha256', ''), ('sha256', ''),
                           ('previous_sha256', None), ('sha256', None),
                           ('allow_any', True)]:
            with self.subTest(key=key, value=value):
                receipt = copy.deepcopy(self.receipt)
                receipt['platform_execution_updates']['general_visual.py'][key] = value
                atomic_json(self.root / 'orchestration_update.json', receipt)
                with self.assertRaises(ValueError):
                    execution_contract(self.root, self.new)
        receipt = copy.deepcopy(self.receipt)
        del receipt['platform_execution_updates']['general_visual.py']['previous_sha256']
        atomic_json(self.root / 'orchestration_update.json', receipt)
        with self.assertRaises(ValueError):
            execution_contract(self.root, self.new)
        self.assert_original_contract()

    def test_unknown_file_and_non_object_updates_are_rejected(self):
        for updates in ([], None, {'scoring.py': {'previous_sha256': 'fixed', 'sha256': 'changed'}},
                        {'general_visual.py': None}):
            with self.subTest(updates=updates):
                receipt = copy.deepcopy(self.receipt)
                receipt['platform_execution_updates'] = updates
                atomic_json(self.root / 'orchestration_update.json', receipt)
                with self.assertRaises(ValueError):
                    execution_contract(self.root, self.new)
        self.assert_original_contract()

    def test_config_protocol_dataset_runtime_and_scorer_changes_are_rejected(self):
        atomic_json(self.root / 'orchestration_update.json', self.receipt)
        changes = [
            ('config', 'datasets', {'mme_en': {'root': '/new/raw'}}),
            ('config', 'workers', 32),
            ('protocol', 'max_tokens', 16384),
            ('dataset', 'preparation', {'image_mode': 'on-demand', 'records_sha256': 'fixed'}),
            ('runtime', 'qwen-vl-utils', 'changed'),
            ('scorer_sources', 'native.py', 'changed'),
            ('framework', 'scoring.py', 'changed'),
        ]
        for area, key, value in changes:
            with self.subTest(area=area, key=key):
                new = copy.deepcopy(self.new)
                new[area][key] = value
                with self.assertRaises(ValueError):
                    execution_contract(self.root, new)
        self.assert_original_contract()

    def test_platform_receipt_composes_with_exact_service_receipt(self):
        new, receipt = copy.deepcopy(self.new), copy.deepcopy(self.receipt)
        new['framework']['serving.py'] = digest('new serving')
        receipt['service_execution_updates'] = {'serving.py': {
            'previous_sha256': self.old['framework']['serving.py'],
            'sha256': new['framework']['serving.py']}}
        atomic_json(self.root / 'orchestration_update.json', receipt)
        execution_contract(self.root, new)
        self.assert_original_contract()
        self.assertEqual(read_json(self.root / 'execution_provenance.json')['service_execution_updates'],
                         receipt['service_execution_updates'])


if __name__ == '__main__':
    unittest.main()
