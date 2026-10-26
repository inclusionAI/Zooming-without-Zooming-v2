"""Transport changes cannot authorize data, model, prompt or scorer changes."""
import copy
import tempfile
import unittest
from pathlib import Path

from zwz_eval.io import atomic_json, lock_contract, read_json
from zwz_eval.runner import execution_contract


class ServiceReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old = {'framework': {'runner.py': 'r0', 'cli.py': 'c0', 'serving.py': 's0',
                                  'scoring.py': 'fixed'},
                    'protocol': {'max_tokens': 8192}, 'dataset': {'sha256': 'fixed'},
                    'config': {'hardware': {'tp': 1}}, 'scorer_sources': {'native.py': 'fixed'}}
        lock_contract(self.root, self.old)
        self.original_bytes = (self.root / 'contract.json').read_bytes()
        self.new = copy.deepcopy(self.old)
        self.new['framework'].update({'runner.py': 'r1', 'cli.py': 'c1',
                                      'serving.py': 's1', 'service_lifecycle.py': 'l1'})
        self.receipt = {'previous_runner_sha256': 'r0', 'runner_sha256': 'r1',
                        'previous_cli_sha256': 'c0', 'cli_sha256': 'c1',
                        'service_execution_updates': {
                            'serving.py': {'previous_sha256': 's0', 'sha256': 's1'},
                            'service_lifecycle.py': {'previous_sha256': None, 'sha256': 'l1'}}}

    def test_exact_update_preserves_original_contract(self):
        atomic_json(self.root / 'orchestration_update.json', self.receipt)
        execution_contract(self.root, self.new)
        self.assertEqual((self.root / 'contract.json').read_bytes(), self.original_bytes)
        self.assertEqual(read_json(self.root / 'execution_provenance.json')['service_execution_updates'],
                         self.receipt['service_execution_updates'])

    def test_missing_receipt_rejected(self):
        with self.assertRaises(ValueError):
            execution_contract(self.root, self.new)
        receipt = copy.deepcopy(self.receipt)
        del receipt['service_execution_updates']['service_lifecycle.py']
        atomic_json(self.root / 'orchestration_update.json', receipt)
        with self.assertRaises(ValueError):
            execution_contract(self.root, self.new)

    def test_wrong_or_empty_hash_rejected(self):
        for field, value in [('previous_sha256', 'wrong'), ('sha256', 'wrong'), ('sha256', '')]:
            with self.subTest(field=field, value=value):
                receipt = copy.deepcopy(self.receipt)
                receipt['service_execution_updates']['serving.py'][field] = value
                atomic_json(self.root / 'orchestration_update.json', receipt)
                with self.assertRaises(ValueError):
                    execution_contract(self.root, self.new)

    def test_unrelated_changes_rejected_even_with_receipt(self):
        atomic_json(self.root / 'orchestration_update.json', self.receipt)
        for area, key, value in [('framework', 'scoring.py', 'changed'),
                                 ('protocol', 'max_tokens', 99),
                                 ('config', 'hardware', {'tp': 2}),
                                 ('dataset', 'sha256', 'changed'),
                                 ('scorer_sources', 'native.py', 'changed')]:
            with self.subTest(area=area):
                new = copy.deepcopy(self.new)
                new[area][key] = value
                with self.assertRaises(ValueError):
                    execution_contract(self.root, new)
        receipt = copy.deepcopy(self.receipt)
        receipt['service_execution_updates']['scoring.py'] = {
            'previous_sha256': 'fixed', 'sha256': 'changed'}
        atomic_json(self.root / 'orchestration_update.json', receipt)
        with self.assertRaises(ValueError):
            execution_contract(self.root, self.new)


if __name__ == '__main__':
    unittest.main()
