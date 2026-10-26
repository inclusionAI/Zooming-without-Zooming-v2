import copy
import tempfile
import unittest
from pathlib import Path

from zwz_eval.io import atomic_json, digest, lock_contract
from zwz_eval.runner import execution_contract


class ConcurrencyReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old = {'framework': {'runner.py': 'r0'},
                    'config': {'workers': 16, 'datasets': {'mme_en': {'root': 'data'}},
                               'main': {'model': 'fixed'}},
                    'protocol': {'max_tokens': 16384}, 'dataset': {'hash': 'fixed'}}
        lock_contract(self.root, self.old)
        self.original = (self.root / 'contract.json').read_bytes()
        self.new = copy.deepcopy(self.old)
        self.new['framework']['runner.py'] = 'r1'
        self.new['config']['workers'] = 256
        self.receipt = {'previous_runner_sha256': 'r0', 'runner_sha256': 'r1',
                        'request_concurrency_update': {
                            'previous_workers': 16, 'workers': 256,
                            'previous_config_sha256': digest(self.old['config']),
                            'config_sha256': digest(self.new['config'])}}

    def check(self, receipt=None, new=None):
        atomic_json(self.root / 'orchestration_update.json', receipt or self.receipt)
        return execution_contract(self.root, new or self.new)

    def test_exact_change_retains_original(self):
        self.assertEqual(self.check(), digest(self.old))
        self.assertEqual((self.root / 'contract.json').read_bytes(), self.original)

    def test_wrong_hash_missing_fields_invalid_capacity_rejected(self):
        for key, value in [('previous_config_sha256', 'wrong'),
                           ('config_sha256', 'wrong'), ('previous_workers', 8),
                           ('workers', True), ('workers', 257), ('workers', 0)]:
            receipt = copy.deepcopy(self.receipt)
            receipt['request_concurrency_update'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.check(receipt)
        receipt = copy.deepcopy(self.receipt)
        del receipt['request_concurrency_update']['config_sha256']
        with self.assertRaises(ValueError):
            self.check(receipt)

    def test_unrelated_model_protocol_data_change_rejected(self):
        for field in ('main', 'datasets'):
            new = copy.deepcopy(self.new)
            new['config'][field] = {'changed': True}
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(new=new)
        new = copy.deepcopy(self.new)
        new['protocol']['max_tokens'] = 99
        with self.assertRaises(ValueError):
            self.check(new=new)

    def test_non_mme_and_missing_receipt_rejected(self):
        with self.assertRaises(ValueError):
            execution_contract(self.root, self.new)
        self.old['config']['datasets'] = {'o3': {'root': 'data'}}
        atomic_json(self.root / 'contract.json', {'sha256': digest(self.old), 'contract': self.old})
        self.new['config']['datasets'] = self.old['config']['datasets']
        self.receipt['request_concurrency_update'].update(
            previous_config_sha256=digest(self.old['config']), config_sha256=digest(self.new['config']))
        with self.assertRaises(ValueError):
            self.check()
