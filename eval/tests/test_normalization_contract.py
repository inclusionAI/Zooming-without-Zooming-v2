import copy
import unittest
from unittest.mock import patch

from zwz_eval.normalization import identities, verify_rows


class NormalizationContract(unittest.TestCase):
    def test_no_prompt_gold_or_order_changes(self):
        rows = [{'query': '<image> X <image>', 'images': ['z.jpg', 'a.jpg'], 'response': 'A', 'system': 'original'},
                {'query': 'other', 'images': ['z.jpg'], 'response': 'B'}]
        expected = identities(rows)
        with patch('zwz_eval.normalization.read_json', return_value={'inputs': {'mmstar': expected}}):
            self.assertTrue(verify_rows('mmstar', rows)['matches_historical_annotations'])
            for change in ('query', 'response', 'images', 'row_order'):
                bad = copy.deepcopy(rows)
                if change == 'row_order':
                    bad.reverse()
                elif change == 'images':
                    bad[0]['images'].reverse()
                else:
                    bad[0][change] += '!'
                with self.assertRaisesRegex(ValueError, 'Historical input mismatch'):
                    verify_rows('mmstar', bad)

    def test_only_explicit_prefix_relocates_absolute_paths(self):
        rows = [{'query': 'q', 'images': ['/old/data/task/z.jpg'], 'response': '1'}]
        with self.assertRaises(ValueError):
            identities(rows)
        self.assertEqual(identities(rows, '/old/data'), identities([{'query': 'q', 'images': ['task/z.jpg'], 'response': '1'}]))
        with self.assertRaises(ValueError):
            identities(rows, '/wrong')

    def test_all_historical_contracts_present_without_actual_items(self):
        from zwz_eval.io import read_json
        from zwz_eval.protocols import RESOURCES, PLATFORM_BENCHMARKS
        data = read_json(RESOURCES / 'normalization_contracts.json')['inputs']
        self.assertEqual(set(data), set(PLATFORM_BENCHMARKS) | {'mimic-' + k for k in ('common', 'counting', 'listing', 'oddone')})
        for contract in data.values():
            self.assertNotIn('query', contract)
            self.assertNotIn('response', contract)
            self.assertNotIn('images', contract)
        self.assertEqual(data['mimic-counting']['rows'], 5000)
