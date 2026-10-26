import tempfile
import unittest
from pathlib import Path

from zwz_eval.config import identifier
from zwz_eval.io import append_jsonl, lock_contract, read_journal, repair_torn_tail, run_lock
from zwz_eval.protocols import MILE_TASKS, mml_specs


class Contracts(unittest.TestCase):
    def test_task_identity(self):
        self.assertEqual(len(set(MILE_TASKS)), 29)
        self.assertEqual(len(mml_specs()), 25)
        self.assertFalse(any('count' in x['family'] for x in mml_specs()))

    def test_path_injection(self):
        for value in ('../escape', '/tmp', '.', 'a/b', ''):
            with self.assertRaises(ValueError):
                identifier(value, 'model')

    def test_contract_and_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            with run_lock(tmp):
                lock_contract(tmp, {'protocol': 1})
                lock_contract(tmp, {'protocol': 1})
                with self.assertRaises(ValueError):
                    lock_contract(tmp, {'protocol': 2})
                with self.assertRaises(RuntimeError):
                    with run_lock(tmp):
                        pass

    def test_torn_journal_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'predictions.jsonl'
            append_jsonl(p, {'uid': 'valid'})
            with p.open('ab') as stream:
                stream.write(b'{"uid":')
            self.assertEqual(len(read_journal(p)), 1)
            repair_torn_tail(p)
            self.assertEqual(len(list(Path(tmp).glob('*.torn-*'))), 1)
            self.assertTrue(p.read_bytes().endswith(b'\n'))


if __name__ == '__main__':
    unittest.main()
