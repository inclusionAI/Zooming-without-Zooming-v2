"""Phase validation must not hash all image bytes or weaken resume contracts."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from zwz_eval.io import atomic_json, lock_contract, read_json
from zwz_eval.runner import execution_manifest, execution_contract


class PhaseValidationTests(unittest.TestCase):
    def test_annotation_only_and_preserve_previous_full_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            light = {'sets': {'all': {'rows': 2, 'sha256': 'ann'}},
                     'images_checked': 0, 'image_sha256': {}, 'manifest_sha256': 'annotation-only-derived'}
            full = {**light, 'images_checked': 2, 'image_sha256': {'a': 'hash'}, 'manifest_sha256': 'full-derived'}
            atomic_json(root / 'contract.json', {'contract': {'dataset': full}})
            with patch('zwz_eval.runner.audit', return_value=light) as audit:
                self.assertEqual(execution_manifest({'datasets': {'erqa': {}}}, 'erqa', root), full)
                audit.assert_called_once_with('erqa', {}, verify_images=False)
            changed = copy.deepcopy(light)
            changed['sets']['all']['sha256'] = 'changed'
            with patch('zwz_eval.runner.audit', return_value=changed):
                with self.assertRaises(ValueError):
                    execution_manifest({'datasets': {'erqa': {}}}, 'erqa', root)

    def test_only_explicit_runner_update_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = {'framework': {'runner.py': 'old', 'scoring.py': 'same'},
                        'protocol': {'max_tokens': 4096}}
            lock_contract(root, original)
            saved = (root / 'contract.json').read_bytes()
            new = copy.deepcopy(original)
            new['framework']['runner.py'] = 'new'
            with self.assertRaises(ValueError):
                execution_contract(root, new)

            atomic_json(root / 'orchestration_update.json', {
                'previous_runner_sha256': 'old', 'runner_sha256': 'new'})
            execution_contract(root, new)
            self.assertEqual((root / 'contract.json').read_bytes(), saved)
            self.assertEqual(read_json(root / 'execution_provenance.json')['runner_sha256'], 'new')
            new['protocol']['max_tokens'] = 100
            with self.assertRaises(ValueError):
                execution_contract(root, new)
            new = copy.deepcopy(original)
            new['framework']['runner.py'] = 'new'
            new['framework']['scoring.py'] = 'changed'
            with self.assertRaises(ValueError):
                execution_contract(root, new)

    def test_cli_audits_only_for_explicit_audit_command(self):
        import ast
        from zwz_eval import cli
        tree = ast.parse(Path(cli.__file__).read_text())
        audits = [n for n in ast.walk(tree) if isinstance(n, ast.If)
                  and ast.unparse(n.test) == "args.command == 'audit'"]
        self.assertEqual(len(audits), 1)
        self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                            and n.func.id == 'audit' for n in ast.walk(audits[0])))

    def test_exact_gui_execution_receipt_preserves_original_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = 'vendor/gui/scripts/eval_grounding_topology.py'
            helper = 'vendor/gui/scripts/replica_lifecycle.py'
            original = {'framework': {'runner.py': 'old'},
                        'scorer_sources': {source: 'before', 'scorer.py': 'same'},
                        'protocol': {'max_tokens': 8192}}
            lock_contract(root, original)
            saved = (root / 'contract.json').read_bytes()
            new = copy.deepcopy(original)
            new['framework']['runner.py'] = 'new'
            new['scorer_sources'].update({source: 'after', helper: 'new-helper'})
            permission = {'previous_runner_sha256': 'old', 'runner_sha256': 'new',
                          'gui_execution_updates': {
                              source: {'previous_sha256': 'before', 'sha256': 'after'},
                              helper: {'previous_sha256': None, 'sha256': 'new-helper'}}}
            atomic_json(root / 'orchestration_update.json', permission)
            execution_contract(root, new)
            self.assertEqual((root / 'contract.json').read_bytes(), saved)
            for area, name, value in [('scorer_sources', 'scorer.py', 'changed'),
                                      ('protocol', 'max_tokens', 100)]:
                bad = copy.deepcopy(new)
                bad[area][name] = value
                with self.assertRaises(ValueError):
                    execution_contract(root, bad)
            permission['gui_execution_updates']['scorer.py'] = {
                'previous_sha256': 'same', 'sha256': 'changed'}
            atomic_json(root / 'orchestration_update.json', permission)
            with self.assertRaises(ValueError):
                execution_contract(root, new)



if __name__ == '__main__':
    unittest.main()
