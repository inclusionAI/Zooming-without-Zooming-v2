"""Public model selection must be explicit without touching scoring semantics."""
import ast
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from zwz_eval.execution import command_plan


SOURCE = (Path(__file__).resolve().parents[1] /
          'zwz_eval/resources/vendor/milebench/score.py')


def selector():
    # Compile only the environment selector, not imports/cache writes/API clients.
    tree = ast.parse(SOURCE.read_text())
    selected = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == 'configured_models')
    namespace = {'os': os}
    exec(compile(ast.Module(body=[selected], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['configured_models']


class MileBenchModelSelection(unittest.TestCase):
    def test_missing_or_blank_selection_fails_without_fallback(self):
        for value in (None, '', ' \n\t '):
            with patch.dict(os.environ, {}, clear=True):
                if value is not None:
                    os.environ['MILEBENCH_MODELS'] = value
                with self.assertRaisesRegex(ValueError, 'Set MILEBENCH_MODELS'):
                    selector()()

    def test_explicit_model_order_is_preserved(self):
        with patch.dict(os.environ, {'MILEBENCH_MODELS': ' model-b\nmodel-a '}):
            self.assertEqual(selector()(), ['model-b', 'model-a'])

    def test_public_execution_passes_only_requested_model(self):
        config = {'model_id': 'user-model', 'workers': 16,
                  'datasets': {'milebench': {'root': '/example/dataset'}},
                  'judge': {'model': 'judge', 'base_urls': ['http://127.0.0.1:9000/v1'],
                            'api_key_env': 'JUDGE_API_KEY'}}
        jobs = command_plan(config, 'milebench', 'score', '/example/output')
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]['env']['MILEBENCH_MODELS'], 'user-model')
        self.assertEqual(jobs[0]['env']['ZWZ_STRICT_ERRORS'], '1')
        with patch.dict(os.environ, {'MILEBENCH_MODELS': jobs[0]['env']['MILEBENCH_MODELS']}):
            self.assertEqual(selector()(), ['user-model'])

    def test_no_named_model_roster_remains(self):
        tree = ast.parse(SOURCE.read_text())
        assignments = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                       for target in node.targets if isinstance(target, ast.Name)}
        self.assertNotIn('_DEFAULT_ALL_MODELS', assignments)
        self.assertIsInstance(assignments['ALL_MODELS'], ast.Call)
        self.assertEqual(assignments['ALL_MODELS'].func.id, 'configured_models')


if __name__ == '__main__':
    unittest.main()
