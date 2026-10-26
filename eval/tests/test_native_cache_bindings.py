import tempfile
import unittest
import types
from unittest.mock import patch
from pathlib import Path

from zwz_eval.execution import bind_score_cache, command_plan, execute_steps
from zwz_eval.io import read_json
from test_adapters import configuration


class NativeCacheBindings(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input = self.root / 'answers.jsonl'
        self.input.write_text('{"model_answer":"A"}\n')
        self.binding = {'path': str(self.root / 'cache_inputs.json'),
                        'inputs': [str(self.input)],
                        'existing': [{'root': str(self.root), 'pattern': 'verdict*.jsonl'}],
                        'signature': 'judge-model-and-parameters-v1'}

    def test_identical_inputs_can_resume(self):
        bind_score_cache(self.binding)
        before = read_json(self.binding['path'])
        (self.root / 'verdict1.jsonl').write_text('{"score":1}\n')
        bind_score_cache(self.binding)
        self.assertEqual(read_json(self.binding['path']), before)

    def test_changed_answer_cannot_reuse_old_judge(self):
        bind_score_cache(self.binding)
        verdict = self.root / 'verdict1.jsonl'
        verdict.write_text('{"score":1}\n')
        self.input.write_text('{"model_answer":"B"}\n')
        with self.assertRaisesRegex(ValueError, 'refusing stale verdict'):
            bind_score_cache(self.binding)
        self.assertEqual(verdict.read_text(), '{"score":1}\n')

    def test_changed_scorer_configuration_is_rejected(self):
        bind_score_cache(self.binding)
        with self.assertRaisesRegex(ValueError, 'refusing stale verdict'):
            bind_score_cache({**self.binding, 'signature': 'another-judge-model'})

    def test_unbound_existing_cache_is_not_adopted(self):
        (self.root / 'verdict1.jsonl').write_text('{"score":1}\n')
        with self.assertRaisesRegex(ValueError, 'Unbound legacy verdict cache'):
            bind_score_cache(self.binding)
        self.assertFalse(Path(self.binding['path']).exists())

    def test_missing_glob_does_not_create_a_binding(self):
        with self.assertRaises(FileNotFoundError):
            bind_score_cache({**self.binding, 'inputs': [str(self.root / 'missing/*.json')]})
        self.assertFalse(Path(self.binding['path']).exists())

    def test_all_native_judge_paths_have_input_bindings(self):
        config = configuration(self.root)
        expected = {'memlens': 4, 'milebench': 1, 'agentvista': 1}
        for bench, count in expected.items():
            jobs = command_plan(config, bench, 'score', self.root / bench)
            bindings = [j['cache_binding'] for j in jobs if 'cache_binding' in j]
            self.assertEqual(len(bindings), count, bench)
            self.assertTrue(all(b['inputs'] and b['existing'] and b['signature'] for b in bindings))

    def test_literal_brackets_in_paths_are_not_glob_syntax(self):
        root = self.root / 'experiment [1]'
        root.mkdir()
        source = root / 'answer.jsonl'
        source.write_text('answer')
        binding = {**self.binding, 'inputs': [str(source)], 'path': str(root / 'binding.json')}
        bind_score_cache(binding)
        self.assertIn(str(source.resolve()), read_json(binding['path'])['inputs_sha256'])

    def test_input_changes_during_subprocess_are_detected(self):
        job = {'argv': ['dummy-python', 'dummy-scorer'], 'cwd': str(self.root),
               'env': {}, 'unset': [], 'cache_binding': self.binding}
        def mutate(*args, **kwargs):
            self.input.write_text('modified while judging')
            return types.SimpleNamespace(returncode=0)
        with patch('zwz_eval.execution.subprocess.run', side_effect=mutate):
            with self.assertRaisesRegex(ValueError, 'refusing stale verdict'):
                execute_steps([job], self.root / 'logs')
