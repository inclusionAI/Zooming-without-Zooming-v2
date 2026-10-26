"""One-command public entrypoints tested together, using synthetic data only."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from zwz_eval.general_visual import EMPTY_THINK
from zwz_eval.io import atomic_json, read_json
from zwz_eval.normalization import fingerprint, identities
from zwz_eval.protocols import PLATFORM_COUNTS
from zwz_eval.raw_platform import MC_SUFFIX, SYSTEM
from zwz_eval.workflow import prepare_suite, run_suite


class OneClickPipeline(unittest.TestCase):
    def test_raw_conversion_config_two_benches_infer_score_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            for name, color in [('first.png', 'red'), ('second.png', 'blue')]:
                Image.new('RGB', (8, 8), color).save(root / name)
            expected = {
                'mmstar': [
                    {'query': '<image>Second? Select from the following choices.\n(A) x\n(B) y',
                     'response': '(A)', 'system': SYSTEM, 'images': ['2.jpg']},
                    {'query': '<image>First? Select from the following choices.\n(A) x\n(B) y',
                     'response': '(B)', 'system': SYSTEM, 'images': ['1.jpg']}],
                'vstar': [
                    {'query': '<image>First?\n(A) x\n(B) y\n' + MC_SUFFIX,
                     'response': 'A', 'system': SYSTEM, 'images': ['first.png']},
                    {'query': '<image>Second?\n(A) x\n(B) y\n' + MC_SUFFIX,
                     'response': 'B', 'system': SYSTEM, 'images': ['second.png']}],
            }
            atomic_json(root / 'mmstar.json', [
                {'index': 1, 'question': 'First? Options: A: x, B: y', 'answer': 'B', 'image': 'first.png'},
                {'index': 2, 'question': 'Second? Options: A: x, B: y', 'answer': 'A', 'image': 'second.png'}])
            atomic_json(root / 'vstar.json', [
                {'text': 'First?\n(A) x\n(B) y\n' + MC_SUFFIX, 'label': 'A', 'image': 'first.png'},
                {'text': 'Second?\n(A) x\n(B) y\n' + MC_SUFFIX, 'label': 'B', 'image': 'second.png'}])
            recipes = {bench: {'schema': 1, 'benchmark': bench, 'order': [
                [fingerprint({k: v for k, v in row.items() if k != 'images'}), row['images']]
                for row in records]} for bench, records in expected.items()}
            contract = {'inputs': {bench: identities(records) for bench, records in expected.items()}}
            atomic_json(root / 'datasets.json', {'output_root': './prepared', 'datasets': {
                bench: {'source': './' + bench + '.json', 'options': {'judge_profile': 'platform'}}
                for bench in expected}})
            endpoint = {'model': 'synthetic-model', 'base_urls': ['http://127.0.0.1:1/v1']}
            atomic_json(root / 'template.json', {'model_id': 'synthetic-model', 'run_id': 'synthetic',
                'output_root': './outputs', 'main': endpoint, 'judge': endpoint, 'workers': 1,
                'judge_profiles': {'platform': {'endpoint': endpoint, 'family': 'qwen3.5-35b-a3b'}},
                'datasets': {bench: {'root': './not-yet-prepared'} for bench in expected}})
            def pixels(paths):
                return [Image.open(path).convert('RGB') for path in paths]
            with patch.dict(PLATFORM_COUNTS, {'mmstar': 2, 'vstar': 2}), \
                 patch('zwz_eval.raw_platform.load_recipe', side_effect=lambda name: recipes[name]), \
                 patch('zwz_eval.normalization.read_json', return_value=contract), \
                 patch('zwz_eval.general_visual.preprocess', side_effect=pixels) as preprocess:
                prepared = prepare_suite(root / 'datasets.json', config_path=root / 'template.json',
                                         output_config=root / 'local.json')
                preprocess.assert_not_called()
                for data in prepared['datasets'].values():
                    self.assertEqual(read_json(Path(data['root']) / 'preparation.json')['image_mode'], 'on-demand')
                calls, judgments = [], []
                class Stub:
                    def __init__(self, *args): pass
                    def health(self): return True
                    def complete(self, messages, **kwargs):
                        calls.append((messages, kwargs))
                        return {'choices': [{'message': {'content': '<answer>A</answer>'}, 'finish_reason': 'stop'}]}
                    def complete_prompt(self, prompt, **kwargs):
                        judgments.append((prompt, kwargs))
                        return {'choices': [{'text': 'Yes'}]}
                with patch('zwz_eval.runner.CompatibleAPI', Stub), \
                     patch('zwz_eval.spatial.CompatibleAPI', Stub), \
                     patch('zwz_eval.general_visual.CompatibleAPI', Stub), \
                     patch('zwz_eval.general_visual.tokenizer_manifest', return_value={'fixture': 'tokenizer'}), \
                     patch('zwz_eval.general_visual.NativeJudgeRenderer', return_value=lambda text: text + EMPTY_THINK), \
                     contextlib.redirect_stdout(io.StringIO()):
                    result = run_suite(root / 'local.json', 'mmstar,vstar', 'all')
                    self.assertTrue(result['complete'])
                    self.assertEqual([r['benchmark'] for r in result['results']], ['mmstar', 'vstar'])
                    counts = len(calls), len(judgments)
                    self.assertTrue(run_suite(root / 'local.json', 'mmstar,vstar', 'all')['complete'])
                    self.assertEqual((len(calls), len(judgments)), counts)
                self.assertEqual(len(calls), 4)
                self.assertEqual(preprocess.call_count, 4)
                self.assertGreater(len(judgments), 0)
                for _, settings in calls:
                    self.assertEqual(settings['max_tokens'], 16384)
                    self.assertFalse(settings['thinking'])
                self.assertTrue(all(settings['max_tokens'] == 96 for _, settings in judgments))
                self.assertTrue(all(prompt.endswith(EMPTY_THINK) for prompt, _ in judgments))
                for bench, records in expected.items():
                    data_root = Path(prepared['datasets'][bench]['root'])
                    stored = read_json(data_root / 'records.json')
                    self.assertEqual([r['query'] for r in stored], [r['query'] for r in records])
                    self.assertEqual([r['response'] for r in stored], [r['response'] for r in records])
                    score = read_json(root / 'outputs/synthetic-model/synthetic' / bench / 'score.json')
                    self.assertEqual(score['status'], 'complete')
                    self.assertEqual(score['macro'], 100)


if __name__ == '__main__':
    unittest.main()
