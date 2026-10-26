import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from zwz_eval.erqa import infer, score, messages, validate_row, review_request
from zwz_eval.io import atomic_json, read_json, read_journal
from zwz_eval.runner import run
from test_adapters import configuration
from test_spatial_pipeline import Client


def audit_verdict(value):
    return json.dumps({'verdict': value, 'selected': None, 'reason': 'Fixture'})


class ERQAPipeline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.new('RGB', (4, 4), 'white').save(self.root / 'frame.png')
        self.config = configuration(self.root)
        self.config['workers'] = 1
        self.row = {'id': 0, 'question': 'ABCDEfghijK', 'gold': 'A',
                    'images': ['frame.png'], 'visual_indices': [], 'question_type': 'fixture'}

    def test_reference_interleaving_and_validation(self):
        row = {**self.row, 'images': ['frame.png']*3, 'visual_indices': [0, 5, 9]}
        content = messages(row, 'erqa', self.root)[0]['content']
        self.assertEqual([p['type'] for p in content], ['image_url', 'text', 'image_url', 'text', 'image_url', 'text'])
        self.assertEqual([p['text'] for p in content if p['type'] == 'text'], ['ABCDE', 'fghi', 'jK'])
        for indices in ([], [0, 0, 0]):
            content = messages({**row, 'visual_indices': indices}, 'erqa', self.root)[0]['content']
            self.assertEqual([p['type'] for p in content], ['image_url']*3+['text'])
            self.assertEqual(content[-1]['text'], row['question'])
        content = messages({**self.row, 'visual_indices': [99]}, 'erqa', self.root)[0]['content']
        self.assertEqual([p['type'] for p in content], ['image_url', 'text'])
        self.assertEqual(content[-1]['text'], self.row['question'])
        with self.assertRaises(ValueError):
            validate_row({**row, 'visual_indices': [0]})

    def test_two_direction_audit_with_semantic_ambiguity(self):
        records = [{**self.row, 'id': i} for i in range(6)]
        atomic_json(self.root / 'records.json', records)
        infer(self.config, self.root, Client(['A', 'B', 'A', 'B', 'A', 'B']))
        judge = Client([audit_verdict(v) for v in ('incorrect', 'correct', 'ambiguous', 'ambiguous', 'correct', 'incorrect')])
        metrics = score(self.config, self.root, judge)
        self.assertEqual(metrics['initial_correct'], 3)
        self.assertEqual(metrics['added'], 1)
        self.assertEqual(metrics['subtracted'], 1)
        self.assertEqual(metrics['ambiguous'], 2)
        self.assertEqual(metrics['primary_accuracy'], .5)
        self.assertEqual([r[1]['max_tokens'] for r in judge.requests], [160, 1024, 160, 1024, 160, 1024])
        self.assertTrue(all(r[1]['thinking'] is False for r in judge.requests))
        self.assertTrue(all(all(isinstance(m['content'], str) for m in r[0]) for r in judge.requests))
        score(self.config, self.root, Client([]))

    def test_unparseable_judge_is_not_semantic_ambiguity(self):
        atomic_json(self.root / 'records.json', [self.row])
        infer(self.config, self.root, Client(['A']))
        with self.assertRaises(RuntimeError):
            score(self.config, self.root, Client(['bad']*3))
        self.assertFalse((self.root / 'scores/metrics.json').exists())
        failed = read_journal(self.root / 'scores/erqa_reviews.jsonl')[0]
        self.assertEqual(failed['status'], 'error')
        self.assertEqual(failed['judge_attempts'], ['bad']*3)
        self.assertEqual(score(self.config, self.root, Client([audit_verdict('ambiguous')]))['primary_accuracy'], 1)

    def test_literal_wrong_review_delimiters_and_initial_newlines(self):
        wrong = review_request(self.row, 'B', False)
        right = review_request(self.row, 'A', True)
        self.assertIn('\\n\\nQUESTION:\\n', wrong['messages'][0]['content'])
        self.assertIn('\n\nQUESTION:\n', right['messages'][0]['content'])
        self.assertEqual(wrong['max_tokens'], 1024)
        self.assertEqual(right['max_tokens'], 160)
        # SHA-256 of prompts rendered from the archived source AST on this fixture.
        row = {'question': 'question', 'gold': 'A'}
        for initial, expected in [(False, '9a54e05cdea39007069edce3bf4f8f1bd10c5460cb408269d2eab9f0c236476d'),
                                  (True, '40f2bbb731a0084d1faa52a04324178ba58b8267fd7d04ad730222dd7011de30')]:
            prompt = review_request(row, 'prediction', initial)['messages'][0]['content']
            self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(), expected)

    def test_full_400_framework_and_resume(self):
        atomic_json(self.root / 'records.json', [{**self.row, 'id': i} for i in range(400)])
        calls = []
        class Stub:
            def __init__(self, *args):
                pass
            def health(self):
                return True
            def complete(self, request, **kwargs):
                calls.append(kwargs)
                text = 'A' if kwargs['max_tokens'] == 4096 else audit_verdict('correct')
                return {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}
        with patch('zwz_eval.runner.CompatibleAPI', Stub), patch('zwz_eval.erqa.CompatibleAPI', Stub), patch('zwz_eval.spatial.CompatibleAPI', Stub):
            directory = run(self.config, 'erqa', 'all')
            self.assertEqual(read_json(directory / 'score.json')['macro'], 100)
            self.assertEqual(len(calls), 800)
            run(self.config, 'erqa', 'all')
            self.assertEqual(len(calls), 800)


if __name__ == '__main__':
    unittest.main()
