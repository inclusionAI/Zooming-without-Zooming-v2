import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from zwz_eval.io import atomic_json, read_json, read_journal
from zwz_eval.protocols import PROTOCOLS
from zwz_eval.references import load
from zwz_eval.runner import run
from zwz_eval.structured import infer, messages, score, review_spar, task_names, validate_set
from test_adapters import configuration
from test_spatial_pipeline import Client


def verdict(score=1, rationale='Matches reference.'):
    return json.dumps({'scores': [{'criterion_id': 'criterion_1', 'extracted_answer': '5',
                                  'score': score, 'rationale': rationale}]})


class StructuredPipeline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.new('RGB', (8, 8), 'white').save(self.root / 'chart.png')
        self.config = configuration(self.root)
        self.config['workers'] = 1
        self.chart = {'id': 'chart-1', 'task_id': 'chart-1', 'prompt': 'Original question?',
                      'golden_answer': '5', 'images': ['chart.png']}

    def test_chart_exact_payload_and_contradiction_retry(self):
        atomic_json(self.root / 'records.json', [self.chart])
        main = Client(['  5  '])
        infer(self.config, 'chartography', self.root, main)
        request, generation = main.requests[0]
        self.assertEqual([p['type'] for p in request[0]['content']], ['text', 'image_url'])
        self.assertEqual(request[0]['content'][0]['text'], self.chart['prompt'])
        self.assertEqual(generation['max_tokens'], 16384)
        self.assertFalse(generation['enable_thinking'])
        self.assertFalse(generation['add_generation_parameters']['enable_thinking'])
        prediction = read_journal(self.root / 'predictions.jsonl')[0]
        self.assertEqual(prediction['raw_response'], '  5  ')
        self.assertEqual(prediction['model_answer'], '5')
        judge = Client([verdict(1, 'does not match'), verdict()])
        result = score(self.config, 'chartography', self.root, judge)
        self.assertEqual(result['primary_accuracy'], 1)
        self.assertEqual(len(judge.requests), 2)
        self.assertEqual(judge.requests[0][1]['max_tokens'], 8192)
        self.assertEqual(judge.requests[0][1]['response_format'], {'type': 'json_object'})
        self.assertEqual(judge.requests[0][1]['guided_json'], load('chartography').JUDGE_GUIDED_JSON)
        self.assertTrue(all(isinstance(p['content'], str) for p in judge.requests[0][0]))
        score(self.config, 'chartography', self.root, Client([]))

    def test_chart_judge_failure_keeps_raw_and_can_resume(self):
        atomic_json(self.root / 'records.json', [self.chart])
        infer(self.config, 'chartography', self.root, Client(['5']))
        with self.assertRaises(RuntimeError):
            score(self.config, 'chartography', self.root, Client(['bad JSON']*5))
        self.assertFalse((self.root / 'scores/metrics.json').exists())
        record = read_journal(self.root / 'scores/chart_judgments.jsonl')[0]
        self.assertEqual(record['judge_attempts'], ['bad JSON']*5)
        self.assertEqual(score(self.config, 'chartography', self.root, Client([verdict()]))['primary_accuracy'], 1)

    def test_chart_empty_response_no_llm_sampling(self):
        atomic_json(self.root / 'records.json', [self.chart])
        infer(self.config, 'chartography', self.root, Client(['']))
        self.assertEqual(score(self.config, 'chartography', self.root, Client([]))['primary_accuracy'], 0)

    def test_spar_macro_is_not_micro_or_reviewed_score(self):
        ref = load('spar')
        records, answers = [], []
        for index, task in enumerate(sorted(task_names())):
            gold = 'B' if task in ref.MCA else '10' if task in ref.NA else 'move_right:1'
            records.append({'id': index, 'question': 'Original question.', 'task': task,
                            'answer': gold, 'images': ['chart.png']})
            answers.append(gold)
        task = sorted(ref.MCA)[0]
        records.append({'id': 20, 'question': 'Extra instance.', 'task': task,
                        'answer': 'B', 'images': ['chart.png']})
        answers.append('A')
        atomic_json(self.root / 'records.json', records)
        main = Client(answers)
        infer(self.config, 'spar', self.root, main)
        metric = score(self.config, 'spar', self.root)
        self.assertAlmostEqual(metric['primary_accuracy'], .975)
        self.assertAlmostEqual(metric['official_micro'], 20/21)
        request, generation = main.requests[0]
        self.assertEqual(request[0]['content'], ref.ANSWER_FORMAT_SYSTEM)
        self.assertEqual(request[1]['content'][0]['text'], ref.official_prompt(records[0]))
        self.assertEqual(generation['top_p'], 1)
        self.assertEqual(generation['max_tokens'], 2048)
        judge = Client(['{"verdict":"correct","recovered_answer":"B","reason":"Equivalent."}'])
        audit = review_spar(self.config, self.root, judge)
        self.assertEqual(audit['reviewed'], 1)
        self.assertEqual(audit['reviewed_micro'], 1)
        self.assertEqual(judge.requests[0][1]['max_tokens'], 160)
        self.assertFalse(judge.requests[0][1]['thinking'])
        self.assertEqual(read_json(self.root / 'scores/metrics.json')['primary_accuracy'], .975)
        review_spar(self.config, self.root, Client([]))

    def test_spar_fractional_score_is_not_rounded_to_zero_or_one(self):
        record = {'id': 1, 'task': 'depth_prediction_oc', 'question': 'Distance?',
                  'answer': '10', 'images': ['chart.png']}
        atomic_json(self.root / 'records.json', [record])
        infer(self.config, 'spar', self.root, Client(['7']))
        value = score(self.config, 'spar', self.root)['primary_accuracy']
        self.assertGreater(value, 0)
        self.assertLess(value, 1)

    def test_full_chart_framework_and_resume(self):
        atomic_json(self.root / 'records.json', [{**self.chart, 'id': i, 'task_id': i} for i in range(100)])
        calls = []
        class Stub:
            def __init__(self, *args):
                pass
            def health(self):
                return True
            def complete(self, request, **kwargs):
                calls.append(kwargs)
                text = verdict() if kwargs['max_tokens'] == 8192 else '5'
                return {'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}]}
        with patch('zwz_eval.runner.CompatibleAPI', Stub), patch('zwz_eval.structured.CompatibleAPI', Stub), patch('zwz_eval.spatial.CompatibleAPI', Stub):
            directory = run(self.config, 'chartography', 'all')
            self.assertEqual(read_json(directory / 'score.json')['macro'], 100)
            self.assertEqual(len(calls), 200)
            run(self.config, 'chartography', 'all')
            self.assertEqual(len(calls), 200)

    def test_fixed_denominators_and_task_set(self):
        with self.assertRaises(ValueError):
            validate_set([self.chart], 'chartography')
        self.assertEqual(PROTOCOLS['spar']['max_tokens'], 2048)
        self.assertEqual(len(task_names()), 20)


if __name__ == '__main__':
    unittest.main()
